from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS, sha256_payload
from rl_quant.training.hold30_coordinator import (
    HOLD30_MAX_UPDATES,
    Hold30CheckpointReference,
    Hold30CohortIdentity,
    Hold30CoordinationError,
    Hold30ValidationRecord,
    Hold30ValidationScore,
    coordinate_hold30_seed_cohort,
    select_hold30_shared_checkpoint,
    verify_hold30_cohort_receipt,
)


def _digest(value: object) -> str:
    return sha256_payload(value)


def _identity() -> Hold30CohortIdentity:
    return Hold30CohortIdentity(
        setting_id="hold30-m02-age-hazard",
        fold_index=0,
        executable_manifest_sha256=_digest("manifest"),
        fold_sha256=_digest("fold"),
        inner_validation_sequence_sha256=_digest("validation"),
    )


def _refs(update: int) -> tuple[Hold30CheckpointReference, ...]:
    return tuple(
        Hold30CheckpointReference(
            seed=seed,
            update=update,
            checkpoint_id=f"seed-{seed}-update-{update:03d}",
            checkpoint_sha256=_digest(("checkpoint", seed, update)),
            checkpoint_receipt_sha256=_digest(("receipt", seed, update)),
        )
        for seed in HOLD30_SEEDS
    )


def _score(update: int, wealth: float, turnover: float = 0.03) -> Hold30ValidationScore:
    return Hold30ValidationScore(
        update=update,
        active_log_wealth=wealth,
        discretionary_turnover=turnover,
        trace_sha256=_digest(("trace", update)),
        inner_validation_sequence_sha256=_identity().inner_validation_sequence_sha256,
    )


def test_coordinator_advances_exact_five_seed_updates_and_stops_on_patience() -> None:
    calls: list[tuple[str, int]] = []
    wealth = {8: 0.0010, 16: 0.0012, 24: 0.0011, 32: 0.0011, 40: 0.0011, 48: 0.0011}

    def advance(update: int):
        calls.append(("advance", update))
        return _refs(update)

    def validate(update: int, checkpoints):
        calls.append(("validate", update))
        assert checkpoints == _refs(update)
        return _score(update, wealth[update])

    outcome = coordinate_hold30_seed_cohort(
        _identity(),
        _refs(0),
        advance_cohort=advance,
        validate_ensemble=validate,
    )

    assert [row.update for row in outcome.validations] == [8, 16, 24, 32, 40, 48]
    assert calls == [
        item
        for update in (8, 16, 24, 32, 40, 48)
        for item in (("advance", update), ("validate", update))
    ]
    assert outcome.stop_reason == "validation_patience_exhausted"
    assert outcome.stopped_update == 48
    assert outcome.selected_validation.update == 32
    receipt = outcome.receipt()
    claimed = receipt["receipt_sha256"]
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    assert claimed == sha256_payload(unsigned)
    assert receipt["outer_access"] is False
    assert receipt["scientific_qualification"] is False
    assert verify_hold30_cohort_receipt(receipt) == outcome


def test_selection_is_earliest_checkpoint_within_one_basis_point_of_maximum() -> None:
    identity = _identity()
    rows = tuple(
        Hold30ValidationRecord(_score(update, value), _refs(update))
        for update, value in (
            (8, 0.00900),
            (16, 0.00800),
            (24, 0.00700),
            (32, 0.00100),
            (40, 0.00109),
            (48, 0.00111),
        )
    )
    # Pre-minimum checkpoints are never deployable candidates. Update 32 is
    # 1.1 bp below the post-minimum maximum and is ineligible; update 40 is
    # 0.2 bp below it and wins by the frozen earliest rule.
    assert select_hold30_shared_checkpoint(rows, identity).update == 40


def test_monotone_validation_runs_to_the_128_update_ceiling() -> None:
    outcome = coordinate_hold30_seed_cohort(
        _identity(),
        _refs(0),
        advance_cohort=_refs,
        validate_ensemble=lambda update, _refs_: _score(update, update / 1_000_000.0),
    )
    assert outcome.stop_reason == "maximum_updates"
    assert outcome.stopped_update == HOLD30_MAX_UPDATES
    assert outcome.selected_validation.update == 32


def test_coordinator_rejects_per_seed_checkpoint_search_and_outer_evidence() -> None:
    mixed = list(_refs(8))
    mixed[-1] = replace(mixed[-1], update=16)
    with pytest.raises(Hold30CoordinationError, match="share one update"):
        coordinate_hold30_seed_cohort(
            _identity(),
            _refs(0),
            advance_cohort=lambda _update: mixed,
            validate_ensemble=lambda update, _refs_: _score(update, 0.0),
        )

    with pytest.raises(ValueError, match="outer-score access"):
        replace(_score(8, 0.0), outer_access=True)


def test_coordinator_rejects_wrong_seed_order_or_unbound_validation() -> None:
    reversed_refs = tuple(reversed(_refs(0)))
    with pytest.raises(Hold30CoordinationError, match="ordered seeds"):
        coordinate_hold30_seed_cohort(
            _identity(),
            reversed_refs,
            advance_cohort=_refs,
            validate_ensemble=lambda update, _refs_: _score(update, 0.0),
        )

    with pytest.raises(Hold30CoordinationError, match="unbound sequence"):
        coordinate_hold30_seed_cohort(
            _identity(),
            _refs(0),
            advance_cohort=_refs,
            validate_ensemble=lambda update, _refs_: replace(
                _score(update, 0.0), inner_validation_sequence_sha256=_digest("other")
            ),
        )


def test_validation_prefix_must_be_contiguous() -> None:
    identity = _identity()
    rows = (
        Hold30ValidationRecord(_score(8, 0.0), _refs(8)),
        Hold30ValidationRecord(_score(24, 0.1), _refs(24)),
    )
    with pytest.raises(Hold30CoordinationError, match="exact cadence prefix"):
        select_hold30_shared_checkpoint(rows, identity)


def test_cohort_receipt_rejects_tampering_or_unknown_fields() -> None:
    outcome = coordinate_hold30_seed_cohort(
        _identity(),
        _refs(0),
        advance_cohort=_refs,
        validate_ensemble=lambda update, _refs_: _score(update, 0.0),
    )
    tampered = outcome.receipt()
    tampered["selected_update"] = 16
    with pytest.raises(Hold30CoordinationError, match="self-hash"):
        verify_hold30_cohort_receipt(tampered)

    unknown = outcome.receipt()
    unknown["outer_score"] = 1.0
    with pytest.raises(Hold30CoordinationError, match="unknown fields"):
        verify_hold30_cohort_receipt(unknown)


def test_outcome_rejects_an_interruption_or_training_past_patience() -> None:
    identity = _identity()
    interrupted = tuple(
        Hold30ValidationRecord(_score(update, float(update)), _refs(update))
        for update in (8, 16, 24, 32)
    )
    with pytest.raises(Hold30CoordinationError, match="interruption"):
        # A caller cannot relabel a resumable prefix as a terminal outcome.
        from rl_quant.training.hold30_coordinator import Hold30CohortOutcome

        Hold30CohortOutcome(
            identity,
            _refs(0),
            interrupted,
            interrupted[-1],
            interrupted[-1].checkpoints,
            32,
            "validation_patience_exhausted",
        )

    overrun = tuple(
        Hold30ValidationRecord(_score(update, 1.0 if update == 8 else 0.0), _refs(update))
        for update in (8, 16, 24, 32, 40, 48)
    )
    with pytest.raises(Hold30CoordinationError, match="continued after"):
        select = overrun[3]
        Hold30CohortOutcome(
            identity,
            _refs(0),
            overrun,
            select,
            overrun[-1].checkpoints,
            48,
            "validation_patience_exhausted",
        )
