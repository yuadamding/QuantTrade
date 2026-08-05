from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
import torch

import rl_quant.datasets.hold30_folds as fold_module
import rl_quant.datasets.hold30_qualification as qualification
from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetError,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.datasets.hold30_folds import (
    materialize_hold30_development_fold,
    materialize_hold30_outer_fold,
)
from rl_quant.datasets.hold30_null_rebuild import _rebuild_c1
from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS, Hold30Fold, sha256_payload
from rl_quant.training.hold30_coordinator import (
    Hold30CheckpointReference,
    Hold30CohortIdentity,
    Hold30ValidationScore,
    coordinate_hold30_seed_cohort,
)


POSITIONS = 400
ASSETS = 301
DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _digest(value: object) -> str:
    return sha256_payload(value)


def _fold(index: int = 0) -> Hold30Fold:
    return Hold30Fold(
        fold_index=index,
        expanding_train=(0, 160),
        train_validation_separation=(160, 191),
        inner_validation=(191, 254),
        validation_support=(254, 285),
        outer_score=(285, 348),
        outer_support=(348, 379),
        embargo=(379, 400),
        training_warmup=(0, 63),
        training_anchors=(63, 129),
        training_support=(129, 159),
        training_terminal_observation=159,
    )


def _fake_folds(_axis: tuple[str, ...]) -> tuple[Hold30Fold, ...]:
    return tuple(_fold(index) for index in range(6))


def _monthly(timestamps: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(timestamps.numel(), dtype=torch.bool)
    prior = None
    for index, raw in enumerate(timestamps.tolist()):
        date = datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
        month = (date.year, date.month)
        if month != prior:
            result[index] = True
            prior = month
    return result


def _sequence() -> tuple[Hold30DatasetSequence, torch.Tensor]:
    dtype = torch.float64
    first = 1_704_067_200_000
    decision_ts = first + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (POSITIONS, 1, ASSETS)
    membership = torch.ones(shape, dtype=torch.bool)
    tradability = torch.ones(shape, dtype=torch.bool)
    returns = torch.zeros((POSITIONS - 1, 1, ASSETS), dtype=dtype)
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False
    mandatory = torch.zeros_like(ordinary)
    caps = torch.full(shape, 0.01, dtype=dtype)
    caps[..., 0] = 1.0
    gross = torch.ones((POSITIONS, 1), dtype=dtype)
    costs = torch.full((POSITIONS - 1, 1), 0.002, dtype=dtype)
    c1_weights = torch.zeros(shape, dtype=dtype)
    c1_weights[..., 1:] = 1.0 / 300.0
    c1_net = torch.zeros((POSITIONS - 1, 1), dtype=dtype)
    known_decision = decision_ts.view(-1, 1, 1).expand(shape).clone()
    known_fill = fill_ts.view(-1, 1, 1).expand(shape).clone()
    versions = torch.zeros(shape, dtype=torch.int64)
    absent = torch.full(shape, -1, dtype=torch.int64)
    evidence = Hold30AsOfEvidence(
        decision_membership_known_at_ms=known_decision.clone(),
        decision_tradability_known_at_ms=known_decision.clone(),
        fill_membership_known_at_ms=known_fill.clone(),
        fill_tradability_known_at_ms=known_fill.clone(),
        corporate_action_factor=torch.ones(shape, dtype=dtype),
        corporate_action_version=versions.clone(),
        corporate_action_known_at_ms=absent.clone(),
        identifier_version=versions.clone(),
        identifier_known_at_ms=absent.clone(),
    )
    provenance = Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("data"),
        raw_market_data_sha256=_digest("raw"),
        universe_events_sha256=_digest("universe"),
        tradability_events_sha256=_digest("tradability"),
        corporate_actions_sha256=_digest("corporate"),
        identifier_events_sha256=_digest("identifier"),
        c1_benchmark_trace_sha256=_digest("stale-c1"),
        risk_limits_sha256=_digest("risk"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="pit-active300-monthly-fold-test-v1",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )
    initial = Hold30DatasetSequence(
        decision_timestamps_ms=decision_ts,
        fill_timestamps_ms=fill_ts,
        asset_ids=("CASH", *(f"PERM-{index:03d}" for index in range(1, ASSETS))),
        decision_state=torch.zeros((*shape, 1), dtype=dtype),
        decision_membership=membership.clone(),
        decision_tradability=tradability.clone(),
        fill_membership=membership.clone(),
        fill_tradability=tradability.clone(),
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=c1_weights,
        c1_benchmark_net_returns=c1_net,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=costs,
        asof_evidence=evidence,
        provenance=provenance,
    )
    schedule = _monthly(decision_ts)
    c1 = _rebuild_c1(initial, returns, schedule)
    sequence = replace(
        initial,
        c1_benchmark_weights=c1.weights,
        c1_benchmark_net_returns=c1.net_returns,
        provenance=replace(provenance, c1_benchmark_trace_sha256=c1.trace_sha256),
    )
    return sequence, schedule


def _artifacts(sequence: Hold30DatasetSequence) -> dict[str, str]:
    return {
        path: getattr(sequence.provenance, field)
        for path, field in qualification.HOLD30_REQUIRED_EXTERNAL_ARTIFACTS.items()
    }


@pytest.fixture
def qualified(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(qualification, "HOLD30_MIN_AXIS_POSITIONS", POSITIONS)
    monkeypatch.setattr(qualification, "render_hold30_folds", _fake_folds)
    monkeypatch.setattr(fold_module, "render_hold30_folds", _fake_folds)
    sequence, monthly = _sequence()
    receipt = qualification.qualify_hold30_dataset(
        sequence,
        monthly_rebalance=monthly,
        external_artifacts=_artifacts(sequence),
    )
    return sequence, monthly, _artifacts(sequence), receipt


def _refs(update: int) -> tuple[Hold30CheckpointReference, ...]:
    return tuple(
        Hold30CheckpointReference(
            seed,
            update,
            f"seed-{seed}-update-{update:03d}",
            _digest(("checkpoint", seed, update)),
            _digest(("checkpoint-receipt", seed, update)),
        )
        for seed in HOLD30_SEEDS
    )


def _cohort_receipt(development) -> dict[str, object]:
    identity = Hold30CohortIdentity(
        setting_id="hold30-m02-age-hazard",
        fold_index=development.fold.fold_index,
        executable_manifest_sha256=_digest("manifest"),
        fold_sha256=development.fold_sha256,
        inner_validation_sequence_sha256=development.inner_validation.axis_id,
    )
    outcome = coordinate_hold30_seed_cohort(
        identity,
        _refs(0),
        advance_cohort=_refs,
        validate_ensemble=lambda update, _refs_: Hold30ValidationScore(
            update,
            1.0 if update == 8 else 0.0,
            0.03,
            _digest(("trace", update)),
            development.inner_validation.axis_id,
        ),
    )
    return outcome.receipt()


def test_development_materialization_has_exact_roles_and_no_outer_sequence(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    development = materialize_hold30_development_fold(
        sequence,
        _fold(),
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
        data_qualification_receipt=receipt,
    )

    assert development.training_absolute_range == (0, 160)
    assert development.validation_absolute_range == (128, 285)
    assert development.outer_absolute_range == (222, 379)
    assert development.training.roles.score_indices.tolist() == list(range(63, 129))
    assert development.inner_validation.roles.score_indices.tolist() == list(range(63, 126))
    assert not hasattr(development, "outer")
    assert len(development.receipt_sha256) == 64


def test_outer_materialization_requires_terminal_shared_selection_and_marker(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    development = materialize_hold30_development_fold(
        sequence,
        _fold(),
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
        data_qualification_receipt=receipt,
    )
    cohort = _cohort_receipt(development)
    outer = materialize_hold30_outer_fold(
        sequence,
        development,
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
        data_qualification_receipt=receipt,
        cohort_selection_receipt=cohort,
        access_marker_sha256=_digest("access-marker"),
    )

    assert outer.absolute_range == (222, 379)
    assert outer.sequence.roles.score_indices.tolist() == list(range(63, 126))
    assert outer.sequence.roles.terminal.nonzero().item() == 156
    assert len(outer.receipt_sha256) == 64

    with pytest.raises(Hold30DatasetError, match="access_marker"):
        materialize_hold30_outer_fold(
            sequence,
            development,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
            data_qualification_receipt=receipt,
            cohort_selection_receipt=cohort,
            access_marker_sha256="bad",
        )


def test_outer_rejects_checkpoint_selection_from_another_validation_axis(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    development = materialize_hold30_development_fold(
        sequence,
        _fold(),
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
        data_qualification_receipt=receipt,
    )
    cohort = _cohort_receipt(development)
    changed = dict(cohort)
    changed_identity = dict(changed["identity"])
    changed_identity["inner_validation_sequence_sha256"] = _digest("other-validation")
    changed["identity"] = changed_identity
    changed["identity_sha256"] = sha256_payload(changed_identity)
    unsigned = dict(changed)
    unsigned.pop("receipt_sha256")
    changed["receipt_sha256"] = sha256_payload(unsigned)

    with pytest.raises(Hold30DatasetError, match="invalid checkpoint selection"):
        materialize_hold30_outer_fold(
            sequence,
            development,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
            data_qualification_receipt=receipt,
            cohort_selection_receipt=changed,
            access_marker_sha256=_digest("access-marker"),
        )


def test_development_rejects_a_fold_not_rendered_from_the_qualified_axis(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    altered = replace(_fold(), inner_validation=(190, 253))
    with pytest.raises(Hold30DatasetError, match="differs from the qualified"):
        materialize_hold30_development_fold(
            sequence,
            altered,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
            data_qualification_receipt=receipt,
        )


def test_outer_reconstructs_and_rejects_a_forged_development_range(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    development = materialize_hold30_development_fold(
        sequence,
        _fold(),
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
        data_qualification_receipt=receipt,
    )
    forged = replace(development, outer_absolute_range=(0, 157))
    with pytest.raises(Hold30DatasetError, match="tampered"):
        materialize_hold30_outer_fold(
            sequence,
            forged,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
            data_qualification_receipt=receipt,
            cohort_selection_receipt=_cohort_receipt(development),
            access_marker_sha256=_digest("access-marker"),
        )


def test_fold_materialization_rechecks_live_parent_tensors(qualified) -> None:
    sequence, monthly, artifacts, receipt = qualified
    sequence.asset_returns[0, 0, 1] = 0.001

    with pytest.raises(Hold30DatasetError, match="changed after|no longer matches|stale"):
        materialize_hold30_development_fold(
            sequence,
            _fold(),
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
            data_qualification_receipt=receipt,
        )
