from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.hold30_random_controls import (
    HOLD30_C7_IDS,
    HOLD30_C8_CANDIDATE_IDS,
    HOLD30_RANDOM_BANK_SIZE,
    Hold30C8GateError,
    Hold30C8Profile,
    Hold30FoldRandomControlReceipts,
    Hold30PointInTimeAllocation,
    Hold30RandomBankEntry,
    Hold30RandomBankIdentity,
    Hold30RandomControlError,
    audit_hold30_random_trace,
    build_hold30_c7_receipt,
    build_hold30_c8_profile,
    build_hold30_cross_fold_control_mapping,
    build_hold30_random_bank_receipt,
    generate_hold30_random_intents,
    hold30_random_key,
    run_hold30_random_control,
    select_hold30_c8_controls,
    verify_hold30_c7_receipt,
    verify_hold30_c8_selection_receipt,
    verify_hold30_cross_fold_control_mapping,
    verify_hold30_random_bank_receipt,
    verify_hold30_random_intents,
)
from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import Hold30Sequence


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(tensor.shape),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _identity(
    mechanism: str,
    mask: torch.Tensor,
    *,
    fold: int = 0,
    classifications: tuple[str, ...] = (),
) -> Hold30RandomBankIdentity:
    variant = (
        "hold30-m01-slow-gate"
        if mechanism == "H1"
        else "hold30-m02-age-hazard"
    )
    decisions, batch, assets = mask.shape
    return Hold30RandomBankIdentity(
        HOLD30_PROTOCOL_GENERATION,
        variant,
        mechanism,  # type: ignore[arg-type]
        fold,
        "random-control-test-axis",
        decisions,
        batch,
        assets,
        0,
        "float64",
        _tensor_digest(mask),
        _digest("source"),
        _digest("generator"),
        _digest("builder"),
        _digest("constraints"),
        _digest("chronology"),
        classifications,  # type: ignore[arg-type]
        _digest("classification-availability"),
    )


def _sequence(positions: int = 95, assets: int = 101) -> Hold30Sequence:
    dtype = torch.float64
    batch = 1
    weights = torch.zeros((batch, assets), dtype=dtype)
    weights[:, 1:] = 1.0 / (assets - 1)
    state = torch.zeros((positions, batch, assets, 1), dtype=dtype)
    mask = torch.ones((positions, batch, assets), dtype=torch.bool)
    returns = torch.zeros((positions - 1, batch, assets), dtype=dtype)
    returns[:, :, 0] = 0.00002
    market = torch.linspace(-0.002, 0.002, positions - 1, dtype=dtype)
    returns[:, :, 1:] = market.view(-1, 1, 1)
    benchmark_return = (weights * returns).sum(-1)
    return Hold30Sequence(
        decision_state=state,
        asset_returns=returns,
        decision_available=mask.clone(),
        fill_membership=mask.clone(),
        fill_availability=mask.clone(),
        benchmark_weights=weights.unsqueeze(0).expand(positions, -1, -1).clone(),
        risk_asset_caps=torch.full(
            (positions, batch, assets),
            0.01,
            dtype=dtype,
        ),
        risk_gross_max=torch.ones((positions, batch), dtype=dtype),
        benchmark_net_returns=benchmark_return,
        initial_ledger=CohortLedger.from_staggered_endowment(
            weights,
            cash_index=0,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        cost_rate=0.002,
        track_entry_units=torch.ones(positions - 1, dtype=torch.bool),
        axis_id="random-control-test-axis",
    )


@pytest.mark.parametrize("mechanism", ["H1", "H2"])
def test_counter_intents_are_exact_return_blind_and_variant_fold_keyed(
    mechanism: str,
) -> None:
    mask = torch.ones((8, 1, 7), dtype=torch.bool)
    mask[3, 0, 6] = False
    identity = _identity(mechanism, mask)
    first = generate_hold30_random_intents(identity, 17, mask)
    second = generate_hold30_random_intents(identity, 17, mask)

    assert first.receipt_payload == second.receipt_payload
    verify_hold30_random_intents(identity, first, mask)
    assert first.rng_key_sha256 == hold30_random_key(identity, 17).hex()
    if mechanism == "H1":
        assert first.target_logits is not None
        assert first.gate is not None
        assert first.entry_scores is None
        assert first.target_logits[3, 0, 6] == 0.0
    else:
        assert first.entry_scores is not None
        assert first.hazard_residual is not None
        assert first.exposure_residual is not None
        assert first.hazard_residual[..., 0].eq(-12.0).all()
        assert first.entry_scores[3, 0, 6] == 0.0

    different_id = generate_hold30_random_intents(identity, 18, mask)
    assert different_id.receipt_sha256 != first.receipt_sha256
    fold_one = _identity(mechanism, mask, fold=1)
    assert hold30_random_key(fold_one, 17) != hold30_random_key(identity, 17)
    other_mechanism = _identity("H2" if mechanism == "H1" else "H1", mask)
    assert hold30_random_key(other_mechanism, 17) != hold30_random_key(identity, 17)
    with pytest.raises(Hold30RandomControlError, match="superseding v2"):
        replace(
            identity,
            protocol_generation="prelockbox-hold30-h0-h3-v1",
        )

    changed_mask = mask.clone()
    changed_mask[0, 0, 1] = False
    with pytest.raises(Hold30RandomControlError, match="decision mask"):
        generate_hold30_random_intents(identity, 17, changed_mask)


@pytest.mark.parametrize("mechanism", ["H1", "H2"])
def test_random_trace_uses_package_builder_and_is_hard_feasible(mechanism: str) -> None:
    sequence = _sequence()
    identity = _identity(mechanism, sequence.decision_available[:-1])
    roles = Hold30ReplayGeometry().roles(sequence.n_positions)

    run = run_hold30_random_control(identity, 9, sequence, roles)

    assert run.hard_feasibility_receipt["hard_feasible"] is True
    assert run.hard_feasibility_receipt["failures"] == []
    assert run.hard_feasibility_receipt["replicate_id"] == 9
    assert run.hard_feasibility_receipt["metric_scope"] == {
        "hard_audit_is_non_gating_for_holding_metrics": True,
        "holding_and_matching_gate_owner": "build_hold30_c8_profile",
        "score_and_holding_masks_required_there": True,
        "entry_tracking_mask_is_bound_above": True,
    }
    assert all(
        float(row.projection_distance.max()) <= 1e-6
        for row in run.trace.transitions
    )
    assert all(
        float(row.pre_cost_weights[:, 1:].max()) <= 0.010001
        for row in run.trace.transitions
    )
    profile = build_hold30_c8_profile(
        identity,
        run.trace,
        hard_feasibility_receipt_sha256=run.hard_feasibility_receipt[
            "receipt_sha256"
        ],
        return_covariance_receipt_sha256=_digest("profile-return-source"),
        classification_availability_receipt_sha256=(
            identity.classification_availability_receipt_sha256
        ),
        score_mask=torch.ones(identity.decision_count, dtype=torch.bool),
        holding_mask=torch.ones(identity.decision_count, dtype=torch.bool),
        bank_id=9,
    )
    assert profile.source_trace_sha256 == run.trace_sha256
    assert profile.beta is not None
    assert "mean_return" not in profile.payload["allowlisted_matching_fields"]

    tampered = list(run.trace.transitions)
    tampered[0] = replace(
        tampered[0],
        pre_cost_weights=tampered[0].pre_cost_weights.clone(),
    )
    tampered[0].pre_cost_weights[0, 1] += 0.1
    bad_trace = replace(run.trace, transitions=tuple(tampered))
    with pytest.raises(Hold30RandomControlError, match="filled delta"):
        audit_hold30_random_trace(identity, run.intent_trace, sequence, bad_trace)


def _entries(identity: Hold30RandomBankIdentity) -> tuple[Hold30RandomBankEntry, ...]:
    return tuple(
        Hold30RandomBankEntry(
            replicate_id,
            hold30_random_key(identity, replicate_id).hex(),
            _digest(f"intent-{replicate_id}"),
            _digest(f"trace-{replicate_id}"),
            _digest(f"feasible-{replicate_id}"),
        )
        for replicate_id in range(HOLD30_RANDOM_BANK_SIZE)
    )


def _profile(
    identity: Hold30RandomBankIdentity,
    *,
    bank_id: int | None,
    entry: Hold30RandomBankEntry | None = None,
    turnover: float = 0.03,
    exposure: float = 0.90,
    beta: float | None = 1.0,
    tracking_error: float | None = 0.10,
    hhi: float = 0.01,
    median: float | None = 30.0,
    survival: float | None = 0.50,
    sold: float = 1.0,
    age30: float = 0.20,
) -> Hold30C8Profile:
    allocations = (
        (
            Hold30PointInTimeAllocation(
                "sector",
                ("A", "B"),
                (0.40, 0.50),
                0.02,
                _digest("sector-source"),
            ),
        )
        if "sector" in identity.classification_dimensions
        else ()
    )
    return Hold30C8Profile(
        bank_identity_sha256=identity.receipt_sha256,
        source_trace_sha256=(
            _digest("learned-trace") if entry is None else entry.trace_sha256
        ),
        hard_feasibility_receipt_sha256=(
            _digest("learned-feasible")
            if entry is None
            else entry.hard_feasibility_receipt_sha256
        ),
        return_covariance_receipt_sha256=_digest("return-covariance"),
        classification_availability_receipt_sha256=_digest(
            "classification-availability"
        ),
        score_mask_sha256=_digest("score-mask"),
        holding_mask_sha256=_digest("holding-mask"),
        bank_id=bank_id,
        discretionary_turnover=turnover,
        risky_exposure=exposure,
        beta=beta,
        tracking_error=tracking_error,
        hhi=hhi,
        median_sale_age=median,
        survival_30=survival,
        sold_notional=sold,
        age30_mass_at_risk=age30,
        allocations=allocations,
    )


def test_complete_bank_c7_and_c8_receipts_are_deterministic_and_tamper_evident() -> None:
    mask = torch.ones((2, 1, 3), dtype=torch.bool)
    identity = _identity("H2", mask)
    entries = _entries(identity)
    bank = build_hold30_random_bank_receipt(identity, entries)
    c7 = build_hold30_c7_receipt(identity, entries, bank)

    assert len(bank["entries"]) == 8192
    assert c7["selected_ids"] == list(HOLD30_C7_IDS)
    verify_hold30_random_bank_receipt(identity, entries, bank)
    verify_hold30_c7_receipt(identity, entries, bank, c7)

    with pytest.raises(Hold30RandomControlError, match="0..8191"):
        build_hold30_random_bank_receipt(identity, entries[:-1])
    failed_entries = list(entries)
    failed_entries[0] = replace(
        failed_entries[0],
        hard_feasibility_failures=("per_name_cap",),
    )
    failed_bank = build_hold30_random_bank_receipt(identity, failed_entries)
    with pytest.raises(Hold30RandomControlError, match="hard-infeasible"):
        build_hold30_c7_receipt(identity, failed_entries, failed_bank)
    tampered = dict(bank)
    tampered["bank_size"] = 8191
    with pytest.raises(Hold30RandomControlError, match="hash mismatch"):
        verify_hold30_random_bank_receipt(identity, entries, tampered)


def test_c8_filter_keeps_all_reasons_and_ranks_distance_then_stable_id() -> None:
    mask = torch.ones((2, 1, 3), dtype=torch.bool)
    identity = _identity("H2", mask, classifications=("sector",))
    entries = _entries(identity)
    bank = build_hold30_random_bank_receipt(identity, entries)
    target = _profile(identity, bank_id=None)
    profiles = {
        entry.replicate_id: _profile(
            identity,
            bank_id=entry.replicate_id,
            entry=entry,
        )
        for entry in entries[64:]
    }
    profiles[64] = _profile(
        identity,
        bank_id=64,
        entry=entries[64],
        turnover=0.04,
        median=None,
        sold=0.01,
        age30=0.0,
    )
    profiles[64] = replace(
        profiles[64],
        allocations=(
            Hold30PointInTimeAllocation(
                "sector",
                ("A", "B"),
                (0.50, 0.50),
                0.02,
                _digest("sector-source"),
            ),
        ),
    )

    receipt = select_hold30_c8_controls(identity, target, entries, bank, profiles)

    assert receipt["selected_ids"] == list(range(65, 129))
    row64 = receipt["candidate_rows"][0]
    assert row64["rejected_reasons"] == [
        "estimability:sold_notional_below_0.10",
        "estimability:age30_mass_at_risk_nonpositive",
        "estimability:median_sale_age_undefined",
        "match:discretionary_turnover",
        "match:median_sale_age_undefined",
        "match:sector:A",
    ]
    assert row64["distance"] is None
    assert receipt["candidate_rows"][1]["distance_rank"] == 0
    verify_hold30_c8_selection_receipt(
        identity,
        target,
        entries,
        bank,
        profiles,
        receipt,
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["selected_ids"][0] = 130
    with pytest.raises(Hold30RandomControlError, match="hash mismatch"):
        verify_hold30_c8_selection_receipt(
            identity,
            target,
            entries,
            bank,
            profiles,
            tampered,
        )


def test_c8_fails_closed_for_target_or_fewer_than_64_feasible_controls() -> None:
    mask = torch.ones((2, 1, 3), dtype=torch.bool)
    identity = _identity("H1", mask)
    entries = _entries(identity)
    bank = build_hold30_random_bank_receipt(identity, entries)
    bad_target = _profile(identity, bank_id=None, sold=0.09, age30=0.0)
    with pytest.raises(Hold30C8GateError) as target_failure:
        select_hold30_c8_controls(identity, bad_target, entries, bank, {})
    assert target_failure.value.receipt["status"] == "target_precondition_failed"
    assert target_failure.value.receipt["candidate_rows"] == []

    target = _profile(identity, bank_id=None)
    profiles = {
        entry.replicate_id: _profile(
            identity,
            bank_id=entry.replicate_id,
            entry=entry,
        )
        for entry in entries[64:127]
    }
    with pytest.raises(Hold30C8GateError) as insufficient:
        select_hold30_c8_controls(identity, target, entries, bank, profiles)
    failure = insufficient.value.receipt
    assert failure["status"] == "insufficient_feasible_controls"
    assert failure["selected_count"] == 63
    assert len(failure["candidate_rows"]) == len(HOLD30_C8_CANDIDATE_IDS)
    assert failure["candidate_rows"][-1]["rejected_reasons"] == ["profile_missing"]
    verify_hold30_c8_selection_receipt(
        identity,
        target,
        entries,
        bank,
        profiles,
        failure,
    )


def _seal(payload: dict) -> dict:
    result = dict(payload)
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    result["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _fold_receipts(fold: int) -> Hold30FoldRandomControlReceipts:
    mask = torch.ones((2, 1, 3), dtype=torch.bool)
    identity = _identity("H2", mask, fold=fold)
    entries = _entries(identity)
    bank = build_hold30_random_bank_receipt(identity, entries)
    c7 = build_hold30_c7_receipt(identity, entries, bank)
    selected = list(range(64 + fold, 128 + fold))
    rows = [
        {
            "replicate_id": replicate_id,
            "distance_rank": (
                selected.index(replicate_id) if replicate_id in selected else None
            ),
            "selected": replicate_id in selected,
        }
        for replicate_id in HOLD30_C8_CANDIDATE_IDS
    ]
    c8 = _seal(
        {
            "schema": "rl-quant.hold30.c8-selection",
            "status": "passed",
            "bank_identity_sha256": identity.receipt_sha256,
            "bank_receipt_sha256": bank["receipt_sha256"],
            "selected_ids": selected,
            "candidate_rows": rows,
        }
    )
    return Hold30FoldRandomControlReceipts(identity, bank, c7, c8)


def test_cross_fold_mapping_freezes_c7_common_ids_and_c8_fold_ranks() -> None:
    folds = tuple(_fold_receipts(fold) for fold in range(6))
    receipt = build_hold30_cross_fold_control_mapping(folds)

    assert len(receipt["mapping"]) == 64 * 6 * 2
    c7_path_7 = [
        row
        for row in receipt["mapping"]
        if row["control_id"] == "C7" and row["aggregate_path"] == 7
    ]
    assert [row["bank_id"] for row in c7_path_7] == [7] * 6
    c8_path_7 = [
        row
        for row in receipt["mapping"]
        if row["control_id"] == "C8" and row["aggregate_path"] == 7
    ]
    assert [row["bank_id"] for row in c8_path_7] == [71, 72, 73, 74, 75, 76]
    assert [row["distance_rank"] for row in c8_path_7] == [7] * 6
    verify_hold30_cross_fold_control_mapping(folds, receipt)

    tampered = json.loads(json.dumps(receipt))
    tampered["mapping"][0]["bank_id"] = 999
    with pytest.raises(Hold30RandomControlError, match="hash mismatch"):
        verify_hold30_cross_fold_control_mapping(folds, tampered)
