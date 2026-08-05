from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import torch

import rl_quant.datasets.hold30_qualification as qualification
from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetError,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.datasets.hold30_null_rebuild import _rebuild_c1
from rl_quant.protocol.hold30_freeze import Hold30Fold


DAY_MS = 86_400_000
HOUR_MS = 3_600_000
POSITIONS = 95
ASSETS = 301


def _digest(character: str) -> str:
    return character * 64


def _provenance() -> Hold30PointInTimeProvenance:
    return Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("a"),
        raw_market_data_sha256=_digest("b"),
        universe_events_sha256=_digest("c"),
        tradability_events_sha256=_digest("d"),
        corporate_actions_sha256=_digest("e"),
        identifier_events_sha256=_digest("f"),
        c1_benchmark_trace_sha256=_digest("1"),
        risk_limits_sha256=_digest("2"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="pit-active300-monthly-v1",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )


def _monthly_schedule(decision_timestamps_ms: torch.Tensor) -> torch.Tensor:
    schedule = torch.zeros(decision_timestamps_ms.numel(), dtype=torch.bool)
    previous = None
    for index, timestamp in enumerate(decision_timestamps_ms.tolist()):
        value = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        month = (value.year, value.month)
        if month != previous:
            schedule[index] = True
            previous = month
    return schedule


def _sequence() -> tuple[Hold30DatasetSequence, torch.Tensor]:
    dtype = torch.float64
    first_decision = 1_735_776_000_000
    decision_ts = first_decision + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill_ts = decision_ts - 6 * HOUR_MS
    fill_ts[0] = decision_ts[0] - HOUR_MS
    shape = (POSITIONS, 1, ASSETS)
    membership = torch.ones(shape, dtype=torch.bool)
    tradability = torch.ones(shape, dtype=torch.bool)
    state = torch.zeros((*shape, 1), dtype=dtype)

    returns = torch.zeros((POSITIONS - 1, 1, ASSETS), dtype=dtype)
    returns[..., 0] = 0.0001
    returns[..., 1:] = torch.arange(1, ASSETS, dtype=dtype) * 1e-7
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    mandatory[30, 0, 1] = True
    returns[30, 0, 1] = -0.10
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False
    ordinary &= ~mandatory

    caps = torch.full(shape, 0.01, dtype=dtype)
    caps[..., 0] = 1.0
    gross = torch.ones((POSITIONS, 1), dtype=dtype)
    costs = torch.full((POSITIONS - 1, 1), 0.002, dtype=dtype)
    stale_c1 = torch.zeros(shape, dtype=dtype)
    stale_c1[..., 0] = 1.0
    stale_net = returns[..., 0].clone()

    decision_known = decision_ts.view(-1, 1, 1).expand(shape).clone()
    fill_known = fill_ts.view(-1, 1, 1).expand(shape).clone()
    corporate_factor = torch.ones(shape, dtype=dtype)
    corporate_factor[31:, 0, 1] = 0.9
    corporate_version = torch.zeros(shape, dtype=torch.int64)
    corporate_version[31:, 0, 1] = 1
    corporate_known = torch.full(shape, -1, dtype=torch.int64)
    corporate_known[31:, 0, 1] = decision_ts[31]
    identifier_version = torch.zeros(shape, dtype=torch.int64)
    identifier_known = torch.full(shape, -1, dtype=torch.int64)
    evidence = Hold30AsOfEvidence(
        decision_membership_known_at_ms=decision_known.clone(),
        decision_tradability_known_at_ms=decision_known.clone(),
        fill_membership_known_at_ms=fill_known.clone(),
        fill_tradability_known_at_ms=fill_known.clone(),
        corporate_action_factor=corporate_factor,
        corporate_action_version=corporate_version,
        corporate_action_known_at_ms=corporate_known,
        identifier_version=identifier_version,
        identifier_known_at_ms=identifier_known,
    )
    initial = Hold30DatasetSequence(
        decision_timestamps_ms=decision_ts,
        fill_timestamps_ms=fill_ts,
        asset_ids=("CASH", *(f"PERM-{index:03d}" for index in range(1, ASSETS))),
        decision_state=state,
        decision_membership=membership.clone(),
        decision_tradability=tradability.clone(),
        fill_membership=membership.clone(),
        fill_tradability=tradability.clone(),
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=stale_c1,
        c1_benchmark_net_returns=stale_net,
        risk_asset_caps=caps,
        risk_gross_max=gross,
        cost_rate=costs,
        asof_evidence=evidence,
        provenance=_provenance(),
    )
    monthly = _monthly_schedule(decision_ts)
    c1 = _rebuild_c1(initial, initial.asset_returns, monthly)
    provenance = replace(
        initial.provenance,
        c1_benchmark_trace_sha256=c1.trace_sha256,
    )
    fresh = replace(
        initial,
        c1_benchmark_weights=c1.weights,
        c1_benchmark_net_returns=c1.net_returns,
        provenance=provenance,
    )
    return fresh, monthly


def _external_artifacts(sequence: Hold30DatasetSequence) -> dict[str, str]:
    return {
        path: getattr(sequence.provenance, field)
        for path, field in qualification.HOLD30_REQUIRED_EXTERNAL_ARTIFACTS.items()
    }


def _fake_folds(axis: tuple[str, ...]) -> tuple[Hold30Fold, ...]:
    end = len(axis)
    return tuple(
        Hold30Fold(
            fold_index=index,
            expanding_train=(0, 64),
            train_validation_separation=(64, 65),
            inner_validation=(65, 66),
            validation_support=(66, 67),
            outer_score=(67, 68),
            outer_support=(68, end if index == 5 else 69),
            embargo=(end if index == 5 else 69, end if index == 5 else 70),
            training_warmup=(0, 63),
            training_anchors=(63, 64),
            training_support=(64, 94),
            training_terminal_observation=94,
        )
        for index in range(6)
    )


@pytest.fixture
def small_qualified_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Production constants/functions remain unchanged. This replaces only the
    # expensive axis floor and renderer inside these in-memory unit fixtures.
    monkeypatch.setattr(qualification, "HOLD30_MIN_AXIS_POSITIONS", POSITIONS)
    monkeypatch.setattr(qualification, "render_hold30_folds", _fake_folds)


def test_qualification_receipt_is_deterministic_self_hashed_and_non_authorizing(
    small_qualified_geometry: None,
) -> None:
    del small_qualified_geometry
    sequence, monthly = _sequence()
    artifacts = _external_artifacts(sequence)
    first = qualification.qualify_hold30_dataset(
        sequence,
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
    )
    second = qualification.qualify_hold30_dataset(
        sequence,
        monthly_rebalance=monthly,
        external_artifacts=artifacts,
    )

    assert first == second
    assert first["passed"] is True
    assert first["launch_authorized"] is False
    assert first["scientific_qualification"] is False
    assert first["economic_contract"] == {
        "primary_cost_rate": 0.002,
        "primary_cost_basis_points": 20,
        "turnover_basis": "executed_one_way",
    }
    assert "exact_primary_20bp_cost" in first["checks"]
    assert first["counts"] == {
        "positions": POSITIONS,
        "batches": 1,
        "assets": ASSETS,
        "active_risky": 300,
        "folds": 6,
    }
    assert len(first["receipt_sha256"]) == 64
    qualification.verify_hold30_data_qualification_receipt(first)
    qualification.verify_hold30_dataset_against_qualification(
        sequence,
        monthly,
        artifacts,
        first,
    )


def test_production_minimum_rejects_synthetic_short_data() -> None:
    sequence, monthly = _sequence()
    with pytest.raises(Hold30DatasetError, match="N >= 1811"):
        qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=monthly,
            external_artifacts=_external_artifacts(sequence),
        )


def test_top2000_missing_hashes_and_stale_c1_fail_closed(
    small_qualified_geometry: None,
) -> None:
    del small_qualified_geometry
    sequence, monthly = _sequence()
    artifacts = _external_artifacts(sequence)

    top2000 = replace(
        sequence,
        provenance=replace(
            sequence.provenance,
            universe_rule_id="future-selected-top2000-static-universe",
        ),
    )
    with pytest.raises(Hold30DatasetError, match="TOP2000"):
        qualification.qualify_hold30_dataset(
            top2000,
            monthly_rebalance=monthly,
            external_artifacts=_external_artifacts(top2000),
        )

    missing = dict(artifacts)
    missing.pop("data/corporate-actions.parquet")
    with pytest.raises(Hold30DatasetError, match="missing"):
        qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=monthly,
            external_artifacts=missing,
        )
    unknown = dict(artifacts)
    unknown["data/local-TOP2000.npy"] = _digest("9")
    with pytest.raises(Hold30DatasetError, match="unknown"):
        qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=monthly,
            external_artifacts=unknown,
        )

    stale = replace(
        sequence,
        c1_benchmark_net_returns=sequence.c1_benchmark_net_returns + 1e-8,
    )
    with pytest.raises(Hold30DatasetError, match="C1 benchmark trace is stale"):
        qualification.qualify_hold30_dataset(
            stale,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
        )


def test_monthly_schedule_tensor_and_receipt_tampering_are_rejected(
    small_qualified_geometry: None,
) -> None:
    del small_qualified_geometry
    sequence, monthly = _sequence()
    artifacts = _external_artifacts(sequence)

    missing_month = monthly.clone()
    missing_month[torch.where(monthly)[0][1]] = False
    with pytest.raises(Hold30DatasetError, match="exactly one frozen event"):
        qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=missing_month,
            external_artifacts=artifacts,
        )

    # Mutating a receipt-bearing tensor after sequence construction cannot be
    # hidden behind the cached axis identity.
    sequence.fill_membership[10, 0, 1] = False
    with pytest.raises(Hold30DatasetError):
        qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=monthly,
            external_artifacts=artifacts,
        )

    clean, monthly = _sequence()
    receipt = qualification.qualify_hold30_dataset(
        clean,
        monthly_rebalance=monthly,
        external_artifacts=_external_artifacts(clean),
    )
    partial = dict(receipt)
    partial.pop("axis")
    with pytest.raises(Hold30DatasetError, match="partial or unknown"):
        qualification.verify_hold30_data_qualification_receipt(partial)

    unknown = dict(receipt)
    unknown["launch_override"] = True
    with pytest.raises(Hold30DatasetError, match="partial or unknown"):
        qualification.verify_hold30_data_qualification_receipt(unknown)

    changed = deepcopy(receipt)
    changed["counts"]["positions"] += 1
    with pytest.raises(Hold30DatasetError, match="self-hash"):
        qualification.verify_hold30_data_qualification_receipt(changed)


def test_exact_20bp_cost_and_live_receipt_binding_reject_all_mutations(
    small_qualified_geometry: None,
) -> None:
    del small_qualified_geometry

    wrong_cost, monthly = _sequence()
    wrong_cost.cost_rate[0, 0] = 0.001
    with pytest.raises(Hold30DatasetError, match=r"exactly 0.002 \(20 bp\)"):
        qualification.qualify_hold30_dataset(
            wrong_cost,
            monthly_rebalance=monthly,
            external_artifacts=_external_artifacts(wrong_cost),
        )

    def qualified():
        sequence, schedule = _sequence()
        artifacts = _external_artifacts(sequence)
        receipt = qualification.qualify_hold30_dataset(
            sequence,
            monthly_rebalance=schedule,
            external_artifacts=artifacts,
        )
        return sequence, schedule, artifacts, receipt

    returns, schedule, artifacts, receipt = qualified()
    returns.asset_returns[0, 0, 1] += 0.001
    with pytest.raises(Hold30DatasetError):
        qualification.verify_hold30_dataset_against_qualification(
            returns,
            schedule,
            artifacts,
            receipt,
        )

    membership, schedule, artifacts, receipt = qualified()
    membership.fill_membership[10, 0, 1] = False
    with pytest.raises(Hold30DatasetError):
        qualification.verify_hold30_dataset_against_qualification(
            membership,
            schedule,
            artifacts,
            receipt,
        )

    cost, schedule, artifacts, receipt = qualified()
    cost.cost_rate[0, 0] = 0.004
    with pytest.raises(Hold30DatasetError, match="exactly 0.002"):
        qualification.verify_hold30_dataset_against_qualification(
            cost,
            schedule,
            artifacts,
            receipt,
        )

    sequence, schedule, artifacts, receipt = qualified()
    changed_schedule = schedule.clone()
    changed_schedule[1] = True
    with pytest.raises(Hold30DatasetError, match="exactly one frozen event"):
        qualification.verify_hold30_dataset_against_qualification(
            sequence,
            changed_schedule,
            artifacts,
            receipt,
        )
