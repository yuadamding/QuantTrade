from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import torch

import rl_quant.datasets.hold30_alpha_qualification as alpha_qualification
import rl_quant.datasets.hold30_folds as fold_module
import rl_quant.datasets.hold30_qualification as base_qualification
from rl_quant.datasets.hold30 import (
    HOLD30_CASH_RETURN_RULE,
    HOLD30_UNIVERSE_MODE,
    Hold30AsOfEvidence,
    Hold30DatasetSequence,
    Hold30PointInTimeProvenance,
)
from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaEvaluationPanel,
    Hold30AlphaEvaluationProvenance,
)
from rl_quant.datasets.hold30_alpha_qualification import (
    Hold30AlphaInMemoryFactorExposureDeclaration,
    Hold30AlphaInMemoryLineageDeclaration,
    Hold30AlphaStructuralQualificationError,
)
from rl_quant.datasets.hold30_null_rebuild import _rebuild_c1
from rl_quant.protocol.hold30_alpha_v3 import HOLD30_ALPHA_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import Hold30Fold, sha256_payload

POSITIONS = 200
ASSETS = 301
DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _digest(value: object) -> str:
    return sha256_payload(value)


def _fold(index: int) -> Hold30Fold:
    return Hold30Fold(
        fold_index=index,
        expanding_train=(0, 95),
        train_validation_separation=(95, 96),
        inner_validation=(96, 97),
        validation_support=(97, 128),
        outer_score=(128, 129),
        outer_support=(129, 160),
        embargo=(160, 161),
        training_warmup=(0, 63),
        training_anchors=(63, 64),
        training_support=(64, 94),
        training_terminal_observation=94,
    )


def _fake_folds(_axis: tuple[str, ...]) -> tuple[Hold30Fold, ...]:
    return tuple(_fold(index) for index in range(6))


def _monthly_schedule(timestamps: torch.Tensor) -> torch.Tensor:
    schedule = torch.zeros(timestamps.numel(), dtype=torch.bool)
    prior = None
    for index, raw in enumerate(timestamps.tolist()):
        date = datetime.fromtimestamp(raw / 1000, tz=UTC)
        month = (date.year, date.month)
        if month != prior:
            schedule[index] = True
            prior = month
    return schedule


def _sequence() -> tuple[Hold30DatasetSequence, torch.Tensor]:
    first = 1_735_776_000_000
    decision = first + torch.arange(POSITIONS, dtype=torch.int64) * DAY_MS
    fill = decision - 6 * HOUR_MS
    fill[0] = decision[0] - HOUR_MS
    shape = (POSITIONS, 1, ASSETS)
    membership = torch.ones(shape, dtype=torch.bool)
    returns = torch.zeros((POSITIONS - 1, 1, ASSETS), dtype=torch.float64)
    rows = torch.arange(POSITIONS - 1, dtype=torch.float64)
    assets = torch.arange(1, ASSETS, dtype=torch.float64)
    returns[:, 0, 0] = 0.00004 + 0.000005 * torch.sin(rows * 0.13)
    returns[:, 0, 1:] = (
        0.0002 * torch.sin(rows[:, None] * 0.09 + assets[None, :] * 0.03)
        + assets[None, :] * 1e-7
    )
    mandatory = torch.zeros_like(returns, dtype=torch.bool)
    ordinary = membership[:-1].clone()
    ordinary[..., 0] = False
    caps = torch.full(shape, 0.01, dtype=torch.float64)
    caps[..., 0] = 1.0
    stale_weights = torch.zeros(shape, dtype=torch.float64)
    stale_weights[..., 0] = 1.0
    decision_known = decision.view(-1, 1, 1).expand(shape).clone()
    fill_known = fill.view(-1, 1, 1).expand(shape).clone()
    zero_versions = torch.zeros(shape, dtype=torch.int64)
    no_event = torch.full(shape, -1, dtype=torch.int64)
    provenance = Hold30PointInTimeProvenance(
        data_snapshot_sha256=_digest("data-snapshot"),
        raw_market_data_sha256=_digest("raw-market"),
        universe_events_sha256=_digest("universe-events"),
        tradability_events_sha256=_digest("tradability-events"),
        corporate_actions_sha256=_digest("corporate-actions"),
        identifier_events_sha256=_digest("identifier-events"),
        c1_benchmark_trace_sha256=_digest("stale-c1"),
        risk_limits_sha256=_digest("risk-limits"),
        universe_mode=HOLD30_UNIVERSE_MODE,
        universe_rule_id="pit-active300-monthly-alpha-v3",
        stable_asset_id_namespace="perm-id-v1",
        benchmark_id="C1",
        cash_asset_id="CASH",
        cash_return_rule=HOLD30_CASH_RETURN_RULE,
    )
    initial = Hold30DatasetSequence(
        decision_timestamps_ms=decision,
        fill_timestamps_ms=fill,
        asset_ids=("CASH", *(f"PERM-{index:03d}" for index in range(1, ASSETS))),
        decision_state=torch.zeros((*shape, 1), dtype=torch.float64),
        decision_membership=membership.clone(),
        decision_tradability=membership.clone(),
        fill_membership=membership.clone(),
        fill_tradability=membership.clone(),
        asset_returns=returns,
        ordinary_return_valid=ordinary,
        mandatory_return_mask=mandatory,
        c1_benchmark_weights=stale_weights,
        c1_benchmark_net_returns=returns[..., 0].clone(),
        risk_asset_caps=caps,
        risk_gross_max=torch.ones((POSITIONS, 1), dtype=torch.float64),
        cost_rate=torch.full((POSITIONS - 1, 1), 0.002, dtype=torch.float64),
        asof_evidence=Hold30AsOfEvidence(
            decision_membership_known_at_ms=decision_known.clone(),
            decision_tradability_known_at_ms=decision_known.clone(),
            fill_membership_known_at_ms=fill_known.clone(),
            fill_tradability_known_at_ms=fill_known.clone(),
            corporate_action_factor=torch.ones(shape, dtype=torch.float64),
            corporate_action_version=zero_versions.clone(),
            corporate_action_known_at_ms=no_event.clone(),
            identifier_version=zero_versions.clone(),
            identifier_known_at_ms=no_event.clone(),
        ),
        provenance=provenance,
    )
    monthly = _monthly_schedule(decision)
    rebuilt = _rebuild_c1(initial, returns, monthly)
    return (
        replace(
            initial,
            c1_benchmark_weights=rebuilt.weights,
            c1_benchmark_net_returns=rebuilt.net_returns,
            provenance=replace(
                provenance,
                c1_benchmark_trace_sha256=rebuilt.trace_sha256,
            ),
        ),
        monthly,
    )


@dataclass(frozen=True)
class _Bundle:
    sequence: Hold30DatasetSequence
    monthly: torch.Tensor
    artifacts: dict[str, str]
    panel: Hold30AlphaEvaluationPanel
    lineage: Hold30AlphaInMemoryLineageDeclaration
    exposures: Hold30AlphaInMemoryFactorExposureDeclaration


@pytest.fixture(scope="module")
def bundle() -> _Bundle:
    sequence, monthly = _sequence()
    artifacts = {
        path: getattr(sequence.provenance, field)
        for path, field in base_qualification.HOLD30_REQUIRED_EXTERNAL_ARTIFACTS.items()
    }
    rows = torch.arange(POSITIONS - 1, dtype=torch.float64)
    provenance = Hold30AlphaEvaluationProvenance(
        risk_free_id="PIT-CASH-TOTAL-RETURN",
        market_benchmark_id="PIT-CAP-WEIGHT-MARKET",
        factor_model_id="PIT-EVALUATOR-FACTORS",
        factor_names=("SIZE", "VALUE"),
        factor_return_conventions=("zero-investment", "zero-investment"),
        risk_free_artifact_sha256=_digest("risk-free-artifact"),
        market_artifact_sha256=_digest("market-artifact"),
        factor_artifact_sha256=_digest("factor-return-artifact"),
        factor_plan_sha256=_digest("factor-plan"),
    )
    panel = Hold30AlphaEvaluationPanel(
        source_axis_id=sequence.axis_id,
        risk_free_returns=sequence.asset_returns[..., sequence.cash_index].clone(),
        risk_free_valid=torch.ones((POSITIONS - 1, 1), dtype=torch.bool),
        market_total_returns=(
            0.0003 + 0.003 * torch.sin(rows * 0.17)
        ).view(-1, 1),
        market_valid=torch.ones((POSITIONS - 1, 1), dtype=torch.bool),
        factor_returns=torch.stack(
            (
                0.001 * torch.sin(rows * 0.11),
                0.001 * torch.cos(rows * 0.07),
            ),
            dim=-1,
        ).view(POSITIONS - 1, 1, 2),
        factor_valid=torch.ones((POSITIONS - 1, 1, 2), dtype=torch.bool),
        provenance=provenance,
    )
    provider_receipt = _digest("provider-snapshot")
    lineage = Hold30AlphaInMemoryLineageDeclaration(
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        source_axis_id=sequence.axis_id,
        provider_id="approved-pit-provider",
        provider_snapshot_receipt_sha256=provider_receipt,
        data_snapshot_sha256=sequence.provenance.data_snapshot_sha256,
        raw_market_data_sha256=sequence.provenance.raw_market_data_sha256,
        universe_events_sha256=sequence.provenance.universe_events_sha256,
        tradability_events_sha256=sequence.provenance.tradability_events_sha256,
        corporate_actions_sha256=sequence.provenance.corporate_actions_sha256,
        identifier_events_sha256=sequence.provenance.identifier_events_sha256,
    )
    exposure_shape = (POSITIONS, 1, ASSETS, 2)
    exposure_values = torch.zeros(exposure_shape, dtype=torch.float64)
    coordinate = torch.linspace(-1.0, 1.0, ASSETS - 1, dtype=torch.float64)
    exposure_values[:, 0, 1:, 0] = coordinate
    exposure_values[:, 0, 1:, 1] = coordinate.square()
    exposure_valid = sequence.decision_membership.unsqueeze(-1).expand(
        exposure_shape
    ).clone()
    exposure_valid[..., sequence.cash_index, :] = False
    known_at = sequence.decision_timestamps_ms.view(-1, 1, 1, 1).expand(
        exposure_shape
    ).clone()
    known_at[..., sequence.cash_index, :] = -1
    exposures = Hold30AlphaInMemoryFactorExposureDeclaration(
        protocol_generation=HOLD30_ALPHA_PROTOCOL_GENERATION,
        source_axis_id=sequence.axis_id,
        factor_model_id=provenance.factor_model_id,
        factor_names=provenance.factor_names,
        values=exposure_values,
        valid=exposure_valid,
        known_at_ms=known_at,
        factor_returns_artifact_sha256=provenance.factor_artifact_sha256,
        factor_plan_sha256=provenance.factor_plan_sha256,
        exposure_artifact_sha256=_digest("factor-exposure-artifact"),
        provider_snapshot_receipt_sha256=provider_receipt,
    )
    return _Bundle(sequence, monthly, artifacts, panel, lineage, exposures)


def _qualify(
    monkeypatch: pytest.MonkeyPatch,
    bundle: _Bundle,
) -> dict[str, object]:
    monkeypatch.setattr(
        base_qualification,
        "HOLD30_MIN_AXIS_POSITIONS",
        POSITIONS,
    )
    monkeypatch.setattr(base_qualification, "render_hold30_folds", _fake_folds)
    monkeypatch.setattr(fold_module, "render_hold30_folds", _fake_folds)
    monkeypatch.setattr(
        alpha_qualification,
        "HOLD30_MIN_AXIS_POSITIONS",
        POSITIONS,
    )
    monkeypatch.setattr(alpha_qualification, "render_hold30_folds", _fake_folds)
    base_receipt = base_qualification.qualify_hold30_dataset(
        bundle.sequence,
        monthly_rebalance=bundle.monthly,
        external_artifacts=bundle.artifacts,
    )
    return alpha_qualification.qualify_hold30_alpha_in_memory_structure(
        bundle.sequence,
        monthly_rebalance=bundle.monthly,
        external_artifacts=bundle.artifacts,
        base_data_qualification_receipt=base_receipt,
        evaluation_panel=bundle.panel,
        lineage_declaration=bundle.lineage,
        factor_exposure_declaration=bundle.exposures,
    )


def test_in_memory_chain_is_structural_only_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
    bundle: _Bundle,
) -> None:
    receipt = _qualify(monkeypatch, bundle)

    assert receipt["alpha_horizons"] == [5, 21, 30, 63]
    assert receipt["counts"] == {
        "positions": POSITIONS,
        "assets": ASSETS,
        "factors": 2,
        "folds": 6,
        "role_chains": 12,
    }
    assert receipt["qualification_scope"] == "in_memory_structural_consistency_only"
    assert receipt["structural_consistency_verified"] is True
    assert receipt["real_data_attested"] is False
    assert receipt["data_qualification_complete"] is False
    assert receipt["file_hash_verification"] is False
    assert receipt["provider_trust_verified"] is False
    assert receipt["outer_values_present"] is True
    assert receipt["outer_access_boundary_enforced"] is False
    assert receipt["production_data_eligible"] is False
    assert receipt["production_preflight_acceptable"] is False
    assert receipt["launch_authorized"] is False
    assert receipt["scientific_qualification"] is False
    assert receipt["promotion_authorized"] is False
    assert receipt["actor_access"] is False
    for fold in receipt["fold_chains"]:
        assert fold["outer_sequence_materialized"] is False
        assert fold["training"]["role"] == "training"
        assert fold["inner_validation"]["role"] == "inner-validation"
        assert len(fold["training"]["objective_inputs_id"]) == 64
        assert len(fold["inner_validation"]["objective_inputs_id"]) == 64
    alpha_qualification.verify_hold30_alpha_in_memory_structural_receipt(
        receipt
    )
    with pytest.raises(
        Hold30AlphaStructuralQualificationError,
        match="never production data bindings",
    ):
        alpha_qualification.require_hold30_alpha_production_data_binding(receipt)

    tampered = deepcopy(receipt)
    tampered["launch_authorized"] = True
    with pytest.raises(
        Hold30AlphaStructuralQualificationError,
        match="authority",
    ):
        alpha_qualification.verify_hold30_alpha_in_memory_structural_receipt(
            tampered
        )


def test_production_floor_rejects_missing_n1811(bundle: _Bundle) -> None:
    with pytest.raises(
        Hold30AlphaStructuralQualificationError,
        match="N >= 1811",
    ):
        alpha_qualification.qualify_hold30_alpha_in_memory_structure(
            bundle.sequence,
            monthly_rebalance=bundle.monthly,
            external_artifacts=bundle.artifacts,
            base_data_qualification_receipt={},
            evaluation_panel=bundle.panel,
            lineage_declaration=bundle.lineage,
            factor_exposure_declaration=bundle.exposures,
        )


def test_generation_synthetic_future_and_actor_access_fail_closed(
    bundle: _Bundle,
) -> None:
    with pytest.raises(Hold30AlphaStructuralQualificationError, match="generation"):
        replace(bundle.lineage, protocol_generation="hold30-alpha-v2")
    with pytest.raises(Hold30AlphaStructuralQualificationError, match="synthetic"):
        replace(bundle.lineage, synthetic_data=True)
    with pytest.raises(Hold30AlphaStructuralQualificationError, match="future-selected"):
        replace(bundle.lineage, future_selected_universe=True)
    with pytest.raises(Hold30AlphaStructuralQualificationError, match="actor-invisible"):
        replace(bundle.exposures, policy_feature_access=True)


def test_missing_factor_exposures_and_return_substitutes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    bundle: _Bundle,
) -> None:
    monkeypatch.setattr(
        alpha_qualification,
        "HOLD30_MIN_AXIS_POSITIONS",
        POSITIONS,
    )
    with pytest.raises(
        Hold30AlphaStructuralQualificationError,
        match="factor exposures",
    ):
        alpha_qualification.qualify_hold30_alpha_in_memory_structure(
            bundle.sequence,
            monthly_rebalance=bundle.monthly,
            external_artifacts=bundle.artifacts,
            base_data_qualification_receipt={},
            evaluation_panel=bundle.panel,
            lineage_declaration=bundle.lineage,
            factor_exposure_declaration=None,  # type: ignore[arg-type]
        )

    substitute = replace(
        bundle.panel,
        market_total_returns=torch.zeros_like(bundle.panel.market_total_returns),
    )
    monkeypatch.setattr(
        alpha_qualification,
        "verify_hold30_dataset_against_qualification",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        alpha_qualification,
        "bind_hold30_alpha_evaluation_panel",
        lambda *_args: SimpleNamespace(receipt_id=_digest("binding")),
    )
    with pytest.raises(
        Hold30AlphaStructuralQualificationError,
        match="constant substitute",
    ):
        alpha_qualification.qualify_hold30_alpha_in_memory_structure(
            bundle.sequence,
            monthly_rebalance=bundle.monthly,
            external_artifacts=bundle.artifacts,
            base_data_qualification_receipt={},
            evaluation_panel=substitute,
            lineage_declaration=bundle.lineage,
            factor_exposure_declaration=bundle.exposures,
        )
