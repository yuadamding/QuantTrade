from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from io import BytesIO

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveProfitabilityCapacityPanelV1,
    MassiveProfitabilityCapacityRowV1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1,
    MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
    MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_DATASET,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SPEC_SHA256,
    MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1,
    MassiveProfitabilityEvaluationPlanV1,
    MassiveProfitabilityEvaluationPredictionV1,
)
from rl_quant.evaluation.massive_profitability_inference_v1 import (
    MassiveProfitabilityCompositeDailyRowV1,
    MassiveProfitabilityCompositePnlV1,
    materialize_massive_profitability_outer_evidence_v1,
    parse_massive_profitability_outer_evidence_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
)

_DIGEST = "a" * 64


def _risk_receipt(fold: int) -> str:
    return semantic_sha256(("risk", fold))


def _plan(tmp_path) -> MassiveProfitabilityEvaluationPlanV1:
    predictions = []
    for fold in range(4):
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1:
            body = {
                "fold_index": fold,
                "setting_id": setting,
                "prediction_semantic_receipt_sha256": semantic_sha256(
                    ("prediction", fold, setting)
                ),
                "outer_test_inventory_sha256": semantic_sha256(("outer", fold)),
                "ensemble": setting != "MV00",
                "seed_inventory": (
                    (0,)
                    if setting == "MV00"
                    else MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
                ),
            }
            predictions.append(
                MassiveProfitabilityEvaluationPredictionV1(
                    **body,  # type: ignore[arg-type]
                    receipt_sha256=semantic_sha256(body),
                )
            )
    tournament_receipt = "b" * 64
    semantic = {
        "schema": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SCHEMA,
        "data_gate_semantic_receipt_sha256": "c" * 64,
        "phase_plan_semantic_receipt_sha256": "d" * 64,
        "tournament_plan_receipt_sha256": tournament_receipt,
        "outer_fold_receipts": tuple(
            semantic_sha256(("fold", fold)) for fold in range(4)
        ),
        "horizon_risk_scaling_receipts": tuple(
            _risk_receipt(fold) for fold in range(4)
        ),
        "predictions": tuple(asdict(row) for row in predictions),
        "score_field": "mean",
        "residual_ridge_lambda": 1e-6,
        "residual_exposure_fields": (
            "intercept",
            "log_source_economic_value",
            "log_trailing_63_session_adv",
            "reversal_5",
            "momentum_21_minus_5",
            "economic_volatility_63",
        ),
        "tail_fraction": 0.20,
        "horizons": MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1,
        "cost_rates": MASSIVE_PROFITABILITY_EVALUATION_COSTS_V1,
        "primary_cost_rate": 0.002,
        "annual_short_borrow_rate": 0.01,
        "bootstrap_block_sessions": 63,
        "bootstrap_replicates": 2_000,
        "bootstrap_seed": MASSIVE_PROFITABILITY_EVALUATION_BOOTSTRAP_SEED_V1,
        "capacity_capital_usd": MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1,
        "primary_capacity_usd": 10_000_000.0,
        "adv_participation_limit": 0.02,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SHA256,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    relative = "massive-profitability-evaluation-plan-v1/test.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=tmp_path,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=1,
        downloaded_at_ms=1,
        schema_sha256=MASSIVE_PROFITABILITY_EVALUATION_PLAN_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=tournament_receipt,
        committed_at_ms=1,
    )
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1
    )
    runtime = dict(semantic)
    runtime.pop("predictions")
    result = MassiveProfitabilityEvaluationPlanV1(
        **runtime,  # type: ignore[arg-type]
        predictions=tuple(predictions),
        loaded_source=loaded,
    )
    result.validate()
    return result


def _composite(*, fold: int, setting: str) -> MassiveProfitabilityCompositePnlV1:
    daily = {
        "MV00": 0.0005,
        "MV02": 0.0010,
        "MV04": 0.0030,
        "MV04-SHUFFLE": 0.0011,
    }[setting]
    start = date(2024, 1, 1) + timedelta(days=100 * fold)
    rows = []
    for offset in range(64):
        body = {
            "session_date": (start + timedelta(days=offset)).isoformat(),
            "gross_return": daily + 0.001,
            "net_returns": (daily + 0.0002, daily, daily - 0.0002),
            "horizon_row_receipts": tuple(
                semantic_sha256((fold, setting, offset, horizon))
                for horizon in MASSIVE_PROFITABILITY_EVALUATION_HORIZONS_V1
            ),
        }
        rows.append(
            MassiveProfitabilityCompositeDailyRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "setting_id": setting,
        "fold_index": fold,
        "rows": tuple(asdict(row) for row in rows),
        "fixed_tranche_pnl_semantic_receipt_sha256": semantic_sha256(
            ("pnl", fold, setting)
        ),
        "risk_scaling_receipt_sha256": _risk_receipt(fold),
        "row_inventory_sha256": inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityCompositePnlV1(
        setting_id=setting,
        fold_index=fold,
        rows=tuple(rows),
        fixed_tranche_pnl_semantic_receipt_sha256=body[
            "fixed_tranche_pnl_semantic_receipt_sha256"
        ],  # type: ignore[arg-type]
        risk_scaling_receipt_sha256=body["risk_scaling_receipt_sha256"],  # type: ignore[arg-type]
        row_inventory_sha256=inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def _capacity(*, fold: int, setting: str) -> MassiveProfitabilityCapacityPanelV1:
    selected_receipt = semantic_sha256(("selected", fold, setting))
    target_receipt = semantic_sha256(("targets", fold))
    rows = []
    for capital in MASSIVE_PROFITABILITY_EVALUATION_CAPITAL_V1:
        body = {
            "setting_id": setting,
            "fold_index": fold,
            "capital_usd": capital,
            "intended_order_count": 100,
            "clipped_order_count": 1,
            "mean_participation": 0.01,
            "participation_p95": 0.015,
            "maximum_participation": 0.03,
            "lost_intended_notional_fraction": 0.01,
            "clipped_mean_daily_net_return_20bp": (
                0.002 if setting == "MV04" else 0.0005
            ),
            "selected_tranches_semantic_receipt_sha256": selected_receipt,
            "target_accounting_inventory_sha256": target_receipt,
        }
        rows.append(
            MassiveProfitabilityCapacityRowV1(
                **body,  # type: ignore[arg-type]
                receipt_sha256=semantic_sha256(body),
            )
        )
    inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": "rl-quant.massive-profitability-capacity-panel-v1",
        "setting_id": setting,
        "fold_index": fold,
        "rows": tuple(asdict(row) for row in rows),
        "selected_tranches_semantic_receipt_sha256": selected_receipt,
        "target_accounting_inventory_sha256": target_receipt,
        "row_inventory_sha256": inventory,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityCapacityPanelV1(
        setting_id=setting,
        fold_index=fold,
        rows=tuple(rows),
        selected_tranches_semantic_receipt_sha256=selected_receipt,
        target_accounting_inventory_sha256=target_receipt,
        row_inventory_sha256=inventory,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def test_outer_evidence_uses_paired_63_session_inference_and_round_trips(
    tmp_path,
) -> None:
    plan = _plan(tmp_path)
    composites = tuple(
        _composite(fold=fold, setting=setting)
        for fold in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    capacity = tuple(
        _capacity(fold=fold, setting=setting)
        for fold in range(4)
        for setting in MASSIVE_PROFITABILITY_EVALUATION_SETTINGS_V1
    )
    evidence = materialize_massive_profitability_outer_evidence_v1(
        root=tmp_path,
        artifact_id="paired",
        evaluation_plan=plan,
        composites=composites,
        capacity_panels=capacity,
        committed_at_ms=2,
    )
    assert evidence.outer_profitability_gate_passed is True
    assert evidence.mv04_minus_mv02_net_20bp_lcb95 > 0.0
    assert evidence.mv04_minus_shuffle_net_20bp_lcb95 > 0.0
    assert evidence.mean_mv04_clipped_10m_net_20bp > 0.0
    assert evidence.profitability_reporting_authorized is False
    assert evidence.lockbox_access_authorized is False
    reloaded = parse_massive_profitability_outer_evidence_v1(
        root=tmp_path, loaded_source=evidence.loaded_source
    )
    assert reloaded.semantic_receipt_sha256 == evidence.semantic_receipt_sha256
