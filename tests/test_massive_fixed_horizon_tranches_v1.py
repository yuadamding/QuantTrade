from __future__ import annotations

import math
from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveFixedHorizonTranchesV1Error,
    MassiveProfitabilityResidualInputRowV1,
    MassiveProfitabilityResidualScoreRowV1,
    MassiveProfitabilityResidualScoresV1,
    build_massive_profitability_residual_scores_v1,
    evaluate_massive_profitability_entry_capacity_v1,
    evaluate_massive_profitability_fixed_tranches_v1,
    select_massive_profitability_tranches_v1,
)
from rl_quant.evaluation.massive_profitability_inference_v1 import (
    build_massive_profitability_composite_pnl_v1,
    build_massive_profitability_horizon_risk_scaling_v1,
)
from rl_quant.features.massive_profitability_target_accounting_authority_v2 import (
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
    MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
    MassiveProfitabilityTargetAccountingAuthorityV2,
)
from rl_quant.features.massive_profitability_target_accounting_v1 import (
    MassiveProfitabilityTargetEconomicPathRowV1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)

_DIGEST = "a" * 64


def _residual_input(index: int) -> MassiveProfitabilityResidualInputRowV1:
    exposure = float(index - 10)
    body = {
        "decision_session_date": "2024-01-02",
        "security_id": f"SEC{index:03d}",
        "raw_scores": tuple(
            0.7 * exposure + (0.25 if index % 2 else -0.25) * (horizon + 1)
            for horizon in range(4)
        ),
        "exposures": (
            1.0,
            exposure,
            exposure**2,
            float(index % 3),
            float(index % 5),
            float(index % 7),
        ),
        "trailing_63_session_adv": 10_000_000.0 + index,
        "prediction_row_receipt_sha256": semantic_sha256(("prediction", index)),
        "feature_row_receipt_sha256": semantic_sha256(("feature", index)),
        "feature_accounting_row_inventory_sha256": semantic_sha256(
            ("accounting", index)
        ),
    }
    return MassiveProfitabilityResidualInputRowV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _residual_scores(
    *, sec000_scores: tuple[float, ...] | None = None
) -> MassiveProfitabilityResidualScoresV1:
    rows = []
    for index in range(10):
        score = float(index)
        scores = (
            sec000_scores if index == 0 and sec000_scores is not None else (score,) * 4
        )
        body = {
            "decision_session_date": "2024-01-02",
            "security_id": f"SEC{index:03d}",
            "raw_scores": scores,
            "residual_scores": scores,
            "trailing_63_session_adv": 5_000_000.0,
            "residual_input_receipt_sha256": semantic_sha256(("input", index)),
        }
        rows.append(
            MassiveProfitabilityResidualScoreRowV1(
                **body,
                receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
            )
        )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    body = {
        "schema": "rl-quant.massive-profitability-residual-scores-v1",
        "setting_id": "MV04",
        "fold_index": 0,
        "rows": tuple(asdict(row) for row in rows),
        "evaluation_plan_semantic_receipt_sha256": _DIGEST,
        "prediction_semantic_receipt_sha256": "b" * 64,
        "input_inventory_sha256": "c" * 64,
        "row_inventory_sha256": row_inventory,
        "ridge_lambda": 1e-6,
        "exposure_fields": (
            "intercept",
            "log_source_economic_value",
            "log_trailing_63_session_adv",
            "reversal_5",
            "momentum_21_minus_5",
            "economic_volatility_63",
        ),
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    return MassiveProfitabilityResidualScoresV1(
        setting_id="MV04",
        fold_index=0,
        rows=tuple(rows),
        evaluation_plan_semantic_receipt_sha256=_DIGEST,
        prediction_semantic_receipt_sha256="b" * 64,
        input_inventory_sha256="c" * 64,
        row_inventory_sha256=row_inventory,
        ridge_lambda=1e-6,
        exposure_fields=body["exposure_fields"],  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )


def _path(
    security_id: str,
    *,
    daily_return: float,
    invalid_offset: int | None = None,
    fallback_offset: int | None = None,
) -> MassiveProfitabilityTargetEconomicPathRowV1:
    economic = tuple(1_000 + offset for offset in range(64))
    values = []
    valid = []
    terminal = []
    kinds = []
    for offset in range(64):
        is_fallback = fallback_offset is not None and offset >= fallback_offset
        is_valid = offset != invalid_offset
        values.append(
            0.0 if is_fallback or not is_valid else 100.0 * (1 + daily_return) ** offset
        )
        valid.append(is_valid)
        terminal.append(is_fallback)
        kinds.append(
            "terminal-disposition"
            if is_fallback
            else "market"
            if is_valid
            else "missing"
        )
    body = {
        "security_id": security_id,
        "economic_at_ms": economic,
        "available_at_ms": economic,
        "values": tuple(values),
        "valid": tuple(valid),
        "terminal": tuple(terminal),
        "mark_kinds": tuple(kinds),
        "mark_receipts": tuple(
            semantic_sha256((security_id, offset)) for offset in range(64)
        ),
        "unresolved_terminal_fallback_session_offset": fallback_offset,
        "conservative_total_loss_fallback": fallback_offset is not None,
    }
    return MassiveProfitabilityTargetEconomicPathRowV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _authority(
    *,
    invalid_security: str | None = None,
    invalid_offset: int | None = None,
    fallback_security: str | None = None,
    fallback_offset: int | None = None,
) -> MassiveProfitabilityTargetAccountingAuthorityV2:
    start = date(2024, 1, 2)
    rows = tuple(
        _path(
            f"SEC{index:03d}",
            daily_return=(index - 4.5) / 1_000.0,
            invalid_offset=(
                invalid_offset if f"SEC{index:03d}" == invalid_security else None
            ),
            fallback_offset=(
                fallback_offset if f"SEC{index:03d}" == fallback_security else None
            ),
        )
        for index in range(10)
    )
    session_dates = tuple(
        (start + timedelta(days=offset)).isoformat() for offset in range(64)
    )
    row_inventory = semantic_sha256(tuple(row.receipt_sha256 for row in rows))
    semantic = {
        "schema": "rl-quant.massive-profitability-target-accounting-authority-v2",
        "origin_receipt_sha256": _DIGEST,
        "origin_plan_semantic_receipt_sha256": "b" * 64,
        "decision_session_date": session_dates[0],
        "session_dates": session_dates,
        "rows": tuple(asdict(row) for row in rows),
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "daily_input_authority_semantic_receipt_sha256": "c" * 64,
        "fill_source_authority_semantic_receipt_sha256": "d" * 64,
        "terminal_authority_semantic_receipt_sha256": "e" * 64,
        "economic_coverage_semantic_receipt_sha256": "f" * 64,
        "scoped_economic_event_inventory_sha256": semantic_sha256(()),
        "row_inventory_sha256": row_inventory,
        "fill_sources_qualified": True,
        "economic_values_data_qualified": True,
        "terminal_accounting_complete": True,
        "conservative_total_loss_target_count": sum(
            row.conservative_total_loss_fallback for row in rows
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    audit_source = semantic_sha256(("audit", semantic_receipt))
    runtime = dict(semantic)
    runtime["rows"] = rows
    result = MassiveProfitabilityTargetAccountingAuthorityV2(
        **runtime,
        semantic_receipt_sha256=semantic_receipt,
        economic_archive_audit_receipt_sha256=audit_source,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_archive_audit_receipt_sha256": audit_source,
            }
        ),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def test_residualization_is_deterministic_and_target_free() -> None:
    rows = tuple(_residual_input(index) for index in range(20))
    first = build_massive_profitability_residual_scores_v1(
        setting_id="MV04",
        fold_index=0,
        evaluation_plan_semantic_receipt_sha256=_DIGEST,
        prediction_semantic_receipt_sha256="b" * 64,
        rows=rows,
    )
    second = build_massive_profitability_residual_scores_v1(
        setting_id="MV04",
        fold_index=0,
        evaluation_plan_semantic_receipt_sha256=_DIGEST,
        prediction_semantic_receipt_sha256="b" * 64,
        rows=rows,
    )
    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.outer_evaluation_authorized is False
    assert first.profitability_reporting_authorized is False
    for horizon in range(4):
        assert abs(sum(row.residual_scores[horizon] for row in first.rows)) < 1e-6
    with pytest.raises(MassiveFixedHorizonTranchesV1Error, match="plan-authorized"):
        select_massive_profitability_tranches_v1(
            residual_scores=first, target_accounting=(_authority(),)
        )


def test_selection_opens_targets_after_ranking_and_is_horizon_direction_safe() -> None:
    scores = _residual_scores()
    authority = _authority(invalid_security="SEC004", invalid_offset=1)
    selected = select_massive_profitability_tranches_v1(
        residual_scores=scores, target_accounting=(authority,)
    )
    assert len(selected.positions) == 16
    assert all(position.security_id != "SEC004" for position in selected.positions)

    with pytest.raises(
        MassiveFixedHorizonTranchesV1Error, match="complete marked path"
    ):
        select_massive_profitability_tranches_v1(
            residual_scores=scores,
            target_accounting=(
                _authority(invalid_security="SEC000", invalid_offset=1),
            ),
        )
    with pytest.raises(MassiveFixedHorizonTranchesV1Error, match="selected short"):
        select_massive_profitability_tranches_v1(
            residual_scores=scores,
            target_accounting=(
                _authority(fallback_security="SEC000", fallback_offset=5),
            ),
        )
    allowed = select_massive_profitability_tranches_v1(
        residual_scores=_residual_scores(sec000_scores=(-1.0, 4.5, 4.5, 4.5)),
        target_accounting=(_authority(fallback_security="SEC000", fallback_offset=63),),
    )
    assert any(
        row.security_id == "SEC000"
        and row.side == "short"
        and row.horizon_sessions < 63
        for row in allowed.positions
    )


def test_fixed_tranches_costs_capacity_and_composite_are_deterministic() -> None:
    authority = _authority()
    selected = select_massive_profitability_tranches_v1(
        residual_scores=_residual_scores(), target_accounting=(authority,)
    )
    pnl = evaluate_massive_profitability_fixed_tranches_v1(
        selected=selected, target_accounting=(authority,)
    )
    assert all(
        row.net_returns[0] >= row.net_returns[1] >= row.net_returns[2]
        for row in pnl.rows
    )
    assert len(pnl.rows) == 64 * 4
    assert any(row.active_position_count == 0 for row in pnl.rows)

    capacity = evaluate_massive_profitability_entry_capacity_v1(
        selected=selected, target_accounting=(authority,)
    )
    assert tuple(row.capital_usd for row in capacity.rows) == (
        1_000_000.0,
        10_000_000.0,
        50_000_000.0,
    )
    assert (
        capacity.rows[0].lost_intended_notional_fraction
        <= capacity.rows[-1].lost_intended_notional_fraction
    )

    scaling = build_massive_profitability_horizon_risk_scaling_v1(
        fold_index=0,
        fit_daily_returns_by_horizon={
            horizon: tuple(((-1) ** index) * horizon / 10_000 for index in range(64))
            for horizon in (1, 5, 21, 63)
        },
        fit_source_receipt_sha256=_DIGEST,
    )
    assert math.isclose(sum(scaling.horizon_weights), 1.0)
    composite = build_massive_profitability_composite_pnl_v1(
        pnl=pnl, risk_scaling=scaling
    )
    assert len(composite.rows) == 64
    assert composite.profitability_reporting_authorized is False


def test_long_unresolved_fallback_retains_total_loss() -> None:
    authority = _authority(fallback_security="SEC009", fallback_offset=5)
    selected = select_massive_profitability_tranches_v1(
        residual_scores=_residual_scores(), target_accounting=(authority,)
    )
    pnl = evaluate_massive_profitability_fixed_tranches_v1(
        selected=selected, target_accounting=(authority,)
    )
    h5_exit = next(
        row
        for row in pnl.rows
        if row.session_date == authority.session_dates[5] and row.horizon_sessions == 5
    )
    assert h5_exit.gross_return < 0.0


def test_artifact_receipts_fail_on_mutation() -> None:
    scores = _residual_scores()
    scores.validate()
    with pytest.raises(MassiveFixedHorizonTranchesV1Error, match="inventory"):
        replace(scores, semantic_receipt_sha256="0" * 64).validate()
