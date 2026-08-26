from __future__ import annotations

import inspect
from dataclasses import asdict
from datetime import date, timedelta

import pytest

from rl_quant.evaluation.massive_fixed_horizon_tranches_v1 import (
    MassiveFixedHorizonTranchesV1Error,
    MassiveProfitabilityResidualInputRowV1,
    MassiveProfitabilitySelectedTranchePositionV1,
    MassiveProfitabilitySelectedTranchesV1,
    build_massive_profitability_residual_scores_v1,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v1 import (
    MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2,
    MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2,
)
from rl_quant.evaluation.massive_profitability_evaluation_plan_v2 import (
    MASSIVE_PROFITABILITY_EVALUATION_PLAN_V2_SPEC_SHA256,
    materialize_massive_profitability_evaluation_plan_v2,
)
from rl_quant.evaluation.massive_profitability_outer_evidence_v2 import (
    MassiveProfitabilityCausalEligibilityDateV2,
    build_massive_profitability_stitched_composite_v2,
    evaluate_massive_profitability_stitched_capacity_v2,
    stitch_massive_profitability_fixed_tranches_v2,
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


def _path(
    *, security_id: str, growth: float
) -> MassiveProfitabilityTargetEconomicPathRowV1:
    values = tuple(100.0 * (1.0 + growth) ** offset for offset in range(64))
    body = {
        "security_id": security_id,
        "economic_at_ms": tuple(1_000 + offset for offset in range(64)),
        "available_at_ms": tuple(1_000 + offset for offset in range(64)),
        "values": values,
        "valid": (True,) * 64,
        "terminal": (False,) * 64,
        "mark_kinds": ("market",) * 64,
        "mark_receipts": tuple(
            semantic_sha256((security_id, offset)) for offset in range(64)
        ),
        "unresolved_terminal_fallback_session_offset": None,
        "conservative_total_loss_fallback": False,
    }
    return MassiveProfitabilityTargetEconomicPathRowV1(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )


def _authority(
    *, start: date, security_id: str
) -> MassiveProfitabilityTargetAccountingAuthorityV2:
    path = _path(security_id=security_id, growth=0.01)
    session_dates = tuple(
        (start + timedelta(days=offset)).isoformat() for offset in range(64)
    )
    semantic = {
        "schema": "rl-quant.massive-profitability-target-accounting-authority-v2",
        "origin_receipt_sha256": semantic_sha256(("origin", start.isoformat())),
        "origin_plan_semantic_receipt_sha256": "b" * 64,
        "decision_session_date": session_dates[0],
        "session_dates": session_dates,
        "rows": (asdict(path),),
        "horizons": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_HORIZONS,
        "daily_input_authority_semantic_receipt_sha256": "c" * 64,
        "fill_source_authority_semantic_receipt_sha256": "d" * 64,
        "terminal_authority_semantic_receipt_sha256": "e" * 64,
        "economic_coverage_semantic_receipt_sha256": "f" * 64,
        "scoped_economic_event_inventory_sha256": semantic_sha256(()),
        "row_inventory_sha256": semantic_sha256((path.receipt_sha256,)),
        "fill_sources_qualified": True,
        "economic_values_data_qualified": True,
        "terminal_accounting_complete": True,
        "conservative_total_loss_target_count": 0,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TARGET_ACCOUNTING_AUTHORITY_V2_SOURCE_SHA256,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    audit = semantic_sha256(("audit", semantic_receipt))
    result = MassiveProfitabilityTargetAccountingAuthorityV2(
        **{**semantic, "rows": (path,)},
        semantic_receipt_sha256=semantic_receipt,
        economic_archive_audit_receipt_sha256=audit,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "economic_archive_audit_receipt_sha256": audit,
            }
        ),  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _selected(
    *,
    fold_index: int,
    authority: MassiveProfitabilityTargetAccountingAuthorityV2,
    target_inventory: str,
) -> MassiveProfitabilitySelectedTranchesV1:
    position_body = {
        "decision_session_date": authority.decision_session_date,
        "security_id": authority.rows[0].security_id,
        "horizon_sessions": 5,
        "side": "long",
        "tail_rank": 1,
        "signed_entry_weight": 0.1,
        "residual_score": 1.0,
        "trailing_63_session_adv": 1_000_000.0,
        "residual_score_row_receipt_sha256": semantic_sha256(("score", fold_index)),
        "target_path_receipt_sha256": authority.rows[0].receipt_sha256,
        "unresolved_terminal_fallback_session_offset": None,
    }
    position = MassiveProfitabilitySelectedTranchePositionV1(
        **position_body,
        receipt_sha256=semantic_sha256(position_body),  # type: ignore[arg-type]
    )
    position_inventory = semantic_sha256((position.receipt_sha256,))
    body = {
        "setting_id": "MV04",
        "fold_index": fold_index,
        "positions": (asdict(position),),
        "residual_scores_semantic_receipt_sha256": semantic_sha256(
            ("residual", fold_index)
        ),
        "target_accounting_inventory_sha256": target_inventory,
        "position_inventory_sha256": position_inventory,
        "path_support_complete": True,
        "direction_safe_terminal_support_complete": True,
        "outer_evaluation_authorized": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilitySelectedTranchesV1(
        setting_id="MV04",
        fold_index=fold_index,
        positions=(position,),
        residual_scores_semantic_receipt_sha256=body[
            "residual_scores_semantic_receipt_sha256"
        ],  # type: ignore[arg-type]
        target_accounting_inventory_sha256=target_inventory,
        position_inventory_sha256=position_inventory,
        path_support_complete=True,
        direction_safe_terminal_support_complete=True,
        semantic_receipt_sha256=semantic_sha256(body),
        outer_evaluation_authorized=True,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    result.validate()
    return result


def _stitched_fixture() -> tuple[
    tuple[MassiveProfitabilitySelectedTranchesV1, ...],
    tuple[MassiveProfitabilityTargetAccountingAuthorityV2, ...],
]:
    authorities = tuple(
        _authority(
            start=date(2024, 1, 2) + timedelta(days=fold), security_id=f"SEC{fold}"
        )
        for fold in range(4)
    )
    inventory = semantic_sha256(
        tuple(sorted(row.semantic_receipt_sha256 for row in authorities))
    )
    selected = tuple(
        _selected(fold_index=fold, authority=authority, target_inventory=inventory)
        for fold, authority in enumerate(authorities)
    )
    return selected, authorities


def test_caller_rows_cannot_authorize_residual_scores(tmp_path) -> None:
    row_body = {
        "decision_session_date": "2024-01-02",
        "security_id": "SEC0",
        "raw_scores": (1.0, 1.0, 1.0, 1.0),
        "exposures": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "trailing_63_session_adv": 1_000_000.0,
        "prediction_row_receipt_sha256": _DIGEST,
        "feature_row_receipt_sha256": "b" * 64,
        "feature_accounting_row_inventory_sha256": "c" * 64,
    }
    row = MassiveProfitabilityResidualInputRowV1(
        **row_body,
        receipt_sha256=semantic_sha256(row_body),  # type: ignore[arg-type]
    )
    with pytest.raises(
        MassiveFixedHorizonTranchesV1Error,
        match="caller-supplied residual rows cannot authorize",
    ):
        build_massive_profitability_residual_scores_v1(
            setting_id="MV04",
            fold_index=0,
            evaluation_plan_semantic_receipt_sha256=_DIGEST,
            prediction_semantic_receipt_sha256="b" * 64,
            rows=(row,),
            evaluation_plan_root=tmp_path,
        )


def test_fixed_horizon_weights_and_causal_threshold_are_frozen() -> None:
    assert MASSIVE_PROFITABILITY_FIXED_HORIZON_WEIGHTS_V2 == (0.25,) * 4
    assert len(set(MASSIVE_PROFITABILITY_FIXED_HORIZON_SCALING_RECEIPTS_V2)) == 4
    assert len(MASSIVE_PROFITABILITY_EVALUATION_PLAN_V2_SPEC_SHA256) == 64
    assert (
        "horizon_risk_scaling_receipts"
        not in inspect.signature(
            materialize_massive_profitability_evaluation_plan_v2
        ).parameters
    )
    eligible = tuple(f"SEC{index:03d}" for index in range(400))
    body = {
        "decision_session_date": "2024-01-02",
        "pit_member_count": 500,
        "required_eligible_count": 400,
        "eligible_security_ids": eligible,
        "eligible_inventory_sha256": semantic_sha256(eligible),
    }
    row = MassiveProfitabilityCausalEligibilityDateV2(
        **body,
        receipt_sha256=semantic_sha256(body),  # type: ignore[arg-type]
    )
    row.validate()


def test_stitched_ledger_has_one_calendar_row_and_keeps_cross_fold_positions() -> None:
    selected, authorities = _stitched_fixture()
    pnl = stitch_massive_profitability_fixed_tranches_v2(
        selected_by_fold=selected,
        target_accounting=authorities,
    )
    keys = tuple((row.session_date, row.horizon_sessions) for row in pnl.rows)
    assert len(keys) == len(set(keys))
    overlap = next(
        row
        for row in pnl.rows
        if row.session_date == "2024-01-03" and row.horizon_sessions == 5
    )
    assert overlap.active_position_count == 2
    assert overlap.entry_fold_net_return_20bp[0] != 0.0
    assert overlap.entry_fold_net_return_20bp[1] != 0.0
    composite = build_massive_profitability_stitched_composite_v2(pnl=pnl)
    assert composite.horizon_weights == (0.25,) * 4
    assert len(composite.rows) == len({row.session_date for row in pnl.rows})


def test_capacity_reports_intended_and_executed_participation() -> None:
    selected, authorities = _stitched_fixture()
    panel = evaluate_massive_profitability_stitched_capacity_v2(
        selected_by_fold=selected,
        target_accounting=authorities,
    )
    large = next(row for row in panel.rows if row.capital_usd == 50_000_000.0)
    # 25% of total composite capital belongs to the H5 sleeve before clipping.
    assert large.maximum_intended_participation == pytest.approx(1.25)
    assert large.maximum_executed_participation == pytest.approx(0.02)
    assert large.clipped_order_count == large.intended_order_count
    assert large.lost_intended_notional_fraction > 0.0
