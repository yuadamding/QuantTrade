from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.training.top2000_m03r_v9_risk_source import (
    M03RV9RiskSourceError,
    M03RV9RiskSourceInventory,
    audit_m03r_v9_risk_source,
)


def _legacy_top2000_inventory() -> M03RV9RiskSourceInventory:
    return M03RV9RiskSourceInventory(
        source_id="top2000_raw_time_partitioned_v1",
        source_schema_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        source_columns=(
            "symbol",
            "available_timestamp_ms",
            "market_cap",
            "financial_assets",
        ),
        sector_exposure_names=(),
        style_risk_exposure_names=(),
        active_beta_exposure_name=None,
        point_in_time_sector_receipt_sha256=None,
        point_in_time_style_risk_receipt_sha256=None,
        point_in_time_active_beta_receipt_sha256=None,
        origin_availability_receipt_sha256=None,
        projector_manifest_sha256=None,
        target_projector_exposure_names_match=False,
    )


def test_legacy_top2000_schema_blocks_worker_before_gpu_allocation() -> None:
    readiness = audit_m03r_v9_risk_source(_legacy_top2000_inventory())
    assert not readiness.predictive_worker_authorized
    assert not readiness.economic_panel_authorized
    assert "missing-point-in-time-sector-classification" in readiness.blocker_codes
    assert "missing-projector-manifest" in readiness.blocker_codes
    with pytest.raises(M03RV9RiskSourceError, match="predictive worker is blocked"):
        readiness.require_predictive_worker_authorized()


def test_complete_point_in_time_inventory_authorizes_only_predictive_worker() -> None:
    ready = replace(
        _legacy_top2000_inventory(),
        sector_exposure_names=("sector-technology", "sector-financials"),
        style_risk_exposure_names=("style-size", "style-volatility"),
        active_beta_exposure_name="active-beta",
        point_in_time_sector_receipt_sha256="c" * 64,
        point_in_time_style_risk_receipt_sha256="d" * 64,
        point_in_time_active_beta_receipt_sha256="e" * 64,
        origin_availability_receipt_sha256="f" * 64,
        projector_manifest_sha256="1" * 64,
        target_projector_exposure_names_match=True,
    )
    readiness = audit_m03r_v9_risk_source(ready)
    readiness.require_predictive_worker_authorized()
    assert readiness.predictive_worker_authorized
    assert not readiness.economic_panel_authorized
    assert not readiness.blocker_codes
