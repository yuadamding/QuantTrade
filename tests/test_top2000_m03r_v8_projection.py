from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_quant.execution.top2000_m03r_v8_projection import (
    M03RV8ProjectionError,
    project_m03r_v8_active_book,
    qualify_m03r_v8_risk_manifest,
)


def _manifest(
    *,
    bound: float = 0.001,
    covariance_variance: float = 1.0e-6,
    beta_scale: float = 0.0,
):
    covariance = torch.eye(4, dtype=torch.float64) * covariance_variance
    covariance[0, 0] = 0.0
    return qualify_m03r_v8_risk_manifest(
        exposure_names=("factor",),
        asset_axis_sha256="1" * 64,
        source_receipt_sha256="2" * 64,
        exposure_loadings=torch.tensor(
            [[0.0], [1.0], [-1.0], [0.0]], dtype=torch.float64
        ),
        exposure_lower_bounds=torch.tensor([-bound], dtype=torch.float64),
        exposure_upper_bounds=torch.tensor([bound], dtype=torch.float64),
        active_beta_loadings=torch.tensor(
            [0.0, beta_scale, -beta_scale, 0.0], dtype=torch.float64
        ),
        daily_return_covariance=covariance,
        cash_index=0,
    )


def _projection_inputs() -> dict[str, torch.Tensor]:
    benchmark = torch.tensor([[0.98, 0.005, 0.005, 0.01]], dtype=torch.float64)
    requested = torch.tensor([[0.98, 0.01, 0.0, 0.01]], dtype=torch.float64)
    return {
        "requested_weights": requested,
        "benchmark_weights": benchmark,
        "trade_mask": torch.ones_like(requested, dtype=torch.bool),
        "risk_asset_caps": torch.tensor([[1.0, 0.01, 0.01, 0.01]], dtype=torch.float64),
        "risk_gross_max": torch.ones(1, dtype=torch.float64),
    }


def test_relaxed_nonzero_factor_band_changes_the_feasible_book() -> None:
    inputs = _projection_inputs()
    manifest = _manifest(bound=0.001)
    reference = project_m03r_v8_active_book(
        **inputs,
        risk_manifest=manifest,
        factor_sector_bound_multiplier=1.0,
    )
    relaxed = project_m03r_v8_active_book(
        **inputs,
        risk_manifest=manifest,
        factor_sector_bound_multiplier=1.5,
    )

    assert reference.radial_scale.item() == pytest.approx(0.1, rel=2.0e-5)
    assert relaxed.radial_scale.item() == pytest.approx(0.15, rel=2.0e-5)
    assert not torch.equal(reference.projected_weights, relaxed.projected_weights)
    assert reference.projected_factor_exposure.abs().max() <= 0.001 + 1.0e-9
    assert relaxed.projected_factor_exposure.abs().max() <= 0.0015 + 1.0e-9


def test_active_beta_and_tracking_error_share_the_same_radial_scale() -> None:
    inputs = _projection_inputs()
    beta_bound = project_m03r_v8_active_book(
        **inputs,
        risk_manifest=_manifest(bound=1.0, beta_scale=20.0),
        factor_sector_bound_multiplier=1.0,
    )
    assert beta_bound.requested_active_beta.item() == pytest.approx(0.2)
    assert beta_bound.projected_active_beta.abs().item() <= 0.10 + 1.0e-8

    te_bound = project_m03r_v8_active_book(
        **inputs,
        risk_manifest=_manifest(bound=1.0, covariance_variance=0.40),
        factor_sector_bound_multiplier=1.0,
    )
    assert te_bound.requested_annual_tracking_error.item() > 0.06
    assert te_bound.projected_annual_tracking_error.item() <= 0.06 + 1.0e-8


def test_manifest_rejects_zero_slabs_and_post_qualification_mutation() -> None:
    with pytest.raises(M03RV8ProjectionError, match="nonzero"):
        _manifest(bound=0.0)

    manifest = _manifest()
    manifest.exposure_loadings[1, 0] += 1.0
    with pytest.raises(M03RV8ProjectionError, match="content changed"):
        manifest.validate()


def test_projection_rejects_unavailable_active_weight_and_unknown_multiplier() -> None:
    inputs = _projection_inputs()
    inputs["trade_mask"] = torch.tensor([[True, False, True, True]])
    with pytest.raises(M03RV8ProjectionError, match="benchmark is infeasible"):
        project_m03r_v8_active_book(
            **inputs,
            risk_manifest=_manifest(),
            factor_sector_bound_multiplier=1.0,
        )
    with pytest.raises(M03RV8ProjectionError, match="multiplier"):
        project_m03r_v8_active_book(
            **_projection_inputs(),
            risk_manifest=_manifest(),
            factor_sector_bound_multiplier=2.0,
        )


def test_forged_qualification_capability_and_receipt_drift_are_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(M03RV8ProjectionError, match="axis identity"):
        replace(manifest, _qualification_issuer=None).validate()
    with pytest.raises(M03RV8ProjectionError, match="receipt identity"):
        replace(manifest, manifest_sha256="0" * 64).validate()
