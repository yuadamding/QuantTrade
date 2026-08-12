from __future__ import annotations

import pytest
import torch

from rl_quant.training.top2000_m03r_v11_residual_operator import (
    M03RV11ResidualOperatorError,
    apply_m03r_v11_residual_operator,
    build_m03r_v11_residual_operator,
)


def _operator() -> object:
    available = torch.ones(8, dtype=torch.bool)
    available[0] = False
    loadings = torch.zeros((8, 6), dtype=torch.float64)
    loadings[1:, 0] = 1.0
    loadings[1:4, 1] = 1.0
    loadings[4:, 2] = 1.0
    loadings[1:, 3] = torch.linspace(-1.0, 1.0, 7)
    loadings[1:, 4] = torch.tensor([-1.0, 0.5, 1.2, -0.3, 0.8, -1.5, 0.2])
    loadings[1:, 5] = torch.tensor([0.2, 1.0, -0.5, 1.5, -1.0, 0.7, -0.2])
    weights = torch.ones(8, dtype=torch.float64)
    weights[0] = 0.0
    weights[7] = 0.0
    return build_m03r_v11_residual_operator(
        origin_state_index=31,
        cash_index=0,
        available_mask=available,
        exposure_loadings=loadings,
        regression_weights=weights,
        projector_exposure_names=(
            "sector-a",
            "sector-b",
            "active-beta",
            "style-return",
            "style-volatility",
        ),
        projector_exposure_families=(
            "sector",
            "sector",
            "active-beta",
            "style-risk",
            "style-risk",
        ),
        asset_axis_sha256="a" * 64,
        source_exposure_receipt_sha256="b" * 64,
    )


def test_unqualified_asset_is_invalid_and_shared_operator_is_orthogonal() -> None:
    operator = _operator()
    assert operator.available_risky_asset_count == 7  # type: ignore[attr-defined]
    assert operator.factor_qualified_risky_asset_count == 6  # type: ignore[attr-defined]
    assert not operator.qualified_asset_mask[7]  # type: ignore[attr-defined]
    value = torch.tensor([0.0, -0.03, 0.02, 0.01, -0.01, 0.04, -0.02, 9.0])
    target = apply_m03r_v11_residual_operator(value, operator)  # type: ignore[arg-type]
    signal = apply_m03r_v11_residual_operator(value * 2.0, operator)  # type: ignore[arg-type]
    assert target.residual[7].item() == 0.0
    assert not target.qualified_asset_mask[7]
    assert target.operator_receipt_sha256 == signal.operator_receipt_sha256
    assert target.weighted_exposure_error < 1.0e-9
    assert signal.weighted_exposure_error < 1.0e-9


def test_collinear_design_fails_before_training() -> None:
    available = torch.tensor([False, True, True, True])
    loadings = torch.ones((4, 3), dtype=torch.float64)
    with pytest.raises(M03RV11ResidualOperatorError, match="rank deficient"):
        build_m03r_v11_residual_operator(
            origin_state_index=0,
            cash_index=0,
            available_mask=available,
            exposure_loadings=loadings,
            regression_weights=torch.tensor([0.0, 1.0, 1.0, 1.0]),
            projector_exposure_names=("sector-a", "sector-b"),
            projector_exposure_families=("sector", "sector"),
            asset_axis_sha256="a" * 64,
            source_exposure_receipt_sha256="b" * 64,
        )


def test_empty_sector_is_omitted_and_supported_reference_is_dropped() -> None:
    available = torch.tensor([False, True, True, True, True, True, True])
    loadings = torch.zeros((7, 6), dtype=torch.float64)
    loadings[1:, 0] = 1.0
    loadings[1:4, 1] = 1.0
    loadings[4:, 2] = 1.0
    # Column 3 is an intentionally unsupported sector.
    loadings[1:, 4] = torch.linspace(-1.0, 1.0, 6)
    loadings[1:, 5] = torch.tensor([-1.0, 0.5, 1.2, -0.3, 0.8, -1.5])
    operator = build_m03r_v11_residual_operator(
        origin_state_index=20,
        cash_index=0,
        available_mask=available,
        exposure_loadings=loadings,
        regression_weights=torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        projector_exposure_names=(
            "sector-a",
            "sector-b",
            "sector-empty",
            "active-beta",
            "style-return",
        ),
        projector_exposure_families=(
            "sector",
            "sector",
            "sector",
            "active-beta",
            "style-risk",
        ),
        asset_axis_sha256="a" * 64,
        source_exposure_receipt_sha256="b" * 64,
    )
    assert operator.dropped_reference_sector == "sector-b"
    assert operator.unsupported_sector_names == ("sector-empty",)
    assert "sector-empty" not in operator.exposure_names
    assert operator.effective_design_rank == len(operator.exposure_names)
