from __future__ import annotations

import pytest
import torch

from rl_quant.training.top2000_m03r_v11_residual_operator import (
    apply_m03r_v11_residual_operator,
)
from rl_quant.training.top2000_m03r_v14_residual_operator import (
    apply_m03r_v14_residual_operator,
    build_m03r_v14_residual_operator,
)


def _operator() -> object:
    available = torch.ones(9, dtype=torch.bool)
    available[0] = False
    loadings = torch.zeros((9, 6), dtype=torch.float64)
    loadings[1:, 0] = 1.0
    loadings[1:5, 1] = 1.0
    loadings[5:, 2] = 1.0
    loadings[1:, 3] = torch.linspace(-1.0, 1.0, 8)
    loadings[1:, 4] = torch.tensor(
        [-1.0, 0.5, 1.2, -0.3, 0.8, -1.5, 0.2, 0.6],
        dtype=torch.float64,
    )
    loadings[1:, 5] = torch.tensor(
        [0.2, 1.0, -0.5, 1.5, -1.0, 0.7, -0.2, 0.4],
        dtype=torch.float64,
    )
    weights = torch.ones(9, dtype=torch.float64)
    weights[0] = 0.0
    return build_m03r_v14_residual_operator(
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


def test_v14_precomputed_map_matches_qualified_qr_and_preserves_gradients() -> None:
    operator = _operator()
    value = torch.tensor(
        [0.0, -0.03, 0.02, 0.01, -0.01, 0.04, -0.02, 0.03, -0.04],
        dtype=torch.float32,
        requires_grad=True,
    )
    expected = apply_m03r_v11_residual_operator(value, operator.base)  # type: ignore[attr-defined]
    observed = apply_m03r_v14_residual_operator(value, operator)  # type: ignore[arg-type]
    assert torch.allclose(observed.residual, expected.residual, rtol=0.0, atol=2.0e-7)
    observed.residual.square().sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert float(value.grad.abs().sum()) > 0.0


def test_v14_apply_does_not_repeat_qr_factorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    monkeypatch.setattr(
        torch.linalg,
        "qr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("apply must reuse the precomputed coefficient map")
        ),
    )
    value = torch.linspace(-0.04, 0.04, 9)
    first = apply_m03r_v14_residual_operator(value, operator)  # type: ignore[arg-type]
    second = apply_m03r_v14_residual_operator(2.0 * value, operator)  # type: ignore[arg-type]
    assert torch.allclose(second.residual, 2.0 * first.residual)
