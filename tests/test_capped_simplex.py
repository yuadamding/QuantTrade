from __future__ import annotations

import pytest
import torch

from rl_quant.protocol.constraints import project_capped_risky_simplex


def _iterative_capped_reference(
    weights: torch.Tensor,
    available: torch.Tensor,
    cap: float,
    cash_index: int = 0,
) -> torch.Tensor:
    """Small independent water-fill oracle; deliberately loops over rows/assets."""
    original_dtype = weights.dtype
    work = weights.float() if weights.dtype in (torch.float16, torch.bfloat16) else weights
    flat_weights = work.reshape(-1, work.shape[-1])
    flat_available = available.reshape(-1, available.shape[-1])
    rows = []
    for row, row_available in zip(flat_weights, flat_available, strict=True):
        active = row_available.clone()
        active[cash_index] = False
        risky = torch.where(active, row, torch.zeros_like(row))
        target = torch.minimum(risky.sum(), active.sum().to(row.dtype) * cap)
        projected = torch.zeros_like(row)
        remaining = target
        for _ in range(row.numel()):
            count = active.sum()
            if not bool(count):
                break
            active_mass = torch.where(active, risky, torch.zeros_like(risky)).sum()
            proportional = remaining * risky / active_mass.clamp_min(torch.finfo(row.dtype).tiny)
            uniform = torch.full_like(risky, 1.0) * remaining / count.to(row.dtype)
            proposal = torch.where(active_mass > 0, proportional, uniform)
            newly_capped = active & (proposal > cap)
            if not bool(newly_capped.any()):
                projected = projected + torch.where(active, proposal, torch.zeros_like(proposal))
                break
            projected = projected + newly_capped.to(row.dtype) * cap
            remaining = remaining - newly_capped.sum().to(row.dtype) * cap
            active = active & ~newly_capped
        projected = projected.clone()
        projected[cash_index] = 1.0 - projected.sum()
        rows.append(projected)
    return torch.stack(rows).reshape_as(weights).to(original_dtype)


def test_capped_projection_redistributes_without_reinflating_risky_weights() -> None:
    requested = torch.tensor([[0.50, 0.30, 0.10, 0.10]], requires_grad=True)
    available = torch.ones_like(requested, dtype=torch.bool)

    projected = project_capped_risky_simplex(
        requested,
        available,
        max_risky_weight=0.20,
        cash_index=0,
    )

    torch.testing.assert_close(projected, torch.tensor([[0.50, 0.20, 0.15, 0.15]]))
    torch.testing.assert_close(projected.sum(dim=-1), torch.ones(1))
    assert float(projected[:, 1:].max()) <= 0.20 + 1e-7

    (projected * torch.arange(4, dtype=projected.dtype)).sum().backward()
    assert requested.grad is not None
    assert bool(torch.isfinite(requested.grad).all())


def test_capped_projection_sends_capacity_shortfall_and_unavailable_mass_to_cash() -> None:
    requested = torch.tensor([[0.01, 0.33, 0.33, 0.33]])
    available = torch.tensor([[True, True, False, True]])

    projected = project_capped_risky_simplex(
        requested,
        available,
        max_risky_weight=0.20,
        cash_index=0,
    )

    torch.testing.assert_close(projected, torch.tensor([[0.60, 0.20, 0.00, 0.20]]))
    torch.testing.assert_close(projected.sum(dim=-1), torch.ones(1))


def test_uncapped_projection_still_removes_unavailable_risky_mass() -> None:
    requested = torch.tensor([[0.10, 0.40, 0.50]])
    available = torch.tensor([[True, True, False]])

    projected = project_capped_risky_simplex(
        requested,
        available,
        max_risky_weight=1.0,
    )

    torch.testing.assert_close(projected, torch.tensor([[0.60, 0.40, 0.00]]))


@pytest.mark.parametrize("bad_cap", [0.0, -0.1, 1.1, float("nan")])
def test_capped_projection_rejects_invalid_limits(bad_cap: float) -> None:
    with pytest.raises(ValueError, match="max_risky_weight"):
        project_capped_risky_simplex(
            torch.tensor([[1.0, 0.0]]),
            torch.ones(1, 2, dtype=torch.bool),
            max_risky_weight=bad_cap,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sort_projection_matches_iterative_reference_randomized(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(91_337)
    for action_count in (2, 3, 7, 31, 127):
        cash_index = action_count // 2
        for cap in (0.005, 0.01, 0.03, 0.10, 0.25, 0.70, 1.0):
            logits = torch.randn(2, 3, action_count, generator=generator)
            available = torch.rand(2, 3, action_count, generator=generator) > 0.30
            available[..., cash_index] = True
            # Retain unavailable mass deliberately. Policy callers normally
            # mask it before softmax, but the projection's own availability
            # contract must remain valid, including at the cap=1 fast path.
            weights = logits.softmax(dim=-1).to(dtype)

            projected = project_capped_risky_simplex(
                weights,
                available,
                max_risky_weight=cap,
                cash_index=cash_index,
            )
            reference = _iterative_capped_reference(weights, available, cap, cash_index)

            tolerance = 3e-6 if dtype == torch.float32 else torch.finfo(torch.bfloat16).eps
            torch.testing.assert_close(projected.float(), reference.float(), atol=tolerance, rtol=tolerance)
            assert projected.dtype == dtype
            assert bool(torch.isfinite(projected).all())
            assert bool((projected >= 0).all())
            risky_mask = available.clone()
            risky_mask[..., cash_index] = False
            assert bool((projected.masked_select(~available) == 0).all())
            # Casting an exact FP32 cap back to BF16 can round upward by at
            # most its representable cap value; this is the tight dtype-aware
            # bound, rather than a loose global epsilon.
            represented_cap = float(torch.tensor(cap, dtype=dtype))
            cap_bound = max(cap, represented_cap)
            if bool(risky_mask.any()):
                assert float(projected.masked_select(risky_mask).max()) <= cap_bound
            sum_tolerance = 1e-5 if dtype == torch.float32 else 4e-3
            expected_risky_mass = torch.minimum(
                torch.where(risky_mask, weights.float(), 0.0).sum(dim=-1),
                risky_mask.sum(dim=-1).float() * cap,
            )
            torch.testing.assert_close(
                torch.where(risky_mask, projected.float(), 0.0).sum(dim=-1),
                expected_risky_mass,
                atol=sum_tolerance,
                rtol=0,
            )
            torch.testing.assert_close(
                projected.float().sum(dim=-1),
                torch.ones(projected.shape[:-1]),
                atol=sum_tolerance,
                rtol=0,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sort_projection_matches_iterative_reference_gradients(dtype: torch.dtype) -> None:
    available = torch.tensor(
        [[True, True, True, True, False, True], [True, True, True, True, True, True]],
        dtype=torch.bool,
    )
    coefficients = torch.tensor([0.30, -0.20, 0.70, 1.10, -0.50, 0.40], dtype=dtype)
    values = torch.tensor(
        [[0.07, 0.61, 0.17, 0.09, 0.04, 0.02], [0.25, 0.05, 0.15, 0.20, 0.30, 0.05]],
        dtype=dtype,
    )
    sort_input = values.clone().requires_grad_()
    iterative_input = values.clone().requires_grad_()

    projected = project_capped_risky_simplex(
        sort_input,
        available,
        max_risky_weight=0.20,
    )
    reference = _iterative_capped_reference(iterative_input, available, 0.20)
    (projected * coefficients).sum().backward()
    (reference * coefficients).sum().backward()

    assert sort_input.grad is not None and iterative_input.grad is not None
    assert bool(torch.isfinite(sort_input.grad).all())
    tolerance = 2e-6 if dtype == torch.float32 else torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(
        sort_input.grad.float(),
        iterative_input.grad.float(),
        atol=tolerance,
        rtol=tolerance,
    )


def test_sort_projection_passes_double_precision_gradcheck() -> None:
    weights = torch.tensor([[0.07, 0.61, 0.17, 0.09, 0.04, 0.02]], dtype=torch.float64, requires_grad=True)
    available = torch.tensor([[True, True, True, True, False, True]])

    assert torch.autograd.gradcheck(
        lambda value: project_capped_risky_simplex(value, available, max_risky_weight=0.20),
        (weights,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )
