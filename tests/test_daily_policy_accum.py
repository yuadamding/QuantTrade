from __future__ import annotations

import pytest
import torch

import rl_quant.training.daily_policy as daily_policy


_REAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, *, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _install_scalar_rollout(monkeypatch, stack_calls: list[tuple[int, ...]]) -> None:
    def fake_stack(episodes, indices, _device):
        stack_calls.append(tuple(indices))
        batch = len(indices)
        return {
            "coefficient": torch.tensor([episodes[index] for index in indices], dtype=torch.float32),
            "ret_valid": torch.ones(batch, 1, 2, dtype=torch.bool),
            "score_mask": torch.ones(batch, 1, dtype=torch.bool),
        }

    def fake_rollout(policy, batch, _cost, **_kwargs):
        nets = policy.weight * batch["coefficient"].reshape(-1, 1)
        zero = nets * 0.0
        return nets, zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(daily_policy, "_stack", fake_stack)
    monkeypatch.setattr(daily_policy, "_daily_rollout", fake_rollout)


def _train_once(
    monkeypatch,
    *,
    batch_days: int,
    accum_steps: int,
    stack_calls: list[tuple[int, ...]],
) -> tuple[_ScalarPolicy, _CountingSGD, list[tuple[int, int]], int, int]:
    _install_scalar_rollout(monkeypatch, stack_calls)
    order = [7, 0, 6, 1, 5, 2, 4, 3]
    sampler_calls: list[tuple[int, int]] = []

    def sampler(n_items: int, requested_batch: int) -> list[int]:
        sampler_calls.append((n_items, requested_batch))
        return order[: min(n_items, requested_batch)]

    reduce_calls = 0

    def reduce_once(_parameters) -> None:
        nonlocal reduce_calls
        reduce_calls += 1

    clip_calls = 0
    def clip_once(parameters, max_norm, *args, **kwargs):
        nonlocal clip_calls
        clip_calls += 1
        return _REAL_CLIP_GRAD_NORM(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_once)
    policy = _ScalarPolicy()
    optimizer = _CountingSGD(policy.parameters(), lr=0.1)
    daily_policy.train_daily_policy(
        policy,
        [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0],
        steps=1,
        lr=0.1,
        optimizer=optimizer,
        batch_days=batch_days,
        accum_steps=accum_steps,
        cost=0.0,
        risk_lambda=0.0,
        entropy_coef=0.0,
        budget_lambda=0.0,
        gate_entropy_coef=0.0,
        missing_label_penalty=0.0,
        schedule="constant",
        grad_clip=1_000.0,
        eval_every=0,
        device=torch.device("cpu"),
        episode_sampler=sampler,
        grad_reduce=reduce_once,
    )
    return policy, optimizer, sampler_calls, reduce_calls, clip_calls


def test_accumulated_policy_batch_matches_one_full_batch(monkeypatch) -> None:
    full_stacks: list[tuple[int, ...]] = []
    full, full_opt, full_samples, full_reduces, full_clips = _train_once(
        monkeypatch, batch_days=8, accum_steps=1, stack_calls=full_stacks
    )

    micro_stacks: list[tuple[int, ...]] = []
    accumulated, accum_opt, accum_samples, accum_reduces, accum_clips = _train_once(
        monkeypatch, batch_days=2, accum_steps=4, stack_calls=micro_stacks
    )

    torch.testing.assert_close(accumulated.weight, full.weight)
    assert full_samples == accum_samples == [(8, 8)]
    assert full_stacks == [(7, 0, 6, 1, 5, 2, 4, 3)]
    assert micro_stacks == [(7, 0), (6, 1), (5, 2), (4, 3)]
    assert len({index for micro in micro_stacks for index in micro}) == 8
    assert full_opt.step_calls == accum_opt.step_calls == 1
    assert full_reduces == accum_reduces == 1
    assert full_clips == accum_clips == 1


def test_accum_one_keeps_legacy_short_batch_sampler_request(monkeypatch) -> None:
    stack_calls: list[tuple[int, ...]] = []
    _install_scalar_rollout(monkeypatch, stack_calls)
    sampler_calls: list[tuple[int, int]] = []

    def sampler(n_items: int, requested_batch: int) -> list[int]:
        sampler_calls.append((n_items, requested_batch))
        return list(range(n_items))

    policy = _ScalarPolicy()
    daily_policy.train_daily_policy(
        policy,
        [1.0, 2.0],
        steps=1,
        batch_days=4,
        accum_steps=1,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        schedule="constant",
        eval_every=0,
        device=torch.device("cpu"),
        episode_sampler=sampler,
    )

    assert sampler_calls == [(2, 4)]
    assert stack_calls == [(0, 1)]


@pytest.mark.parametrize("accum_steps", [0, -1, False, 1.5])
def test_policy_accumulation_must_be_a_positive_integer(accum_steps) -> None:
    with pytest.raises(ValueError, match="accum_steps must be a positive integer"):
        daily_policy.train_daily_policy(
            _ScalarPolicy(),
            [1.0],
            steps=0,
            batch_days=1,
            accum_steps=accum_steps,
            device=torch.device("cpu"),
        )
