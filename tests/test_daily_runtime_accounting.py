"""Focused regressions for compact daily storage and canonical portfolio accounting."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

from rl_quant.datasets.daily import build_daily_episodes, build_daily_raw_episodes, to_daily_raw_records
from rl_quant.datasets.streaming import LazyDay, LazyWindow
from rl_quant.models.decision_policy import DecisionPolicyConfig, DecisionPolicyHead, _NewsAggregator
from rl_quant.training.context_pretrain import encode_days
from rl_quant.training.daily_policy import _stack as _stack_daily
from rl_quant.training.decision_policy import _rollout


def _storage_nbytes(value: torch.Tensor) -> int:
    return value.untyped_storage().nbytes()


def _assert_compact(value: torch.Tensor) -> None:
    assert _storage_nbytes(value) == value.numel() * value.element_size()


class _ToyEncoder(nn.Module):
    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor, cov: torch.Tensor):
        del bar_mask
        batch, actions = bars.shape[:2]
        blocks = cov.shape[1]
        base = torch.arange(batch * blocks * actions * 3, device=bars.device, dtype=bars.dtype)
        per_stock = base.reshape(batch, blocks, actions, 3)
        market = per_stock.mean(dim=2)
        return per_stock, market


class _CovMaskEncoder(_ToyEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.seen_cov_valid: torch.Tensor | None = None

    def forward(self, bars, bar_mask, cov, cov_valid):
        self.seen_cov_valid = cov_valid.detach().clone()
        return super().forward(bars, bar_mask, cov)


def _encoded_day(*, days: int = 1, blocks: int = 4, actions: int = 3, steps: int = 8) -> dict:
    del days
    return {
        "date": "2022-01-03",
        "bars": torch.randn(actions, steps, 2),
        "bar_mask": torch.ones(actions, steps, dtype=torch.bool),
        "cov_blocks": torch.randn(blocks, actions, 2),
        "news_raw": torch.randn(blocks, actions, 2, 1),
        "news_mask": torch.ones(blocks, actions, 2, dtype=torch.bool),
        "avail": torch.ones(blocks, actions, dtype=torch.bool),
        "ret": torch.randn(blocks, actions),
        "ret_valid": torch.ones(blocks, actions, dtype=torch.bool),
        "day_open": torch.ones(actions),
        "day_close": torch.arange(actions, dtype=torch.float32) + 10.0,
        "day_close_valid": torch.ones(actions, dtype=torch.bool),
        "session_close_block": torch.tensor(blocks - 1),
    }


def test_encode_days_last_only_owns_eod_storage_and_dtype() -> None:
    day = _encoded_day()
    encoded = encode_days(
        _ToyEncoder(), [day], torch.device("cpu"), batch=1, last_only=True, output_dtype=torch.bfloat16
    )
    got = encoded[0]

    assert got["market"].shape == (3,)
    assert got["per_stock"].shape == (3, 3)
    assert got["avail"].shape == (3,)
    assert got["news_raw"].shape == (3, 2, 1)
    assert got["news_mask"].shape == (3, 2)
    assert got["market"].dtype == torch.bfloat16
    assert got["per_stock"].dtype == torch.bfloat16
    for key in ("market", "per_stock", "avail", "news_raw", "news_mask", "ret", "ret_valid"):
        _assert_compact(got[key])

    # The adapter accepts the already-EOD shape and returns owned fields rather than tail views.
    record = to_daily_raw_records(encoded)[0]
    assert record["market"].shape == (3,)
    assert record["per_stock"].shape == (3, 3)
    for key in ("market", "per_stock", "avail", "news_raw", "news_mask"):
        _assert_compact(record[key])


def test_encode_days_forwards_optional_covariate_validity() -> None:
    day = _encoded_day()
    day["cov_valid_blocks"] = torch.rand_like(day["cov_blocks"]) > 0.5
    encoder = _CovMaskEncoder()

    encode_days(encoder, [day], torch.device("cpu"), batch=1, last_only=True)

    assert encoder.seen_cov_valid is not None
    assert torch.equal(encoder.seen_cov_valid[0], day["cov_valid_blocks"])


def test_encode_days_last_only_lazy_overrides_do_not_pin_block_storage() -> None:
    day = _encoded_day()
    window = {
        key: value.unsqueeze(0)
        for key, value in day.items()
        if isinstance(value, torch.Tensor)
    }
    window.update({"dates": [day["date"]], "window": "w0", "n_days": 1, "n_blocks": 4})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "window.pt"
        torch.save(window, path)
        lazy_window = LazyWindow(path, {k: window[k] for k in ("dates", "window", "n_days", "n_blocks")})
        encoded = encode_days(
            _ToyEncoder(), [LazyDay(lazy_window, 0)], torch.device("cpu"), batch=1, last_only=True
        )

    got = encoded[0]
    assert isinstance(got, LazyDay)
    for key in ("market", "per_stock", "avail", "news_raw", "news_mask", "ret", "ret_valid"):
        _assert_compact(got[key])
    assert _storage_nbytes(got["per_stock"]) < 4 * 3 * 3 * got["per_stock"].element_size()


def _daily_record(index: int, *, actions: int = 3, steps: int = 8) -> dict:
    close = torch.tensor([1.0, 10.0 + index, 20.0 + 2 * index])
    return {
        "date": f"d{index}",
        "day_close": close,
        "day_open": close,
        "market": torch.tensor([float(index), 0.0]),
        "per_stock": torch.full((actions, 2), float(index)),
        "bars": torch.arange(actions * steps * 2, dtype=torch.float32).reshape(actions, steps, 2) + index * 100,
        "bar_mask": torch.ones(actions, steps, dtype=torch.bool),
        "news_raw": torch.zeros(actions, 1, 1),
        "news_mask": torch.zeros(actions, 1, dtype=torch.bool),
        "avail": torch.ones(actions, dtype=torch.bool),
    }


def test_daily_raw_uses_one_step_reward_aux_horizon_and_prefix_score_mask() -> None:
    records = [_daily_record(i) for i in range(9)]
    episodes = build_daily_raw_episodes(records, episode_len=9, horizon=4, score_start=3)
    assert len(episodes) == 1
    episode = episodes[0]

    assert episode["n_blocks"] == 7  # N - (exec_delay=1 + canonical horizon=1)
    torch.testing.assert_close(episode["ret"], episode["real_ret"])
    assert torch.equal(episode["ret_valid"], episode["real_ret_valid"])
    assert not torch.allclose(episode["ret"], episode["aux_ret"])
    assert episode["aux_ret_valid"][:, 1:].any(dim=1).tolist() == [True, True, True, True, False, False, False]
    assert episode["score_mask"].tolist() == [False, False, False, True, True, True, True]

    aux_limited = build_daily_raw_episodes(
        records, episode_len=9, horizon=4, require_aux_labels=True
    )
    assert aux_limited[0]["n_blocks"] == 4


def test_daily_raw_overlap_scores_each_warmed_decision_once_in_recent_tail() -> None:
    records = [_daily_record(i) for i in range(300)]
    episodes = build_daily_raw_episodes(
        records,
        episode_len=252,
        stride=15,
        horizon=21,
        exec_delay=1,
        score_tail=15,
    )

    scored_ids: list[str] = []
    for episode in episodes:
        local_scored = torch.where(episode["score_mask"])[0]
        assert bool((local_scored >= 252 - 15).all())
        scored_ids.extend(
            decision_id
            for decision_id, scored in zip(episode["decision_ids"], episode["score_mask"].tolist(), strict=True)
            if scored
        )

    # N - (exec_delay + one-step reward horizon) = 298 usable decisions.
    # The first 237 are causal history; all remaining decisions are covered,
    # including the one-row stride remainder at the data boundary.
    assert scored_ids == [f"d{i}" for i in range(237, 298)]
    assert len(scored_ids) == len(set(scored_ids))


def test_daily_raw_overlapping_episodes_share_one_chronological_backing() -> None:
    records = [_daily_record(i, actions=7) for i in range(80)]
    episodes = build_daily_raw_episodes(records, episode_len=30, stride=3, horizon=5)
    assert len(episodes) > 10

    # Logical episode volume is highly overlapping, but every tensor field has exactly one backing allocation.
    # data_ptr() differs by slice offset, so compare the storage allocation pointer rather than the view pointer.
    shared_fields = (
        "market", "per_stock", "news_raw", "news_mask", "avail", "ret", "ret_valid",
        "real_ret", "real_ret_valid", "aux_ret", "aux_ret_valid", "past_ret", "past_ret_valid",
        "bars", "bar_mask",
    )
    for key in shared_fields:
        storage_ptrs = {episode[key].untyped_storage().data_ptr() for episode in episodes}
        assert len(storage_ptrs) == 1, f"{key} was copied per overlapping episode"
        logical_bytes = sum(episode[key].numel() * episode[key].element_size() for episode in episodes)
        assert episodes[0][key].untyped_storage().nbytes() < logical_bytes

    # Compatibility aliases share the same reward storage too.
    assert episodes[0]["ret"].untyped_storage().data_ptr() == episodes[0]["real_ret"].untyped_storage().data_ptr()
    assert episodes[0]["ret_valid"].untyped_storage().data_ptr() == episodes[0]["real_ret_valid"].untyped_storage().data_ptr()


def test_daily_adapter_reuses_compact_eod_but_copies_oversized_block_view() -> None:
    encoded = _encoded_day(blocks=5, actions=4)
    compact_market = torch.randn(3)
    compact_per_stock = torch.randn(4, 3)
    encoded["market"] = compact_market
    encoded["per_stock"] = compact_per_stock

    record = to_daily_raw_records([encoded])[0]
    assert record["market"].untyped_storage().data_ptr() == compact_market.untyped_storage().data_ptr()
    assert record["per_stock"].untyped_storage().data_ptr() == compact_per_stock.untyped_storage().data_ptr()

    full_blocks = torch.randn(5, 4, 3)
    encoded["per_stock"] = full_blocks
    copied = to_daily_raw_records([encoded])[0]["per_stock"]
    assert copied.untyped_storage().data_ptr() != full_blocks.untyped_storage().data_ptr()
    _assert_compact(copied)
    torch.testing.assert_close(copied, full_blocks[-1])

    # Distributed TOP2000 overlays each EOD row as a view of one full date
    # timeline.  Keep that view until the episode builder performs the single
    # chronological stack; cloning here would create a redundant full copy
    # one day at a time before stacking it again.
    chronology = torch.randn(9, 4, 3)
    encoded["per_stock"] = chronology[4]
    shared = to_daily_raw_records([encoded])[0]["per_stock"]
    assert shared.untyped_storage().data_ptr() == chronology.untyped_storage().data_ptr()


def test_disabled_news_stays_scalar_backed_through_daily_episode_assembly() -> None:
    days = []
    for index in range(15):
        day = _encoded_day(blocks=4, actions=3)
        day["date"] = f"2022-01-{index + 3:02d}"
        day["day_close"] = torch.arange(3, dtype=torch.float32) + 10.0 + index
        day["news_raw"] = torch.zeros((), dtype=torch.float32).expand(4, 3, 32, 1)
        day["news_mask"] = torch.zeros((), dtype=torch.bool).expand(4, 3, 32)
        days.append(day)

    encoded = encode_days(_ToyEncoder(), days, torch.device("cpu"), batch=2, last_only=True)
    for day in encoded:
        assert day["news_raw"].untyped_storage().nbytes() == day["news_raw"].element_size()
        assert day["news_mask"].untyped_storage().nbytes() == day["news_mask"].element_size()

    records = to_daily_raw_records(encoded)
    episodes = build_daily_raw_episodes(records, episode_len=6, stride=6, horizon=2)
    assert len(episodes) == 2
    episode = episodes[0]
    assert episode["news_raw"].shape == (6, 3, 32, 1)
    assert episode["news_mask"].shape == (6, 3, 32)
    assert episode["news_raw"].untyped_storage().nbytes() == episode["news_raw"].element_size()
    assert episode["news_mask"].untyped_storage().nbytes() == episode["news_mask"].element_size()
    assert not bool(episode["news_raw"].any())
    assert not bool(episode["news_mask"].any())

    # TOP2000 uses two episodes per rank. Create one scalar directly on the destination instead of densifying via
    # either torch.stack or a cross-device copy.
    batch = _stack_daily(episodes, [0, 1], torch.device("cpu"))
    assert batch["news_raw"].shape == (2, 6, 3, 32, 1)
    assert batch["news_mask"].shape == (2, 6, 3, 32)
    assert batch["news_raw"].untyped_storage().nbytes() == batch["news_raw"].element_size()
    assert batch["news_mask"].untyped_storage().nbytes() == batch["news_mask"].element_size()


def test_all_masked_news_aggregator_skips_article_projection_with_exact_norm_semantics() -> None:
    aggregator = _NewsAggregator(raw_dim=1, out_dim=5)
    with torch.no_grad():
        aggregator.norm.bias.copy_(torch.tensor([0.4, -0.3, 0.2, 0.1, -0.5]))
        aggregator.norm.weight.copy_(torch.tensor([0.7, 1.2, 0.8, 1.1, 0.9]))

    projection_calls = 0

    def count_projection(_module, _inputs, _output) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handle = aggregator.proj.register_forward_hook(count_projection)
    scores = torch.zeros((), dtype=torch.float32).expand(12, 7, 32, 1)
    mask = torch.zeros((), dtype=torch.bool).expand(12, 7, 32)
    output = aggregator(scores, mask)
    handle.remove()

    assert projection_calls == 0
    assert output.shape == (12, 7, 5)
    torch.testing.assert_close(output, aggregator.norm.bias.view(1, 1, 5).expand_as(output))

    output.sum().backward()
    torch.testing.assert_close(aggregator.norm.bias.grad, torch.full((5,), 12 * 7.0))
    torch.testing.assert_close(aggregator.norm.weight.grad, torch.zeros(5))
    for parameter in aggregator.proj.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


def test_daily_single_episode_stack_avoids_host_copy() -> None:
    episodes = build_daily_raw_episodes([_daily_record(i) for i in range(8)], episode_len=6, horizon=2)
    batch = _stack_daily(episodes, [0], torch.device("cpu"))

    assert batch["ret"].shape == (1, *episodes[0]["ret"].shape)
    assert batch["ret"].untyped_storage().data_ptr() == episodes[0]["ret"].untyped_storage().data_ptr()
    torch.testing.assert_close(batch["ret"][0], episodes[0]["ret"])


def test_daily_batch_preserves_compact_bfloat16_context_on_device() -> None:
    records = [_daily_record(i) for i in range(8)]
    episodes = build_daily_raw_episodes(records, episode_len=6, stride=6, horizon=2)
    episodes[0]["market"] = episodes[0]["market"].bfloat16()
    episodes[0]["per_stock"] = episodes[0]["per_stock"].bfloat16()

    batch = _stack_daily(episodes, [0], torch.device("cpu"))

    assert episodes[0]["market"].dtype == torch.bfloat16
    assert episodes[0]["per_stock"].dtype == torch.bfloat16
    assert batch["market"].dtype == torch.bfloat16
    assert batch["per_stock"].dtype == torch.bfloat16
    assert batch["per_stock"].element_size() * 2 == torch.empty((), dtype=torch.float32).element_size()


def test_generic_daily_builder_stacks_only_final_raw_block() -> None:
    records = [_daily_record(i) for i in range(5)]
    episode = build_daily_episodes(records, episode_len=5, raw_block_steps=2)[0]
    expected = torch.stack([record["bars"][:, -2:] for record in records[:3]])

    assert episode["bars"].shape == (3, 3, 2, 2)
    torch.testing.assert_close(episode["bars"], expected)


class _PolicyAdapter(nn.Module):
    def __init__(self, policy: DecisionPolicyHead, *, batched: bool) -> None:
        super().__init__()
        self.policy = policy
        self.batched = batched
        self.batch_calls = 0
        self.step_calls = 0

    def encode_raw_policy_step(self, bars, bar_mask, step):
        self.step_calls += 1
        return self.policy.encode_raw_policy_step(bars, bar_mask, step)

    def encode_raw_policy_context(self, bars, bar_mask, target_steps):
        if not self.batched:
            raise AssertionError("step-only adapter should not expose the batched method")
        self.batch_calls += 1
        return self.policy.encode_raw_policy_context(bars, bar_mask, target_steps)

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)


class _StepOnlyPolicy(nn.Module):
    def __init__(self, policy: DecisionPolicyHead) -> None:
        super().__init__()
        self.policy = policy
        self.step_calls = 0

    def encode_raw_policy_step(self, bars, bar_mask, step):
        self.step_calls += 1
        return self.policy.encode_raw_policy_step(bars, bar_mask, step)

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)


def _policy_batch() -> dict:
    torch.manual_seed(3)
    batch, blocks, actions, block_steps = 2, 3, 4, 2
    ret = 0.01 * torch.randn(batch, blocks, actions)
    ret[:, :, 0] = 0.0
    return {
        "market": torch.randn(batch, blocks, 4),
        "per_stock": torch.randn(batch, blocks, actions, 4),
        "bars": torch.randn(batch, actions, blocks * block_steps, 2),
        "bar_mask": torch.ones(batch, actions, blocks * block_steps, dtype=torch.bool),
        "news_raw": torch.randn(batch, blocks, actions, 2, 1),
        "news_mask": torch.ones(batch, blocks, actions, 2, dtype=torch.bool),
        "ret": ret,
        "ret_valid": torch.ones(batch, blocks, actions, dtype=torch.bool),
        "avail": torch.ones(batch, blocks, actions, dtype=torch.bool),
    }


def test_evaluation_batches_raw_encoding_without_changing_rollout() -> None:
    torch.manual_seed(4)
    policy = DecisionPolicyHead(
        DecisionPolicyConfig(
            context_dim=4,
            bar_feature_dim=2,
            raw_policy_dim=4,
            raw_block_seconds=2,
            raw_policy_layers=0,
            raw_policy_heads=1,
            news_raw_dim=1,
            news_embed_dim=2,
            token_dim=8,
            n_heads=2,
            n_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        )
    ).eval()
    batch = _policy_batch()
    batched = _PolicyAdapter(policy, batched=True).eval()
    step_only = _StepOnlyPolicy(policy).eval()

    got = _rollout(batched, batch, cost=0.001)
    expected = _rollout(step_only, batch, cost=0.001)

    assert batched.batch_calls == 1
    assert batched.step_calls == 0
    assert step_only.step_calls == batch["per_stock"].shape[1]
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


class _DriftPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_previous: list[torch.Tensor] = []

    def encode_raw_policy_step(self, bars, bar_mask, step):
        del bar_mask, step
        return torch.zeros(bars.shape[0], bars.shape[1], 1)

    def forward(self, market, per_stock, raw_policy_ctx, news, news_mask, prev_weights, available):
        del market, per_stock, raw_policy_ctx, news, news_mask, available
        self.seen_previous.append(prev_weights.detach().clone())
        weights = torch.full_like(prev_weights, 0.5)
        gate = (prev_weights[:, 0] > 0.9).to(prev_weights.dtype)
        return weights, gate


def test_rollout_drifts_held_weights_and_charges_terminal_liquidation() -> None:
    batch = {
        "market": torch.zeros(1, 2, 1),
        "per_stock": torch.zeros(1, 2, 2, 1),
        "bars": torch.zeros(1, 2, 2, 1),
        "bar_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "news_raw": torch.zeros(1, 2, 2, 1, 1),
        "news_mask": torch.ones(1, 2, 2, 1, dtype=torch.bool),
        "ret": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "ret_valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "avail": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    policy = _DriftPolicy().eval()

    nets, _, _, cash_weight, turnover, _ = _rollout(policy, batch, cost=0.1)
    # ret[0] contains close information after decision 1.  It belongs in
    # execution accounting, not in the policy's decision-1 observation.
    torch.testing.assert_close(policy.seen_previous[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(policy.seen_previous[1], torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(cash_weight[0], torch.tensor([0.5, 1.0 / 3.0]))
    torch.testing.assert_close(turnover[0], torch.tensor([0.5, 2.0 / 3.0]))
    torch.testing.assert_close(nets[0], torch.tensor([0.45, -0.1 * 2.0 / 3.0]))

    nets_open, _, _, _, turn_open, _ = _rollout(policy, batch, cost=0.1, terminal_liquidate=False)
    torch.testing.assert_close(turn_open[0], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(nets_open[0], torch.tensor([0.45, 0.0]))

    # An unlabeled cache tail must not swallow the exit cost; charge it on the final scored transition.
    tailed = {
        "market": torch.zeros(1, 3, 1),
        "per_stock": torch.zeros(1, 3, 2, 1),
        "bars": torch.zeros(1, 2, 3, 1),
        "bar_mask": torch.ones(1, 2, 3, dtype=torch.bool),
        "news_raw": torch.zeros(1, 3, 2, 1, 1),
        "news_mask": torch.ones(1, 3, 2, 1, dtype=torch.bool),
        "ret": torch.tensor([[[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]),
        "ret_valid": torch.tensor([[[True, True], [True, True], [True, False]]]),
        "avail": torch.ones(1, 3, 2, dtype=torch.bool),
    }
    tail_nets, _, _, _, tail_turn, _ = _rollout(policy, tailed, cost=0.1)
    torch.testing.assert_close(tail_turn[0], torch.tensor([0.5, 2.0 / 3.0, 0.0]))
    torch.testing.assert_close(tail_nets[0], torch.tensor([0.45, -0.1 * 2.0 / 3.0, 0.0]))
