"""Package-owned causal state provider for the Hold-30 daily policy.

The chronological runtime deliberately owns only economic tensors.  This
module binds the raw-bar/frozen-context inputs that produce the actor state and
recomputes the exact origin state with autograd during Pass B.  A detached
``Hold30Sequence.decision_state`` is therefore never mistaken for a trained
raw/temporal encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Iterable, Sequence

import torch

from rl_quant.models.daily_policy import DailyCrossSectionPolicy
from rl_quant.training.hold30_runtime import Hold30Policy, Hold30Sequence


def _require_sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Hold30DailyPolicyInputs:
    """Causal model inputs for every decision-bearing position.

    Tensor layout is ``[batch, decision, asset, ...]``. ``day_bars_fn`` must
    return the raw-OHLCV bar tensor and validity mask for one absolute decision
    index.  The two external digests bind its immutable backing store and the
    frozen Stage-1 context cache; a callable name is not treated as evidence.
    """

    market_context: torch.Tensor
    stock_context: torch.Tensor
    news_raw: torch.Tensor
    news_mask: torch.Tensor
    available: torch.Tensor
    past_return: torch.Tensor
    past_return_valid: torch.Tensor
    day_bars_fn: Callable[[int], tuple[torch.Tensor, torch.Tensor]]
    source_axis_id: str
    raw_bars_sha256: str
    frozen_context_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("source_axis_id", self.source_axis_id)
        _require_sha256("raw_bars_sha256", self.raw_bars_sha256)
        _require_sha256("frozen_context_sha256", self.frozen_context_sha256)
        if not callable(self.day_bars_fn):
            raise TypeError("day_bars_fn must be callable")
        if self.market_context.ndim != 3:
            raise ValueError("market_context must have shape [batch, decision, context]")
        if self.stock_context.ndim != 4:
            raise ValueError("stock_context must have shape [batch, decision, asset, context]")
        batch, decisions, assets, context = self.stock_context.shape
        if tuple(self.market_context.shape) != (batch, decisions, context):
            raise ValueError("market_context and stock_context axes do not agree")
        expected = (batch, decisions, assets)
        if tuple(self.available.shape) != expected or self.available.dtype != torch.bool:
            raise ValueError("available must be boolean [batch, decision, asset]")
        if tuple(self.past_return.shape) != expected or not self.past_return.is_floating_point():
            raise ValueError("past_return must be floating [batch, decision, asset]")
        if tuple(self.past_return_valid.shape) != expected or self.past_return_valid.dtype != torch.bool:
            raise ValueError("past_return_valid must be boolean [batch, decision, asset]")
        if self.news_raw.ndim != 5 or self.news_raw.shape[:3] != expected:
            raise ValueError("news_raw must have shape [batch, decision, asset, article, feature]")
        if tuple(self.news_mask.shape) != tuple(self.news_raw.shape[:-1]) or self.news_mask.dtype != torch.bool:
            raise ValueError("news_mask must be boolean and match news_raw without its feature axis")
        floating = (
            self.market_context,
            self.stock_context,
            self.news_raw,
            self.past_return,
        )
        reference = self.stock_context
        if not all(
            value.is_floating_point()
            and value.device == reference.device
            and bool(torch.isfinite(value).all())
            for value in floating
        ):
            raise ValueError("all floating state inputs must be finite and share one device")
        for value in (self.news_mask, self.available, self.past_return_valid):
            if value.device != reference.device:
                raise ValueError("state masks must share the context device")

    @property
    def batch_size(self) -> int:
        return int(self.stock_context.shape[0])

    @property
    def n_decisions(self) -> int:
        return int(self.stock_context.shape[1])

    @property
    def num_assets(self) -> int:
        return int(self.stock_context.shape[2])


class Hold30DailyPolicyStateProvider:
    """Recompute rolling 63-session actor states from the current parameters."""

    trains_upstream_encoder = True

    def __init__(self, inputs: Hold30DailyPolicyInputs) -> None:
        if not isinstance(inputs, Hold30DailyPolicyInputs):
            raise TypeError("inputs must be Hold30DailyPolicyInputs")
        self.inputs = inputs

    @property
    def binding_config(self) -> dict[str, object]:
        inputs = self.inputs
        payload: dict[str, object] = {
            "schema_version": 2,
            "provider": f"{type(self).__module__}.{type(self).__qualname__}",
            "replay_batching": "union-raw-days-v1",
            "source_axis_id": inputs.source_axis_id,
            "raw_bars_sha256": inputs.raw_bars_sha256,
            "frozen_context_sha256": inputs.frozen_context_sha256,
            "batch_size": inputs.batch_size,
            "decision_count": inputs.n_decisions,
            "asset_count": inputs.num_assets,
            "context_dim": int(inputs.stock_context.shape[-1]),
            "news_shape": list(inputs.news_raw.shape[3:]),
        }
        payload["binding_sha256"] = _payload_sha256(payload)
        return payload

    @staticmethod
    def _policy(policy: Hold30Policy) -> DailyCrossSectionPolicy:
        if not isinstance(policy, DailyCrossSectionPolicy):
            raise TypeError("Hold30DailyPolicyStateProvider requires DailyCrossSectionPolicy")
        if policy.hold30_switches is None:
            raise ValueError("the daily policy must bind a registered Hold-30 setting")
        if policy.config.dropout != 0.0:
            raise ValueError("Hold-30 canonical/replay equality requires dropout=0")
        return policy

    def _validate_sequence(
        self,
        policy: DailyCrossSectionPolicy,
        sequence: Hold30Sequence,
    ) -> None:
        inputs = self.inputs
        if inputs.source_axis_id != sequence.axis_id:
            raise ValueError("state-provider source axis does not match the economic sequence")
        if inputs.n_decisions != sequence.n_positions - 1:
            raise ValueError("state-provider decision count does not match the sequence")
        if inputs.batch_size != sequence.batch_size or inputs.num_assets != sequence.num_assets:
            raise ValueError("state-provider batch/asset axes do not match the sequence")
        if inputs.stock_context.shape[-1] != policy.config.context_dim:
            raise ValueError("frozen context width does not match the policy")
        if inputs.news_raw.shape[-1] != policy.config.news_raw_dim:
            raise ValueError("news feature width does not match the policy")
        expected_available = sequence.decision_available[:-1].permute(1, 0, 2)
        if not torch.equal(inputs.available, expected_available):
            raise ValueError("state-provider availability differs from the economic decision mask")
        if sequence.decision_state.shape[-1] != policy.token_dim:
            raise ValueError("sequence decision-state placeholder width must equal policy token width")

    @staticmethod
    def _origins(origins: Iterable[int] | torch.Tensor, n_decisions: int) -> tuple[int, ...]:
        values = torch.as_tensor(origins, dtype=torch.long, device="cpu")
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("origins must be a non-empty one-dimensional integer sequence")
        result = tuple(int(value) for value in values.tolist())
        if any(not 0 <= value < n_decisions for value in result):
            raise ValueError("an origin lies outside the decision axis")
        return result

    def _token_variants(
        self,
        policy: DailyCrossSectionPolicy,
        start: int,
        end: int,
        raw_absolute_days: frozenset[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.inputs
        if not 0 <= start < end <= inputs.n_decisions:
            raise ValueError("token interval lies outside the decision axis")
        if any(day < start or day >= end for day in raw_absolute_days):
            raise ValueError("a requested raw day lies outside the token interval")

        def day_bars(local_index: int) -> tuple[torch.Tensor, torch.Tensor]:
            bars, mask = inputs.day_bars_fn(start + int(local_index))
            expected_bars = (
                inputs.batch_size,
                inputs.num_assets,
                policy.config.session_seconds,
                policy.config.bar_feature_dim,
            )
            if tuple(bars.shape) != expected_bars:
                raise ValueError(
                    f"day_bars_fn returned bars {tuple(bars.shape)}; expected {expected_bars}"
                )
            if tuple(mask.shape) != expected_bars[:-1] or mask.dtype != torch.bool:
                raise ValueError("day_bars_fn returned an invalid bar mask")
            if bars.device != inputs.stock_context.device or mask.device != bars.device:
                raise ValueError("raw bars and masks must share the state-input device")
            if not bars.is_floating_point() or not bool(torch.isfinite(bars).all()):
                raise ValueError("raw bars must be finite and floating point")
            return bars, mask

        raw_mask = [absolute in raw_absolute_days for absolute in range(start, end)]
        raw_tokens = policy._episode_tokens(  # noqa: SLF001 - package-owned token API
            inputs.market_context[:, start:end],
            inputs.stock_context[:, start:end],
            day_bars,
            inputs.news_raw[:, start:end],
            inputs.news_mask[:, start:end],
            inputs.past_return[:, start:end],
            inputs.past_return_valid[:, start:end],
            raw_mask,
            reload_ckpt=False,
        )
        no_raw_tokens = policy._episode_tokens(  # noqa: SLF001 - package-owned token API
            inputs.market_context[:, start:end],
            inputs.stock_context[:, start:end],
            day_bars,
            inputs.news_raw[:, start:end],
            inputs.news_mask[:, start:end],
            inputs.past_return[:, start:end],
            inputs.past_return_valid[:, start:end],
            [False] * (end - start),
            reload_ckpt=False,
        )
        return raw_tokens, no_raw_tokens

    @staticmethod
    def _select_window_tokens(
        policy: DailyCrossSectionPolicy,
        raw_tokens: torch.Tensor,
        no_raw_tokens: torch.Tensor,
    ) -> torch.Tensor:
        length = raw_tokens.shape[1]
        recent = int(policy.config.raw_recent_days)
        raw_start = 0 if recent <= 0 else max(0, length - recent)
        selector = torch.arange(length, device=raw_tokens.device) >= raw_start
        selector = selector.view(1, length, 1, 1)
        return torch.where(selector, raw_tokens, no_raw_tokens)

    def _state_from_tokens(
        self,
        policy: DailyCrossSectionPolicy,
        raw_tokens: torch.Tensor,
        no_raw_tokens: torch.Tensor,
        *,
        absolute_end: int,
        absolute_start: int,
    ) -> torch.Tensor:
        tokens = self._select_window_tokens(policy, raw_tokens, no_raw_tokens)
        available = self.inputs.available[:, absolute_start:absolute_end]
        state = policy.temporal_state(tokens, available)
        return state[:, -1]

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> Sequence[torch.Tensor] | torch.Tensor:
        daily_policy = self._policy(policy)
        self._validate_sequence(daily_policy, sequence)
        raw_tokens, no_raw_tokens = self._token_variants(
            daily_policy,
            0,
            self.inputs.n_decisions,
            frozenset(range(self.inputs.n_decisions)),
        )
        states: list[torch.Tensor] = []
        lookback = int(daily_policy.config.daily_lookback)
        for decision in range(self.inputs.n_decisions):
            start = max(0, decision + 1 - lookback)
            states.append(
                self._state_from_tokens(
                    daily_policy,
                    raw_tokens[:, start : decision + 1],
                    no_raw_tokens[:, start : decision + 1],
                    absolute_start=start,
                    absolute_end=decision + 1,
                )
            )
        return torch.stack(states)

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        daily_policy = self._policy(policy)
        self._validate_sequence(daily_policy, sequence)
        if isinstance(origin, bool) or not isinstance(origin, int):
            raise TypeError("origin must be an integer")
        return self.replay_origin_states(policy, sequence, torch.tensor([origin]))[0]

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Recompute one origin batch with one union-day raw-token pass.

        Each rolling window remains an independent temporal sequence.  The
        windows are left aligned and right padded before flattening
        ``origin x economic_batch`` into one true temporal batch; selecting
        each last real row therefore preserves the scalar window's positional
        encoding and causal history exactly.  Only the union of days selected
        by ``raw_recent_days`` is loaded and encoded.
        """

        daily_policy = self._policy(policy)
        self._validate_sequence(daily_policy, sequence)
        origin_values = self._origins(origins, self.inputs.n_decisions)
        lookback = int(daily_policy.config.daily_lookback)
        starts = tuple(max(0, origin + 1 - lookback) for origin in origin_values)
        ends = tuple(origin + 1 for origin in origin_values)
        union_start = min(starts)
        union_end = max(ends)

        recent = int(daily_policy.config.raw_recent_days)
        raw_days: set[int] = set()
        for start, end in zip(starts, ends, strict=True):
            raw_start = start if recent <= 0 else max(start, end - recent)
            raw_days.update(range(raw_start, end))
        raw_tokens, no_raw_tokens = self._token_variants(
            daily_policy,
            union_start,
            union_end,
            frozenset(raw_days),
        )

        max_length = max(end - start for start, end in zip(starts, ends, strict=True))
        token_windows: list[torch.Tensor] = []
        available_windows: list[torch.Tensor] = []
        lengths: list[int] = []
        for start, end in zip(starts, ends, strict=True):
            relative_start = start - union_start
            relative_end = end - union_start
            length = end - start
            raw_start = start if recent <= 0 else max(start, end - recent)
            absolute_days = torch.arange(start, end, device=raw_tokens.device)
            selector = (absolute_days >= raw_start).view(1, length, 1, 1)
            tokens = torch.where(
                selector,
                raw_tokens[:, relative_start:relative_end],
                no_raw_tokens[:, relative_start:relative_end],
            )
            available = self.inputs.available[:, start:end]
            if length < max_length:
                token_padding = tokens.new_zeros(
                    (
                        self.inputs.batch_size,
                        max_length - length,
                        self.inputs.num_assets,
                        tokens.shape[-1],
                    )
                )
                available_padding = torch.zeros(
                    (
                        self.inputs.batch_size,
                        max_length - length,
                        self.inputs.num_assets,
                    ),
                    dtype=torch.bool,
                    device=available.device,
                )
                tokens = torch.cat((tokens, token_padding), dim=1)
                available = torch.cat((available, available_padding), dim=1)
            token_windows.append(tokens)
            available_windows.append(available)
            lengths.append(length)

        origin_count = len(origin_values)
        batch = self.inputs.batch_size
        assets = self.inputs.num_assets
        token_dim = int(raw_tokens.shape[-1])
        batched_tokens = torch.stack(token_windows).reshape(
            origin_count * batch,
            max_length,
            assets,
            token_dim,
        )
        batched_available = torch.stack(available_windows).reshape(
            origin_count * batch,
            max_length,
            assets,
        )
        temporal = daily_policy.temporal_state(batched_tokens, batched_available).reshape(
            origin_count,
            batch,
            max_length,
            assets,
            token_dim,
        )
        return torch.stack(
            [temporal[index, :, length - 1] for index, length in enumerate(lengths)]
        )


__all__ = [
    "Hold30DailyPolicyInputs",
    "Hold30DailyPolicyStateProvider",
]
