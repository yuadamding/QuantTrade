"""Evaluation-only five-member runtime adapter for Hold-30 ensembles.

The economic runtime owns one shared portfolio and age ledger.  This module
owns only the five frozen model paths: each member receives its own causal
state, but all five see the exact same pretrade weights, age summaries, and
decision-availability mask.  Raw model outputs are aggregated once through
the frozen output-space ensemble rule before the common execution path.

Canonical state exposed to :class:`Hold30ChronologicalRuntime` has layout
``[batch, asset, member, feature]``.  ``EnsemblePolicy`` performs the sole
permutation to ``[member, batch, asset, feature]`` required by
``decide_hold30_ensemble``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import torch
from torch import nn

from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_ensemble import decide_hold30_ensemble
from rl_quant.training.hold30_runtime import (
    Hold30DecisionStateProvider,
    Hold30Policy,
    Hold30Sequence,
)


HOLD30_ENSEMBLE_MEMBERS = 5
HOLD30_ENSEMBLE_STATE_LAYOUT = "decision,batch,asset,member,feature"


class Hold30EvaluationOnlyError(RuntimeError):
    """An evaluation-only ensemble was used by a training/replay path."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _binding_config(provider: object) -> dict[str, Any]:
    value = getattr(provider, "binding_config", None)
    if callable(value):
        value = value()
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError("each ensemble state provider must expose a mapping binding_config")
    result = dict(value)
    required = {
        "source_axis_id",
        "raw_bars_sha256",
        "frozen_context_sha256",
        "batch_size",
        "decision_count",
        "asset_count",
        "binding_sha256",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(
            "ensemble state-provider binding is missing: " + ", ".join(missing)
        )
    claimed = result["binding_sha256"]
    if not isinstance(claimed, str):
        raise ValueError("ensemble state-provider binding_sha256 is malformed")
    unsigned = dict(result)
    del unsigned["binding_sha256"]
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != claimed:
        raise ValueError("ensemble state-provider binding self-hash mismatch")
    return result


def _decision_available(provider: object) -> torch.Tensor:
    value = getattr(provider, "decision_available", None)
    if value is None:
        inputs = getattr(provider, "inputs", None)
        value = getattr(inputs, "available", None)
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.dtype != torch.bool:
        raise TypeError(
            "each ensemble state provider must expose boolean decision_available "
            "or inputs.available with layout [batch, decision, asset]"
        )
    return value


def _semantic_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    # A provider's self-hash is derived evidence, not provenance.  Every other
    # binding field must agree so additional future provenance fields fail
    # closed automatically.
    return {key: item for key, item in value.items() if key != "binding_sha256"}


def _member_states(
    value: Sequence[torch.Tensor] | torch.Tensor,
    *,
    member_index: int,
    sequence: Hold30Sequence,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        states = value
    else:
        rows = tuple(value)
        if not rows:
            raise ValueError(f"member {member_index} returned no canonical states")
        if not all(isinstance(row, torch.Tensor) for row in rows):
            raise TypeError(f"member {member_index} canonical states must be tensors")
        states = torch.stack(rows)
    expected_prefix = (
        sequence.n_positions - 1,
        sequence.batch_size,
        sequence.num_assets,
    )
    if states.ndim != 4 or tuple(states.shape[:3]) != expected_prefix:
        raise ValueError(
            f"member {member_index} canonical state must have shape "
            "[decision, batch, asset, feature]"
        )
    if not states.is_floating_point() or not bool(torch.isfinite(states).all()):
        raise ValueError(f"member {member_index} canonical state must be finite and floating")
    return states


class EnsemblePolicy(nn.Module):
    """Exactly five frozen member policies with one aggregate Hold30 intent."""

    def __init__(
        self,
        mechanism: str,
        member_policies: Sequence[nn.Module],
        *,
        cash_index: int = 0,
    ) -> None:
        super().__init__()
        members = tuple(member_policies)
        if len(members) != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("Hold-30 evaluation requires exactly five member policies")
        if len({id(member) for member in members}) != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("Hold-30 ensemble members must be five distinct policy objects")
        if mechanism not in {"H0", "H1", "H2", "H3"}:
            raise ValueError("mechanism must be H0, H1, H2, or H3")
        if isinstance(cash_index, bool) or not isinstance(cash_index, int) or cash_index < 0:
            raise ValueError("cash_index must be a non-negative integer")
        for member in members:
            if not isinstance(member, nn.Module) or not callable(
                getattr(member, "hold30_intent", None)
            ):
                raise TypeError("every ensemble member must be a torch module with hold30_intent")
            switches = getattr(member, "hold30_switches", None)
            if switches is not None and getattr(switches, "mechanism", None) != mechanism:
                raise ValueError("an ensemble member binds a different Hold-30 mechanism")
            member.eval()
            member.requires_grad_(False)
            member.zero_grad(set_to_none=True)
        self.mechanism = mechanism
        self.cash_index = cash_index
        self.members = nn.ModuleList(members)
        super().train(False)

    def train(self, mode: bool = True) -> "EnsemblePolicy":
        if mode:
            raise Hold30EvaluationOnlyError("EnsemblePolicy cannot enter training mode")
        super().train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> "EnsemblePolicy":
        if requires_grad:
            raise Hold30EvaluationOnlyError("EnsemblePolicy parameters must remain frozen")
        super().requires_grad_(False)
        return self

    def _assert_frozen(self) -> None:
        if self.training or any(member.training for member in self.members):
            raise Hold30EvaluationOnlyError("all Hold-30 ensemble members must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise Hold30EvaluationOnlyError("all Hold-30 ensemble parameters must remain frozen")

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        """Evaluate each member once and return only the aggregate intent."""

        self._assert_frozen()
        if not isinstance(state_t, torch.Tensor) or state_t.ndim != 4:
            raise ValueError("ensemble state must have layout [batch, asset, member, feature]")
        if state_t.shape[2] != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("ensemble state member axis must have length five")
        batch, assets, _members, _features = state_t.shape
        if tuple(prev_weights.shape) != (batch, assets):
            raise ValueError("shared pretrade weights must have shape [batch, asset]")
        if tuple(available.shape) != (batch, assets) or available.dtype != torch.bool:
            raise ValueError("shared availability must be boolean [batch, asset]")
        if age_summaries is None or age_summaries.ndim != 3:
            raise ValueError("shared age summaries must have shape [batch, asset, feature]")
        if tuple(age_summaries.shape[:2]) != (batch, assets):
            raise ValueError("shared age summaries do not match the economic axes")
        if not 0 <= self.cash_index < assets:
            raise ValueError("cash_index is outside the shared asset axis")
        member_states = state_t.permute(2, 0, 1, 3)
        with torch.no_grad():
            decision = decide_hold30_ensemble(
                self.mechanism,
                tuple(self.members),
                member_states,
                prev_weights,
                available,
                age_summaries,
                cash_index=self.cash_index,
            )
        return decision.aggregate_intent

    def forward(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor,
    ) -> Hold30Intent:
        return self.hold30_intent(state_t, prev_weights, available, age_summaries)


class EnsembleStateProvider:
    """Stack five corresponding canonical member states for evaluation only."""

    trains_upstream_encoder = False

    def __init__(self, member_providers: Sequence[Hold30DecisionStateProvider]) -> None:
        providers = tuple(member_providers)
        if len(providers) != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("Hold-30 evaluation requires exactly five member state providers")
        if len({id(provider) for provider in providers}) != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("Hold-30 member state providers must be distinct objects")
        for provider in providers:
            if not callable(getattr(provider, "canonical_states", None)):
                raise TypeError("every ensemble state provider must implement canonical_states")
        bindings = tuple(_binding_config(provider) for provider in providers)
        semantic = tuple(_semantic_binding(binding) for binding in bindings)
        if any(value != semantic[0] for value in semantic[1:]):
            raise ValueError("ensemble state-provider axes or provenance differ across members")
        masks = tuple(_decision_available(provider) for provider in providers)
        if any(
            mask.shape != masks[0].shape
            or mask.device != masks[0].device
            or not torch.equal(mask, masks[0])
            for mask in masks[1:]
        ):
            raise ValueError("ensemble state-provider decision masks differ across members")
        self.member_providers = providers
        self._bindings = bindings
        self._decision_available = masks[0].detach().clone()

    @property
    def binding_config(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "provider": f"{type(self).__module__}.{type(self).__qualname__}",
            "evaluation_only": True,
            "member_count": HOLD30_ENSEMBLE_MEMBERS,
            "state_layout": HOLD30_ENSEMBLE_STATE_LAYOUT,
            "member_bindings": list(self._bindings),
        }
        payload["binding_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
        return payload

    def _validate_live_bindings(self, sequence: Hold30Sequence) -> None:
        expected_mask = sequence.decision_available[:-1].permute(1, 0, 2)
        if not torch.equal(self._decision_available, expected_mask):
            raise ValueError("ensemble provider mask differs from the economic decision mask")
        for index, provider in enumerate(self.member_providers):
            binding = _binding_config(provider)
            if _semantic_binding(binding) != _semantic_binding(self._bindings[index]):
                raise ValueError(f"ensemble member {index} binding changed after construction")
            mask = _decision_available(provider)
            if not torch.equal(mask, self._decision_available):
                raise ValueError(f"ensemble member {index} decision mask changed after construction")
            if binding["source_axis_id"] != sequence.axis_id:
                raise ValueError(f"ensemble member {index} source axis differs from the sequence")
        expected_axes = (
            sequence.batch_size,
            sequence.n_positions - 1,
            sequence.num_assets,
        )
        if tuple(self._decision_available.shape) != expected_axes:
            raise ValueError("ensemble provider axes differ from the economic sequence")

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> torch.Tensor:
        if not isinstance(policy, EnsemblePolicy):
            raise TypeError("EnsembleStateProvider requires EnsemblePolicy")
        self._validate_live_bindings(sequence)
        states: list[torch.Tensor] = []
        with torch.no_grad():
            for index, (provider, member) in enumerate(
                zip(self.member_providers, policy.members, strict=True)
            ):
                states.append(
                    _member_states(
                        provider.canonical_states(member, sequence),
                        member_index=index,
                        sequence=sequence,
                    )
                )
        reference = states[0]
        if any(
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            for value in states[1:]
        ):
            raise ValueError("ensemble member canonical state axes, dtype, or device differ")
        # [decision, batch, asset, member, feature]; each runtime row is
        # therefore [batch, asset, member, feature].
        return torch.stack(states, dim=3)

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        del policy, sequence, origin
        raise Hold30EvaluationOnlyError("ensemble state replay is not permitted")

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        del policy, sequence, origins
        raise Hold30EvaluationOnlyError("ensemble state replay/update is not permitted")


__all__ = [
    "EnsemblePolicy",
    "EnsembleStateProvider",
    "HOLD30_ENSEMBLE_MEMBERS",
    "HOLD30_ENSEMBLE_STATE_LAYOUT",
    "Hold30EvaluationOnlyError",
]
