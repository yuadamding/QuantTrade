from __future__ import annotations

import hashlib
import json

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.evaluation.hold30_ensemble_runtime import (
    EnsemblePolicy,
    EnsembleStateProvider,
    Hold30EvaluationOnlyError,
)
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_ensemble import decide_hold30_ensemble
from rl_quant.training.hold30_runtime import Hold30ChronologicalRuntime, Hold30Sequence


_AXIS_ID = "a" * 64


class _MemberPolicy(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(value))
        self.calls = 0
        self.shared_ids: list[tuple[int, int, int]] = []
        self.states: list[torch.Tensor] = []

    def hold30_intent(
        self,
        state: torch.Tensor,
        weights: torch.Tensor,
        available: torch.Tensor,
        age: torch.Tensor,
    ) -> Hold30Intent:
        self.calls += 1
        self.shared_ids.append((id(weights), id(available), id(age)))
        self.states.append(state.detach().clone())
        return Hold30Intent(
            entry_scores=state[..., 0] + self.bias,
            hazard_residual=state[..., 1] - self.bias,
            exposure_residual=state[..., 2].mean(dim=-1) + self.bias,
        )


class _StateProvider:
    trains_upstream_encoder = True

    def __init__(
        self,
        states: torch.Tensor,
        available: torch.Tensor,
        *,
        axis_id: str = _AXIS_ID,
        raw_digest: str = "b" * 64,
        context_digest: str = "c" * 64,
        axis_asset_count: int | None = None,
    ) -> None:
        self.states = states
        self.decision_available = available
        self.calls = 0
        batch, decisions, assets = available.shape
        self.binding_config = {
            "schema_version": 2,
            "provider": "tests._StateProvider",
            "source_axis_id": axis_id,
            "raw_bars_sha256": raw_digest,
            "frozen_context_sha256": context_digest,
            "batch_size": batch,
            "decision_count": decisions,
            "asset_count": assets if axis_asset_count is None else axis_asset_count,
        }
        encoded = json.dumps(
            self.binding_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.binding_config["binding_sha256"] = hashlib.sha256(encoded).hexdigest()

    def canonical_states(self, policy: torch.nn.Module, sequence: Hold30Sequence) -> torch.Tensor:
        del policy, sequence
        self.calls += 1
        return self.states

    def replay_origin_states(self, *args: object) -> torch.Tensor:
        del args
        raise AssertionError("member replay must never be called")


def _sequence(*, decisions: int = 2, batch: int = 2, assets: int = 3) -> Hold30Sequence:
    positions = decisions + 1
    dtype = torch.float32
    weights = torch.tensor([[0.8, 0.1, 0.1]], dtype=dtype).expand(batch, -1).clone()
    available = torch.ones((positions, batch, assets), dtype=torch.bool)
    benchmark = weights.unsqueeze(0).expand(positions, -1, -1).clone()
    return Hold30Sequence(
        decision_state=torch.zeros((positions, batch, assets, 4), dtype=dtype),
        asset_returns=torch.zeros((decisions, batch, assets), dtype=dtype),
        decision_available=available,
        fill_membership=available.clone(),
        fill_availability=available.clone(),
        benchmark_weights=benchmark,
        risk_asset_caps=torch.ones_like(benchmark),
        risk_gross_max=torch.ones((positions, batch), dtype=dtype),
        benchmark_net_returns=torch.zeros((decisions, batch), dtype=dtype),
        initial_ledger=CohortLedger.from_weights(weights, cash_index=0),
        cost_rate=0.0,
        axis_id=_AXIS_ID,
    )


def _members() -> tuple[_MemberPolicy, ...]:
    return tuple(_MemberPolicy(float(index) / 10.0) for index in range(5))


def _providers(sequence: Hold30Sequence, *, feature: int = 4) -> tuple[_StateProvider, ...]:
    decisions = sequence.n_positions - 1
    available = sequence.decision_available[:-1].permute(1, 0, 2).clone()
    return tuple(
        _StateProvider(
            torch.full(
                (decisions, sequence.batch_size, sequence.num_assets, feature),
                float(index),
            ),
            available.clone(),
        )
        for index in range(5)
    )


def _assert_intents_close(actual: Hold30Intent, expected: Hold30Intent) -> None:
    for name in (
        "entry_scores",
        "target_logits",
        "gate",
        "hazard_residual",
        "exposure_residual",
    ):
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        assert (actual_value is None) == (expected_value is None)
        if actual_value is not None and expected_value is not None:
            torch.testing.assert_close(actual_value, expected_value)


def test_ensemble_policy_and_provider_require_five_distinct_members() -> None:
    members = _members()
    with pytest.raises(ValueError, match="exactly five"):
        EnsemblePolicy("H2", members[:4])
    with pytest.raises(ValueError, match="distinct"):
        EnsemblePolicy("H2", (members[0],) * 5)

    sequence = _sequence()
    providers = _providers(sequence)
    with pytest.raises(ValueError, match="exactly five"):
        EnsembleStateProvider(providers[:4])
    with pytest.raises(ValueError, match="distinct"):
        EnsembleStateProvider((providers[0],) * 5)


def test_state_provider_stacks_canonical_member_states_on_declared_axis() -> None:
    sequence = _sequence()
    members = _members()
    providers = _providers(sequence)
    ensemble = EnsemblePolicy("H2", members)
    provider = EnsembleStateProvider(providers)

    state = provider.canonical_states(ensemble, sequence)

    assert state.shape == (2, 2, 3, 5, 4)
    assert [member_provider.calls for member_provider in providers] == [1] * 5
    for member_index in range(5):
        torch.testing.assert_close(
            state[..., member_index, :],
            torch.full((2, 2, 3, 4), float(member_index)),
        )
    assert provider.binding_config["state_layout"] == (
        "decision,batch,asset,member,feature"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"axis_asset_count": 4}, "axes or provenance"),
        ({"raw_digest": "e" * 64}, "axes or provenance"),
    ],
)
def test_state_provider_rejects_differing_axes_or_provenance(
    change: dict[str, object],
    message: str,
) -> None:
    sequence = _sequence()
    providers = list(_providers(sequence))
    original = providers[-1]
    providers[-1] = _StateProvider(
        original.states,
        original.decision_available,
        **change,
    )
    with pytest.raises(ValueError, match=message):
        EnsembleStateProvider(providers)


def test_state_provider_rejects_differing_member_masks() -> None:
    sequence = _sequence()
    providers = list(_providers(sequence))
    bad_mask = providers[-1].decision_available.clone()
    bad_mask[0, 0, 1] = False
    providers[-1] = _StateProvider(providers[-1].states, bad_mask)
    with pytest.raises(ValueError, match="decision masks differ"):
        EnsembleStateProvider(providers)


def test_state_provider_rejects_a_tampered_member_binding_hash() -> None:
    sequence = _sequence()
    providers = list(_providers(sequence))
    providers[-1].binding_config["asset_count"] = 99
    with pytest.raises(ValueError, match="self-hash"):
        EnsembleStateProvider(providers)


def test_policy_permutation_calls_each_member_once_on_identical_economic_state() -> None:
    torch.manual_seed(7)
    members = _members()
    ensemble = EnsemblePolicy("H2", members)
    state = torch.randn(2, 3, 5, 4)
    weights = torch.tensor([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    available = torch.ones_like(weights, dtype=torch.bool)
    age = torch.randn(2, 3, 5)

    result = ensemble.hold30_intent(state, weights, available, age)

    assert isinstance(result, Hold30Intent)
    assert [member.calls for member in members] == [1] * 5
    assert all(member.shared_ids == [(id(weights), id(available), id(age))] for member in members)
    for member_index, member in enumerate(members):
        torch.testing.assert_close(member.states[0], state[:, :, member_index, :])


def test_policy_numerically_matches_direct_raw_output_aggregation() -> None:
    torch.manual_seed(11)
    wrapped_members = _members()
    direct_members = _members()
    ensemble = EnsemblePolicy("H2", wrapped_members)
    for member in direct_members:
        member.eval()
        member.requires_grad_(False)
    state = torch.randn(2, 3, 5, 4)
    weights = torch.tensor([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    available = torch.ones_like(weights, dtype=torch.bool)
    age = torch.randn(2, 3, 5)

    actual = ensemble.hold30_intent(state, weights, available, age)
    expected = decide_hold30_ensemble(
        "H2",
        direct_members,
        state.permute(2, 0, 1, 3),
        weights,
        available,
        age,
    ).aggregate_intent

    _assert_intents_close(actual, expected)


def test_ensemble_is_frozen_and_rejects_training_replay_or_bad_layout() -> None:
    members = _members()
    ensemble = EnsemblePolicy("H2", members)
    assert not ensemble.training
    assert all(not member.training for member in members)
    assert all(not parameter.requires_grad for parameter in ensemble.parameters())
    with pytest.raises(Hold30EvaluationOnlyError, match="training mode"):
        ensemble.train()
    with pytest.raises(Hold30EvaluationOnlyError, match="remain frozen"):
        ensemble.requires_grad_(True)

    weights = torch.tensor([[0.8, 0.1, 0.1]])
    available = torch.ones_like(weights, dtype=torch.bool)
    age = torch.zeros(1, 3, 5)
    with pytest.raises(ValueError, match="member axis"):
        ensemble.hold30_intent(torch.zeros(1, 3, 4, 5), weights, available, age)

    sequence = _sequence()
    provider = EnsembleStateProvider(_providers(sequence))
    with pytest.raises(ValueError, match="differentiable decision-state provider"):
        Hold30ChronologicalRuntime(
            "H2",
            state_provider=provider,
            require_trainable_state_provider=True,
        )
    with pytest.raises(Hold30EvaluationOnlyError, match="replay/update"):
        provider.replay_origin_states(
            ensemble,
            sequence,
            torch.tensor([0]),
        )


def test_provider_fails_if_live_mask_changes_after_binding() -> None:
    sequence = _sequence()
    providers = list(_providers(sequence))
    provider = EnsembleStateProvider(providers)
    ensemble = EnsemblePolicy("H2", _members())
    providers[2].decision_available[0, 0, 1] = False

    with pytest.raises(ValueError, match="changed after construction"):
        provider.canonical_states(ensemble, sequence)


def test_aggregate_intent_is_accepted_by_the_shared_economic_runtime() -> None:
    sequence = _sequence()
    members = _members()
    ensemble = EnsemblePolicy("H2", members)
    provider = EnsembleStateProvider(_providers(sequence))
    runtime = Hold30ChronologicalRuntime("H2", state_provider=provider)
    canonical_states = provider.canonical_states(ensemble, sequence)

    state = runtime.decide(
        ensemble,
        sequence,
        runtime.initial_state(sequence),
        decision_state=canonical_states[0],
    )

    assert state.pending_intent is not None
    assert isinstance(state.pending_intent.intent, Hold30Intent)
    assert [member.calls for member in members] == [1] * 5
