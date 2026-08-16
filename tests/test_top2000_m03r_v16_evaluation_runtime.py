from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import rl_quant.training.top2000_m03r_v16_evaluation_runtime as runtime
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)


@dataclass(frozen=True)
class _Sequence:
    decision_state: torch.Tensor


class _Policy:
    def __init__(self, training: bool) -> None:
        self.training = training
        self.source_policy = object()
        self.v16_setting = object()

    def train(self, mode: bool = True) -> _Policy:
        self.training = mode
        return self

    def eval(self) -> _Policy:
        return self.train(False)


@pytest.mark.parametrize("initial_training", (True, False))
def test_v16_inner_validation_restores_the_prior_policy_mode(
    monkeypatch: pytest.MonkeyPatch,
    initial_training: bool,
) -> None:
    cache = SimpleNamespace(
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        validate_unmodified=lambda: None,
    )
    risk = SimpleNamespace(
        cache_sha256=cache.cache_sha256,
        action_hash=cache.action_hash,
        exposures=object(),
        validate=lambda: None,
    )
    slab = SimpleNamespace(
        receipt=SimpleNamespace(
            cache_sha256=cache.cache_sha256,
            asset_axis_sha256=cache.action_hash,
            risk_source_receipt_sha256="c" * 64,
        ),
        require_fast_identity=lambda: None,
    )
    risk.receipt_sha256 = slab.receipt.risk_source_receipt_sha256
    built = SimpleNamespace(
        sequence=_Sequence(torch.zeros((345, 1, 3, 1))),
        identity=SimpleNamespace(receipt_sha256="d" * 64),
    )

    class _Provider:
        def __init__(self, _inputs: Any) -> None:
            pass

        def replay_origin_states(self, policy: Any, sequence: Any, origins: Any) -> Any:
            del policy, sequence, origins
            assert model.training is False
            return torch.zeros((63, 1, 3, 2))

    batch = SimpleNamespace(policy_state_binding_sha256="e" * 64)
    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        lambda *_args, **_kwargs: built,
    )
    monkeypatch.setattr(
        runtime,
        "move_and_bind_m03r_v16_sequence",
        lambda sequence, **_kwargs: sequence,
    )
    monkeypatch.setattr(
        runtime, "top2000_m03r_v7_decision_inputs", lambda _sequence: object()
    )
    monkeypatch.setattr(runtime, "Top2000M03RV7DecisionStateProvider", _Provider)
    monkeypatch.setattr(
        runtime,
        "build_m03r_v16_batch_from_origin_states",
        lambda *_args, **_kwargs: batch,
    )
    monkeypatch.setattr(runtime, "model_state_sha256", lambda _policy: "e" * 64)

    model = _Policy(initial_training)
    result = runtime.build_m03r_v16_inner_validation_batch(
        cache,
        render_m03r_v16_fold_geometries(1001)[0],
        risk,
        slab,
        model,
        device=torch.device("cpu"),
    )
    assert result is batch
    assert model.training is initial_training
