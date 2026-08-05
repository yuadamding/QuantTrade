from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_quant.training.hold30 import Hold30CanonicalRow, Hold30OriginReplay
from rl_quant.training.hold30_driver import (
    Hold30StateProviderBinding,
    Hold30TrainingSweep,
    Hold30TrialIdentity,
    run_hold30_trial,
    verify_hold30_run,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _TrainableProvider:
    trains_upstream_encoder = True
    hold30_provider_config = {"fixture": "differentiable-causal-state-v1"}


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))


class _Adapter:
    state_provider = _TrainableProvider()
    require_trainable_state_provider = True

    def canonical_pass(self, policy, sequence, roles):
        del policy
        rows = [Hold30CanonicalRow(utility=0.0) for _ in range(roles.n_positions - 1)]
        return {"update": sequence["update"]}, rows

    def replay_origins(self, policy, sequence, canonical_state, origins, roles):
        del roles
        assert canonical_state["update"] == sequence["update"]
        result = []
        for origin in origins.tolist():
            scale = policy.score.new_tensor(float(origin - 60))
            utility = (policy.score * scale).expand(31) / 31.0
            zero = policy.score * 0.0
            result.append(
                Hold30OriginReplay(
                    origin=origin,
                    utility_rows=utility,
                    discretionary_turnover=zero,
                    early_sale_mass=zero,
                    gate=zero,
                    gate_entropy=zero,
                )
            )
        return result


def _identity() -> Hold30TrialIdentity:
    return Hold30TrialIdentity(
        setting_id="hold30-a06-no-turn-penalty",
        fold_index=0,
        seed=17,
        executable_manifest_sha256=_digest("manifest"),
        fold_sha256=_digest("fold-0"),
    )


def _binding() -> Hold30StateProviderBinding:
    config = {"fixture": "differentiable-causal-state-v1"}
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return Hold30StateProviderBinding(
        provider_id="distributed-test-provider",
        provider_config=config,
        provider_config_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _sweeps() -> tuple[Hold30TrainingSweep, ...]:
    # Production performs 128 optimizer updates over the same chronological
    # fold sequence.  Three are enough to prove that identity/resume contract.
    sequence_sha = _digest("one-reused-chronological-sequence")
    return tuple(
        Hold30TrainingSweep(
            sweep_index=index,
            sweep_id=f"update-{index:03d}",
            sequence_sha256=sequence_sha,
            sequence={"update": index},
            n_positions=96,
        )
        for index in range(3)
    )


def _trainables() -> tuple[_Policy, torch.optim.AdamW]:
    policy = _Policy()
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-5
    )
    return policy, optimizer


def _distributed_worker(
    rank: int,
    port: int,
    root: str,
    resume: bool,
    max_sweeps: int | None,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.set_num_threads(1)
        policy, optimizer = _trainables()
        run_hold30_trial(
            policy,
            optimizer,
            _Adapter(),
            _identity(),
            _sweeps(),
            root,
            state_provider_binding=_binding(),
            resume=resume,
            max_sweeps=max_sweeps,
            world_size=2,
            rank=rank,
            qualification_update_override=3,
        )
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _spawn(root: Path, *, resume: bool = False, max_sweeps: int | None = None) -> None:
    mp.spawn(
        _distributed_worker,
        args=(_free_port(), str(root), resume, max_sweeps),
        nprocs=2,
        join=True,
    )


def _terminal_payload(root: Path) -> dict:
    return torch.load(root / "final-model.pt", map_location="cpu", weights_only=True)


def _terminal_checkpoint_receipt(root: Path) -> dict:
    return json.loads(
        (root / "checkpoints/update-000003.receipt.json").read_text(encoding="utf-8")
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_sum_reduction_matches_single_process_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    single_root = tmp_path / "single"
    single_policy, single_optimizer = _trainables()
    run_hold30_trial(
        single_policy,
        single_optimizer,
        _Adapter(),
        _identity(),
        _sweeps(),
        single_root,
        state_provider_binding=_binding(),
        qualification_update_override=3,
    )

    uninterrupted_root = tmp_path / "distributed-uninterrupted"
    _spawn(uninterrupted_root)
    uninterrupted_receipt = verify_hold30_run(
        uninterrupted_root,
        expected_identity=_identity(),
        allow_qualification_only=True,
    )
    assert uninterrupted_receipt["optimization_sweeps_complete"] is True
    assert uninterrupted_receipt["validation_checkpoint_selected"] is False
    assert uninterrupted_receipt["artifact_graph"]["checkpoints"][-1]["completed_sweeps"] == 3

    resumed_root = tmp_path / "distributed-resumed"
    _spawn(resumed_root, max_sweeps=1)
    _spawn(resumed_root, resume=True)
    verify_hold30_run(
        resumed_root,
        expected_identity=_identity(),
        allow_qualification_only=True,
    )

    single_state = _terminal_payload(single_root)["policy_state"]["score"]
    uninterrupted_state = _terminal_payload(uninterrupted_root)["policy_state"]["score"]
    resumed_state = _terminal_payload(resumed_root)["policy_state"]["score"]
    torch.testing.assert_close(uninterrupted_state, single_state, rtol=0.0, atol=1e-15)
    assert torch.equal(resumed_state, uninterrupted_state)

    single_metrics = json.loads((single_root / "metrics.json").read_text())["metrics"]
    distributed_metrics = json.loads(
        (uninterrupted_root / "metrics.json").read_text()
    )["metrics"]
    for single_row, distributed_row in zip(single_metrics, distributed_metrics, strict=True):
        for name in (
            "anchor_count",
            "objective",
            "utility_rows_replayed",
            "repeated_calendar_rows",
            "calendar_objective",
        ):
            assert distributed_row["metrics"][name] == pytest.approx(
                single_row["metrics"][name], abs=1e-15
            )

    uninterrupted_terminal = _terminal_checkpoint_receipt(uninterrupted_root)
    resumed_terminal = _terminal_checkpoint_receipt(resumed_root)
    assert resumed_terminal["policy_state_sha256"] == uninterrupted_terminal["policy_state_sha256"]
    assert resumed_terminal["optimizer_state_sha256"] == uninterrupted_terminal["optimizer_state_sha256"]
    assert resumed_terminal["rng_state_sha256s"] == uninterrupted_terminal["rng_state_sha256s"]
    assert len(resumed_terminal["rng_state_sha256s"]) == 2
