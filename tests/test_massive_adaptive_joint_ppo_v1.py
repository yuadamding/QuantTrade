from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch

from rl_quant.training.massive_adaptive_joint_ppo_v1 import (
    load_joint_ppo_cpu_checkpoint, rank_sample_indices, weighted_rank_loss,
)


def test_odd_batch_partition_keeps_every_transition_once():
    order = torch.randperm(63, generator=torch.Generator().manual_seed(17))
    parts = [rank_sample_indices(order, rank) for rank in (0, 1)]
    assert [len(part) for part in parts] == [32, 31]
    assert torch.equal(torch.cat(parts).sort().values, torch.arange(63))
    assert not set(parts[0].tolist()).intersection(parts[1].tolist())


def test_unequal_rank_sums_equal_global_mean_not_mean_of_means():
    values = torch.arange(63, dtype=torch.float64).square().requires_grad_()
    combined = (weighted_rank_loss(values[::2]) + weighted_rank_loss(values[1::2])) / 2
    torch.testing.assert_close(combined, values.mean(), rtol=0, atol=1e-12)
    combined.backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 1 / 63), rtol=0, atol=1e-15)
    assert abs(float((values[::2].mean() + values[1::2].mean()) / 2 - values.mean())) > 0.1


@pytest.mark.parametrize("order,rank", [
    (torch.arange(64), 0), (torch.zeros(63, dtype=torch.int64), 1),
    (torch.arange(63).float(), 0), (torch.arange(63), 2), (torch.arange(63), True),
])
def test_partition_rejects_padding_duplicates_and_invalid_ranks(order, rank):
    with pytest.raises(ValueError):
        rank_sample_indices(order, rank)


def run_child(argv, log, timeout):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="17",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def command(mode, root):
    return [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=2", "--max-restarts=0", "--monitor-interval=1",
            str(Path(__file__).with_name("massive_adaptive_joint_ppo_probe.py")), mode, str(root)]


@pytest.fixture(scope="module")
def joint_run(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        pytest.skip("Requires an allocated two-GPU LSF regression worker")
    root = tmp_path_factory.mktemp("joint-ppo")
    for mode in ("qualify", "resume"):
        log = root / (mode + ".log")
        code = run_child(command(mode, root), log, 360)
        assert code == 0, log.read_text()[-14000:]
    return root


def test_two_rank_actor_critic_matches_global_batch_and_rank_one_contributes(joint_run):
    info = json.loads((joint_run / "reference.json").read_text())
    assert info["rank_one_changes_gradient"] and info["replicas_equal"]
    assert info["full_batch_parameter_max_error"] <= 2e-5
    assert info["first_update"]["samples_per_rank"] == [32, 31]
    assert info["first_update"]["optimizer_steps"] == 4
    assert info["total_economic_transitions"] == 126
    assert info["real_training_authorized"] is False


def test_fresh_process_resume_restores_both_optimizers_rng_and_economic_episode(joint_run):
    result = json.loads((joint_run / "resume.json").read_text())
    assert all(result[key] for key in ("fresh_process", "exact_model", "exact_optimizers",
                                      "exact_rank_rng", "exact_actions", "exact_transitions", "exact_economic_state"))
    assert result["native_v5_qualified"] is False


def test_distributed_checkpoint_supports_nonmaterializing_cpu_inference(joint_run):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in joint_run.iterdir()}
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("massive_adaptive_joint_ppo_probe.py")),
                             "cpu", str(joint_run)], capture_output=True, timeout=40, check=True)
    info = json.loads(result.stdout)
    assert info["cpu_reload"] and info["economic_cursor"] == 63 and not info["native_v5_qualified"]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in joint_run.iterdir()}


def test_modified_committed_joint_checkpoint_is_rejected(joint_run, tmp_path):
    info = json.loads((joint_run / "reference.json").read_text())
    raw = bytearray((joint_run / "joint.pt").read_bytes())
    raw[len(raw) // 2] ^= 1
    path = tmp_path / "changed.pt"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes changed"):
        load_joint_ppo_cpu_checkpoint(path, info["checkpoint_sha256"])


def test_rank_one_failure_aborts_whole_trial_without_restart(joint_run):
    started = time.monotonic()
    log = joint_run / "failure.log"
    code = run_child(command("failure", joint_run), log, 70)
    assert code != 0
    assert time.monotonic() - started < 70
    assert not (joint_run / "incorrectly-survived.json").exists()
    assert "exitcode" in log.read_text() and "23" in log.read_text()
