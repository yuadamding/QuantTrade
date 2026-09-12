from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from massive_adaptive_single_gpu_probe import load_checkpoint, write_bytes


def child(mode, root, log, *, cpu=False, timeout=360):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="17",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    argv = [sys.executable, "-B", str(Path(__file__).with_name("massive_adaptive_single_gpu_probe.py")),
            mode, str(root)]
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
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


@pytest.fixture(scope="module")
def single_run(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        pytest.skip("Requires an allocated single-GPU LSF regression worker")
    root = tmp_path_factory.mktemp("single-ppo")
    for mode in ("qualify", "resume"):
        log = root / (mode + ".log")
        assert child(mode, root, log) == 0, log.read_text()[-14000:]
    return root


def test_single_gpu_updates_actual_actor_and_critic_with_unchanged_global_budget(single_run):
    result = json.loads((single_run / "reference.json").read_text())
    assert result["runtime"]["visible_devices"] == 1
    assert result["actor_updated"] and result["critic_updated"]
    assert result["optimizer_steps_per_update"] == 4 and result["global_batch"] == 63
    assert result["peak_cuda_allocated_bytes"] > 0
    assert result["first_update"]["chronology_cursor"] == 63
    assert result["second_update"]["chronology_cursor"] == result["total_economic_transitions"] == 126


def test_fresh_process_restores_single_gpu_model_optimizers_rng_and_economics(single_run):
    result = json.loads((single_run / "resume.json").read_text())
    assert all(result[key] for key in ("fresh_process", "exact_model", "exact_optimizers", "exact_rng",
                                       "exact_actions", "exact_transitions", "exact_economic_state"))


def test_single_gpu_checkpoint_reloads_for_cpu_inference_without_evidence_writes(single_run, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in single_run.iterdir()}
    log = tmp_path / "cpu.log"
    assert child("cpu", single_run, log, cpu=True, timeout=50) == 0, log.read_text()[-14000:]
    assert json.loads(log.read_text())["cpu_inference"] is True
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in single_run.iterdir()}


def test_single_gpu_changed_checkpoint_bytes_rejected(single_run, tmp_path):
    info = json.loads((single_run / "reference.json").read_text())
    raw = bytearray((single_run / "single.pt").read_bytes())
    raw[len(raw) // 2] ^= 1
    path = tmp_path / "modified.pt"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes changed"):
        load_checkpoint(path, info["checkpoint_sha256"])


def test_single_gpu_checkpoint_cannot_be_overwritten_or_restored_into_wrong_config(single_run, tmp_path):
    path = single_run / "single.pt"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_bytes(path, b"replacement")
    assert path.read_bytes() == before
    log = tmp_path / "wrong-config.log"
    assert child("wrong-config", single_run, log, timeout=60) != 0
    assert "checkpoint and trainer roots differ" in log.read_text()


def test_single_gpu_synthetic_result_does_not_authorize_real_data(single_run):
    result = json.loads((single_run / "reference.json").read_text())
    checkpoint = load_checkpoint(single_run / "single.pt", result["checkpoint_sha256"])
    assert not result["real_sources_opened"] and not result["real_training_authorized"]
    assert not result["native_v5_qualified"] and not checkpoint.source_data_qualified
    assert not checkpoint.development_rl_training_authorized
    assert not checkpoint.profitability_reporting_authorized and not checkpoint.outer_evaluation_authorized
