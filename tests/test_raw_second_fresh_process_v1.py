"""Run in its OWN pytest process on LSF, after GPU-kernel pytest exits.

The orchestrating parent never initializes CUDA. Train, restore and verify
children exit sequentially, compatible with exclusive_process GPU allocation.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.lsf_gpu


def test_fresh_gpu_resume_and_cpu_inference_without_live_parent_cuda(tmp_path):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert not torch.cuda.is_initialized(), "Invoke this module in a separate pytest process"
    script = Path(__file__).with_name("raw_second_resume_probe.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run([sys.executable, "-B", str(script), "prepare", str(tmp_path)],
                             env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    hashes = json.loads((tmp_path / "hashes.json").read_text())
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    for mode in ("resume", "cpu-inference"):
        child_env = {**env, **({"CUDA_VISIBLE_DEVICES": ""} if mode == "cpu-inference" else {})}
        argv = [sys.executable, "-B", str(script), str(tmp_path / "catalog.json"), str(tmp_path / "resume.pt"),
                hashes["checkpoint_sha"], str(tmp_path / "reference.pt"), mode, str(tmp_path / "frozen.pt"), hashes["frozen_sha"]]
        result = subprocess.run(argv, env=child_env, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("prepare", ["prepare", "prepare-partitions"])
def test_actual_held_out_report_replays_in_a_fresh_gpu_process(tmp_path, prepare):
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert not torch.cuda.is_initialized(), "Invoke this module in a separate pytest process"
    script = Path(__file__).with_name("raw_second_experiment_probe.py")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run([sys.executable, "-B", str(script), prepare, str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-B", str(script), "verify", str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
