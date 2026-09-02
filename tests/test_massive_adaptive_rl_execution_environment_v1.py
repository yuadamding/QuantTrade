from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    build_seeded_massive_adaptive_ppo_model_v1,
    massive_adaptive_ppo_initial_model_state_receipt_v1,
)
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    MassiveAdaptiveRLExecutionEnvironmentV1Error,
    capture_massive_adaptive_rl_execution_environment_v1,
    load_massive_adaptive_rl_execution_environment_authority_v1,
    massive_adaptive_rl_deterministic_execution_v1,
    materialize_massive_adaptive_rl_execution_environment_authority_v1,
    verify_massive_adaptive_rl_execution_environment_integrity_v1,
    verify_massive_adaptive_rl_execution_environment_replay_v1,
)
from rl_quant.workflows import massive_adaptive_rl_execution_environment_v1 as module
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_process_state_v1 import (
    preserve_massive_adaptive_rl_process_rng_state_v1,
)


def _patch_clean_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    actual_git = module._git

    def clean_git(repository_root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
            return ""
        if arguments[:2] == ("ls-files", "--others"):
            return ""
        return actual_git(repository_root, *arguments)

    monkeypatch.setattr(module, "_git", clean_git)


def test_seeded_model_forces_cpu_float32_despite_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
    finally:
        torch.set_default_dtype(previous)

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_execution_environment_captures_and_restores_deterministic_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_repository(monkeypatch)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="execution-environment-capture",
        execution_device_specification="cpu",
    )
    initial = massive_adaptive_ppo_initial_model_state_receipt_v1(seed=17)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )

    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        first = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )
        second = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )

    assert first.semantic_receipt_sha256 == second.semantic_receipt_sha256
    assert first.source_data_qualified
    assert not first.source_transaction_verified
    assert first.runtime_environment_replayed
    assert first.execution_device_type == "cpu"
    assert first.parameter_dtype == "torch.float32"
    assert first.deterministic_algorithms
    assert not first.float32_matmul_tf32
    assert not first.cudnn_tf32
    assert not first.cudnn_benchmark
    assert first.cudnn_deterministic
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    ) == previous

    promoted = replace(
        first,
        deterministic_algorithms=False,
        semantic_receipt_sha256="0" * 64,
    )
    promoted = replace(
        promoted,
        semantic_receipt_sha256=semantic_sha256(promoted.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLExecutionEnvironmentV1Error,
        match="differs",
    ):
        promoted.validate()


def test_execution_environment_persists_integrity_and_replays_exact_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_repository(monkeypatch)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="execution-environment-persistence",
        execution_device_specification="cpu",
    )
    initial = massive_adaptive_ppo_initial_model_state_receipt_v1(seed=17)
    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        captured = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )
        materialized = (
            materialize_massive_adaptive_rl_execution_environment_authority_v1(
                root=tmp_path,
                artifact_id="execution-environment-persistence-fold0",
                authority=captured,
                committed_at_ms=10_000,
            )
        )

    assert materialized.source_transaction_verified
    assert not materialized.runtime_environment_replayed
    assert materialized.tracked_source_inventory == captured.tracked_source_inventory

    def forbidden_git(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("portable integrity verification invoked git")

    with monkeypatch.context() as isolated:
        isolated.setattr(module, "_git", forbidden_git)
        integrity = verify_massive_adaptive_rl_execution_environment_integrity_v1(
            root=tmp_path,
            artifact_id="execution-environment-persistence-fold0",
            verified_at_ms=10_001,
        )
    loaded = load_massive_adaptive_rl_execution_environment_authority_v1(
        root=tmp_path,
        artifact_id="execution-environment-persistence-fold0",
        verified_at_ms=10_002,
    )
    assert integrity.semantic_receipt_sha256 == captured.semantic_receipt_sha256
    assert loaded.semantic_receipt_sha256 == integrity.semantic_receipt_sha256
    assert integrity.source_transaction_verified
    assert not integrity.runtime_environment_replayed

    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        replayed = verify_massive_adaptive_rl_execution_environment_replay_v1(
            authority=integrity,
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )
    assert replayed.source_transaction_verified
    assert replayed.runtime_environment_replayed
    assert replayed.semantic_receipt_sha256 == captured.semantic_receipt_sha256

    clean_git = module._git

    def dirty_git(repository_root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
            return " M src/rl_quant/workflows/runtime_change.py"
        return clean_git(repository_root, *arguments)

    monkeypatch.setattr(module, "_git", dirty_git)
    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        with pytest.raises(
            MassiveAdaptiveRLExecutionEnvironmentV1Error,
            match="did not replay",
        ):
            verify_massive_adaptive_rl_execution_environment_replay_v1(
                authority=integrity,
                manifest=manifest,
                initial_model_state_receipt_sha256=initial,
                device="cpu",
            )


def test_dirty_or_untracked_runtime_source_is_explicitly_nonauthorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_repository(monkeypatch)
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="execution-environment-dirty-source",
        execution_device_specification="cpu",
    )
    initial = massive_adaptive_ppo_initial_model_state_receipt_v1(seed=17)
    clean_git = module._git

    def dirty_git(repository_root: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
            return " M src/rl_quant/workflows/example.py"
        return clean_git(repository_root, *arguments)

    monkeypatch.setattr(module, "_git", dirty_git)
    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        dirty = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )
    dirty.validate()
    assert not dirty.tracked_worktree_clean
    assert dirty.tracked_worktree_status
    assert not dirty.source_data_qualified

    monkeypatch.setattr(module, "_git", clean_git)
    monkeypatch.setattr(
        module,
        "_untracked_runtime_source_inventory",
        lambda _root: (
            ("src/rl_quant/untracked_runtime.py", semantic_sha256("untracked")),
        ),
    )
    with massive_adaptive_rl_deterministic_execution_v1(device="cpu"):
        untracked = capture_massive_adaptive_rl_execution_environment_v1(
            manifest=manifest,
            initial_model_state_receipt_sha256=initial,
            device="cpu",
        )
    untracked.validate()
    assert untracked.tracked_worktree_clean
    assert untracked.untracked_runtime_source_count == 1
    assert not untracked.source_data_qualified


def test_process_rng_preservation_restores_all_cpu_generators() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    with preserve_massive_adaptive_rl_process_rng_state_v1():
        random.seed(404)
        np.random.seed(505)
        torch.manual_seed(606)
        random.random()
        np.random.random()
        torch.rand(3)

    numpy_after = np.random.get_state()
    assert random.getstate() == python_before
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
