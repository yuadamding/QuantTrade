from __future__ import annotations

from dataclasses import replace

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
    massive_adaptive_rl_deterministic_execution_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
)


def test_seeded_model_forces_cpu_float32_despite_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = build_seeded_massive_adaptive_ppo_model_v1(seed=17)
    finally:
        torch.set_default_dtype(previous)

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_execution_environment_captures_and_restores_deterministic_switches() -> None:
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
