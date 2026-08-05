from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rl_quant.protocol.hold30_alpha_v3 import HOLD30_ALPHA_MECH8_IDS
from rl_quant.training.hold30_alpha_driver import (
    HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS,
    HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS,
    HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS,
    HOLD30_ALPHA_SYNTHETIC_A06_BINDING,
    Hold30AlphaDriverError,
    Hold30AlphaProductionPreflightBindings,
    build_hold30_alpha_synthetic_objective_config,
    qualify_hold30_alpha_full_policy_cpu_restart_parity,
    qualify_hold30_alpha_full_policy_cpu_two_rank_parity,
    require_hold30_alpha_executable_plan,
    resolve_hold30_alpha_synthetic_route,
    run_hold30_alpha_synthetic_qualification,
    verify_hold30_alpha_synthetic_run,
)
from rl_quant.training.hold30_alpha_pilot_plan import (
    build_hold30_alpha_pilot_training_plan,
)
from rl_quant.training.hold30_alpha_plan import (
    unresolved_hold30_alpha_training_plan,
)

RUNNABLE = HOLD30_ALPHA_MECH8_IDS


def _pilot_plan():
    return build_hold30_alpha_pilot_training_plan(
        a06_optimizer_spec_receipt_sha256=HOLD30_ALPHA_SYNTHETIC_A06_BINDING
    )


def test_driver_routes_all_eight_settings_and_consumes_exact_pilot_configs() -> None:
    routes = tuple(
        resolve_hold30_alpha_synthetic_route(setting_id)
        for setting_id in HOLD30_ALPHA_MECH8_IDS
    )
    assert tuple(route.setting_id for route in routes) == HOLD30_ALPHA_MECH8_IDS
    assert routes[0].mechanism == "H0"
    assert all(route.mechanism == "H2" for route in routes[1:])
    assert tuple(route.objective_kind for route in routes[:2]) == (
        "absolute-net-log-return",
        "absolute-net-log-return",
    )
    assert all(route.objective_kind == "v3-global-two-pass" for route in routes[2:])
    assert routes[6].separate_overlay is True
    assert all(route.runnable_in_synthetic_driver for route in routes)
    assert all(route.blocker is None for route in routes)

    pilot = {config.setting_id: config for config in _pilot_plan().objective_configs}
    assert (
        build_hold30_alpha_synthetic_objective_config(HOLD30_ALPHA_MECH8_IDS[0]) is None
    )
    assert (
        build_hold30_alpha_synthetic_objective_config(HOLD30_ALPHA_MECH8_IDS[1]) is None
    )
    for setting_id in HOLD30_ALPHA_MECH8_IDS[2:]:
        synthetic = build_hold30_alpha_synthetic_objective_config(setting_id)
        assert synthetic is not None
        expected = replace(
            pilot[setting_id],
            qualification_math_test_only=True,
            total_sharpe_epsilon=(
                1e-6
                if setting_id == "hold30a-a06-sharpe-overlay"
                else pilot[setting_id].total_sharpe_epsilon
            ),
        )
        assert asdict(synthetic) == asdict(expected)


def test_production_preflight_reports_data_image_and_ddp_without_writing(
    tmp_path: Path,
) -> None:
    plan = _pilot_plan()
    output = tmp_path / "must-not-exist"
    with pytest.raises(Hold30AlphaDriverError) as caught:
        require_hold30_alpha_executable_plan(plan)
    message = str(caught.value)
    for blocker in HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS:
        assert blocker in message
    for field in (
        *HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS,
        "container_image_digest",
        "ddp_pass_b_gradient_parity_receipt_sha256",
    ):
        assert f"missing-binding:{field}" in message
    assert not output.exists()

    complete_bindings = Hold30AlphaProductionPreflightBindings(
        **{name: "b" * 64 for name in HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS},
        container_image_digest="sha256:" + "c" * 64,
        ddp_pass_b_gradient_parity_receipt_sha256="d" * 64,
    )
    with pytest.raises(Hold30AlphaDriverError) as complete:
        require_hold30_alpha_executable_plan(plan, complete_bindings)
    assert "missing-binding" not in str(complete.value)
    for blocker in HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS:
        assert blocker in str(complete.value)

    with pytest.raises(Hold30AlphaDriverError, match="pilot-training-plan"):
        require_hold30_alpha_executable_plan(
            unresolved_hold30_alpha_training_plan(), complete_bindings
        )


@pytest.mark.parametrize("setting_id", RUNNABLE)
def test_runnable_settings_close_one_update_synthetic_artifact_graph(
    setting_id: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / setting_id
    receipt = run_hold30_alpha_synthetic_qualification(setting_id, root)
    assert receipt == verify_hold30_alpha_synthetic_run(root)
    assert receipt["setting_id"] == setting_id
    assert receipt["qualification_only"] is True
    assert receipt["launch_authorized"] is False
    assert receipt["production_data_consumed"] is False
    assert receipt["gpu_consumed"] is False
    assert receipt["updates"] == 1
    assert receipt["scored_sessions"] == 63
    assert len(receipt["pilot_training_plan_receipt_sha256"]) == 64
    assert receipt["initial_model_state_sha256"] != receipt["final_model_state_sha256"]
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["checkpoint_gate_eligibility"] in {True, False}
    assert metrics["minimum_update_satisfied"] is False
    assert metrics["checkpoint_eligible"] is False

    with pytest.raises(Hold30AlphaDriverError, match="refusing to overwrite"):
        run_hold30_alpha_synthetic_qualification(setting_id, root)


def test_a06_persists_disjoint_optimizer_evidence_and_tampering_breaks_receipt(
    tmp_path: Path,
) -> None:
    a06 = tmp_path / "a06"
    receipt = run_hold30_alpha_synthetic_qualification(
        "hold30a-a06-sharpe-overlay", a06
    )
    assert len(receipt["a06_initial_optimizer_state_receipt_sha256"]) == 64
    assert len(receipt["a06_optimizer_spec_receipt_sha256"]) == 64
    assert len(receipt["a06_post_update_optimizer_state_receipt_sha256"]) == 64
    assert (
        receipt["a06_initial_optimizer_state_receipt_sha256"]
        != receipt["a06_post_update_optimizer_state_receipt_sha256"]
    )
    metrics = json.loads((a06 / "metrics.json").read_text(encoding="utf-8"))
    evidence = metrics["optimizer_update_evidence"]
    assert evidence["gradient_isolation_verified"] is True
    assert evidence["three_stream_contract_verified"] is True
    assert evidence["gradient_reduction"] == "SUM"
    assert evidence["alpha_core_optimizer_steps"] == 1
    assert evidence["overlay_optimizer_steps"] == 1
    assert (
        evidence["post_update_evaluation_point_id"]
        != evidence["pre_update_evaluation_point_id"]
    )
    assert (
        evidence["post_update_optimizer_state_receipt"]
        ["parent_state_receipt_sha256"]
        == evidence["initial_optimizer_state_receipt_sha256"]
    )

    root = tmp_path / "tamper"
    run_hold30_alpha_synthetic_qualification("hold30a-m03-alpha-core", root)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["gradient_norm"] *= 2.0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(Hold30AlphaDriverError, match="hash mismatch"):
        verify_hold30_alpha_synthetic_run(root)


def test_distinct_shard_full_policy_cpu_parity_is_exact_and_non_gpu() -> None:
    receipt = qualify_hold30_alpha_full_policy_cpu_two_rank_parity()
    assert receipt.setting_ids == HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS
    assert receipt.world_size == 2
    assert receipt.local_paths_per_rank == 1
    assert receipt.exact_gradient_parity is True
    assert receipt.exact_parameter_parity is True
    assert receipt.exact_optimizer_state_parity is True
    assert receipt.gpu_consumed is False
    assert receipt.h100_parity_claimed is False
    assert receipt.launch_authorized is False
    assert len(receipt.setting_evidence_sha256) == 8
    assert len(receipt.receipt_id) == 64


def test_full_policy_all_eight_two_update_restart_receipt_is_non_gpu() -> None:
    receipt = qualify_hold30_alpha_full_policy_cpu_restart_parity()
    assert receipt.setting_ids == HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS
    assert receipt.updates == 2
    assert receipt.checkpoint_update == 1
    assert receipt.exact_model_state_parity is True
    assert receipt.exact_optimizer_state_parity is True
    assert receipt.exact_update_receipt_parity is True
    assert receipt.gpu_consumed is False
    assert receipt.h100_parity_claimed is False
    assert receipt.launch_authorized is False
    assert len(receipt.setting_evidence_sha256) == 8
    assert len(receipt.receipt_id) == 64
