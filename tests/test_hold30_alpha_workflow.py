from __future__ import annotations

import copy
from pathlib import Path

import pytest

import rl_quant.workflows.hold30_alpha_prelockbox as workflow
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
)
from rl_quant.workflows.hold30_alpha_prelockbox import (
    HOLD30_ALPHA_SOFTWARE_SCHEMA,
    Hold30AlphaQualificationError,
    qualify_hold30_alpha_software,
    resolve_hold30_alpha_qualification_inventory,
    verify_hold30_alpha_software_receipt,
)


def test_v3_inventory_binds_every_alpha_source_test_and_document() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = resolve_hold30_alpha_qualification_inventory(root)
    assert set(inventory.component_sources) == {
        source for source, _test in workflow.V3_COMPONENT_SOURCE_TESTS
    }
    assert "tests/test_hold30_alpha_evaluation.py" in inventory.component_tests
    assert "tests/test_hold30_alpha_workflow.py" in inventory.component_tests
    assert "docs/prelockbox_hold30_alpha_mech8_v3.md" in inventory.evidence_files
    assert "docs/prelockbox_hold30_alpha_evaluation_v3.md" in inventory.evidence_files
    assert "docs/daily_hold30_policy_rfc.md" in inventory.evidence_files
    assert "docs/prelockbox_hold30_mech8_v2.md" in inventory.evidence_files
    assert "docs/prelockbox_hold30_active_alpha_m03r_v7.md" in inventory.evidence_files
    assert (
        "docs/prelockbox_hold30_active_alpha_m03r_v7_experiment.md"
        in inventory.evidence_files
    )
    discovered_runtime = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "rl_quant").rglob("*hold30*.py")
    } - set(inventory.excluded_runtime_history)
    assert discovered_runtime == set(inventory.component_sources) | set(
        inventory.integration_sources
    ) - set(workflow.V3_REUSED_NON_HOLD30_SOURCES)
    assert set(inventory.component_tests) | set(inventory.integration_tests) == {
        path.relative_to(root).as_posix()
        for path in (root / "tests").glob("test_hold30*.py")
    }
    assert inventory.excluded_runtime_history == (
        "src/rl_quant/workflows/hold30_prelockbox.py",
    )
    assert set(inventory.qualified_files) >= set(inventory.component_sources)
    assert {
        "src/rl_quant/models/hold30_confidence_v6.py",
        "src/rl_quant/models/hold30_exit_action_v6.py",
        "src/rl_quant/execution/hold30_exit_v6.py",
        "src/rl_quant/protocol/hold30_m03r_confidence.py",
        "src/rl_quant/training/hold30_m03r_confidence_fit.py",
        "src/rl_quant/training/hold30_m03r_confidence_objective_v6.py",
        "src/rl_quant/protocol/hold30_alpha_m03r_v7.py",
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_schedule.py",
        "src/rl_quant/training/hold30_alpha_m03r_v7.py",
        "src/rl_quant/training/hold30_alpha_m03r_v7_routes.py",
        "src/rl_quant/training/hold30_alpha_m03r_v7_schedule.py",
    } <= set(inventory.integration_sources)
    assert {
        "tests/test_hold30_confidence_v6.py",
        "tests/test_hold30_m03r_v6_exit_action.py",
        "tests/test_hold30_m03r_v6_model_integration.py",
        "tests/test_hold30_m03r_confidence_fit.py",
        "tests/test_hold30_m03r_confidence_objective_v6.py",
        "tests/test_hold30_alpha_m03r_v7_protocol.py",
        "tests/test_hold30_alpha_m03r_v7_objective.py",
        "tests/test_hold30_alpha_m03r_v7_routes.py",
        "tests/test_hold30_alpha_m03r_v7_schedule.py",
    } <= set(inventory.integration_tests)


def test_v3_software_receipt_is_disjoint_non_authorizing_and_self_hashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]

    def passing_gate(gate_id, argv, *, cwd, timeout_seconds):
        assert cwd == root
        assert timeout_seconds == 17
        return {
            "gate_id": gate_id,
            "argv": list(argv),
            "exit_code": 0,
            "passed": True,
            "started_ns": 1,
            "finished_ns": 2,
            "output_sha256": "0" * 64,
            "output_tail": "",
        }

    monkeypatch.setattr(workflow, "_run_gate", passing_gate)
    receipt = qualify_hold30_alpha_software(root, timeout_seconds=17)
    verify_hold30_alpha_software_receipt(receipt)
    assert receipt["schema"] == HOLD30_ALPHA_SOFTWARE_SCHEMA
    assert receipt["protocol_generation"] == HOLD30_ALPHA_PROTOCOL_GENERATION
    assert receipt["protocol_contract"]["setting_ids"] == list(HOLD30_ALPHA_MECH8_IDS)
    assert receipt["superseded_v2_receipts_accepted"] is False
    assert receipt["launch_authorized"] is False
    assert receipt["data_qualification"] is False
    assert receipt["gpu_capacity_qualification"] is False
    assert receipt["scientific_plan_resolved"] is False
    assert receipt["executable_authorization_capable"] is False
    assert receipt["end_to_end_v3_training_driver_qualified"] is False
    assert receipt["qualification_scope"] == "local_cpu_component_integration_only"
    assert receipt["model_contract"]["scientific_plan_resolved"] is False
    assert (
        receipt["model_contract"]["typed_training_plan_resolved_for_executable"]
        is False
    )
    assert (
        receipt["model_contract"]["A06_overlay_coefficient_and_routing_frozen"] is False
    )
    assert receipt["model_contract"]["unresolved_objective_settings"] == list(
        HOLD30_ALPHA_MECH8_IDS[2:]
    )

    tampered = copy.deepcopy(receipt)
    tampered["launch_authorized"] = True
    with pytest.raises(Hold30AlphaQualificationError, match="identity/authority"):
        verify_hold30_alpha_software_receipt(tampered)


def test_v2_receipt_cannot_cross_authorize_v3() -> None:
    v2_like = {
        "schema_version": 1,
        "protocol_generation": "prelockbox-hold30-mech8-v2",
        "qualification_sha256": "0" * 64,
    }
    with pytest.raises(Hold30AlphaQualificationError, match="partial or unknown"):
        verify_hold30_alpha_software_receipt(v2_like)


def test_v3_cli_exposes_only_local_software_qualification() -> None:
    parser = workflow.build_parser()
    choices = parser._actions[1].choices
    assert choices == ("qualify-software",)
    with pytest.raises(SystemExit):
        parser.parse_args(["render-manifest", "--output", "ignored.json"])
    pyproject = (
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    )
    assert (
        "quanttrade-hold30-alpha-prelockbox = "
        '"rl_quant.workflows.hold30_alpha_prelockbox:main"'
    ) in pyproject
