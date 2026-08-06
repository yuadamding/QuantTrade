from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.workflows import hold30_prelockbox as workflow
from rl_quant.workflows.hold30_prelockbox import (
    HOLD30_COMPONENT_TESTS,
    HOLD30_EVIDENCE_FILES,
    HOLD30_INTEGRATION_SOURCE_FILES,
    SOFTWARE_GATE_FILES,
    Hold30QualificationError,
    _hold30_model_evidence,
    _read_axis,
    _resolve_qualification_inventory,
    _write_new_json,
    build_parser,
    qualify_hold30_software,
)


def _materialize_inventory_tree(root: Path, *, omit: set[str] | None = None) -> None:
    omitted = omit or set()
    paths = {
        *(source for source, _test in HOLD30_COMPONENT_TESTS),
        *(test for _source, test in HOLD30_COMPONENT_TESTS),
        *HOLD30_INTEGRATION_SOURCE_FILES,
        *SOFTWARE_GATE_FILES,
        *workflow.COMPATIBILITY_GATE_FILES,
        *HOLD30_EVIDENCE_FILES,
    }
    for name in sorted(paths - omitted):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# qualification fixture\n", encoding="utf-8")


def test_cli_requires_an_explicit_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_has_no_launch_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["launch"])


def test_axis_reader_rejects_non_string_arrays(tmp_path: Path) -> None:
    path = tmp_path / "axis.json"
    path.write_text(json.dumps(["2020-01-01", 2]), encoding="utf-8")
    with pytest.raises(Hold30QualificationError, match="array of timestamps"):
        _read_axis(path)


def test_evidence_writer_is_exclusive_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_new_json(path, {"passed": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"passed": True}
    with pytest.raises(FileExistsError):
        _write_new_json(path, {"passed": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"passed": True}


def test_inventory_contains_every_landed_hold30_component_and_test() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = _resolve_qualification_inventory(root)

    assert "src/rl_quant/datasets/hold30.py" in inventory.component_sources
    assert "src/rl_quant/models/hold30_ensemble.py" in inventory.component_sources
    assert "src/rl_quant/models/hold30_confidence_v6.py" in inventory.component_sources
    assert (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7.py" in inventory.component_sources
    )
    assert (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_schedule.py"
        in inventory.component_sources
    )
    assert (
        "src/rl_quant/training/hold30_alpha_m03r_v7.py" in inventory.component_sources
    )
    assert (
        "src/rl_quant/training/hold30_alpha_m03r_v7_routes.py"
        in inventory.component_sources
    )
    assert (
        "src/rl_quant/training/hold30_alpha_m03r_v7_schedule.py"
        in inventory.component_sources
    )
    assert "src/rl_quant/training/hold30_runtime.py" in inventory.component_sources
    assert "tests/test_hold30_dataset.py" in inventory.hold30_tests
    assert "tests/test_hold30_ensemble.py" in inventory.hold30_tests
    assert "tests/test_hold30_confidence_v6.py" in inventory.hold30_tests
    assert "tests/test_hold30_alpha_m03r_v7_protocol.py" in inventory.hold30_tests
    assert "tests/test_hold30_alpha_m03r_v7_objective.py" in inventory.hold30_tests
    assert "tests/test_hold30_alpha_m03r_v7_routes.py" in inventory.hold30_tests
    assert "tests/test_hold30_alpha_m03r_v7_schedule.py" in inventory.hold30_tests
    assert "docs/prelockbox_hold30_active_alpha_m03r_v7.md" in HOLD30_EVIDENCE_FILES
    assert (
        "docs/prelockbox_hold30_active_alpha_m03r_v7_experiment.md"
        in HOLD30_EVIDENCE_FILES
    )
    assert "tests/test_hold30_mechanisms.py" in inventory.hold30_tests
    assert "tests/test_hold30_workflow.py" in inventory.hold30_tests
    assert set(inventory.component_sources) == {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "rl_quant").rglob("*hold30*.py")
    }


def test_inventory_fails_closed_when_a_required_test_is_missing(tmp_path: Path) -> None:
    missing = "tests/test_hold30_dataset.py"
    _materialize_inventory_tree(tmp_path, omit={missing})

    with pytest.raises(
        Hold30QualificationError, match="required qualification files are absent"
    ):
        _resolve_qualification_inventory(tmp_path)


def test_future_hold30_driver_requires_and_binds_its_conventional_test(
    tmp_path: Path,
) -> None:
    _materialize_inventory_tree(tmp_path)
    driver = tmp_path / "src/rl_quant/training/hold30_aux_driver.py"
    driver.write_text("# package-owned driver\n", encoding="utf-8")

    with pytest.raises(
        Hold30QualificationError, match="requires tests/test_hold30_aux_driver.py"
    ):
        _resolve_qualification_inventory(tmp_path)

    driver_test = tmp_path / "tests/test_hold30_aux_driver.py"
    driver_test.write_text("# driver qualification\n", encoding="utf-8")
    inventory = _resolve_qualification_inventory(tmp_path)
    assert "src/rl_quant/training/hold30_aux_driver.py" in inventory.component_sources
    assert "tests/test_hold30_aux_driver.py" in inventory.hold30_tests
    assert (
        "src/rl_quant/training/hold30_aux_driver.py" in inventory.static_hygiene_files
    )


def test_bad_compact_parameter_counts_block_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "hold30_parameter_counts",
        lambda _setting_id: SimpleNamespace(
            context_encoder=266_496,
            actor_path=7_000_001,
            total_unique=7_266_497,
        ),
    )

    with pytest.raises(Hold30QualificationError, match="compact parameter caps failed"):
        _hold30_model_evidence()


def test_software_receipt_binds_inventory_and_exact_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]

    def passing_gate(
        gate_id: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: int,
    ) -> dict[str, object]:
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
    receipt = qualify_hold30_software(root, timeout_seconds=17)

    assert receipt["passed"] is True
    assert receipt["launch_authorized"] is False
    assert receipt["scientific_qualification"] is False
    assert receipt["gpu_capacity_qualification"] is False
    model_contract = receipt["model_contract"]
    assert len(model_contract["settings"]) == 8
    assert model_contract["context_config"]["d_model"] == 128
    assert all(
        row["parameter_counts"]["total_unique"] <= 7_000_000
        for row in model_contract["settings"]
    )
    assert set(receipt["qualified_file_sha256s"]) == set(
        receipt["qualification_inventory"]["qualified_files"]
    )
