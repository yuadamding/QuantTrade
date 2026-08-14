from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_quant.training import top2000_m03r_v12_h3_requalification as requal


def _fields(root: Path) -> dict[str, object]:
    parent = root / "runs" / "parent" / "workers"
    continuation = root / "continuations" / "c01"
    return {
        "continuation_id": "qt-m03r-v12-h3-parent-requal-c01",
        "parent_run_id": "qt-m03r-v12-h3-parent",
        "parent_job_name": "qt-m03r-v12-parent-a05",
        "parent_job_uid": "parent-job-uid",
        "parent_attach_config_path": str(root / "launch" / "attach.json"),
        "parent_attach_config_file_sha256": "1" * 64,
        "parent_lifecycle_source_sha256": "2" * 64,
        "parent_terminal_evidence_path": str(root / "evidence" / "terminal.json"),
        "parent_terminal_evidence_file_sha256": "3" * 64,
        "parent_terminal_job_path": str(root / "evidence" / "job.json"),
        "parent_terminal_job_file_sha256": "4" * 64,
        "parent_terminal_pods_path": str(root / "evidence" / "pods.json"),
        "parent_terminal_pods_file_sha256": "5" * 64,
        "parent_supervisor_error_path": str(root / "evidence" / "error.json"),
        "parent_supervisor_error_file_sha256": "6" * 64,
        "parent_cleanup_receipt_path": str(root / "evidence" / "cleanup.json"),
        "parent_cleanup_receipt_file_sha256": "7" * 64,
        "prior_failed_continuation_id": "qt-m03r-v12-h3-parent-requal-c00",
        "prior_failure_receipt_path": str(
            root / "continuations" / "c00" / "attempt-error.json"
        ),
        "prior_failure_receipt_file_sha256": "a" * 64,
        "parent_output_root": str(parent),
        "parent_pythonpath": str(root / "package" / "source" / "src"),
        "corrected_lifecycle_source_path": str(
            continuation / "bundle" / "lifecycle.py"
        ),
        "corrected_lifecycle_source_file_sha256": "8" * 64,
        "requalification_source_path": str(
            continuation / "bundle" / "requalification.py"
        ),
        "requalification_source_file_sha256": "9" * 64,
        "output_root": str(continuation / "evidence"),
    }


def test_v12_h3_requalification_plan_is_cpu_only_and_content_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requal, "APPROVED_ROOT", tmp_path)
    plan = requal.build_requalification_plan(**_fields(tmp_path))
    assert requal._validate_plan(plan) == plan
    assert plan["selected_horizon_sessions"] == 3
    assert plan["no_new_kubernetes_job"] is True
    assert plan["training_reexecuted"] is False
    assert plan["economic_optimizer_updates"] == 0
    assert plan["outer_2026_accessed"] is False


def test_v12_h3_requalification_plan_rejects_rehashing_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requal, "APPROVED_ROOT", tmp_path)
    plan = requal.build_requalification_plan(**_fields(tmp_path))
    unsigned = dict(plan)
    del unsigned["receipt_sha256"]
    unsigned["outer_2026_accessed"] = True
    changed = {**unsigned, "receipt_sha256": requal._content_sha256(unsigned)}
    with pytest.raises(
        requal.M03RV12H3RequalificationError,
        match="semantics drifted",
    ):
        requal._validate_plan(changed)


def test_v12_h3_requalification_plan_requires_disjoint_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requal, "APPROVED_ROOT", tmp_path)
    fields = _fields(tmp_path)
    fields["output_root"] = str(Path(str(fields["parent_output_root"])) / "repair")
    with pytest.raises(
        requal.M03RV12H3RequalificationError,
        match="disjoint",
    ):
        requal.build_requalification_plan(**fields)


def test_v12_h3_requalification_plan_rejects_failed_source_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requal, "APPROVED_ROOT", tmp_path)
    fields = _fields(tmp_path)
    fields["corrected_lifecycle_source_file_sha256"] = fields[
        "parent_lifecycle_source_sha256"
    ]
    with pytest.raises(
        requal.M03RV12H3RequalificationError,
        match="must differ",
    ):
        requal.build_requalification_plan(**fields)


def test_v12_h3_requalification_compares_typed_cleanup_as_canonical_json() -> None:
    typed = {"first_owned_pod_uids": (), "second_owned_pod_uids": ("pod",)}
    json_value = json.loads(json.dumps(typed))
    assert requal._canonical(typed) == requal._canonical(json_value)
