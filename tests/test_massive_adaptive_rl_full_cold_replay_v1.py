from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import inspect
from pathlib import Path
import subprocess
import sys
from typing import TypeVar

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_full_cold_replay_v1 as cold
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
    MassiveAdaptiveRLPrequentialRunV5,
)
from rl_quant.workflows.massive_adaptive_rl_full_cold_replay_v1 import (
    MassiveAdaptiveRLFullColdReplayV1Error,
    MassiveAdaptiveRLProtectedEvidenceFileV1,
    _issue_massive_adaptive_rl_verified_replay_evidence_v1,
    _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1,
    load_massive_adaptive_rl_full_cold_replay_authority_v1,
    massive_adaptive_rl_protected_evidence_inventory_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _replay_roots(
    manifest_receipt: str,
    execution_receipt: str,
) -> tuple[
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    MassiveAdaptiveRLPrequentialRunV5,
]:
    registration = _typed_shell(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        semantic_receipt_sha256=_digest("registration"),
        manifest_v5_receipt_sha256=manifest_receipt,
    )
    execution = _typed_shell(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        semantic_receipt_sha256=execution_receipt,
        manifest_v5_receipt_sha256=manifest_receipt,
        manifest_v5_registration_authority_receipt_sha256=(
            registration.semantic_receipt_sha256
        ),
    )
    run = _typed_shell(
        MassiveAdaptiveRLPrequentialRunV5,
        experiment_id="cold-replay",
        manifest_v5_receipt_sha256=manifest_receipt,
        execution_implementation_registration_authority_receipt_sha256=(
            execution_receipt
        ),
        semantic_receipt_sha256=_digest("report-boundary"),
        prequential_state_head_stage="profitability-report-published",
        sealed_outer_fold_indices=(0, 1, 2, 3),
        profitability_reporting_authorized=True,
        end_to_end_profitability_execution_complete=False,
        next_required_stage="full-cold-replay-verification",
        profitability_report_authority_receipt_sha256=_digest("report"),
        profitability_report_source_receipt_sha256=_digest("report-source"),
        profitability_report_commit_receipt_sha256=_digest("report-commit"),
        profitability_report_committed_at_ms=10,
        prequential_state_head_receipt_sha256=_digest("report-state"),
        prequential_state_head_source_receipt_sha256=_digest("report-state-source"),
        prequential_state_head_commit_receipt_sha256=_digest("report-state-commit"),
        prequential_state_head_committed_at_ms=20,
        outer_fold_seal_authority_receipts=tuple(
            _digest(("seal", fold_index)) for fold_index in range(4)
        ),
        policy_schedule_disposition="policy-prefix-qualified",
        profitability_gates_passed=False,
    )
    return registration, execution, run


def test_cold_replay_proof_persistence_is_internal_and_opaque() -> None:
    assert not hasattr(cold, "run_or_resume_massive_adaptive_rl_full_cold_replay_v1")
    assert (
        "_persist_or_replay_massive_adaptive_rl_full_cold_replay_v1" not in cold.__all__
    )
    parameters = set(
        inspect.signature(
            _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1
        ).parameters
    )
    assert "verified_replay" in parameters
    assert not parameters.intersection(
        {
            "replayed_run",
            "evidence_inventory_before",
            "evidence_inventory_after",
            "environment",
            "forecasts",
            "actions",
            "targets",
            "transitions",
            "returns",
            "metrics",
            "outer_fold_seals",
        }
    )


def test_protected_inventory_excludes_only_operational_completion_files(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="cold-replay-inventory"
    )
    artifact_root = tmp_path / "artifact"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    source_root.mkdir()
    experiment = artifact_root / "adaptive-rl" / manifest.experiment_id
    report = experiment / "profitability-report-authority-v2" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("report\n")
    report_state = (
        experiment
        / "prequential-experiment-state-v1"
        / "013-profitability-report-published.json"
    )
    report_state.parent.mkdir(parents=True)
    report_state.write_text("state\n")
    lock = experiment / "orchestration-lease-v1" / "orchestration.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("lock\n")
    completion = experiment / "full-cold-replay-authority-v1" / "completion.json"
    completion.parent.mkdir(parents=True)
    completion.write_text("completion\n")
    final_state = (
        experiment
        / "prequential-experiment-state-v1"
        / "014-full-cold-replay-verified.json"
    )
    final_state.write_text("final\n")
    source = (
        source_root
        / "massive-adaptive"
        / "decision-tensor-v1"
        / f"{manifest.experiment_id}-v5-outer-fold-0.json"
    )
    source.parent.mkdir(parents=True)
    source.write_text("source\n")
    unrelated = source.with_name("other-experiment.json")
    unrelated.write_text("other\n")

    rows = massive_adaptive_rl_protected_evidence_inventory_v1(
        artifact_root=artifact_root,
        source_root=source_root,
        manifest=manifest,
    )
    observed = {(row.root_role, row.relative_path) for row in rows}
    assert ("artifact", report.relative_to(artifact_root).as_posix()) in observed
    assert (
        "artifact",
        report_state.relative_to(artifact_root).as_posix(),
    ) in observed
    assert ("source", source.relative_to(source_root).as_posix()) in observed
    assert all("orchestration.lock" not in row.relative_path for row in rows)
    assert all("full-cold-replay-authority-v1" not in row.relative_path for row in rows)
    assert all("014-full-cold-replay" not in row.relative_path for row in rows)
    assert all("other-experiment" not in row.relative_path for row in rows)


def test_cold_replay_requires_unchanged_inventory_and_generic_load_is_nonauthorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="cold-replay"
    )
    registration, execution, run = _replay_roots(
        manifest.semantic_receipt_sha256,
        _digest("execution"),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "development_protocol_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(MassiveAdaptiveRLPrequentialRunV5, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialRunV5,
        "semantic_unsigned",
        lambda value: {"receipt": value.semantic_receipt_sha256},
    )
    monkeypatch.setattr(
        cold,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(
        cold,
        "issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        cold,
        "massive_adaptive_rl_manifest_v5_writer_scope_v1",
        lambda **_: nullcontext(),
    )
    inventory = (
        MassiveAdaptiveRLProtectedEvidenceFileV1(
            root_role="artifact",
            relative_path=(
                "adaptive-rl/cold-replay/profitability-report-authority-v2/report.json"
            ),
            size_bytes=10,
            content_sha256=_digest("report-bytes"),
        ),
    )
    changed = (replace(inventory[0], size_bytes=11),)
    with pytest.raises(
        MassiveAdaptiveRLFullColdReplayV1Error,
        match="changed protected experiment evidence",
    ):
        _issue_massive_adaptive_rl_verified_replay_evidence_v1(
            expected_run=run,
            replayed_run=run,
            evidence_inventory_before=inventory,
            evidence_inventory_after=changed,
        )

    verified_replay = _issue_massive_adaptive_rl_verified_replay_evidence_v1(
        expected_run=run,
        replayed_run=run,
        evidence_inventory_before=inventory,
        evidence_inventory_after=inventory,
    )
    replayed = _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        verified_replay=verified_replay,
    )
    assert replayed.development_full_cold_replay_verified
    assert not replayed.end_to_end_profitability_execution_complete
    assert not replayed.positive_profitability_authorization_eligible

    files_before = tuple(
        sorted(
            (path.relative_to(tmp_path).as_posix(), path.read_bytes())
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
    )
    cold_replayed = _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1(
        root=tmp_path,
        manifest=manifest,
        manifest_registration=registration,
        execution_registration=execution,
        verified_replay=verified_replay,
        allow_materialize=False,
    )
    files_after = tuple(
        sorted(
            (path.relative_to(tmp_path).as_posix(), path.read_bytes())
            for path in tmp_path.rglob("*")
            if path.is_file()
        )
    )
    assert cold_replayed.semantic_receipt_sha256 == replayed.semantic_receipt_sha256
    assert files_after == files_before

    loaded = load_massive_adaptive_rl_full_cold_replay_authority_v1(
        root=tmp_path,
        manifest=manifest,
    )
    assert loaded.semantic_receipt_sha256 == replayed.semantic_receipt_sha256
    assert not loaded.runtime_cold_replay_replayed
    assert not loaded.development_full_cold_replay_verified
    assert not loaded.end_to_end_profitability_execution_complete

    repository = Path(__file__).resolve().parents[1]
    child = subprocess.run(
        (
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import build_massive_adaptive_rl_experiment_manifest_v5",
                    "from rl_quant.workflows.massive_adaptive_rl_full_cold_replay_v1 import load_massive_adaptive_rl_full_cold_replay_authority_v1",
                    "manifest = build_massive_adaptive_rl_experiment_manifest_v5(experiment_id='cold-replay')",
                    "authority = load_massive_adaptive_rl_full_cold_replay_authority_v1(root=sys.argv[1], manifest=manifest)",
                    "assert not authority.runtime_cold_replay_replayed",
                    "assert not authority.development_full_cold_replay_verified",
                    "assert not authority.end_to_end_profitability_execution_complete",
                    "print(authority.semantic_receipt_sha256)",
                )
            ),
            str(tmp_path),
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == replayed.semantic_receipt_sha256


def test_cold_replay_persistence_rejects_caller_forged_verifier_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="cold-replay"
    )
    registration, execution, run = _replay_roots(
        manifest.semantic_receipt_sha256,
        _digest("execution"),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "development_protocol_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    inventory = (
        MassiveAdaptiveRLProtectedEvidenceFileV1(
            root_role="artifact",
            relative_path=(
                "adaptive-rl/cold-replay/profitability-report-authority-v2/report.json"
            ),
            size_bytes=10,
            content_sha256=_digest("report-bytes"),
        ),
    )
    forged = cold._MassiveAdaptiveRLVerifiedReplayEvidenceV1(
        expected_run_receipt_sha256=run.semantic_receipt_sha256,
        replayed_run=run,
        evidence_inventory_before=inventory,
        evidence_inventory_after=inventory,
        _seal=object(),
    )
    with pytest.raises(
        MassiveAdaptiveRLFullColdReplayV1Error,
        match="not issued by the package verifier",
    ):
        _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=execution,
            verified_replay=forged,
        )


@pytest.mark.parametrize(
    "present_suffixes",
    (("",), ("", ".receipt.json"), (".receipt.json", ".commit.json")),
)
def test_cold_replay_never_repairs_a_partial_completion_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    present_suffixes: tuple[str, ...],
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="cold-replay"
    )
    registration, execution, run = _replay_roots(
        manifest.semantic_receipt_sha256,
        _digest("execution"),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
        "development_protocol_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
        "development_execution_registered",
        property(lambda _: True),
    )
    monkeypatch.setattr(MassiveAdaptiveRLPrequentialRunV5, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLPrequentialRunV5,
        "semantic_unsigned",
        lambda value: {"receipt": value.semantic_receipt_sha256},
    )
    monkeypatch.setattr(
        cold,
        "massive_adaptive_rl_experiment_materialization_lock_v1",
        lambda **_: nullcontext(),
    )
    inventory = (
        MassiveAdaptiveRLProtectedEvidenceFileV1(
            root_role="artifact",
            relative_path=(
                "adaptive-rl/cold-replay/profitability-report-authority-v2/report.json"
            ),
            size_bytes=10,
            content_sha256=_digest("report-bytes"),
        ),
    )
    verified_replay = _issue_massive_adaptive_rl_verified_replay_evidence_v1(
        expected_run=run,
        replayed_run=run,
        evidence_inventory_before=inventory,
        evidence_inventory_after=inventory,
    )
    relative = cold.full_cold_replay_authority_relative_path_v1(manifest=manifest)
    payload = tmp_path / relative
    payload.parent.mkdir(parents=True)
    for suffix in present_suffixes:
        payload.with_name(payload.name + suffix).write_bytes(
            f"partial:{suffix}".encode()
        )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(
        MassiveAdaptiveRLFullColdReplayV1Error,
        match="transaction is incomplete",
    ):
        _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            execution_registration=execution,
            verified_replay=verified_replay,
        )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
