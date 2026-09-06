from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    initial_validation_inputs_authority_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import (
    massive_adaptive_rl_execution_implementation_registration_v1 as implementation,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
    load_massive_adaptive_rl_execution_implementation_registration_v1,
    massive_adaptive_rl_preimplementation_economic_evidence_v1,
    run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256,
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
)
from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_scope_v1 import (
    massive_adaptive_rl_vertical_qualification_scope_v1,
)
from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_runner_v1 import (
    MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1,
    MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1,
    _run_vertical_qualification_process_v1,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _qualified_capture_body(
    *,
    root: Path,
    manifest,
    registration,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    run_qualification = implementation._vertical_qualification
    monkeypatch.setattr(
        implementation,
        "_vertical_qualification",
        lambda *, repository_root, v5_native_vertical_complete: run_qualification(
            repository_root=repository_root,
            v5_native_vertical_complete=False,
        ),
    )
    body = implementation._capture_body(
        root=root,
        manifest=manifest,
        manifest_registration=registration,
    )
    missing = body["missing_v5_native_implementation_paths"]
    assert isinstance(missing, tuple)
    implementation_inventory = tuple(body["implementation_inventory"]) + tuple(
        (name, _digest(("v5-native", name))) for name in missing
    )
    test_inventory = tuple(
        (name, _digest(("vertical-test", name)))
        for name in body["vertical_qualification_test_paths"]
    )
    body.update(
        {
            "source_worktree_clean": True,
            "source_worktree_status": (),
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "float32_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_cpu_threads": 1,
            "torch_interop_threads": 1,
            "process_thread_environment": (
                ("OMP_NUM_THREADS", "1"),
                ("MKL_NUM_THREADS", "1"),
                ("OPENBLAS_NUM_THREADS", "1"),
                ("NUMEXPR_NUM_THREADS", "1"),
                ("PYTHONHASHSEED", "0"),
            ),
            "implementation_inventory": implementation_inventory,
            "implementation_inventory_sha256": semantic_sha256(
                implementation_inventory
            ),
            "missing_v5_native_implementation_paths": (),
            "v5_native_vertical_complete": True,
            "vertical_qualification_test_inventory": test_inventory,
            "vertical_qualification_test_inventory_sha256": semantic_sha256(
                test_inventory
            ),
            "missing_vertical_qualification_test_paths": (),
            "vertical_qualification_exit_code": 0,
            "vertical_qualification_passed_node_count": len(
                implementation._VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
            ),
            "vertical_qualification_nonpass_outcome_labels": (),
            "vertical_qualification_normalized_output_sha256": _digest(
                "vertical-qualification-normalized-output"
            ),
            "vertical_qualification_passed": True,
            "source_data_qualified": True,
        }
    )
    body["vertical_qualification_receipt_sha256"] = (
        implementation._vertical_qualification_receipt(body)
    )
    return body


def test_execution_registration_is_separate_create_only_and_replay_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    training_lineage = (_digest("training-state"), _digest("four-fold-fit"))
    monkeypatch.setattr(
        implementation, "_training_lineage_v1", lambda **_: training_lineage
    )
    body = _qualified_capture_body(
        root=tmp_path,
        manifest=manifest,
        registration=registration,
        monkeypatch=monkeypatch,
    )
    replay_roots: list[object | None] = []

    def capture(**kwargs):
        replay_roots.append(kwargs.get("registered_authority"))
        return dict(body)

    monkeypatch.setattr(implementation, "_capture_body", capture)

    authority = (
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
        )
    )
    assert authority.development_execution_registered
    assert authority.manifest_v5_registration_committed_at_ms == registration_time
    assert authority.training_state_receipt_sha256 == training_lineage[0]
    assert authority.four_fold_fit_authority_receipt_sha256 == training_lineage[1]
    assert (
        authority.vertical_qualification_specification_sha256
        == MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256
    )
    assert authority.vertical_qualification_receipt_sha256 == (
        implementation._vertical_qualification_receipt(authority.semantic_unsigned())
    )
    assert authority.source_transaction_committed_at_ms is not None
    assert authority.source_transaction_committed_at_ms > registration_time
    assert not authority.outer_access_authorized
    assert not authority.profitability_reporting_authorized

    generic = load_massive_adaptive_rl_execution_implementation_registration_v1(
        root=tmp_path,
        experiment_id=manifest.experiment_id,
        verified_at_ms=authority.source_transaction_committed_at_ms,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_implementation_replayed
    assert not generic.development_execution_registered

    resumed = (
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            allow_materialize=False,
        )
    )
    assert resumed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert resumed.development_execution_registered
    assert replay_roots[0] is None
    assert all(root is not None for root in replay_roots[1:])


def test_execution_registration_rejects_untracked_source_and_late_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration-fail-closed"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    monkeypatch.setattr(
        implementation,
        "_training_lineage_v1",
        lambda **_: (_digest("training-state"), _digest("four-fold-fit")),
    )
    body = _qualified_capture_body(
        root=tmp_path,
        manifest=manifest,
        registration=registration,
        monkeypatch=monkeypatch,
    )
    body.update(
        {
            "source_worktree_clean": False,
            "source_worktree_status": ("?? src/rl_quant/unregistered.py",),
            "source_data_qualified": False,
        }
    )
    monkeypatch.setattr(implementation, "_capture_body", lambda **_: dict(body))
    with pytest.raises(
        MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
        match="not scientifically qualified",
    ):
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
        )

    late = tmp_path / "adaptive-rl" / manifest.experiment_id / "validation-outcome-v3"
    late.mkdir(parents=True)
    body.update(
        {
            "source_worktree_clean": True,
            "source_worktree_status": (),
            "source_data_qualified": True,
        }
    )
    with pytest.raises(
        MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
        match="must precede every validation input",
    ):
        run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "adaptive-rl/{experiment}/validation-release-v1",
        "adaptive-rl/{experiment}/frozen-fc06-v2",
        "adaptive-rl/{experiment}/outer-access-commitment-v2",
        "adaptive-rl/{experiment}/prequential-experiment-state-v1",
        "massive-adaptive/rl-policy-selection-authority-v3",
        "massive-adaptive/rl-outer-evidence-authority-v4",
        "massive-adaptive/rl-profitability-report-authority-v1",
    ),
)
def test_execution_registration_scan_covers_future_and_legacy_evidence(
    tmp_path: Path,
    relative: str,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration-evidence-scan"
    )
    path = tmp_path / relative.format(experiment=manifest.experiment_id)
    path.mkdir(parents=True)
    found = massive_adaptive_rl_preimplementation_economic_evidence_v1(
        root=tmp_path,
        manifest=manifest,
    )
    assert str(path.relative_to(tmp_path)) in found


def test_execution_registration_scan_rejects_initial_validation_input(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration-initial-input-scan"
    )
    relative = initial_validation_inputs_authority_relative_path_v1(
        manifest=manifest.base_manifest
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    assert relative in massive_adaptive_rl_preimplementation_economic_evidence_v1(
        root=tmp_path,
        manifest=manifest,
    )


def test_execution_registration_requires_complete_v5_native_vertical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="execution-registration-native-inventory"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    registration_time = registration.source_transaction_committed_at_ms
    assert registration_time is not None
    monkeypatch.setattr(
        implementation,
        "_training_lineage_v1",
        lambda **_: (_digest("training-state"), _digest("four-fold-fit")),
    )
    result = implementation._vertical_qualification(
        repository_root=tmp_path,
        v5_native_vertical_complete=True,
    )
    assert result["missing_vertical_qualification_test_paths"] == (
        "tests/test_massive_adaptive_rl_v5_vertical.py",
    )
    assert result["vertical_qualification_exit_code"] is None
    assert result["vertical_qualification_passed_node_count"] == 0
    assert result["vertical_qualification_nonpass_outcome_labels"] == ("not-run",)
    assert result["vertical_qualification_passed"] is False
    assert result["vertical_qualification_receipt_sha256"] == (
        implementation._vertical_qualification_receipt(result)
    )


def test_synthetic_qualification_registration_is_scoped_and_nonrecursive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="v5-vertical-qualification-bootstrap"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    monkeypatch.setattr(
        implementation,
        "_training_lineage_v1",
        lambda **_: (_digest("training-state"), _digest("four-fold-fit")),
    )
    body = _qualified_capture_body(
        root=tmp_path,
        manifest=manifest,
        registration=registration,
        monkeypatch=monkeypatch,
    )
    body.update(
        {
            "registration_mode": "synthetic-vertical-qualification",
            "missing_vertical_qualification_test_paths": (),
            "vertical_qualification_exit_code": None,
            "vertical_qualification_passed_node_count": 0,
            "vertical_qualification_nonpass_outcome_labels": ("not-run",),
            "vertical_qualification_normalized_output_sha256": _digest(
                "synthetic-qualification-not-run"
            ),
            "vertical_qualification_passed": False,
            "source_data_qualified": True,
        }
    )
    body["vertical_qualification_receipt_sha256"] = (
        implementation._vertical_qualification_receipt(body)
    )

    def capture(**kwargs):
        assert kwargs["registration_mode"] == "synthetic-vertical-qualification"
        return dict(body)

    monkeypatch.setattr(implementation, "_capture_body", capture)

    with massive_adaptive_rl_vertical_qualification_scope_v1():
        authority = (
            run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1(
                root=tmp_path,
                manifest=manifest,
                manifest_registration=registration,
            )
        )
        assert authority.registration_mode == "synthetic-vertical-qualification"
        assert authority.development_execution_registered
        assert not authority.vertical_qualification_passed
        assert authority.vertical_qualification_nonpass_outcome_labels == ("not-run",)

    assert not authority.development_execution_registered


def test_synthetic_qualification_registration_rejects_unreserved_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="not-a-qualification-experiment"
    )
    registration = run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=tmp_path,
        manifest=manifest,
    )
    monkeypatch.setattr(
        implementation,
        "_training_lineage_v1",
        lambda **_: (_digest("training-state"), _digest("four-fold-fit")),
    )

    with (
        massive_adaptive_rl_vertical_qualification_scope_v1(),
        pytest.raises(
            MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
            match="package-owned synthetic qualification process",
        ),
    ):
        implementation._capture_body(
            root=tmp_path,
            manifest=manifest,
            manifest_registration=registration,
            registration_mode="synthetic-vertical-qualification",
        )


def test_vertical_qualification_receipt_redacts_duration_and_disables_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_path = tmp_path / "tests" / "test_massive_adaptive_rl_v5_vertical.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_vertical():\n    assert True\n", encoding="utf-8")
    outputs = iter((b"12 passed in 0.11s\n", b"12 passed in 9.87s\n"))

    def completed(command, **kwargs):
        assert command[4:6] == ("-p", "no:cacheprovider")
        assert command[6:] == implementation._VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["timeout"] == (
            MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1
        )
        return implementation.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=next(outputs),
            stderr=b"",
        )

    monkeypatch.setattr(implementation, "_run_vertical_qualification_process_v1", completed)
    first = implementation._vertical_qualification(
        repository_root=tmp_path,
        v5_native_vertical_complete=True,
    )
    second = implementation._vertical_qualification(
        repository_root=tmp_path,
        v5_native_vertical_complete=True,
    )

    assert first["vertical_qualification_passed"] is True
    assert first["vertical_qualification_passed_node_count"] == 12
    assert first["vertical_qualification_nonpass_outcome_labels"] == ()
    assert (
        first["vertical_qualification_normalized_output_sha256"]
        == second["vertical_qualification_normalized_output_sha256"]
    )
    assert (
        first["vertical_qualification_receipt_sha256"]
        == second["vertical_qualification_receipt_sha256"]
    )


def test_registered_qualification_replay_does_not_launch_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_path = tmp_path / "tests" / "test_massive_adaptive_rl_v5_vertical.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_vertical():\n    assert True\n", encoding="utf-8")
    qualification = implementation._vertical_qualification(
        repository_root=tmp_path,
        v5_native_vertical_complete=False,
    )
    qualification.update(
        {
            "missing_vertical_qualification_test_paths": (),
            "vertical_qualification_exit_code": 0,
            "vertical_qualification_passed_node_count": len(
                implementation._VERTICAL_QUALIFICATION_REQUIRED_NODE_IDS
            ),
            "vertical_qualification_nonpass_outcome_labels": (),
            "vertical_qualification_normalized_output_sha256": _digest(
                "qualified-output"
            ),
            "vertical_qualification_passed": True,
        }
    )
    qualification["vertical_qualification_receipt_sha256"] = (
        implementation._vertical_qualification_receipt(qualification)
    )
    authority = SimpleNamespace(**qualification)
    monkeypatch.setattr(
        implementation,
        "_run_vertical_qualification_process_v1",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("pytest reran")),
    )

    replayed = implementation._replay_registered_vertical_qualification(
        repository_root=tmp_path,
        authority=authority,
    )

    assert replayed == qualification


def test_vertical_qualification_rejects_skipped_required_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_path = tmp_path / "tests" / "test_massive_adaptive_rl_v5_vertical.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_vertical():\n    assert True\n", encoding="utf-8")

    def completed(command, **kwargs):
        return implementation.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=b"11 passed, 1 skipped in 0.10s\n",
            stderr=b"",
        )

    monkeypatch.setattr(implementation, "_run_vertical_qualification_process_v1", completed)
    result = implementation._vertical_qualification(
        repository_root=tmp_path,
        v5_native_vertical_complete=True,
    )

    assert result["vertical_qualification_passed"] is False
    assert result["vertical_qualification_passed_node_count"] == 11
    assert result["vertical_qualification_nonpass_outcome_labels"] == ("skipped",)


def test_qualification_budget_contains_training_and_fresh_replay() -> None:
    assert MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1 == 2 * 60 * 60
    assert MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1 == 6 * 60 * 60
    # Training alone has exceeded an hour; leave four hours outside the nested
    # verifier for setup, training, economics, and in-process reconstruction.
    assert (
        MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1
        - MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1
    ) == 4 * 60 * 60


def test_qualification_timeout_is_not_a_failed_profitability_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    test_path = tmp_path / "tests" / "test_massive_adaptive_rl_v5_vertical.py"
    test_path.parent.mkdir()
    test_path.write_text("# Test inventory only; no economic qualification.\n")
    before = tuple(tmp_path.rglob("*"))

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial")

    monkeypatch.setattr(implementation, "_run_vertical_qualification_process_v1", timeout)
    with pytest.raises(
        MassiveAdaptiveRLExecutionImplementationRegistrationV1Error,
        match="timed out after 21600s; operational failure, not an economic result",
    ) as raised:
        implementation._vertical_qualification(
            repository_root=tmp_path, v5_native_vertical_complete=True
        )

    assert isinstance(raised.value.__cause__, subprocess.TimeoutExpired)
    assert tuple(tmp_path.rglob("*")) == before
    output = capsys.readouterr()
    assert output.out == ""
    assert "elapsed_seconds=" in output.err
    assert "timeout_seconds=21600" in output.err


@pytest.mark.skipif(os.name != "posix", reason="qualification uses POSIX isolation")
def test_qualification_process_preserves_output_and_nonzero_exit(tmp_path: Path) -> None:
    result = _run_vertical_qualification_process_v1(
        (
            sys.executable,
            "-c",
            "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(3)",
        ),
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=10,
    )
    assert result.returncode == 3
    assert result.stdout == b"stdout\n"
    assert result.stderr == b"stderr\n"


@pytest.mark.skipif(sys.platform != "linux", reason="inspects owned Linux child state")
@pytest.mark.parametrize("leader_exits", (False, True))
def test_qualification_timeout_stops_descendants_even_after_leader_exit(
    tmp_path: Path, leader_exits: bool
) -> None:
    # This is a real process tree, not a mocked timeout. The child deliberately
    # ignores TERM and holds the output pipes. Killing only the leader leaks it.
    code = """
import os
import signal
import sys
import time
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('descendant=' + str(os.getpid()), flush=True)
    time.sleep(60)
    os._exit(0)
if sys.argv[1] == 'True':
    os._exit(0)
os.waitpid(child, 0)
"""
    started_at = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _run_vertical_qualification_process_v1(
            (sys.executable, "-c", code, str(leader_exits)),
            cwd=tmp_path,
            env=dict(os.environ),
            timeout=2,
        )
    assert time.monotonic() - started_at < 10
    output = raised.value.output
    assert isinstance(output, bytes)
    descendant_pid = int(output.decode("ascii").strip().removeprefix("descendant="))
    # init may not have reaped the orphan yet; a zombie cannot keep computing
    # or writing evidence. Inspect only this test-owned PID, never other jobs.
    state_path = Path(f"/proc/{descendant_pid}/stat")
    deadline = time.monotonic() + 2
    while state_path.exists():
        try:
            state = state_path.read_text().split(")", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        if time.monotonic() >= deadline:
            pytest.fail("qualification left its descendant running after timeout")
        time.sleep(0.01)
