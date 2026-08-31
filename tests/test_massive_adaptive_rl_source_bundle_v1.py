from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MassiveAdaptiveRLSourceBundleV1Error,
    authorize_massive_adaptive_rl_source_bundle_v1,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    MassiveAdaptiveRLExperimentRunnerV2Error,
    run_massive_adaptive_rl_experiment_v2,
    verify_massive_adaptive_rl_experiment_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    build_massive_adaptive_rl_experiment_manifest_v2,
    main,
    write_massive_adaptive_rl_experiment_manifest_v2,
)


_GLOBAL = {
    "session-authority": "authorities/session-authority.json",
    "condition-authority": "authorities/condition-authority.json",
    "persisted-partition-inventory": "authorities/persisted-partition-inventory.json",
    "identity-authority": "authorities/identity-authority.json",
    "economic-event-archive": "authorities/economic-event-archive.json",
    "daily-input-authority": "authorities/daily-input-authority.json",
    "fill-source-authority": "authorities/fill-source-authority.json",
    "split-plan": "authorities/adaptive-split-plan.json",
}
_FOLD = {
    "training-window-inventory": "training-window-inventory.json",
    "supervised-checkpoint-inventory": "supervised-checkpoint-inventory.json",
    "calibration-inventory": "calibration-inventory.json",
    "fit-forecast-archive-inventory": "fit-forecast-archive-inventory.json",
    "decision-root-inventory": "decision-root-inventory.json",
    "context-origin-inventory": "context-origin-inventory.json",
}


@dataclass(frozen=True)
class _SyntheticRuntimeSource:
    semantic_receipt_sha256: str

    def validate(self) -> None:
        assert len(self.semantic_receipt_sha256) == 64


def _persist_synthetic_source_graph(root: Path):
    runtimes = {}
    paths = {(role, None): path for role, path in _GLOBAL.items()}
    for fold_index in range(4):
        paths.update(
            {
                (role, fold_index): f"folds/fold-{fold_index}/{name}"
                for role, name in _FOLD.items()
            }
        )
    for key, relative_path in paths.items():
        receipt = semantic_sha256({"role": key[0], "fold_index": key[1]})
        payload = {"semantic_receipt_sha256": receipt}
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_file_bytes(payload))
        runtimes[key] = bind_massive_adaptive_rl_source_authority_v1(
            role=key[0],
            fold_index=key[1],
            authority=_SyntheticRuntimeSource(receipt),
            source_data_qualified=True,
            runtime_source_replayed=True,
        )
    return runtimes


def test_source_bundle_generic_reload_is_nonauthorizing_and_tamper_evident(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="synthetic-source-bundle"
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    committed = materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest,
        runtime_sources=runtimes,
    )
    assert committed.committed_source_data_qualified
    assert not committed.source_data_qualified

    generic = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest,
    )
    assert generic.persisted_source_replayed
    assert not generic.runtime_source_replayed
    assert not generic.source_data_qualified
    authorized = authorize_massive_adaptive_rl_source_bundle_v1(
        source_bundle=generic,
        runtime_sources=runtimes,
    )
    assert authorized.source_data_qualified
    assert not authorized.profitability_reporting_authorized
    assert not authorized.lockbox_access_authorized

    first_key = next(iter(runtimes))
    unqualified = dict(runtimes)
    unqualified[first_key] = replace(
        unqualified[first_key], source_data_qualified=False
    )
    with pytest.raises(
        MassiveAdaptiveRLSourceBundleV1Error,
        match="unqualified|role-bound",
    ):
        authorize_massive_adaptive_rl_source_bundle_v1(
            source_bundle=generic,
            runtime_sources=unqualified,
        )

    target = tmp_path / _GLOBAL["session-authority"]
    payload = json.loads(target.read_bytes())
    payload["unexpected"] = True
    target.write_bytes(canonical_json_file_bytes(payload))
    with pytest.raises(MassiveAdaptiveRLSourceBundleV1Error, match="changed"):
        load_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest,
        )


def test_source_bundle_rejects_incomplete_runtime_graph(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="incomplete-source-bundle"
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    runtimes.pop(next(iter(runtimes)))
    with pytest.raises(MassiveAdaptiveRLSourceBundleV1Error, match="complete"):
        materialize_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_sources=runtimes,
        )


def test_source_bundle_rejects_unbound_and_role_substituted_runtime_sources(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="role-bound-source-bundle"
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    first_key = next(iter(runtimes))
    unbound = dict(runtimes)
    unbound[first_key] = _SyntheticRuntimeSource(
        runtimes[first_key].semantic_receipt_sha256
    )
    with pytest.raises(MassiveAdaptiveRLSourceBundleV1Error, match="role-bound"):
        materialize_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_sources=unbound,
        )

    identity_key = ("identity-authority", None)
    fill_key = ("fill-source-authority", None)
    substituted = dict(runtimes)
    substituted[identity_key] = runtimes[fill_key]
    with pytest.raises(
        MassiveAdaptiveRLSourceBundleV1Error,
        match="role, fold, or qualification",
    ):
        materialize_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_sources=substituted,
        )


def test_source_bundle_rejects_symlinked_source_artifact(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="symlinked-source-bundle"
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    target = tmp_path / _GLOBAL["session-authority"]
    backing = target.with_name("session-authority-backing.json")
    target.rename(backing)
    target.symlink_to(backing.name)

    with pytest.raises(MassiveAdaptiveRLSourceBundleV1Error, match="symlink"):
        materialize_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_sources=runtimes,
        )


def test_run_resume_and_verify_cli_stop_at_typed_runtime_boundary(
    tmp_path: Path, capsys
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="cli-source-boundary"
    )
    manifest_path = tmp_path / "manifest.json"
    write_massive_adaptive_rl_experiment_manifest_v2(
        path=manifest_path,
        manifest=manifest,
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest,
        runtime_sources=runtimes,
    )
    artifact_root = tmp_path / "artifacts"
    common = [
        "--manifest",
        str(manifest_path),
        "--source-root",
        str(tmp_path),
        "--artifact-root",
        str(artifact_root),
    ]
    assert main(["run", *common]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["current_stage"] == "source-bundle-replayed"
    assert first["blocker_code"] == "typed-runtime-source-replay-required"
    assert main(["resume", *common]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second["semantic_receipt_sha256"] == first["semantic_receipt_sha256"]
    assert main(["verify", *common]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["semantic_receipt_sha256"] == first["semantic_receipt_sha256"]


def test_run_cli_persists_source_replay_failure(tmp_path: Path, capsys) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="missing-source-run"
    )
    manifest_path = tmp_path / "manifest.json"
    write_massive_adaptive_rl_experiment_manifest_v2(
        path=manifest_path,
        manifest=manifest,
    )
    common = [
        "--manifest",
        str(manifest_path),
        "--source-root",
        str(tmp_path / "absent"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert main(["run", *common]) == 2
    failed = json.loads(capsys.readouterr().out)
    assert failed["current_stage"] == "failed"
    assert failed["blocker_code"] == "source-bundle-replay-failed"
    assert main(["verify", *common]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["semantic_receipt_sha256"] == failed["semantic_receipt_sha256"]


def test_runner_v2_binds_device_and_persists_retryable_runtime_blocker(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runner-v2-source-boundary",
        execution_device_specification="cpu",
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    runtimes = _persist_synthetic_source_graph(tmp_path)
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        runtime_sources=runtimes,
    )
    artifact_root = tmp_path / "artifacts"
    first = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=False,
    )
    assert first.current_stage.value == "blocked"
    assert first.next_required_stage is not None
    assert first.next_required_stage.value == "fit-forecasts-authorized"
    assert first.blocker_code == "typed-runtime-source-replay-required"
    assert not first.execution_complete
    assert len(first.state_receipts) == 3

    resumed = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )
    assert resumed.semantic_receipt_sha256 == first.semantic_receipt_sha256
    assert verify_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
    ).semantic_receipt_sha256 == first.semantic_receipt_sha256

    with pytest.raises(MassiveAdaptiveRLExperimentRunnerV2Error, match="device"):
        run_massive_adaptive_rl_experiment_v2(
            manifest_path=manifest_path,
            source_root=tmp_path,
            artifact_root=artifact_root,
            device="cuda:0",
            resume=True,
        )


def test_runner_v2_source_absence_is_blocked_and_resumable(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runner-v2-delayed-source"
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    artifact_root = tmp_path / "artifacts"
    blocked = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=False,
    )
    assert blocked.blocker_code == "source-bundle-temporarily-absent"
    assert blocked.next_required_stage is not None
    assert blocked.next_required_stage.value == "source-bundle-replayed"

    runtimes = _persist_synthetic_source_graph(tmp_path)
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
        runtime_sources=runtimes,
    )
    resumed = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )
    assert resumed.blocker_code == "typed-runtime-source-replay-required"
    assert resumed.next_required_stage is not None
    assert resumed.next_required_stage.value == "fit-forecasts-authorized"


def test_runner_v2_classifies_malformed_bundle_as_integrity_failure(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="runner-v2-malformed-source"
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    bundle_path = (
        tmp_path
        / "adaptive-rl"
        / "source-bundle-v1"
        / f"{manifest.experiment_id}.json"
    )
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_bytes(b"not-json\n")

    result = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        device="cpu",
        resume=False,
    )
    assert result.current_stage.value == "failed"
    assert result.blocker_code == "source-bundle-integrity-failed"
    assert not result.execution_complete
