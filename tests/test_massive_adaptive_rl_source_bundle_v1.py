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
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
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
    source_data_qualified: bool = True

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
        runtimes[key] = _SyntheticRuntimeSource(receipt)
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
    with pytest.raises(MassiveAdaptiveRLSourceBundleV1Error, match="unqualified"):
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
