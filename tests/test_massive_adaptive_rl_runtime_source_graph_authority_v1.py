from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_graph_authority_v1 as graph_module,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    runtime_source_graph_authority_path_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1,
    authorize_massive_adaptive_rl_source_bundle_v1,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    run_massive_adaptive_rl_experiment_v2,
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
    source_transport_qualified: bool = True
    daily_input_data_qualified: bool = True
    source_data_qualified: bool = True
    source_paths_replayed: bool = True
    candidate_source_data_qualified: bool = True
    source_geometry_replayed: bool = True

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
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_file_bytes({"semantic_receipt_sha256": receipt})
        )
        runtimes[key] = bind_massive_adaptive_rl_source_authority_v1(
            role=key[0],
            fold_index=key[1],
            authority=_SyntheticRuntimeSource(receipt),
            source_data_qualified=True,
            runtime_source_replayed=True,
        )
    return runtimes


def _authorized_source_bundle(root: Path, experiment_id: str):
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=experiment_id
    )
    runtimes = _persist_synthetic_source_graph(root)
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
        runtime_sources=runtimes,
    )
    generic = load_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
    )
    authorized = authorize_massive_adaptive_rl_source_bundle_v1(
        source_bundle=generic,
        runtime_sources=runtimes,
    )
    return manifest, runtimes, authorized


def _allow_synthetic_domain_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "_DOMAIN_RUNTIME_TYPES",
        {
            role: _SyntheticRuntimeSource
            for role in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1
        },
    )
    monkeypatch.setattr(
        graph_module,
        "_DIRECT_DOMAIN_SPECIFICATIONS",
        {role: None for role in graph_module._DIRECT_DOMAIN_SPECIFICATIONS},
    )


def test_runtime_source_graph_generic_reload_is_nonauthorizing_and_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-replay"
    )

    committed = materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    assert not committed.persisted_graph_replayed
    assert not committed.runtime_graph_replayed
    assert not committed.source_data_qualified

    generic_source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
    )
    generic = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=generic_source_bundle,
    )
    assert generic.persisted_graph_replayed
    assert not generic.runtime_graph_replayed
    assert not generic.source_data_qualified

    authorized = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic,
        source_bundle=generic_source_bundle,
        runtime_sources=runtimes,
    )
    assert authorized.runtime_graph_replayed
    assert authorized.source_data_qualified
    assert (
        authorized.source_bundle_receipt_sha256 == source_bundle.semantic_receipt_sha256
    )
    assert len(authorized.rows) == 32
    assert not authorized.profitability_reporting_authorized
    assert not authorized.lockbox_access_authorized

    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    run = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        device="cpu",
        resume=False,
    )
    assert run.blocker_code == "runtime-source-graph-replay-required"
    assert (
        run.runtime_source_graph_authority_receipt_sha256
        == authorized.semantic_receipt_sha256
    )
    assert not run.runtime_source_graph_replayed

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="create-only",
    ):
        materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=source_bundle,
            runtime_sources=runtimes,
        )

    graph_path = runtime_source_graph_authority_path_v1(
        source_root=tmp_path,
        experiment_id=manifest.experiment_id,
    )
    backing_path = graph_path.with_name("runtime-source-graph-backing.json")
    graph_path.rename(backing_path)
    graph_path.symlink_to(backing_path.name)
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="absent or not regular",
    ):
        load_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=generic_source_bundle,
        )


def test_runtime_source_graph_rejects_old_arbitrary_role_wrappers(
    tmp_path: Path,
) -> None:
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-domain-types"
    )

    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="concrete domain type",
    ):
        materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=source_bundle,
            runtime_sources=runtimes,
        )


def test_runtime_source_graph_rejects_incomplete_and_changed_runtime_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path, "runtime-source-graph-tamper"
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    generic_source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path,
        manifest=manifest.base_manifest,
    )
    generic = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=generic_source_bundle,
    )

    incomplete = dict(runtimes)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="does not replay",
    ):
        authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            authority=generic,
            source_bundle=generic_source_bundle,
            runtime_sources=incomplete,
        )

    changed = dict(runtimes)
    key = next(iter(changed))
    changed_receipt = semantic_sha256({"changed": True})
    changed[key] = bind_massive_adaptive_rl_source_authority_v1(
        role=key[0],
        fold_index=key[1],
        authority=_SyntheticRuntimeSource(changed_receipt),
        source_data_qualified=True,
        runtime_source_replayed=True,
    )
    with pytest.raises(
        MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
        match="replay",
    ):
        authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
            authority=generic,
            source_bundle=generic_source_bundle,
            runtime_sources=changed,
        )
