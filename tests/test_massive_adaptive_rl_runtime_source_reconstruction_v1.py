from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    materialize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v2 as runner_module
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_reconstruction_v1 as reconstruction,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    run_massive_adaptive_rl_experiment_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from test_massive_adaptive_rl_fit_forecast_v1 import _rl_fit_fixture
from test_massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    _allow_synthetic_domain_types,
    _authorized_source_bundle,
)
from test_massive_adaptive_source_authorized_training_v1 import _sessions
from test_massive_trade_replay import _conditions


def test_package_snapshot_round_trips_production_authority_and_arrays() -> None:
    session_authority = _sessions()
    payload = reconstruction._snapshot_payload(session_authority)
    restored = reconstruction._parse_snapshot(payload)

    assert restored == session_authority
    encoded = reconstruction._encode_value(
        {
            "tensor": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
            "array": np.asarray([[3, 4]], dtype=np.int64),
        }
    )
    decoded = reconstruction._decode_value(encoded)
    assert isinstance(decoded, dict)
    assert torch.equal(decoded["tensor"], torch.tensor([[1.25, -2.5]]))
    assert np.array_equal(decoded["array"], np.asarray([[3, 4]], dtype=np.int64))


def test_package_snapshot_rejects_changed_implementation_identity() -> None:
    payload = deepcopy(reconstruction._snapshot_payload(_sessions()))
    encoded = payload["encoded_value"]
    assert isinstance(encoded, dict)
    encoded["implementation_source_sha256"] = semantic_sha256("changed-source")
    body = {
        key: value for key, value in payload.items() if key != "snapshot_receipt_sha256"
    }
    payload["snapshot_receipt_sha256"] = semantic_sha256(body)

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="implementation differs",
    ):
        reconstruction._parse_snapshot(payload)


def test_dependency_closure_rejects_unreferenced_production_authority() -> None:
    sessions = _sessions()
    primary = (
        (
            "session-authority",
            None,
            "root",
            sessions.receipt_sha256,
            sessions,
        ),
    )

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="incomplete or contains extras",
    ):
        reconstruction._complete_object_graph(
            primary=primary,
            dependencies=(_conditions(),),
        )


def test_dependency_index_is_create_only_and_rehashes_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="minimal-replay-dependency-index",
    )
    sessions = _sessions()
    session_spec = reconstruction.MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[
        "session-authority"
    ]
    monkeypatch.setattr(
        reconstruction,
        "MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1",
        {"session-authority": session_spec},
    )
    graph = SimpleNamespace(
        runtime_authority_receipt_sha256=semantic_sha256("runtime-witness"),
        source_data_qualified=True,
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        source_bundle_receipt_sha256=semantic_sha256("source-bundle"),
        semantic_receipt_sha256=semantic_sha256("persisted-graph"),
        validate=lambda: None,
        runtime_authority=lambda **_kwargs: sessions,
    )

    committed = (
        reconstruction.materialize_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_source_graph_authority=graph,
            replay_dependencies=(),
        )
    )
    loaded = reconstruction.load_massive_adaptive_rl_replay_dependency_index_v1(
        source_root=tmp_path,
        manifest=manifest,
    )

    assert loaded == committed
    assert len(loaded.rows) == 1
    assert loaded.rows[0].semantic_receipt_sha256 == sessions.receipt_sha256
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="create-only",
    ):
        reconstruction.materialize_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
            runtime_source_graph_authority=graph,
            replay_dependencies=(),
        )

    snapshot = tmp_path / loaded.rows[0].relative_path
    snapshot.write_bytes(canonical_json_file_bytes({"tampered": True}))
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="hash differs",
    ):
        reconstruction.load_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=tmp_path,
            manifest=manifest,
        )


def test_rl_fit_dependency_closure_includes_full_tensor_root_inventory(
    tmp_path: Path,
) -> None:
    checkpoint, window, tensor, roots, plan, split_plan, model_spec = _rl_fit_fixture(
        tmp_path,
        outer_fold_index=0,
        block_index=0,
        block_sessions=21,
    )
    for authority in (checkpoint, tensor, plan):
        restored = reconstruction._parse_snapshot(
            reconstruction._snapshot_payload(authority)
        )
        assert type(restored) is type(authority)
        assert reconstruction._receipt(restored) == reconstruction._receipt(authority)
    archive = materialize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=tmp_path,
        artifact_id="reconstruction-fit-forecast",
        checkpoint=checkpoint,
        training_window_plan=window,
        inference_tensor=tensor,
        inference_decision_roots=roots,
        inference_plan=plan,
        split_plan=split_plan,
        model_spec=model_spec,
        committed_at_ms=71_000,
    )
    replayed_archive = reconstruction._parse_snapshot(
        reconstruction._snapshot_payload(archive)
    )
    assert type(replayed_archive) is type(archive)
    assert reconstruction._receipt(replayed_archive) == archive.semantic_receipt_sha256
    root_by_date = {row.decision_session_date: row for row in roots}
    dependencies = reconstruction._expected_dependencies(
        value=archive,
        objects_by_receipt={tensor.semantic_receipt_sha256: tensor},
        primary_decisions_by_date=root_by_date,
    )

    expected_root_receipts = {
        root_by_date[date].semantic_receipt_sha256
        for date in tensor.decision_session_dates
    }
    assert expected_root_receipts <= set(dependencies)
    assert len(expected_root_receipts) > len(archive.origin_session_dates)


def test_dependency_snapshot_path_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_file_bytes({"receipt": semantic_sha256("x")}))
    link = tmp_path / "link.json"
    link.symlink_to(target.name)

    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="symlink",
    ):
        reconstruction._resolve_regular_file(
            root=tmp_path.resolve(),
            relative_path=link.name,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    parent_link = tmp_path / "reconstruction"
    parent_link.symlink_to(outside.name, target_is_directory=True)
    with pytest.raises(
        reconstruction.MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
        match="symlink",
    ):
        reconstruction._create_only_output_path(
            root=tmp_path.resolve(),
            relative_path="reconstruction/object.json",
        )


def test_runner_reconstructs_before_stopping_at_execution_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_synthetic_domain_types(monkeypatch)
    manifest, runtimes, source_bundle = _authorized_source_bundle(
        tmp_path,
        "runner-package-reconstruction",
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtimes,
    )
    generic_graph = (
        runner_module.load_massive_adaptive_rl_runtime_source_graph_authority_v1(
            source_root=tmp_path,
            manifest=manifest,
            source_bundle=runner_module.load_massive_adaptive_rl_source_bundle_v1(
                source_root=tmp_path,
                manifest=manifest.base_manifest,
            ),
        )
    )
    authorized_graph = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic_graph,
        source_bundle=runner_module.load_massive_adaptive_rl_source_bundle_v1(
            source_root=tmp_path,
            manifest=manifest.base_manifest,
        ),
        runtime_sources=runtimes,
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    artifact_root = tmp_path / "artifacts"
    initial = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=False,
    )
    assert initial.blocker_code == "runtime-source-replay-dependency-index-required"

    dependency_index = reconstruction.replay_dependency_index_path_v1(
        source_root=tmp_path,
        experiment_id=manifest.experiment_id,
    )
    dependency_index.parent.mkdir(parents=True, exist_ok=True)
    dependency_index.write_bytes(canonical_json_file_bytes({"test": True}))
    runtime_sources = SimpleNamespace(
        runtime_source_graph_authority=authorized_graph,
        semantic_receipt_sha256=semantic_sha256("reconstructed-runtime-sources"),
    )
    monkeypatch.setattr(
        runner_module,
        "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1",
        lambda **_kwargs: runtime_sources,
    )

    result = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path,
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )

    assert result.blocker_code == "four-fold-execution-backend-required"
    assert result.runtime_source_graph_replayed
    assert result.source_data_qualified
