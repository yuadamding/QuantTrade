from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
    MassiveAdaptiveRLExperimentManifestV5Error,
    build_massive_adaptive_rl_experiment_manifest_v5,
    load_massive_adaptive_rl_experiment_manifest_v5,
    write_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import main


def test_manifest_v5_preregisters_one_prequential_writer(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5"
    )
    assert manifest.initial_validation_fold_indices == (0, 1)
    assert manifest.withheld_validation_fold_indices == (2, 3)
    assert manifest.validation_release_prerequisite_outer_fold_indices == (
        None,
        None,
        0,
        1,
    )
    assert (
        manifest.outer_to_validation_release_edges
        == MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1
    )
    assert manifest.prequential_stage_sequence == (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
    )
    assert manifest.authority_generation_names == (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1
    )
    assert manifest.authoritative_writer_generation == (
        "massive-adaptive-rl-experiment-runner-v5"
    )
    assert not manifest.legacy_manifest_v4_materialization_authorized
    assert manifest.diagnostic_only_continuation_required
    assert not manifest.validation_outcome_access_authorized
    assert not manifest.outer_access_authorized
    assert not manifest.profitability_reporting_authorized
    assert not manifest.lockbox_access_authorized
    assert not manifest.live_trading_authorized

    path = tmp_path / "manifest-v5.json"
    write_massive_adaptive_rl_experiment_manifest_v5(path=path, manifest=manifest)
    assert load_massive_adaptive_rl_experiment_manifest_v5(path) == manifest
    with pytest.raises(
        MassiveAdaptiveRLExperimentManifestV5Error, match="create-only"
    ):
        write_massive_adaptive_rl_experiment_manifest_v5(path=path, manifest=manifest)


def test_manifest_v5_rejects_changed_release_or_stop_semantics() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-mutation"
    )
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV5Error):
        replace(
            manifest,
            outer_to_validation_release_edges=((1, 2), (0, 3)),
        ).validate()
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV5Error):
        replace(manifest, diagnostic_only_continuation_required=False).validate()
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV5Error):
        replace(manifest, legacy_manifest_v4_materialization_authorized=True).validate()


def test_manifest_v5_binds_protocol_and_writer_hashes() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-hashes"
    )
    assert (
        manifest.specification_sha256
        == MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256
    )
    assert (
        manifest.implementation_source_sha256
        == MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256
    )
    assert (
        manifest.authoritative_writer_specification_sha256
        == MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256
    )
    assert (
        manifest.authoritative_writer_implementation_source_sha256
        == MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256
    )
    assert (
        manifest.initial_boundary_predecessor_implementation_source_sha256
        == MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256
    )
    for value in (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)


def test_manifest_v5_cli_creates_validates_and_uses_only_v5_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest-v5.json"
    assert (
        main(
            [
                "manifest-v5",
                "--experiment-id",
                "manifest-v5-cli",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    receipt = capsys.readouterr().out.strip()
    assert main(["validate", "--manifest", str(path)]) == 0
    assert capsys.readouterr().out.strip() == receipt

    from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v5 as runner

    def dispatched(**_: object) -> object:
        raise RuntimeError("v5-runner-dispatched")

    monkeypatch.setattr(runner, "run_massive_adaptive_rl_experiment_v5", dispatched)
    with pytest.raises(RuntimeError, match="v5-runner-dispatched"):
        main(
            [
                "run",
                "--manifest",
                str(path),
                "--source-root",
                str(tmp_path / "sources"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )
