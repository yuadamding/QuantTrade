from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    massive_adaptive_rl_fixed_control_scientific_inventory_v1,
)
from rl_quant.workflows import massive_adaptive_rl_manifest_v5 as manifest_module
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
    MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256,
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
    assert manifest.prequential_stage_sequence[:3] == (
        "trained",
        "execution-implementation-registered",
        "initial-validation-inputs-committed",
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
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV5Error, match="create-only"):
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


def test_manifest_v5_binds_scientific_protocol_and_defers_physical_implementation() -> (
    None
):
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-hashes"
    )
    assert (
        manifest.specification_sha256
        == MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256
    )
    assert len(manifest.scientific_protocol_projection_sha256) == 64
    projection = manifest_module.massive_adaptive_rl_scientific_protocol_projection_v1(
        manifest.base_manifest
    )
    economics = projection["economics"]
    assert isinstance(economics, dict)
    assert economics["fixed_control_scientific_inventory"] == (
        massive_adaptive_rl_fixed_control_scientific_inventory_v1()
    )
    assert economics["fixed_control_scientific_inventory_sha256"] == semantic_sha256(
        economics["fixed_control_scientific_inventory"]
    )
    controls = economics["fixed_control_scientific_inventory"]["constant_controls"]
    assert tuple(row["control_id"] for row in controls) == (
        "FC00",
        "FC01",
        "FC02",
        "FC03",
        "FC04",
        "FC05",
        "FC07",
        "FC08",
        "FC09",
        "FC10",
        "FC11",
        "FC12",
    )
    payload = manifest.semantic_unsigned()
    assert "implementation_source_sha256" not in payload
    assert "authoritative_writer_implementation_source_sha256" not in payload
    assert "initial_boundary_predecessor_implementation_source_sha256" not in payload
    assert "initial_validation_inputs_implementation_source_sha256" not in payload
    assert (
        "validation_execution_environment_implementation_source_sha256" not in payload
    )
    assert "experiment_global_lock_implementation_source_sha256" not in payload
    assert "manifest_v5_registration_implementation_source_sha256" not in payload
    assert "base_manifest_v4_receipt_sha256" not in payload
    assert "authoritative_writer_specification_sha256" not in payload
    assert "execution_implementation_registration_specification_sha256" not in payload
    for value in (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V5_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_INITIAL_BOUNDARY_PREDECESSOR_V4_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_EXECUTION_IMPLEMENTATION_REGISTRATION_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_SCIENTIFIC_PROTOCOL_V1_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)


def test_manifest_v5_receipt_does_not_depend_on_runner_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-implementation-decoupling"
    )
    monkeypatch.setattr(
        manifest_module,
        "MASSIVE_ADAPTIVE_RL_EXPERIMENT_RUNNER_V5_SOURCE_SHA256",
        "f" * 64,
    )
    after = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-implementation-decoupling"
    )
    assert after.semantic_receipt_sha256 == before.semantic_receipt_sha256


def test_manifest_v5_receipt_ignores_nested_compatibility_source_hashes() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="manifest-v5-nested-source-decoupling"
    )
    base_v4 = manifest.base_manifest
    base_v3 = base_v4.base_manifest
    base_v2 = base_v3.base_manifest

    changed_v2 = replace(
        base_v2,
        workflow_implementation_source_sha256="a" * 64,
        semantic_receipt_sha256="0" * 64,
    )
    changed_v2 = replace(
        changed_v2,
        semantic_receipt_sha256=semantic_sha256(changed_v2.semantic_unsigned()),
    )
    changed_v3 = replace(
        base_v3,
        base_manifest=changed_v2,
        profitability_report_implementation_source_sha256="b" * 64,
        implementation_source_sha256="c" * 64,
        semantic_receipt_sha256="0" * 64,
    )
    changed_v3 = replace(
        changed_v3,
        semantic_receipt_sha256=semantic_sha256(changed_v3.semantic_unsigned()),
    )
    changed_v4 = replace(
        base_v4,
        base_manifest=changed_v3,
        implementation_source_sha256="d" * 64,
        semantic_receipt_sha256="0" * 64,
    )
    changed_v4 = replace(
        changed_v4,
        semantic_receipt_sha256=semantic_sha256(changed_v4.semantic_unsigned()),
    )
    changed = replace(manifest, base_manifest=changed_v4)
    changed.validate()
    assert changed.semantic_receipt_sha256 == manifest.semantic_receipt_sha256


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

    from rl_quant.workflows import (
        massive_adaptive_rl_execution_implementation_registration_v1 as implementation,
    )
    from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
        run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1,
    )

    artifact_root = tmp_path / "implementation-artifacts"
    manifest = load_massive_adaptive_rl_experiment_manifest_v5(path)
    run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
        root=artifact_root,
        manifest=manifest,
    )
    implementation_receipt = semantic_sha256("implementation-registration")
    implementation_result = SimpleNamespace(
        semantic_unsigned=lambda: {"schema": "implementation-registration"},
        semantic_receipt_sha256=implementation_receipt,
        source_receipt_sha256=semantic_sha256("implementation-source"),
        source_transaction_receipt_sha256=semantic_sha256("implementation-commit"),
        source_transaction_committed_at_ms=3,
        scientific_execution_fingerprint_sha256=semantic_sha256(
            "execution-fingerprint"
        ),
        runtime_implementation_replayed=True,
        development_execution_registered=True,
    )
    monkeypatch.setattr(
        implementation,
        "run_or_resume_massive_adaptive_rl_execution_implementation_registration_v1",
        lambda **_: implementation_result,
    )
    assert (
        main(
            [
                "register-implementation",
                "--manifest",
                str(path),
                "--artifact-root",
                str(artifact_root),
            ]
        )
        == 0
    )
    implementation_output = json.loads(capsys.readouterr().out)
    assert implementation_output["semantic_receipt_sha256"] == implementation_receipt
    assert implementation_output["development_execution_registered"] is True

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
