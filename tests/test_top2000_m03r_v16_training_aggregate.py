from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_fit import (
    M03RV16NumericalTrainingFailure,
    M03RV16TrainingAdequacy,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16PanelSchedule,
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackageArtifacts,
    build_m03r_v16_package_plan,
)
from rl_quant.workflows import top2000_m03r_v16_training_aggregate as aggregate


def _surfaces() -> tuple[object, M03RV16ExecutionAuthorization]:
    digest = "a" * 64
    artifacts = M03RV16PackageArtifacts(
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        dependency_lock_sha256=digest,
        cache_artifact_sha256=digest,
        cache_manifest_sha256=digest,
        asset_axis_sha256=digest,
        risk_artifact_sha256=digest,
        risk_source_manifest_file_sha256=digest,
        risk_source_receipt_sha256=digest,
        exposure_receipt_sha256=digest,
        projector_manifest_file_sha256=digest,
        projector_manifest_sha256=digest,
        projector_binding_sha256=digest,
        worker_source_sha256=digest,
        operator_source_sha256=digest,
        initial_parameter_state_file_sha256=digest,
        initial_parameter_state_sha256=digest,
        initial_parameter_architecture_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        structural_action_operator_root_sha256=digest,
        structural_target_operator_root_sha256=digest,
        structural_target_root_sha256=(digest, digest, digest),
        image_reference=f"registry.invalid/q@sha256:{digest}",
        image_digest_sha256=digest,
    )
    schedule = M03RV16PanelSchedule(
        protocol_common_data_sha256=digest,
        cache_sha256=digest,
        asset_axis_sha256=digest,
        fold_geometry_sha256=tuple(
            row.receipt_sha256 for row in render_m03r_v16_fold_geometries(1001)
        ),
    )
    package = build_m03r_v16_package_plan(artifacts, schedule)
    authorization = M03RV16ExecutionAuthorization(
        package_plan_sha256=package.package_plan_sha256,
        package_plan_file_sha256="b" * 64,
        source_archive_sha256=digest,
        source_manifest_sha256=digest,
        worker_source_sha256=digest,
        structural_slab_file_sha256=digest,
        structural_slab_receipt_sha256=digest,
        image_reference=artifacts.image_reference,
    )
    authorization.validate(package)
    return package, authorization


def _adequacy(setting: int, fold: int, *, adequate: bool) -> M03RV16TrainingAdequacy:
    value = M03RV16TrainingAdequacy(
        setting_index=setting,
        fold_index=fold,
        epoch_fit_receipt_sha256=tuple(
            f"{setting * 100 + fold * 10 + epoch + 1:064x}" for epoch in range(8)
        ),
        final_prediction_to_target_std_ratio=0.5 if adequate else 0.0,
        recent_rank_ic_slope=0.0,
        recent_robust_loss_relative_improvement=0.0,
        recent_encoder_clip_fraction=0.0,
        recent_selection_head_clip_fraction=0.0,
        status="adequate" if adequate else "collapsed-output",
    )
    value.validate()
    return value


@pytest.mark.parametrize("primary_adequate", [False, True])
def test_v16_training_panel_authorizes_qualification_only_after_primary_adequacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_adequate: bool,
) -> None:
    package, authorization = _surfaces()
    monkeypatch.setattr(aggregate, "load_m03r_v16_package_plan", lambda *a, **k: package)
    monkeypatch.setattr(
        aggregate,
        "load_m03r_v16_execution_authorization",
        lambda *a, **k: authorization,
    )
    terminals: dict[Path, dict[str, object]] = {}
    terminal_paths: list[Path] = []
    terminal_files: list[str] = []
    for setting in range(3):
        root = tmp_path / f"setting-{setting}"
        root.mkdir()
        path = root / "training-terminal.json"
        unsigned: dict[str, object] = {
            "schema": aggregate.M03R_V16_TRAINING_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "worker_plan_sha256": package.panel.workers[setting].receipt_sha256,  # type: ignore[union-attr]
            "training_activation_receipt_sha256": "d" * 64,
            "setting_index": setting,
            "fold_terminal_file_sha256": tuple(
                f"{setting * 10 + fold + 1:064x}" for fold in range(5)
            ),
            "fold_training_adequacy_status": tuple(
                "adequate"
                if (primary_adequate or setting != 2 or fold != 0)
                else "still-improving"
                for fold in range(5)
            ),
            "qualification_tail_accessed": False,
            "outer_qualification_authorized": False,
            "three_seed_confirmation_may_be_minted": False,
            "source_tree_root_sha256": "e" * 64,
            "rendered_manifest_sha256": "a" * 64,
            "pod_template_sha256": "b" * 64,
            "launch_authority_receipt_sha256": "c" * 64,
            "admitted_job_authority_receipt_sha256": "f" * 64,
            "job_uid": f"job-{setting}",
            "pod_runtime_attestation_receipt_sha256": "9" * 64,
            "pod_uid": f"pod-{setting}",
        }
        terminals[path] = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
        path.write_bytes(canonical_json_file_bytes(terminals[path]))
        terminal_paths.append(path)
        terminal_files.append(file_sha256(path))

    monkeypatch.setattr(
        aggregate,
        "_read",
        lambda path, expected: terminals[path],
    )
    monkeypatch.setattr(
        aggregate,
        "_recompute_fold",
        lambda root, fold_sha, *, setting_index, fold_index, **kwargs: (
            _adequacy(
                setting_index,
                fold_index,
                adequate=(primary_adequate or setting_index != 2 or fold_index != 0),
            ),
            {
                "checkpoint_file_sha256": f"{setting_index * 10 + fold_index + 30:064x}",
                "checkpoint_source_array_sha256": "f" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        aggregate,
        "Top2000M03RV16PredictivePolicy",
        lambda setting: object(),
    )
    monkeypatch.setattr(
        aggregate,
        "load_m03r_v16_epoch_checkpoint_for_evaluation",
        lambda *args, **kwargs: object(),
    )

    result = aggregate.aggregate_m03r_v16_training_panel(
        package_plan_path=tmp_path / "package.json",
        package_plan_file_sha256="1" * 64,
        execution_authorization_path=tmp_path / "authorization.json",
        execution_authorization_file_sha256="2" * 64,
        training_terminal_paths=tuple(terminal_paths),  # type: ignore[arg-type]
        training_terminal_file_sha256=tuple(terminal_files),  # type: ignore[arg-type]
        output_root=tmp_path / "aggregate",
    )

    assert result["outer_qualification_outcomes_accessed"] is False
    if primary_adequate:
        assert result["outer_qualification_authorized"] is True
        assert result["next_research_action"] == "qualification-only-execution"
        assert (tmp_path / "aggregate" / "qualification-activation.json").is_file()
    else:
        assert result["outer_qualification_authorized"] is False
        assert result["next_research_action"] == "fit-pathology-investigation"
        assert "qualification_activation_receipt_sha256" not in result
        assert not (tmp_path / "aggregate" / "qualification-activation.json").exists()


def test_v16_numerical_setting_outcome_routes_panel_without_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, authorization = _surfaces()
    monkeypatch.setattr(aggregate, "load_m03r_v16_package_plan", lambda *a, **k: package)
    monkeypatch.setattr(
        aggregate,
        "load_m03r_v16_execution_authorization",
        lambda *a, **k: authorization,
    )
    paths: list[Path] = []
    hashes: list[str] = []
    for setting in range(2):
        root = tmp_path / f"setting-{setting}"
        root.mkdir()
        path = root / "training-terminal.json"
        unsigned: dict[str, object] = {
            "schema": aggregate.M03R_V16_TRAINING_TERMINAL_SCHEMA,
            "package_plan_sha256": package.package_plan_sha256,  # type: ignore[union-attr]
            "authorization_receipt_sha256": authorization.receipt_sha256,
            "worker_plan_sha256": package.panel.workers[setting].receipt_sha256,  # type: ignore[union-attr]
            "training_activation_receipt_sha256": "d" * 64,
            "setting_index": setting,
            "fold_terminal_file_sha256": tuple(
                f"{setting * 10 + fold + 1:064x}" for fold in range(5)
            ),
            "fold_training_adequacy_status": ("adequate",) * 5,
            "qualification_tail_accessed": False,
            "outer_qualification_authorized": False,
            "three_seed_confirmation_may_be_minted": False,
            "source_tree_root_sha256": "e" * 64,
            "rendered_manifest_sha256": "a" * 64,
            "pod_template_sha256": "b" * 64,
            "launch_authority_receipt_sha256": "c" * 64,
            "admitted_job_authority_receipt_sha256": "f" * 64,
            "job_uid": f"job-{setting}",
            "pod_runtime_attestation_receipt_sha256": "9" * 64,
            "pod_uid": f"pod-{setting}",
        }
        path.write_bytes(
            canonical_json_file_bytes(
                {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
            )
        )
        paths.append(path)
        hashes.append(file_sha256(path))
    numerical_root = tmp_path / "setting-2"
    numerical_root.mkdir()
    numerical = M03RV16NumericalTrainingFailure(
        package_plan_sha256=package.package_plan_sha256,  # type: ignore[union-attr]
        authorization_receipt_sha256=authorization.receipt_sha256,
        training_activation_receipt_sha256="d" * 64,
        worker_plan_sha256=package.panel.workers[2].receipt_sha256,  # type: ignore[union-attr]
        source_tree_root_sha256="e" * 64,
        rendered_manifest_sha256="a" * 64,
        pod_template_sha256="b" * 64,
        launch_authority_receipt_sha256="c" * 64,
        admitted_job_authority_receipt_sha256="f" * 64,
        pod_runtime_attestation_receipt_sha256="9" * 64,
        job_uid="job-2",
        pod_uid="pod-2",
        setting_index=2,
        setting_id=package.panel.workers[2].setting_id,  # type: ignore[union-attr]
        fold_index=1,
        update_index=3,
        failure_phase="optimizer-update",
        error_type="M03RV16NumericalTrainingError",
        error="V16 score gradients are non-finite",
        model_state_sha256="1" * 64,
        optimizer_state_sha256="2" * 64,
    )
    failure_path = numerical_root / "training-numerical-failure.json"
    failure_unsigned = asdict(numerical)
    failure_path.write_bytes(
        canonical_json_file_bytes(
            {
                **failure_unsigned,
                "receipt_sha256": numerical.receipt_sha256,
            }
        )
    )
    paths.append(failure_path)
    hashes.append(file_sha256(failure_path))
    monkeypatch.setattr(
        aggregate,
        "_recompute_fold",
        lambda root, fold_sha, *, setting_index, fold_index, **kwargs: (
            _adequacy(setting_index, fold_index, adequate=True),
            {
                "checkpoint_file_sha256": f"{setting_index * 10 + fold_index + 30:064x}",
                "checkpoint_source_array_sha256": "f" * 64,
            },
        ),
    )
    result = aggregate.aggregate_m03r_v16_training_panel(
        package_plan_path=tmp_path / "package.json",
        package_plan_file_sha256="1" * 64,
        execution_authorization_path=tmp_path / "authorization.json",
        execution_authorization_file_sha256="2" * 64,
        training_terminal_paths=tuple(paths),  # type: ignore[arg-type]
        training_terminal_file_sha256=tuple(hashes),  # type: ignore[arg-type]
        output_root=tmp_path / "aggregate-numerical",
    )
    assert result["next_research_action"] == "numerical-investigation"
    assert result["outer_qualification_authorized"] is False
    assert result["training_outcome_kind"] == (
        "training-terminal",
        "training-terminal",
        "numerical-failure",
    )
    assert result["primary_aggregate_adequacy"] == "numerically-invalid"
    assert not (
        tmp_path / "aggregate-numerical/qualification-activation.json"
    ).exists()
