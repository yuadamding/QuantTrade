from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_four_fold_fit_v1 as four_fold_fit
from rl_quant.workflows.massive_adaptive_rl_execution_environment_v1 import (
    MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MassiveAdaptiveRLExperimentStageV2,
    advance_massive_adaptive_rl_experiment_state_v2,
    register_massive_adaptive_rl_experiment_state_v2,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable,
    MassiveAdaptiveRLFourFoldFitV1Error,
    _four_fold_fit_execution_lease,
    advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1,
    advance_massive_adaptive_rl_four_fold_fit_state_v1,
    build_massive_adaptive_rl_four_fold_fit_authority_v1,
    build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1,
    load_massive_adaptive_rl_four_fold_fit_authority_v1,
    load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1,
    materialize_massive_adaptive_rl_four_fold_fit_authority_v1,
    materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1,
    run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1,
    run_or_resume_massive_adaptive_rl_four_fold_fit_v1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_inputs_v1 import (
    MassiveAdaptiveRLFoldFitInputsAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_fold_fit_v1 import (
    MassiveAdaptiveRLFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MassiveAdaptiveRLRuntimeSourcesV1,
)


_T = TypeVar("_T")


def _digest(value: str) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _typed_roots(monkeypatch: pytest.MonkeyPatch):
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="four-fold-fit-canary",
    )
    runtime_sources = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV1,
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("runtime-sources"),
        runtime_source_graph_authority=SimpleNamespace(
            runtime_authority_receipt_sha256=_digest("runtime-witness")
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLRuntimeSourcesV1,
        "validate",
        lambda _self: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldFitInputsAuthorityV1,
        "validate",
        lambda _self: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldFitAuthorityV1,
        "validate",
        lambda _self: None,
    )

    environments = []
    inputs = []
    fits = []
    for fold_index in range(4):
        environment = _typed_shell(
            MassiveAdaptiveRLExecutionEnvironmentAuthorityV1,
            semantic_receipt_sha256=_digest(f"environment-{fold_index}"),
            git_commit=_digest("git-commit"),
            git_tree=_digest("git-tree"),
            tracked_source_inventory_sha256=_digest("tracked-source"),
            dependency_lock_sha256=_digest("lock"),
            python_version="3.11",
            python_implementation="CPython",
            pytorch_version="test",
            numpy_version="test",
            cuda_runtime_version=None,
            cudnn_version=None,
            execution_device_type="cpu",
            gpu_name=None,
            gpu_compute_capability=None,
            nvidia_driver_version=None,
            parameter_dtype="torch.float32",
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            float32_matmul_tf32=False,
            cudnn_tf32=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            cublas_workspace_config="",
            torch_cpu_threads=1,
            torch_interop_threads=1,
            process_thread_environment=(),
            training_seed=17,
            model_initialization_specification_sha256=_digest("model-init"),
            initial_model_state_receipt_sha256=_digest("initial-model"),
            source_data_qualified=True,
            source_transaction_verified=True,
            runtime_environment_replayed=True,
        )
        fit_input = _typed_shell(
            MassiveAdaptiveRLFoldFitInputsAuthorityV1,
            experiment_id=manifest.experiment_id,
            outer_fold_index=fold_index,
            manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
            runtime_sources_receipt_sha256=(
                runtime_sources.semantic_receipt_sha256
            ),
            runtime_graph_witness_receipt_sha256=(
                runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
            ),
            execution_environment_authority=environment,
            source_data_qualified=True,
            runtime_inputs_replayed=True,
            development_rl_training_inputs_authorized=True,
            semantic_receipt_sha256=_digest(f"fit-input-{fold_index}"),
            _loaded_source=object(),
        )
        fit = _typed_shell(
            MassiveAdaptiveRLFoldFitAuthorityV1,
            experiment_id=manifest.experiment_id,
            outer_fold_index=fold_index,
            manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
            runtime_sources_receipt_sha256=(
                runtime_sources.semantic_receipt_sha256
            ),
            runtime_graph_witness_receipt_sha256=(
                runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
            ),
            fit_inputs_authority=fit_input,
            execution_environment_authority=environment,
            source_data_qualified=True,
            runtime_fit_replayed=True,
            development_rl_training_authorized=True,
            semantic_receipt_sha256=_digest(f"fit-{fold_index}"),
            _loaded_source=object(),
        )
        environments.append(environment)
        inputs.append(fit_input)
        fits.append(fit)
    return manifest, runtime_sources, tuple(inputs), tuple(fits)


def _roots():
    manifest = SimpleNamespace(
        experiment_id="four-fold-fit-canary",
        semantic_receipt_sha256=_digest("manifest"),
        validate=lambda: None,
    )
    runtime_sources = SimpleNamespace(
        semantic_receipt_sha256=_digest("runtime-sources"),
        runtime_source_graph_authority=SimpleNamespace(
            runtime_authority_receipt_sha256=_digest("runtime-witness")
        ),
        validate=lambda: None,
    )
    fingerprint = _digest("scientific-execution")
    inputs = []
    fits = []
    for fold_index in range(4):
        environment = SimpleNamespace(
            semantic_receipt_sha256=_digest(f"environment-{fold_index}"),
            scientific_execution_fingerprint_sha256=fingerprint,
            physical_worker_compatibility_sha256=_digest("worker-class"),
            development_execution_authorized=True,
        )
        fit_input = SimpleNamespace(
            experiment_id=manifest.experiment_id,
            outer_fold_index=fold_index,
            manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
            runtime_sources_receipt_sha256=runtime_sources.semantic_receipt_sha256,
            runtime_graph_witness_receipt_sha256=(
                runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
            ),
            execution_environment_authority=environment,
            source_data_qualified=True,
            runtime_inputs_replayed=True,
            development_rl_training_inputs_authorized=True,
            semantic_receipt_sha256=_digest(f"fit-input-{fold_index}"),
            validate=lambda: None,
        )
        inputs.append(fit_input)
        fits.append(
            SimpleNamespace(
                experiment_id=manifest.experiment_id,
                outer_fold_index=fold_index,
                manifest_v3_receipt_sha256=manifest.semantic_receipt_sha256,
                runtime_sources_receipt_sha256=(
                    runtime_sources.semantic_receipt_sha256
                ),
                runtime_graph_witness_receipt_sha256=(
                    runtime_sources.runtime_source_graph_authority.runtime_authority_receipt_sha256
                ),
                fit_inputs_authority=fit_input,
                execution_environment_authority=environment,
                source_data_qualified=True,
                runtime_fit_replayed=True,
                development_rl_training_authorized=True,
                semantic_receipt_sha256=_digest(f"fit-{fold_index}"),
                validate=lambda: None,
            )
        )
    return manifest, runtime_sources, tuple(inputs), tuple(fits)


def test_four_fold_fit_aggregates_exact_order_and_one_scientific_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources, inputs, fits = _typed_roots(monkeypatch)
    input_authority = build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fold_fit_inputs=inputs,
    )
    assert input_authority.fold_indices == (0, 1, 2, 3)
    assert input_authority.development_rl_training_inputs_authorized
    assert not input_authority.source_transaction_verified
    assert not input_authority.development_stage_authorized
    downgraded = replace(
        input_authority,
        source_data_qualified=False,
        runtime_inputs_replayed=False,
        development_rl_training_inputs_authorized=False,
        semantic_receipt_sha256="0" * 64,
    )
    downgraded = replace(
        downgraded,
        semantic_receipt_sha256=semantic_sha256(downgraded.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="inputs"):
        downgraded.validate()

    persisted_inputs = (
        materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=tmp_path,
            authority=input_authority,
            committed_at_ms=1_000,
        )
    )
    replayed_input_indices: list[int] = []

    def replay_input(**kwargs: object):
        fold_index = int(kwargs["outer_fold_index"])
        replayed_input_indices.append(fold_index)
        return inputs[fold_index]

    monkeypatch.setattr(
        four_fold_fit,
        "prepare_or_resume_massive_adaptive_rl_fold_fit_inputs_v1",
        replay_input,
    )
    replayed_inputs = load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        root=tmp_path,
        manifest=manifest,
        runtime_sources=runtime_sources,
        verified_at_ms=1_001,
        device="cpu",
    )
    assert replayed_input_indices == [0, 1, 2, 3]
    assert replayed_inputs.semantic_receipt_sha256 == (
        persisted_inputs.semantic_receipt_sha256
    )
    assert replayed_inputs.development_stage_authorized

    fit_authority = build_massive_adaptive_rl_four_fold_fit_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fit_inputs_authority=replayed_inputs,
        fold_fits=fits,
    )
    assert fit_authority.fold_fit(2) is fits[2]
    persisted_fit = materialize_massive_adaptive_rl_four_fold_fit_authority_v1(
        root=tmp_path,
        authority=fit_authority,
        committed_at_ms=2_000,
    )
    replayed_fit_indices: list[int] = []

    def replay_fit(**kwargs: object):
        fold_index = int(kwargs["outer_fold_index"])
        replayed_fit_indices.append(fold_index)
        return fits[fold_index]

    monkeypatch.setattr(
        four_fold_fit,
        "load_massive_adaptive_rl_fold_fit_authority_v1",
        replay_fit,
    )
    replayed_fit = load_massive_adaptive_rl_four_fold_fit_authority_v1(
        root=tmp_path,
        manifest=manifest,
        runtime_sources=runtime_sources,
        verified_at_ms=2_001,
        device="cpu",
    )
    assert replayed_fit_indices == [0, 1, 2, 3]
    assert replayed_fit.semantic_receipt_sha256 == persisted_fit.semantic_receipt_sha256
    assert replayed_fit.development_rl_training_authorized
    assert replayed_fit.development_stage_authorized
    assert not replayed_fit.profitability_reporting_authorized

    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path / "state",
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    source_replayed = advance_massive_adaptive_rl_experiment_state_v2(
        artifact_root=tmp_path / "state",
        previous=registered,
        stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
        stage_artifact_receipt_sha256=_digest("source-bundle"),
    )
    inputs_state = advance_massive_adaptive_rl_four_fold_fit_inputs_state_v1(
        artifact_root=tmp_path / "state",
        previous=source_replayed,
        authority=replayed_inputs,
    )
    fit_state = advance_massive_adaptive_rl_four_fold_fit_state_v1(
        artifact_root=tmp_path / "state",
        previous=inputs_state,
        authority=replayed_fit,
    )
    assert inputs_state.stage is MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED
    assert (
        fit_state.stage
        is MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED
    )
    assert fit_state.stage_artifact_receipt_sha256 == (
        replayed_fit.semantic_receipt_sha256
    )


def test_four_fold_fit_rejects_receipt_shaped_duck_types(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources, typed_inputs, _typed_fits = _typed_roots(
        monkeypatch
    )
    _fake_manifest, _fake_sources, inputs, _fits = _roots()
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="exact"):
        build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fold_fit_inputs=inputs,
        )
    input_authority = build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fold_fit_inputs=typed_inputs,
    )
    persisted_inputs = (
        materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=tmp_path,
            authority=input_authority,
            committed_at_ms=1_000,
        )
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="exact"):
        build_massive_adaptive_rl_four_fold_fit_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fit_inputs_authority=persisted_inputs,
            fold_fits=_fits,
        )


def test_four_fold_fit_rejects_missing_fold_and_mixed_scientific_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources, inputs, _fits = _typed_roots(monkeypatch)
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="inputs"):
        build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fold_fit_inputs=inputs[:3],
        )

    object.__setattr__(
        inputs[3].execution_environment_authority,
        "git_commit",
        _digest("other-execution"),
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="inputs"):
        build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fold_fit_inputs=inputs,
        )

    object.__setattr__(
        inputs[3].execution_environment_authority,
        "git_commit",
        _digest("git-commit"),
    )
    object.__setattr__(inputs[3], "_loaded_source", None)
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="inputs"):
        build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fold_fit_inputs=inputs,
        )

    object.__setattr__(inputs[3], "_loaded_source", object())
    object.__setattr__(
        inputs[3].execution_environment_authority,
        "gpu_name",
        "other-worker-class",
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="inputs"):
        build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fold_fit_inputs=inputs,
        )


def test_completed_four_fold_stages_do_not_recreate_missing_aggregates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime_sources, inputs, _fits = _typed_roots(monkeypatch)
    input_authority = build_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
        manifest=manifest,
        runtime_sources=runtime_sources,
        fold_fit_inputs=inputs,
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="aggregate"):
        run_or_resume_massive_adaptive_rl_four_fold_fit_inputs_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            artifact_root=tmp_path,
            committed_at_ms=1_000,
            device="cpu",
            allow_materialize=False,
        )
    persisted_inputs = (
        materialize_massive_adaptive_rl_four_fold_fit_inputs_authority_v1(
            root=tmp_path,
            authority=input_authority,
            committed_at_ms=1_100,
        )
    )
    monkeypatch.setattr(
        four_fold_fit,
        "load_massive_adaptive_rl_four_fold_fit_inputs_authority_v1",
        lambda **_kwargs: persisted_inputs,
    )
    with pytest.raises(MassiveAdaptiveRLFourFoldFitV1Error, match="aggregate"):
        run_or_resume_massive_adaptive_rl_four_fold_fit_v1(
            manifest=manifest,
            runtime_sources=runtime_sources,
            fit_inputs_authority=persisted_inputs,
            artifact_root=tmp_path,
            committed_at_ms=2_000,
            device="cpu",
            allow_materialize=False,
        )


def test_four_fold_root_stage_lease_rejects_a_concurrent_owner(tmp_path) -> None:
    with _four_fold_fit_execution_lease(
        root=tmp_path,
        experiment_id="four-fold-lease-canary",
        stage="fit",
    ):
        with pytest.raises(
            MassiveAdaptiveRLFourFoldFitExecutionLeaseUnavailable,
            match="already held",
        ):
            with _four_fold_fit_execution_lease(
                root=tmp_path,
                experiment_id="four-fold-lease-canary",
                stage="fit",
            ):
                raise AssertionError("concurrent root-stage lease unexpectedly acquired")
