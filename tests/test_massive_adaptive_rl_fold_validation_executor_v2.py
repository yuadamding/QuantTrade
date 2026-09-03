from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import asdict, replace
from io import BytesIO
import inspect
from typing import get_type_hints

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    publish_massive_source_object,
)
from rl_quant.evaluation import (
    massive_adaptive_rl_fold_validation_executor_v2 as executor,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_execution_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
    MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error,
    MassiveAdaptiveRLValidationExecutionStageV1,
    fold_validation_execution_authority_relative_path_v1,
    load_massive_adaptive_rl_fold_validation_execution_authority_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v1 import (
    _CanonicalValidationStageV1,
    _canonical_validation_stages_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v2 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SPEC_SHA256,
    MassiveAdaptiveRLFoldValidationExecutorV2Error,
    MassiveAdaptiveRLFoldValidationRecoveryGenerationRequired,
    _existing_stage_prefix,
    _next_publication_time_ms,
    _transaction_exists_v2,
    run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v1 import (
    validation_primary_trace_relative_path_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MassiveAdaptiveRLExperimentManifestV4,
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
)
from rl_quant.workflows import (
    massive_adaptive_rl_validation_execution_environment_v1 as environment_module,
)
from rl_quant.workflows.massive_adaptive_rl_validation_execution_environment_v1 import (
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET,
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
    MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
    MassiveAdaptiveRLValidationExecutionEnvironmentLeaseUnavailable,
    MassiveAdaptiveRLValidationExecutionEnvironmentV1Error,
    _validation_execution_environment_lease_v1,
    load_massive_adaptive_rl_validation_execution_environment_v1,
    run_or_resume_massive_adaptive_rl_validation_execution_environment_v1,
    validation_execution_environment_relative_path_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    policy_selection_authority_relative_path_v3,
    policy_selection_v2_witness_relative_path_v3,
)


def _digest(value: object) -> str:
    return semantic_sha256(value)


class _BarrierPathStub:
    def __init__(self, manifest_receipt: str, receipt: str) -> None:
        self.manifest_v4_receipt_sha256 = manifest_receipt
        self.semantic_receipt_sha256 = receipt
        self.expected_candidate_checkpoint_authority_receipt_inventories = (
            (_digest("environment-checkpoint"),),
            (),
            (),
            (),
        )

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class _ReceiptOnly:
    semantic_receipt_sha256: str


@dataclass(frozen=True)
class _FoldFitOnly:
    outer_fold_index: int


@dataclass(frozen=True)
class _OutcomeOnly:
    fold_index: int


@dataclass(frozen=True)
class _SourcesOnly:
    base_authority_v1: object
    runtime_chronology_authority: object


@dataclass(frozen=True)
class _RegistryOnly:
    base_registry_v1: object


@dataclass(frozen=True)
class _BarrierOnly:
    manifest_v4_receipt_sha256: str
    semantic_receipt_sha256: str
    source_transaction_committed_at_ms: int
    base_authority_v1: object
    sources: _SourcesOnly
    registry: _RegistryOnly

    def validate(self) -> None:
        return None

    def validation_sources(self, _fold_index: int) -> _SourcesOnly:
        return self.sources

    def validation_registry(self, _fold_index: int) -> _RegistryOnly:
        return self.registry


def _generic_environment(
    *, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1:
    tracked = (("src/rl_quant/example.py", _digest("source")),)
    distributions = (("numpy", "2.2.6"), ("torch", "2.6.0"))
    thread_environment = (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("PYTHONHASHSEED", "17"),
    )
    provisional = MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1(
        experiment_id=manifest.experiment_id,
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        runtime_sources_v2_receipt_sha256=_digest("runtime-v2"),
        source_bundle_v2_receipt_sha256=_digest("bundle-v2"),
        runtime_source_graph_v2_receipt_sha256=_digest("graph-v2"),
        runtime_source_graph_v2_witness_receipt_sha256=_digest("graph-witness-v2"),
        replay_dependency_index_v2_receipt_sha256=_digest("index-v2"),
        four_fold_fit_authority_receipt_sha256=_digest("fit"),
        four_fold_validation_inputs_v2_receipt_sha256=_digest("inputs-v2"),
        four_fold_validation_inputs_v2_source_receipt_sha256=_digest("inputs-source"),
        four_fold_validation_inputs_v2_commit_receipt_sha256=_digest("inputs-commit"),
        four_fold_validation_inputs_v2_committed_at_ms=100,
        git_commit="1" * 40,
        git_tree="2" * 40,
        tracked_worktree_clean=True,
        tracked_worktree_status=(),
        tracked_source_inventory=tracked,
        tracked_source_inventory_sha256=semantic_sha256(tracked),
        untracked_runtime_source_inventory=(),
        untracked_runtime_source_count=0,
        dependency_lock_sha256=_digest("lock"),
        installed_distribution_inventory=distributions,
        installed_distribution_inventory_sha256=semantic_sha256(distributions),
        python_version="3.11.13",
        python_implementation="CPython",
        pytorch_version="2.6.0",
        numpy_version="2.2.6",
        torch_build_configuration_sha256=_digest("torch-build"),
        numpy_build_configuration_sha256=_digest("numpy-build"),
        platform_machine="x86_64",
        cpu_model="test-cpu",
        cpu_capability="AVX2",
        cpu_instruction_inventory_sha256=_digest("cpu-flags"),
        execution_device_specification="cpu",
        parameter_dtype="torch.float32",
        observation_dtype="torch.float32",
        deterministic_algorithms=True,
        deterministic_warn_only=False,
        float32_matmul_tf32=False,
        cudnn_tf32=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        torch_cpu_threads=1,
        torch_interop_threads=1,
        process_thread_environment=thread_environment,
        evaluator_implementation_inventory=(
            environment_module._EVALUATOR_IMPLEMENTATION_INVENTORY
        ),
        evaluator_implementation_inventory_sha256=semantic_sha256(
            environment_module._EVALUATOR_IMPLEMENTATION_INVENTORY
        ),
        executor_implementation_source_sha256=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256
        ),
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def _generic_execution(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV4,
    environment: MassiveAdaptiveRLValidationExecutionEnvironmentAuthorityV1,
) -> MassiveAdaptiveRLFoldValidationExecutionAuthorityV1:
    names = (
        "primary-v1",
        "primary-v2",
        "ladder-v1",
        "ladder-v2",
        "fc06-v1",
        "fc06-v2",
        "fold-validation-v1",
        "fold-validation-v2",
        "selection-v2-computation",
        "selection-v3",
    )
    selection_receipt = _digest("selection-v3")
    selection_source = _digest("selection-source")
    selection_commit = _digest("selection-commit")
    previous = _digest("environment-commit")
    stages: list[MassiveAdaptiveRLValidationExecutionStageV1] = []
    for ordinal, name in enumerate(names):
        commit = (
            selection_commit
            if ordinal == len(names) - 1
            else _digest(("commit", ordinal))
        )
        stage = MassiveAdaptiveRLValidationExecutionStageV1(
            stage_ordinal=ordinal,
            stage_name=name,
            relative_payload_path=f"validation/stage-{ordinal}.json",
            authority_receipt_sha256=(
                selection_receipt
                if ordinal == len(names) - 1
                else _digest(("authority", ordinal))
            ),
            source_receipt_sha256=(
                selection_source
                if ordinal == len(names) - 1
                else _digest(("source", ordinal))
            ),
            commit_receipt_sha256=commit,
            committed_at_ms=200 + ordinal * 7,
            observed_published_at_ms=2_000 + ordinal * 11,
            previous_stage_commit_receipt_sha256=previous,
        )
        stages.append(stage)
        previous = commit
    stage_tuple = tuple(stages)
    provisional = MassiveAdaptiveRLFoldValidationExecutionAuthorityV1(
        experiment_id=manifest.experiment_id,
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        training_manifest_v3_receipt_sha256=(
            manifest.base_manifest.semantic_receipt_sha256
        ),
        fold_index=0,
        runtime_sources_v2_receipt_sha256=environment.runtime_sources_v2_receipt_sha256,
        four_fold_fit_authority_receipt_sha256=(
            environment.four_fold_fit_authority_receipt_sha256
        ),
        four_fold_validation_inputs_v2_receipt_sha256=(
            environment.four_fold_validation_inputs_v2_receipt_sha256
        ),
        checkpoint_authority_receipts=(_digest("checkpoint"),),
        fixed_control_selection_authority_receipt_sha256=_digest("fc06"),
        validation_execution_environment_receipt_sha256=(
            environment.semantic_receipt_sha256
        ),
        validation_execution_environment_source_receipt_sha256=_digest(
            "environment-source"
        ),
        validation_execution_environment_commit_receipt_sha256=_digest(
            "environment-commit"
        ),
        validation_execution_environment_committed_at_ms=150,
        validation_execution_environment_observed_published_at_ms=1_500,
        scientific_execution_fingerprint_sha256=(
            environment.scientific_execution_fingerprint_sha256
        ),
        policy_selection_v3_receipt_sha256=selection_receipt,
        policy_selection_v3_source_receipt_sha256=selection_source,
        policy_selection_v3_commit_receipt_sha256=selection_commit,
        policy_selection_v3_committed_at_ms=stages[-1].committed_at_ms,
        stages=stage_tuple,
        stage_inventory_sha256=semantic_sha256(
            tuple(asdict(stage) for stage in stage_tuple)
        ),
        execution_started_at_ms=stages[0].committed_at_ms,
        execution_completed_at_ms=stages[-1].committed_at_ms,
        execution_observed_started_at_ms=stages[0].observed_published_at_ms,
        execution_observed_completed_at_ms=stages[-1].observed_published_at_ms,
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def _publish(
    *, root, relative: str, body: dict[str, object], dataset: str, schema: str, at: int
) -> None:
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=root,
        relative_payload_path=relative,
        dataset_id=dataset,
        source_object_key=relative,
        requested_at_ms=at,
        downloaded_at_ms=at,
        schema_sha256=schema,
        entitlement_receipt_sha256=str(body["semantic_receipt_sha256"]),
        committed_at_ms=at,
    )


def test_executor_v2_surface_owns_time_device_and_environment() -> None:
    signature = inspect.signature(
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2
    )
    assert tuple(signature.parameters) == (
        "root",
        "manifest",
        "runtime_sources_v2",
        "four_fold_fit_authority",
        "four_fold_validation_inputs_v2",
        "fold_index",
        "allow_materialize",
    )
    assert not {
        "committed_at_ms",
        "environment",
        "execution_environment",
        "device",
        "actions",
        "targets",
        "metrics",
        "candidates",
        "selected_checkpoint",
        "artifact_id",
    }.intersection(signature.parameters)
    hints = get_type_hints(
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2
    )
    assert hints["manifest"] is MassiveAdaptiveRLExperimentManifestV4
    assert hints["runtime_sources_v2"] is MassiveAdaptiveRLRuntimeSourcesV2
    assert hints["four_fold_fit_authority"] is MassiveAdaptiveRLFourFoldFitAuthorityV1
    assert hints["return"] is MassiveAdaptiveRLFoldValidationExecutionAuthorityV1


def test_publication_clock_uses_real_resume_time(monkeypatch) -> None:
    monkeypatch.setattr(executor, "_wall_clock_ms", lambda: 10_000)
    assert _next_publication_time_ms(previous_at_ms=100) == 10_000
    assert _next_publication_time_ms(previous_at_ms=10_000) == 10_001


def test_stage_prefix_allows_real_gaps_but_rejects_nonmonotonic_time(tmp_path) -> None:
    stages = tuple(
        _CanonicalValidationStageV1(
            name=f"stage-{index}",
            relative_path=f"executor-v2/stage-{index}.json",
            commit_offset_ms=index,
        )
        for index in range(3)
    )
    for stage, at in zip(stages[:2], (200, 900)):
        body = {"semantic_receipt_sha256": _digest((stage.name, at))}
        _publish(
            root=tmp_path,
            relative=stage.relative_path,
            body=body,
            dataset="executor-v2-stage",
            schema=_digest("executor-v2-stage-schema"),
            at=at,
        )
    loaded = _existing_stage_prefix(
        root=tmp_path,
        stages=stages,
        environment_committed_at_ms=100,
        verified_at_ms=1_000,
    )
    assert tuple(row.commit.committed_at_ms for row in loaded) == (200, 900)

    body = {"semantic_receipt_sha256": _digest("late-stage-with-old-time")}
    _publish(
        root=tmp_path,
        relative=stages[2].relative_path,
        body=body,
        dataset="executor-v2-stage",
        schema=_digest("executor-v2-stage-schema"),
        at=800,
    )
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationExecutorV2Error,
        match="chronology differs",
    ):
        _existing_stage_prefix(
            root=tmp_path,
            stages=stages,
            environment_committed_at_ms=100,
            verified_at_ms=1_000,
        )


def test_partial_stage_requires_a_new_recovery_generation(tmp_path) -> None:
    relative = "executor-v2/partial.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("partial", encoding="utf-8")
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationRecoveryGenerationRequired,
        match="cannot be repaired",
    ):
        _transaction_exists_v2(root=tmp_path, relative=relative)


def test_validation_environment_generic_reload_is_nonauthorizing(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-environment-generic"
    )
    authority = _generic_environment(manifest=manifest)
    authority.validate()
    barrier = _BarrierPathStub(
        manifest.semantic_receipt_sha256,
        authority.four_fold_validation_inputs_v2_receipt_sha256,
    )
    relative = validation_execution_environment_relative_path_v1(
        manifest=manifest,
        four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
    )
    _publish(
        root=tmp_path,
        relative=relative,
        body=environment_module._payload(authority),
        dataset=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_DATASET
        ),
        schema=(
            MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        at=101,
    )
    loaded = load_massive_adaptive_rl_validation_execution_environment_v1(
        root=tmp_path,
        manifest=manifest,
        four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
        verified_at_ms=102,
    )
    assert loaded.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert loaded.source_transaction_verified
    assert not loaded.runtime_environment_replayed
    assert not loaded.development_stage_authorized
    assert loaded.execution_device_specification == "cpu"
    assert loaded.torch_cpu_threads == 1
    assert loaded.torch_interop_threads == 1


def test_validation_environment_lease_is_exclusive_and_does_not_mask_body_errors(
    tmp_path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-environment-lease"
    )
    with _validation_execution_environment_lease_v1(
        root=tmp_path, manifest=manifest
    ):
        with pytest.raises(
            MassiveAdaptiveRLValidationExecutionEnvironmentLeaseUnavailable
        ):
            with _validation_execution_environment_lease_v1(
                root=tmp_path, manifest=manifest
            ):
                pass
    with pytest.raises(OSError, match="environment body failed"):
        with _validation_execution_environment_lease_v1(
            root=tmp_path, manifest=manifest
        ):
            raise OSError("environment body failed")


def test_missing_validation_environment_cannot_be_backfilled_after_an_outcome(
    tmp_path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-environment-late-backfill"
    )
    barrier = _BarrierPathStub(
        manifest.semantic_receipt_sha256,
        _digest("late-environment-inputs"),
    )
    checkpoint = barrier.expected_candidate_checkpoint_authority_receipt_inventories[
        0
    ][0]
    outcome = tmp_path / validation_primary_trace_relative_path_v1(
        manifest=manifest,
        fold_index=0,
        checkpoint_authority_receipt_sha256=checkpoint,
    )
    outcome.parent.mkdir(parents=True)
    outcome.write_text("partial downstream evidence", encoding="utf-8")
    with pytest.raises(
        MassiveAdaptiveRLValidationExecutionEnvironmentV1Error,
        match="cannot be created after outcomes",
    ):
        run_or_resume_massive_adaptive_rl_validation_execution_environment_v1(
            root=tmp_path,
            manifest=manifest,
            runtime_sources_v2=object(),  # type: ignore[arg-type]
            four_fold_fit_authority=object(),  # type: ignore[arg-type]
            four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
            executor_implementation_source_sha256=(
                MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256
            ),
            committed_at_ms=200,
            allow_materialize=True,
        )


def test_fold_execution_authority_accepts_real_time_gaps_and_chains_predecessors(
    tmp_path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-execution-authority-generic"
    )
    environment = _generic_environment(manifest=manifest)
    authority = _generic_execution(manifest=manifest, environment=environment)
    authority.validate()
    assert tuple(stage.committed_at_ms for stage in authority.stages) == tuple(
        200 + index * 7 for index in range(10)
    )
    relative = fold_validation_execution_authority_relative_path_v1(
        manifest=manifest, fold_index=0
    )
    from rl_quant.evaluation import (
        massive_adaptive_rl_fold_validation_execution_authority_v1 as authority_module,
    )

    _publish(
        root=tmp_path,
        relative=relative,
        body=authority_module._payload(authority),
        dataset=MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_DATASET,
        schema=(
            MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        at=authority.execution_completed_at_ms + 1,
    )
    loaded = load_massive_adaptive_rl_fold_validation_execution_authority_v1(
        root=tmp_path,
        manifest=manifest,
        fold_index=0,
        verified_at_ms=authority.execution_completed_at_ms + 2,
    )
    assert loaded.source_transaction_verified
    assert not loaded.runtime_execution_replayed
    assert not loaded.policy_freezing_authorized
    assert not loaded.development_stage_authorized

    broken = list(authority.stages)
    broken[4] = replace(
        broken[4], previous_stage_commit_receipt_sha256=_digest("wrong-predecessor")
    )
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1Error,
        match="authority differs",
    ):
        replace(authority, stages=tuple(broken)).validate()


@pytest.mark.parametrize("failure_ordinal", range(22))
def test_executor_v2_resumes_after_every_fold3_scientific_stage(
    tmp_path,
    monkeypatch,
    failure_ordinal: int,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=f"executor-v2-interruption-{failure_ordinal}"
    )
    checkpoints = tuple(
        _ReceiptOnly(_digest(("checkpoint", index))) for index in range(4)
    )
    fixed_selection = _ReceiptOnly(_digest("fixed-selection"))
    roots = type(
        "Roots",
        (),
        {
            "fold_fit": _FoldFitOnly(outer_fold_index=3),
            "checkpoints": checkpoints,
            "fixed_fit": object(),
            "fixed_selection": fixed_selection,
        },
    )()
    sources = _SourcesOnly(
        base_authority_v1=object(),
        runtime_chronology_authority=object(),
    )
    registry = _RegistryOnly(base_registry_v1=object())
    barrier = _BarrierOnly(
        manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("barrier-v2"),
        source_transaction_committed_at_ms=100,
        base_authority_v1=object(),
        sources=sources,
        registry=registry,
    )
    stages = _canonical_validation_stages_v1(
        manifest=manifest,
        fold_index=3,
        checkpoint_authority_receipts=tuple(
            checkpoint.semantic_receipt_sha256 for checkpoint in checkpoints
        ),
        fixed_control_selection_authority_receipt_sha256=(
            fixed_selection.semantic_receipt_sha256
        ),
    )
    failure_path = stages[failure_ordinal].relative_path
    clock = {"value": 1_000}
    failed = {"value": False}

    def now_ms() -> int:
        clock["value"] += 10
        return clock["value"]

    def ensure_transaction(relative: str, at: int):
        path = tmp_path / relative
        if not path.exists():
            body = {
                "semantic_receipt_sha256": _digest(("stage", relative)),
            }
            _publish(
                root=tmp_path,
                relative=relative,
                body=body,
                dataset="executor-v2-interruption-stage",
                schema=_digest("executor-v2-interruption-schema"),
                at=at,
            )
        loaded = executor._load_stage(
            root=tmp_path,
            relative=relative,
            verified_at_ms=max(at, clock["value"]),
        )
        if relative == failure_path and not failed["value"]:
            failed["value"] = True
            raise RuntimeError("injected scientific-stage interruption")
        return loaded

    monkeypatch.setattr(executor, "_wall_clock_ms", now_ms)
    monkeypatch.setattr(executor, "_validate_execution_roots", lambda **_kw: roots)
    monkeypatch.setattr(
        executor, "_fold_validation_execution_lease_v1", lambda **_kw: nullcontext()
    )
    monkeypatch.setattr(
        executor,
        "preserve_massive_adaptive_rl_process_rng_state_v1",
        lambda **_kw: nullcontext(),
    )
    monkeypatch.setattr(
        executor,
        "massive_adaptive_rl_deterministic_execution_v1",
        lambda **_kw: nullcontext(),
    )
    monkeypatch.setattr(
        executor,
        "validate_massive_adaptive_rl_validation_outcome_barrier_v2",
        lambda **_kw: None,
    )

    class Environment:
        semantic_receipt_sha256 = _digest("environment")
        source_receipt_sha256 = _digest("environment-source")
        source_transaction_receipt_sha256 = _digest("environment-commit")
        scientific_execution_fingerprint_sha256 = _digest("fingerprint")
        development_stage_authorized = True

        def __init__(self, committed_at_ms: int) -> None:
            self.source_transaction_committed_at_ms = committed_at_ms

    def environment(**kw):
        relative = validation_execution_environment_relative_path_v1(
            manifest=manifest,
            four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
        )
        loaded = ensure_transaction(relative, kw["committed_at_ms"])
        return Environment(loaded.commit.committed_at_ms)

    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_validation_execution_environment_v1",
        environment,
    )

    def primary(**kw):
        checkpoint = kw["checkpoint"]
        relative = next(
            stage.relative_path
            for stage in stages
            if stage.name == "primary-v1"
            and checkpoint.semantic_receipt_sha256 in stage.relative_path
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    def ladder(**kw):
        checkpoint = kw["checkpoint"]
        relative = next(
            stage.relative_path
            for stage in stages
            if stage.name == "ladder-v1"
            and checkpoint.semantic_receipt_sha256 in stage.relative_path
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    def fixed(**kw):
        relative = next(
            stage.relative_path for stage in stages if stage.name == "fc06-v1"
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    def outcome(**kw):
        relative = next(
            stage.relative_path
            for stage in stages
            if stage.name
            == {
                "ppo-primary": "primary-v2",
                "ppo-cost-ladder": "ladder-v2",
                "fc06-primary": "fc06-v2",
            }[kw["outcome_kind"]]
            and kw["subject_receipt_sha256"] in stage.relative_path
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    def fold_v1(**kw):
        relative = next(
            stage.relative_path
            for stage in stages
            if stage.name == "fold-validation-v1"
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    def fold_v2(**kw):
        relative = next(
            stage.relative_path
            for stage in stages
            if stage.name == "fold-validation-v2"
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return _OutcomeOnly(fold_index=3)

    monkeypatch.setattr(executor, "_primary_v1", primary)
    monkeypatch.setattr(executor, "_ladder_v1", ladder)
    monkeypatch.setattr(executor, "_fixed_v1", fixed)
    monkeypatch.setattr(executor, "_outcome_v2", outcome)
    monkeypatch.setattr(executor, "_fold_v1", fold_v1)
    monkeypatch.setattr(executor, "_fold_v2", fold_v2)

    class Selection:
        development_stage_authorized = True
        semantic_receipt_sha256 = _digest("selection-v3")
        source_receipt_sha256 = _digest("selection-source")
        source_transaction_receipt_sha256 = _digest("selection-commit")

        def __init__(self, committed_at_ms: int) -> None:
            self.source_transaction_committed_at_ms = committed_at_ms

    monkeypatch.setattr(
        executor, "MassiveAdaptiveRLPolicySelectionAuthorityV3", Selection
    )

    def selection(**kw):
        v2_relative = policy_selection_v2_witness_relative_path_v3(
            manifest=manifest, fold_index=3
        )
        v2 = ensure_transaction(v2_relative, kw["committed_at_ms"])
        v3_relative = policy_selection_authority_relative_path_v3(
            manifest=manifest, fold_index=3
        )
        v3_was_present = (tmp_path / v3_relative).exists()
        v3 = ensure_transaction(
            v3_relative,
            max(kw["committed_at_ms"] + 1, v2.commit.committed_at_ms + 1),
        )
        if v3_was_present:
            assert kw["committed_at_ms"] >= v3.commit.committed_at_ms
        return Selection(v3.commit.committed_at_ms)

    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_policy_selection_authority_v3",
        selection,
    )

    class Completion:
        development_stage_authorized = True
        policy_selection_v3 = Selection(0)

    def completion(**kw):
        relative = fold_validation_execution_authority_relative_path_v1(
            manifest=manifest, fold_index=3
        )
        ensure_transaction(relative, kw["committed_at_ms"])
        return Completion()

    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_fold_validation_execution_authority_v1",
        completion,
    )

    with pytest.raises(RuntimeError, match="injected scientific-stage interruption"):
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
            root=tmp_path,
            manifest=manifest,
            runtime_sources_v2=object(),  # type: ignore[arg-type]
            four_fold_fit_authority=object(),  # type: ignore[arg-type]
            four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
            fold_index=3,
        )

    clock["value"] = 100_000
    result = run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
        root=tmp_path,
        manifest=manifest,
        runtime_sources_v2=object(),  # type: ignore[arg-type]
        four_fold_fit_authority=object(),  # type: ignore[arg-type]
        four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
        fold_index=3,
    )
    assert type(result) is Completion
    loaded = tuple(
        executor._load_stage(
            root=tmp_path,
            relative=stage.relative_path,
            verified_at_ms=clock["value"],
        )
        for stage in stages
    )
    times = tuple(stage.commit.committed_at_ms for stage in loaded)
    assert all(right > left for left, right in zip(times, times[1:]))
    completion_loaded = executor._load_stage(
        root=tmp_path,
        relative=fold_validation_execution_authority_relative_path_v1(
            manifest=manifest, fold_index=3
        ),
        verified_at_ms=clock["value"],
    )
    assert completion_loaded.commit.committed_at_ms > times[-1]
    assert completion_loaded.commit.committed_at_ms >= 100_000

    clock["value"] = 200_000
    replayed = run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2(
        root=tmp_path,
        manifest=manifest,
        runtime_sources_v2=object(),  # type: ignore[arg-type]
        four_fold_fit_authority=object(),  # type: ignore[arg-type]
        four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
        fold_index=3,
        allow_materialize=False,
    )
    assert type(replayed) is Completion


def test_new_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V2_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
    assert (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTION_AUTHORITY_V1_SCHEMA
        == "rl-quant.massive-adaptive-rl-fold-validation-execution-authority-v1"
    )
    assert (
        MASSIVE_ADAPTIVE_RL_VALIDATION_EXECUTION_ENVIRONMENT_AUTHORITY_V1_SCHEMA
        == "rl-quant.massive-adaptive-rl-validation-execution-environment-authority-v1"
    )
