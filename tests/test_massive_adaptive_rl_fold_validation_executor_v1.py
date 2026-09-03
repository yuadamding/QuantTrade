from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from io import BytesIO
import inspect
from typing import get_type_hints

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    canonical_json_file_bytes,
    publish_massive_source_object,
)
from rl_quant.evaluation import (
    massive_adaptive_rl_fold_validation_executor_v1 as executor,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_executor_v1 import (
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256,
    MassiveAdaptiveRLFoldValidationExecutionLeaseUnavailable,
    MassiveAdaptiveRLFoldValidationExecutorV1Error,
    _CanonicalValidationStageV1,
    _canonical_validation_stages_v1,
    _execution_anchor_ms,
    _fold_validation_execution_lease_v1,
    run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
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


def _digest(value: object) -> str:
    return semantic_sha256(value)


@dataclass(frozen=True)
class _ReceiptOnly:
    semantic_receipt_sha256: str


@dataclass(frozen=True)
class _FoldOnly:
    fold_index: int = 3


@dataclass(frozen=True)
class _FoldFitStub:
    outer_fold_index: int


@dataclass(frozen=True)
class _SourcesStub:
    base_authority_v1: object
    runtime_chronology_authority: object


@dataclass(frozen=True)
class _RegistryStub:
    base_registry_v1: object


@dataclass(frozen=True)
class _BarrierStub:
    source_transaction_committed_at_ms: int
    base_authority_v1: object
    sources: _SourcesStub
    registry: _RegistryStub

    def validation_sources(self, _fold_index: int) -> _SourcesStub:
        return self.sources

    def validation_registry(self, _fold_index: int) -> _RegistryStub:
        return self.registry


def _publish_stage(*, root, relative: str, committed_at_ms: int) -> None:
    body = {"relative": relative, "committed_at_ms": committed_at_ms}
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(body)),
        root=root,
        relative_payload_path=relative,
        dataset_id="fold-validation-executor-test",
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=_digest("fold-validation-executor-test-schema"),
        entitlement_receipt_sha256=_digest(body),
        committed_at_ms=committed_at_ms,
    )


def test_executor_surface_and_fold3_plan_are_package_owned() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-validation-executor-canary"
    )
    checkpoints = tuple(_digest(("checkpoint", index)) for index in range(4))
    stages = _canonical_validation_stages_v1(
        manifest=manifest,
        fold_index=3,
        checkpoint_authority_receipts=checkpoints,
        fixed_control_selection_authority_receipt_sha256=_digest("fc06"),
    )

    assert len(stages) == 22
    assert tuple(row.commit_offset_ms for row in stages) == tuple(range(22))
    assert len({row.relative_path for row in stages}) == len(stages)
    assert tuple(row.name for row in stages[:4]) == (
        "primary-v1",
        "primary-v2",
        "ladder-v1",
        "ladder-v2",
    )
    assert tuple(row.name for row in stages[-6:]) == (
        "fc06-v1",
        "fc06-v2",
        "fold-validation-v1",
        "fold-validation-v2",
        "selection-v2-computation",
        "selection-v3",
    )
    assert all(manifest.semantic_receipt_sha256 in row.relative_path for row in stages)

    signature = inspect.signature(
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1
    )
    assert tuple(signature.parameters) == (
        "root",
        "manifest",
        "runtime_sources_v2",
        "four_fold_fit_authority",
        "four_fold_validation_inputs_v2",
        "fold_index",
        "committed_at_ms",
        "allow_materialize",
    )
    assert not {
        "environment",
        "actions",
        "targets",
        "metrics",
        "candidates",
        "selected_checkpoint",
        "device",
        "artifact_id",
    }.intersection(signature.parameters)
    hints = get_type_hints(
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1
    )
    assert hints["manifest"] is MassiveAdaptiveRLExperimentManifestV4
    assert hints["runtime_sources_v2"] is MassiveAdaptiveRLRuntimeSourcesV2
    assert hints["four_fold_fit_authority"] is MassiveAdaptiveRLFourFoldFitAuthorityV1
    assert (
        hints["four_fold_validation_inputs_v2"]
        is MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2
    )


def test_executor_requires_exact_v2_roots_before_any_evaluation(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-validation-v1-runtime-rejected"
    )
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationExecutorV1Error,
        match="exact V2 experiment roots",
    ):
        run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1(
            root=tmp_path,
            manifest=manifest,
            runtime_sources_v2=object(),  # type: ignore[arg-type]
            four_fold_fit_authority=object(),  # type: ignore[arg-type]
            four_fold_validation_inputs_v2=object(),  # type: ignore[arg-type]
            fold_index=0,
            committed_at_ms=1,
        )


def test_executor_resume_anchor_is_immutable_and_gaps_fail_closed(tmp_path) -> None:
    stages = tuple(
        _CanonicalValidationStageV1(
            name=f"stage-{index}",
            relative_path=f"executor/stage-{index}.json",
            commit_offset_ms=index,
        )
        for index in range(3)
    )
    assert (
        _execution_anchor_ms(
            root=tmp_path,
            stages=stages,
            requested_first_commit_ms=11,
            barrier_committed_at_ms=10,
        )
        == 11
    )
    _publish_stage(root=tmp_path, relative=stages[0].relative_path, committed_at_ms=11)
    assert (
        _execution_anchor_ms(
            root=tmp_path,
            stages=stages,
            requested_first_commit_ms=99,
            barrier_committed_at_ms=10,
        )
        == 11
    )
    _publish_stage(root=tmp_path, relative=stages[2].relative_path, committed_at_ms=13)
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationExecutorV1Error,
        match="missing upstream stage",
    ):
        _execution_anchor_ms(
            root=tmp_path,
            stages=stages,
            requested_first_commit_ms=99,
            barrier_committed_at_ms=10,
        )


def test_executor_rejects_noncanonical_resume_chronology(tmp_path) -> None:
    stages = tuple(
        _CanonicalValidationStageV1(
            name=f"stage-{index}",
            relative_path=f"executor/stage-{index}.json",
            commit_offset_ms=index,
        )
        for index in range(2)
    )
    _publish_stage(root=tmp_path, relative=stages[0].relative_path, committed_at_ms=11)
    _publish_stage(root=tmp_path, relative=stages[1].relative_path, committed_at_ms=13)
    with pytest.raises(
        MassiveAdaptiveRLFoldValidationExecutorV1Error,
        match="chronology is not canonical",
    ):
        _execution_anchor_ms(
            root=tmp_path,
            stages=stages,
            requested_first_commit_ms=99,
            barrier_committed_at_ms=10,
        )


def test_executor_fold_lease_rejects_a_concurrent_owner(tmp_path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-validation-executor-lease"
    )
    with _fold_validation_execution_lease_v1(
        root=tmp_path,
        manifest=manifest,
        fold_index=3,
    ):
        with pytest.raises(MassiveAdaptiveRLFoldValidationExecutionLeaseUnavailable):
            with _fold_validation_execution_lease_v1(
                root=tmp_path,
                manifest=manifest,
                fold_index=3,
            ):
                pass


def test_executor_orders_all_fold3_computations_and_returns_only_selection_v3(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="fold-validation-executor-order"
    )
    checkpoints = tuple(
        _ReceiptOnly(semantic_receipt_sha256=_digest(("checkpoint", index)))
        for index in range(4)
    )
    fixed_selection = _ReceiptOnly(semantic_receipt_sha256=_digest("fc06"))
    roots = executor._FoldExecutionRootsV1(  # type: ignore[arg-type]
        fold_fit=_FoldFitStub(outer_fold_index=3),
        checkpoints=checkpoints,
        fixed_fit=object(),
        fixed_selection=fixed_selection,
    )
    sources = _SourcesStub(
        base_authority_v1=object(),
        runtime_chronology_authority=object(),
    )
    registry = _RegistryStub(base_registry_v1=object())
    barrier = _BarrierStub(
        source_transaction_committed_at_ms=90,
        base_authority_v1=object(),
        sources=sources,
        registry=registry,
    )
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(executor, "_validate_execution_roots", lambda **_kw: roots)
    monkeypatch.setattr(executor, "_execution_anchor_ms", lambda **_kw: 100)
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
        "validate_massive_adaptive_rl_validation_outcome_barrier_v2",
        lambda **kw: calls.append(("barrier", kw["outcome_committed_at_ms"])),
    )

    def leaf(name):
        def execute(**kw):
            calls.append(
                (
                    name,
                    kw["committed_at_ms"],
                    kw.get("outcome_kind"),
                    kw["allow_materialize"],
                )
            )
            return _FoldOnly()

        return execute

    monkeypatch.setattr(executor, "_primary_v1", leaf("primary-v1"))
    monkeypatch.setattr(executor, "_ladder_v1", leaf("ladder-v1"))
    monkeypatch.setattr(executor, "_fixed_v1", leaf("fc06-v1"))
    monkeypatch.setattr(executor, "_outcome_v2", leaf("outcome-v2"))
    monkeypatch.setattr(executor, "_fold_v1", leaf("fold-v1"))
    monkeypatch.setattr(executor, "_fold_v2", leaf("fold-v2"))

    class SelectionV3:
        source_transaction_committed_at_ms = 121
        development_stage_authorized = True

    monkeypatch.setattr(
        executor, "MassiveAdaptiveRLPolicySelectionAuthorityV3", SelectionV3
    )

    def select(**kw):
        calls.append(("selection-v3", kw["committed_at_ms"], kw["allow_materialize"]))
        return SelectionV3()

    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_policy_selection_authority_v3",
        select,
    )
    result = run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v1(
        root=tmp_path,
        manifest=manifest,
        runtime_sources_v2=object(),  # type: ignore[arg-type]
        four_fold_fit_authority=object(),  # type: ignore[arg-type]
        four_fold_validation_inputs_v2=barrier,  # type: ignore[arg-type]
        fold_index=3,
        committed_at_ms=100,
        allow_materialize=False,
    )

    assert type(result) is SelectionV3
    assert [row[1] for row in calls if row[0] == "barrier"] == [
        100,
        102,
        104,
        106,
        108,
        110,
        112,
        114,
        116,
    ]
    assert [row[:3] for row in calls if row[0] != "barrier"] == [
        ("primary-v1", 100, None),
        ("outcome-v2", 101, "ppo-primary"),
        ("ladder-v1", 102, None),
        ("outcome-v2", 103, "ppo-cost-ladder"),
        ("primary-v1", 104, None),
        ("outcome-v2", 105, "ppo-primary"),
        ("ladder-v1", 106, None),
        ("outcome-v2", 107, "ppo-cost-ladder"),
        ("primary-v1", 108, None),
        ("outcome-v2", 109, "ppo-primary"),
        ("ladder-v1", 110, None),
        ("outcome-v2", 111, "ppo-cost-ladder"),
        ("primary-v1", 112, None),
        ("outcome-v2", 113, "ppo-primary"),
        ("ladder-v1", 114, None),
        ("outcome-v2", 115, "ppo-cost-ladder"),
        ("fc06-v1", 116, None),
        ("outcome-v2", 117, "fc06-primary"),
        ("fold-v1", 118, None),
        ("fold-v2", 119, None),
        ("selection-v3", 120, False),
    ]
    assert all(row[-1] is False for row in calls if len(row) == 4)
    assert not (tmp_path / "massive-adaptive").exists()


def test_executor_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
