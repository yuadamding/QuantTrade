from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
from typing import TypeVar

import pytest

from rl_quant.evaluation import (
    massive_adaptive_rl_four_fold_validation_executor_v1 as executor,
)
from rl_quant.evaluation.massive_adaptive_rl_fold_validation_execution_authority_v1 import (
    MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_executor_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256,
    _four_fold_validation_execution_lease_v1,
    run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training import (
    massive_adaptive_rl_four_fold_policy_selection_v1 as selection,
)
from rl_quant.training.massive_adaptive_rl_four_fold_policy_selection_v1 import (
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLFourFoldPolicySelectionV1Error,
    MassiveAdaptiveRLFourFoldSelectionDispositionV1,
    authorize_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
    build_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
    load_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
    materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v3 import (
    MassiveAdaptiveRLPolicySelectionAuthorityV3,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
)


_T = TypeVar("_T")


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _roots(
    monkeypatch: pytest.MonkeyPatch,
    *,
    eligible: tuple[bool, bool, bool, bool] = (True, True, True, True),
):
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=f"four-fold-selection-{''.join(map(str, map(int, eligible)))}"
    )
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV3, "validate", lambda _: None
    )
    monkeypatch.setattr(
        selection,
        "validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility",
        lambda **_: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLRuntimeSourcesV2,
        "source_data_qualified",
        True,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        "source_receipt_sha256",
        property(lambda _: _digest("inputs-source")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        "source_transaction_receipt_sha256",
        property(lambda _: _digest("inputs-commit")),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        "source_transaction_committed_at_ms",
        property(lambda _: 100),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
        "source_receipt_sha256",
        property(lambda row: _digest(("execution-source", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
        "source_transaction_receipt_sha256",
        property(lambda row: _digest(("execution-commit", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
        "source_transaction_committed_at_ms",
        property(lambda row: 200 + row.fold_index),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLPolicySelectionAuthorityV3,
        "development_stage_authorized",
        property(lambda _: True),
    )

    runtime = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV2,
        experiment_id=manifest.experiment_id,
        semantic_receipt_sha256=_digest("runtime-v2"),
        source_bundle_v2_receipt_sha256=_digest("bundle-v2"),
        runtime_source_graph_v2_receipt_sha256=_digest("graph-v2"),
        runtime_source_graph_v2_witness_receipt_sha256=_digest("graph-witness-v2"),
        replay_dependency_index_v2_receipt_sha256=_digest("index-v2"),
        training_source_projection_sha256=_digest("training-projection"),
        validation_source_projection_sha256=_digest("validation-projection"),
    )
    fit = _typed_shell(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        experiment_id=manifest.experiment_id,
        manifest_v3_receipt_sha256=manifest.base_manifest.semantic_receipt_sha256,
        semantic_receipt_sha256=_digest("four-fold-fit"),
        source_data_qualified=True,
    )
    expected_checkpoints = tuple(
        tuple(_digest(("checkpoint", fold, item)) for item in range(fold + 1))
        for fold in range(4)
    )
    inputs = _typed_shell(
        MassiveAdaptiveRLFourFoldValidationInputsAuthorityV2,
        semantic_receipt_sha256=_digest("inputs-v2"),
        expected_candidate_checkpoint_authority_receipt_inventories=(
            expected_checkpoints
        ),
        source_data_qualified=True,
    )
    executions = []
    for fold_index in range(4):
        is_eligible = eligible[fold_index]
        selection_v3 = _typed_shell(
            MassiveAdaptiveRLPolicySelectionAuthorityV3,
            fold_index=fold_index,
            semantic_receipt_sha256=_digest(("selection", fold_index)),
            expected_candidate_checkpoint_authority_receipts=(
                expected_checkpoints[fold_index]
            ),
            selected_checkpoint_authority_receipt_sha256=(
                expected_checkpoints[fold_index][-1]
            ),
            selected_checkpoint_receipt_sha256=_digest(
                ("selected-checkpoint", fold_index)
            ),
            selected_model_state_receipt_sha256=_digest(("selected-model", fold_index)),
            selected_update_index=fold_index + 1,
            selected_candidate_validation_eligible=is_eligible,
            validation_eligibility_failures=(
                ()
                if is_eligible
                else ("primary-incremental-rl-log-wealth-strictly-positive",)
            ),
            source_data_qualified=True,
        )
        execution = _typed_shell(
            MassiveAdaptiveRLFoldValidationExecutionAuthorityV1,
            experiment_id=manifest.experiment_id,
            manifest_v4_receipt_sha256=manifest.semantic_receipt_sha256,
            training_manifest_v3_receipt_sha256=(
                manifest.base_manifest.semantic_receipt_sha256
            ),
            fold_index=fold_index,
            runtime_sources_v2_receipt_sha256=runtime.semantic_receipt_sha256,
            four_fold_fit_authority_receipt_sha256=fit.semantic_receipt_sha256,
            four_fold_validation_inputs_v2_receipt_sha256=(
                inputs.semantic_receipt_sha256
            ),
            checkpoint_authority_receipts=expected_checkpoints[fold_index],
            validation_execution_environment_receipt_sha256=_digest("environment"),
            validation_execution_environment_source_receipt_sha256=_digest(
                "environment-source"
            ),
            validation_execution_environment_commit_receipt_sha256=_digest(
                "environment-commit"
            ),
            scientific_execution_fingerprint_sha256=_digest("scientific-environment"),
            policy_selection_v3_source_receipt_sha256=_digest(
                ("selection-source", fold_index)
            ),
            policy_selection_v3_commit_receipt_sha256=_digest(
                ("selection-commit", fold_index)
            ),
            source_data_qualified=True,
            semantic_receipt_sha256=_digest(("execution", fold_index)),
            _runtime_selection_v3=selection_v3,
        )
        executions.append(execution)
    return manifest, runtime, fit, inputs, tuple(executions)


@pytest.mark.parametrize(
    ("eligible", "expected_disposition", "positive"),
    (
        (
            (True, True, True, True),
            MassiveAdaptiveRLFourFoldSelectionDispositionV1.FOUR_FOLD_SELECTIONS_QUALIFIED,
            True,
        ),
        (
            (True, False, True, True),
            MassiveAdaptiveRLFourFoldSelectionDispositionV1.NO_QUALIFIED_POLICY,
            False,
        ),
    ),
)
def test_four_fold_selection_is_create_only_runtime_derived_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible: tuple[bool, bool, bool, bool],
    expected_disposition: MassiveAdaptiveRLFourFoldSelectionDispositionV1,
    positive: bool,
) -> None:
    manifest, runtime, fit, inputs, executions = _roots(monkeypatch, eligible=eligible)
    active = build_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        manifest=manifest,
        runtime_sources_v2=runtime,
        four_fold_fit_authority=fit,
        four_fold_validation_inputs_v2=inputs,
        fold_executions=executions,
    )
    assert active.selection_disposition == expected_disposition.value
    assert not active.development_stage_authorized
    assert not active.positive_profitability_authorization_eligible
    assert not active.final_policy_freezing_authorized
    assert not active.outer_access_authorized

    generic = materialize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        root=tmp_path,
        manifest=manifest,
        authority=active,
        committed_at_ms=300,
    )
    assert generic.source_transaction_verified
    assert not generic.development_stage_authorized
    assert not generic.positive_profitability_authorization_eligible
    assert (
        load_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
            root=tmp_path, manifest=manifest, verified_at_ms=301
        ).semantic_receipt_sha256
        == generic.semantic_receipt_sha256
    )

    replayed = authorize_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
        authority=generic,
        manifest=manifest,
        runtime_sources_v2=runtime,
        four_fold_fit_authority=fit,
        four_fold_validation_inputs_v2=inputs,
        fold_executions=executions,
    )
    assert replayed.development_stage_authorized
    assert replayed.positive_profitability_authorization_eligible is positive
    assert replayed.all_selected_policies_validation_eligible is positive
    assert not replayed.final_policy_freezing_authorized
    assert not replayed.outer_access_authorized


def test_four_fold_selection_rejects_mixed_execution_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, runtime, fit, inputs, executions = _roots(monkeypatch)
    object.__setattr__(
        executions[-1],
        "validation_execution_environment_receipt_sha256",
        _digest("different-environment"),
    )
    with pytest.raises(
        MassiveAdaptiveRLFourFoldPolicySelectionV1Error,
        match="mixed or incomplete",
    ):
        build_massive_adaptive_rl_four_fold_policy_selection_authority_v1(
            manifest=manifest,
            runtime_sources_v2=runtime,
            four_fold_fit_authority=fit,
            four_fold_validation_inputs_v2=inputs,
            fold_executions=executions,
        )


def test_four_fold_runner_owns_inputs_fold_order_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="four-fold-runner"
    )
    runtime = _typed_shell(MassiveAdaptiveRLRuntimeSourcesV2)
    fit = _typed_shell(MassiveAdaptiveRLFourFoldFitAuthorityV1)

    class Inputs:
        development_stage_authorized = True

    inputs = Inputs()
    calls: list[object] = []
    monkeypatch.setattr(executor, "_validate_roots", lambda **_: None)
    monkeypatch.setattr(
        executor,
        "_four_fold_validation_execution_lease_v1",
        lambda **_: nullcontext(),
    )
    monkeypatch.setattr(executor, "_wall_clock_ms", lambda: 1_000)

    def validation_inputs(**kwargs):
        calls.append(("inputs", kwargs))
        return inputs

    class FoldResult:
        development_stage_authorized = True

        def __init__(self, fold_index: int) -> None:
            self.fold_index = fold_index
            self.source_transaction_committed_at_ms = 1_100 + fold_index

    def fold(**kwargs):
        calls.append(("fold", kwargs["fold_index"]))
        return FoldResult(kwargs["fold_index"])

    result = object()

    def aggregate(**kwargs):
        calls.append(("aggregate", kwargs))
        return result

    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_four_fold_validation_inputs_v2",
        validation_inputs,
    )
    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_fold_validation_and_selection_v2",
        fold,
    )
    monkeypatch.setattr(
        executor,
        "run_or_resume_massive_adaptive_rl_four_fold_policy_selection_authority_v1",
        aggregate,
    )
    observed = run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1(
        root=tmp_path,
        manifest=manifest,
        runtime_sources_v2=runtime,
        four_fold_fit_authority=fit,
    )
    assert observed is result
    assert [row for row in calls if row[0] == "fold"] == [
        ("fold", 0),
        ("fold", 1),
        ("fold", 2),
        ("fold", 3),
    ]
    aggregate_call = calls[-1][1]
    assert tuple(row.fold_index for row in aggregate_call["fold_executions"]) == (
        0,
        1,
        2,
        3,
    )
    assert aggregate_call["committed_at_ms"] == 1_104


def test_four_fold_runner_api_has_no_outcome_choice_surface() -> None:
    parameters = inspect.signature(
        run_or_resume_massive_adaptive_rl_four_fold_validation_and_selection_v1
    ).parameters
    assert tuple(parameters) == (
        "root",
        "manifest",
        "runtime_sources_v2",
        "four_fold_fit_authority",
        "allow_materialize",
    )
    forbidden = {
        "fold_index",
        "four_fold_validation_inputs_v2",
        "environment",
        "device",
        "actions",
        "targets",
        "metrics",
        "candidates",
        "selected_checkpoint",
        "committed_at_ms",
        "artifact_id",
    }
    assert not forbidden.intersection(parameters)


def test_four_fold_lease_does_not_mask_body_oserror(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="four-fold-lease"
    )
    with pytest.raises(OSError, match="scientific body failed"):
        with _four_fold_validation_execution_lease_v1(root=tmp_path, manifest=manifest):
            raise OSError("scientific body failed")


def test_four_fold_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_VALIDATION_EXECUTOR_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SOURCE_SHA256,
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_V1_SPEC_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
    assert (
        MASSIVE_ADAPTIVE_RL_FOUR_FOLD_POLICY_SELECTION_AUTHORITY_V1_SCHEMA
        == "rl-quant.massive-adaptive-rl-four-fold-policy-selection-authority-v1"
    )
