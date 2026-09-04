from __future__ import annotations

from datetime import date, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from rl_quant.data_sources.massive.session_calendar import (
    FIVE_MINUTES_NS,
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.evaluation import (
    massive_adaptive_rl_prequential_validation_inputs_v1 as prequential,
)
from rl_quant.evaluation.massive_adaptive_rl_four_fold_validation_inputs_v2 import (
    four_fold_validation_inputs_authority_relative_path_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256,
    MassiveAdaptiveRLPolicyScheduleDispositionV1,
    MassiveAdaptiveRLPrequentialValidationInputsV1Error,
    _forbidden_prequential_artifacts,
    build_massive_adaptive_rl_initial_validation_inputs_authority_v1,
    build_massive_adaptive_rl_prequential_validation_plan_v1,
    load_massive_adaptive_rl_initial_validation_inputs_authority_v1,
    materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1,
    policy_schedule_disposition_v1,
    run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_inputs_v2 import (
    MassiveAdaptiveRLValidationEnvironmentRegistryV2,
    MassiveAdaptiveRLValidationSourcesAuthorityV2,
    validation_sources_authority_relative_path_v2,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1,
    build_massive_adaptive_split_plan_v1,
)
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    build_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v2 import (
    MassiveAdaptiveRLRuntimeSourcesV2,
)


_T = TypeVar("_T")


def _typed_shell(authority_type: type[_T], /, **values: object) -> _T:
    result = object.__new__(authority_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _split_plan():
    source_receipt = semantic_sha256("prequential-test-calendar")
    first = date(2015, 1, 1)
    dates = tuple(
        (first + timedelta(days=index)).isoformat()
        for index in range(MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1)
    )
    sessions = build_massive_session_authority(
        tuple(
            MassiveExchangeSession(
                session_date=session_date,
                exchange="XNYS",
                regular_open_ns=index * 100 * FIVE_MINUTES_NS,
                regular_close_ns=(index * 100 + 72) * FIVE_MINUTES_NS,
                scheduled_five_minute_intervals=72,
                special_session_reason=None,
                calendar_source_receipt_sha256=source_receipt,
            )
            for index, session_date in enumerate(dates)
        ),
        calendar_source_receipt_sha256=source_receipt,
    )
    return build_massive_adaptive_split_plan_v1(
        candidate_session_dates=dates,
        session_authority=sessions,
    )


def test_registered_geometry_requires_prequential_release() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="prequential-plan"
    )
    plan = build_massive_adaptive_rl_prequential_validation_plan_v1(
        manifest=manifest,
        split_plan=_split_plan(),
    )
    assert plan.validation_session_date_inventories[2] == (
        plan.outer_session_date_inventories[0]
    )
    assert plan.validation_session_date_inventories[3] == (
        plan.outer_session_date_inventories[1]
    )
    assert plan.released_validation_folds(sealed_outer_fold_indices=()) == (0, 1)
    assert plan.released_validation_folds(sealed_outer_fold_indices=(0,)) == (
        0,
        1,
        2,
    )
    assert plan.released_validation_folds(sealed_outer_fold_indices=(0, 1)) == (
        0,
        1,
        2,
        3,
    )
    with pytest.raises(
        MassiveAdaptiveRLPrequentialValidationInputsV1Error,
        match="canonical prefix",
    ):
        plan.released_validation_folds(sealed_outer_fold_indices=(1,))


@pytest.mark.parametrize(
    ("eligibility", "expected"),
    (
        (
            (True, True, True, True),
            MassiveAdaptiveRLPolicyScheduleDispositionV1.FOUR_FOLD_POLICY_SCHEDULE_QUALIFIED,
        ),
        (
            (True, True, True, False),
            MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE,
        ),
        (
            (False, False, False, False),
            MassiveAdaptiveRLPolicyScheduleDispositionV1.DIAGNOSTIC_ONLY_POLICY_SCHEDULE,
        ),
    ),
)
def test_manifest_v4_ineligible_schedule_continues_diagnostically(
    eligibility: tuple[bool, ...],
    expected: MassiveAdaptiveRLPolicyScheduleDispositionV1,
) -> None:
    assert policy_schedule_disposition_v1(eligibility) is expected
    assert (
        MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        == "select-deterministic-top-ranked-continue-sealed-outer-diagnostic-positive-authorization-prohibited-v1"
    )


@pytest.mark.parametrize("eligibility", ((True, True, True), (True, True, True, 1)))
def test_policy_schedule_rejects_incomplete_or_nonboolean_evidence(
    eligibility: tuple[object, ...],
) -> None:
    with pytest.raises(
        MassiveAdaptiveRLPrequentialValidationInputsV1Error,
        match="eligibility inventory",
    ):
        policy_schedule_disposition_v1(eligibility)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ("fold-two-source", "legacy-all-four-barrier"))
def test_initial_boundary_rejects_early_or_legacy_artifacts(
    tmp_path: Path, kind: str
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id=f"prequential-forbidden-{kind}"
    )
    if kind == "fold-two-source":
        relative = validation_sources_authority_relative_path_v2(
            manifest=manifest, fold_index=2
        )
    else:
        relative = four_fold_validation_inputs_authority_relative_path_v2(
            manifest=manifest
        )
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    assert relative in _forbidden_prequential_artifacts(
        root=tmp_path, manifest=manifest
    )


def test_initial_input_runner_materializes_only_folds_zero_and_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="prequential-initial-runner"
    )
    runtime = _typed_shell(MassiveAdaptiveRLRuntimeSourcesV2)
    fit = _typed_shell(MassiveAdaptiveRLFourFoldFitAuthorityV1)
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        prequential,
        "validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility",
        lambda **_: None,
    )
    monkeypatch.setattr(prequential, "_wall_clock_ms", lambda: 1_000)
    observed: list[tuple[str, int]] = []

    def prepare_source(**kwargs):
        fold_index = kwargs["fold_index"]
        observed.append(("source", fold_index))
        return SimpleNamespace(
            fold_index=fold_index,
            source_transaction_committed_at_ms=kwargs["committed_at_ms"],
        )

    def prepare_registry(**kwargs):
        fold_index = kwargs["validation_sources_v2"].fold_index
        observed.append(("registry", fold_index))
        return SimpleNamespace(
            fold_index=fold_index,
            source_transaction_committed_at_ms=kwargs["committed_at_ms"],
        )

    marker = object()
    monkeypatch.setattr(
        prequential,
        "prepare_or_resume_massive_adaptive_rl_validation_sources_v2",
        prepare_source,
    )
    monkeypatch.setattr(
        prequential,
        "prepare_or_resume_massive_adaptive_rl_validation_environment_registry_v2",
        prepare_registry,
    )
    monkeypatch.setattr(
        prequential,
        "build_massive_adaptive_rl_initial_validation_inputs_authority_v1",
        lambda **_: SimpleNamespace(
            validation_registry_v2_committed_at_ms=(1_002, 1_003)
        ),
    )
    monkeypatch.setattr(
        prequential,
        "materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1",
        lambda **_: marker,
    )
    assert (
        run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1(
            root=tmp_path,
            manifest=manifest,
            runtime_sources_v2=runtime,
            four_fold_fit_authority=fit,
        )
        is marker
    )
    assert observed == [
        ("source", 0),
        ("source", 1),
        ("registry", 0),
        ("registry", 1),
    ]


def test_initial_authority_is_create_only_and_generic_reload_is_nonauthorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="prequential-initial-authority"
    )
    split_plan = _split_plan()
    runtime = _typed_shell(
        MassiveAdaptiveRLRuntimeSourcesV2,
        _base_runtime_sources_v1=SimpleNamespace(split_plan=split_plan),
        semantic_receipt_sha256=semantic_sha256("runtime-v2"),
        source_bundle_v2_receipt_sha256=semantic_sha256("bundle-v2"),
        runtime_source_graph_v2_receipt_sha256=semantic_sha256("graph-v2"),
        runtime_source_graph_v2_witness_receipt_sha256=semantic_sha256(
            "graph-witness-v2"
        ),
        replay_dependency_index_v2_receipt_sha256=semantic_sha256("index-v2"),
        training_source_projection_sha256=semantic_sha256("training-projection"),
        validation_source_projection_sha256=semantic_sha256(
            "validation-projection"
        ),
        source_data_qualified=True,
    )
    fit = _typed_shell(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        semantic_receipt_sha256=semantic_sha256("fit"),
        source_data_qualified=True,
    )
    sources = tuple(
        _typed_shell(
            MassiveAdaptiveRLValidationSourcesAuthorityV2,
            fold_index=fold_index,
            semantic_receipt_sha256=semantic_sha256(("source", fold_index)),
            validation_decision_session_dates=(
                split_plan.outer_folds[fold_index].inner_validation_session_dates
            ),
            source_data_qualified=True,
        )
        for fold_index in (0, 1)
    )
    registries = tuple(
        _typed_shell(
            MassiveAdaptiveRLValidationEnvironmentRegistryV2,
            fold_index=fold_index,
            semantic_receipt_sha256=semantic_sha256(("registry", fold_index)),
            validation_sources_v2_receipt_sha256=(
                sources[fold_index].semantic_receipt_sha256
            ),
            validation_context_receipt_sha256=semantic_sha256(
                ("context", fold_index)
            ),
            source_data_qualified=True,
        )
        for fold_index in (0, 1)
    )
    monkeypatch.setattr(MassiveAdaptiveRLRuntimeSourcesV2, "validate", lambda _: None)
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLFourFoldFitAuthorityV1,
        "fold_fit",
        lambda _, fold_index: SimpleNamespace(
            candidate_checkpoint_authority_receipts=tuple(
                semantic_sha256(("checkpoint", fold_index, candidate_index))
                for candidate_index in range(fold_index + 1)
            )
        ),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationSourcesAuthorityV2, "validate", lambda _: None
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        "validate",
        lambda _: None,
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationSourcesAuthorityV2,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        "development_stage_authorized",
        property(lambda _: True),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationSourcesAuthorityV2,
        "source_receipt_sha256",
        property(lambda row: semantic_sha256(("source-object", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationSourcesAuthorityV2,
        "source_transaction_receipt_sha256",
        property(lambda row: semantic_sha256(("source-commit", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationSourcesAuthorityV2,
        "source_transaction_committed_at_ms",
        property(lambda _: 10),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        "source_receipt_sha256",
        property(lambda row: semantic_sha256(("registry-object", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        "source_transaction_receipt_sha256",
        property(lambda row: semantic_sha256(("registry-commit", row.fold_index))),
    )
    monkeypatch.setattr(
        MassiveAdaptiveRLValidationEnvironmentRegistryV2,
        "source_transaction_committed_at_ms",
        property(lambda _: 20),
    )
    monkeypatch.setattr(
        prequential,
        "validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility",
        lambda **_: None,
    )
    authority = build_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        manifest=manifest,
        runtime_sources_v2=runtime,
        four_fold_fit_authority=fit,
        validation_sources_v2=sources,
        validation_environment_registries_v2=registries,
    )
    committed = materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        root=tmp_path,
        manifest=manifest,
        authority=authority,
        committed_at_ms=30,
    )
    assert committed.development_stage_authorized
    assert (
        committed.prequential_validation_plan.semantic_receipt_sha256
        == authority.prequential_validation_plan_receipt_sha256
    )
    generic = load_massive_adaptive_rl_initial_validation_inputs_authority_v1(
        root=tmp_path,
        manifest=manifest,
        verified_at_ms=31,
    )
    assert generic.source_transaction_verified
    assert not generic.runtime_inputs_replayed
    assert not generic.development_stage_authorized
    with pytest.raises(
        MassiveAdaptiveRLPrequentialValidationInputsV1Error,
        match="has not been exactly replayed",
    ):
        _ = generic.prequential_validation_plan
    with pytest.raises(
        MassiveAdaptiveRLPrequentialValidationInputsV1Error,
        match="already exists",
    ):
        materialize_massive_adaptive_rl_initial_validation_inputs_authority_v1(
            root=tmp_path,
            manifest=manifest,
            authority=authority,
            committed_at_ms=31,
        )


def test_initial_input_runner_has_no_release_or_outcome_choice_surface() -> None:
    parameters = inspect.signature(
        run_or_resume_massive_adaptive_rl_initial_validation_inputs_v1
    ).parameters
    assert tuple(parameters) == (
        "root",
        "manifest",
        "runtime_sources_v2",
        "four_fold_fit_authority",
        "allow_materialize",
    )
    assert not {
        "fold_index",
        "sealed_outer_fold_indices",
        "outer_evidence",
        "validation_environment",
        "actions",
        "targets",
        "metrics",
    }.intersection(parameters)


def test_prequential_protocol_hashes_are_bound() -> None:
    for value in (
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_PLAN_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_INITIAL_VALIDATION_INPUTS_V1_SPEC_SHA256,
        MASSIVE_ADAPTIVE_RL_PREQUENTIAL_VALIDATION_INPUTS_V1_SOURCE_SHA256,
    ):
        assert len(value) == 64
        int(value, 16)
