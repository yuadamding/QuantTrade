from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import date, timedelta

import pytest

from rl_quant.data_sources.massive.session_calendar import (
    FIVE_MINUTES_NS,
    MassiveExchangeSession,
    build_massive_session_authority,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1Error,
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256,
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.features.massive_adaptive_target_archive_v1 import (
    MassiveAdaptiveTargetArchiveV1Error,
    materialize_massive_adaptive_target_archive_canary_v1,
    parse_massive_adaptive_target_archive_v1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    parse_massive_adaptive_checkpoint_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1,
    MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1Error,
    build_massive_adaptive_split_plan_v1,
)
from rl_quant.training.massive_adaptive_supervised_trainer_v1 import (
    MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
    train_and_publish_massive_adaptive_alpha_canary_v1,
    train_and_publish_massive_adaptive_alpha_v1,
)
from rl_quant.training.massive_adaptive_training_authority_v1 import (
    MassiveAdaptiveTrainingAuthorityV1Error,
)
from tests.test_massive_adaptive_decision_tensor_v1 import (
    _feature,
    _origin,
    _paths,
    _targets,
)


def _sessions():
    source = semantic_sha256("adaptive-session-calendar")
    start = date(2024, 9, 2)
    open_offset = 1_700_000_000_000_000_000
    rows = []
    for index in range(MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1):
        session_date = (start + timedelta(days=index)).isoformat()
        regular_open = open_offset + index * 86_400 * 1_000_000_000
        rows.append(
            MassiveExchangeSession(
                session_date=session_date,
                exchange="XNYS",
                regular_open_ns=regular_open,
                regular_close_ns=regular_open + 78 * FIVE_MINUTES_NS,
                scheduled_five_minute_intervals=78,
                special_session_reason=None,
                calendar_source_receipt_sha256=source,
            )
        )
    return build_massive_session_authority(
        tuple(rows), calendar_source_receipt_sha256=source
    )


def _context(feature, origin) -> MassiveAdaptiveContextOriginAuthorityV1:
    security_ids = tuple(row.security_id for row in feature.rows)
    body = {
        "schema": "rl-quant.massive-adaptive-context-origin-authority-v1",
        "decision_session_date": feature.decision_session_date,
        "decision_at_ms": origin.decision_at_ms,
        "membership_effective_at_ms": origin.membership_effective_at_ms,
        "membership_available_at_ms": origin.membership_available_at_ms,
        "source_session_date": feature.source_session_date,
        "feature_cutoff_at_ms": feature.feature_cutoff_at_ms,
        "feature_input_session_dates": feature.input_session_dates,
        "security_ids": security_ids,
        "universe_ranks": tuple(range(1, len(security_ids) + 1)),
        "decision_clock_receipt_sha256": origin.decision_clock_receipt_sha256,
        "session_authority_receipt_sha256": (
            origin.session_authority_receipt_sha256
        ),
        "identity_authority_receipt_sha256": semantic_sha256(
            (feature.decision_session_date, "context-identity")
        ),
        "context_universe_rule_receipt_sha256": (
            MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.context_universe_rule.receipt_sha256
        ),
        "membership_group_inventory_sha256": semantic_sha256(
            (feature.decision_session_date, "context-membership")
        ),
        "membership_row_inventory_sha256": semantic_sha256(
            (feature.decision_session_date, "context-membership-rows")
        ),
        "feature_semantic_receipt_sha256": feature.semantic_receipt_sha256,
        "feature_row_inventory_sha256": feature.row_inventory_sha256,
        "feature_source_input_inventory_sha256": (
            feature.source_input_inventory_sha256
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SOURCE_SHA256
        ),
        "source_paths_replayed": True,
        # Synthetic fixtures prove wiring only; they may not self-promote.
        "source_data_qualified": False,
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    provisional = MassiveAdaptiveContextOriginAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _source_target(origin) -> MassiveAdaptiveSourceTargetsV1:
    paths = _paths(origin)
    targets = _targets(origin)
    body = {
        "schema": "rl-quant.massive-adaptive-source-targets-v1",
        "decision_session_date": origin.decision_session_date,
        "fill_session_date": origin.decision_session_date,
        "security_ids": origin.security_ids,
        "paths": paths,
        "targets": targets,
        "origin_authority_receipt_sha256": origin.semantic_receipt_sha256,
        "decision_clock_receipt_sha256": origin.decision_clock_receipt_sha256,
        "session_authority_receipt_sha256": (
            origin.session_authority_receipt_sha256
        ),
        "identity_authority_receipt_sha256": semantic_sha256(
            (origin.decision_session_date, "action-identity")
        ),
        "daily_input_authority_receipt_sha256": semantic_sha256(
            (origin.decision_session_date, "daily-input")
        ),
        "fill_source_receipt_sha256": targets.fill_source_receipt_sha256,
        "terminal_authority_receipt_sha256": (
            targets.terminal_authority_receipt_sha256
        ),
        "economic_coverage_receipt_sha256": (
            targets.economic_coverage_receipt_sha256
        ),
        "economic_path_inventory_sha256": semantic_sha256(
            tuple(path.receipt_sha256 for path in paths)
        ),
        "target_receipt_sha256": targets.semantic_receipt_sha256,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_SOURCE_TARGETS_V1_SOURCE_SHA256
        ),
        "source_paths_replayed": True,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    provisional = MassiveAdaptiveSourceTargetsV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _fixture(tmp_path):
    session_authority = _sessions()
    candidate_dates = tuple(row.session_date for row in session_authority.sessions)
    split_plan = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=candidate_dates,
        session_authority=session_authority,
    )
    features = tuple(_feature(index) for index in range(2))
    origins = tuple(
        _origin(
            feature,
            session_authority_receipt_sha256=session_authority.receipt_sha256,
        )
        for feature in features
    )
    contexts = tuple(
        _context(feature, origin)
        for feature, origin in zip(features, origins, strict=True)
    )
    source_targets = tuple(_source_target(origin) for origin in origins)
    committed = materialize_massive_adaptive_decision_tensor_v1(
        root=tmp_path,
        artifact_id="source-training",
        features=features,
        action_origins=origins,
        committed_at_ms=20_000,
    )
    generic = parse_massive_adaptive_decision_tensor_v1(
        root=tmp_path, loaded_source=committed.loaded_source
    )
    return (
        session_authority,
        split_plan,
        features,
        contexts,
        origins,
        source_targets,
        generic,
    )


def test_split_plan_freezes_126_session_purges_and_embargo() -> None:
    session_authority = _sessions()
    candidates = tuple(row.session_date for row in session_authority.sessions)
    plan = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=candidates,
        session_authority=session_authority,
    )

    assert MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1 == 126
    assert len(plan.outer_folds) == 4
    assert not plan.candidate_source_data_qualified
    assert not plan.development_training_authorized
    assert all(len(row.inner_purge_session_dates) == 126 for row in plan.outer_folds)
    assert all(len(row.outer_purge_session_dates) == 126 for row in plan.outer_folds)
    assert len(plan.outer_to_lockbox_embargo_session_dates) == 126
    assert len(plan.lockbox_session_dates) == 252
    with pytest.raises(MassiveAdaptiveSplitPlanV1Error):
        build_massive_adaptive_split_plan_v1(
            candidate_session_dates=candidates[:1000],
            session_authority=session_authority,
        )


def test_decision_root_rejects_cross_clock_substitution(tmp_path) -> None:
    (
        _sessions_root,
        _split,
        features,
        contexts,
        origins,
        _targets_root,
        _tensor,
    ) = _fixture(tmp_path)
    root = build_massive_adaptive_decision_root_v1(
        context_origin=contexts[0],
        action_origin=origins[0],
        features=features[0],
    )
    assert root.action_security_ids == origins[0].security_ids
    assert set(root.action_security_ids) < set(root.context_security_ids)

    changed = replace(
        contexts[0],
        decision_clock_receipt_sha256=semantic_sha256("different-clock"),
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    changed.validate()
    with pytest.raises(MassiveAdaptiveDecisionRootV1Error, match="roots differ"):
        build_massive_adaptive_decision_root_v1(
            context_origin=changed,
            action_origin=origins[0],
            features=features[0],
        )


@pytest.mark.parametrize(
    "receipt_field",
    (
        "identity_authority_receipt_sha256",
        "daily_input_authority_receipt_sha256",
        "fill_source_receipt_sha256",
        "terminal_authority_receipt_sha256",
        "economic_coverage_receipt_sha256",
    ),
)
def test_target_archive_rejects_cross_experiment_substitution(
    tmp_path, receipt_field: str
) -> None:
    (
        _session_authority,
        _split,
        features,
        contexts,
        origins,
        source_targets,
        _tensor,
    ) = _fixture(tmp_path)
    decision_roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=origin,
            features=feature,
        )
        for context, origin, feature in zip(
            contexts, origins, features, strict=True
        )
    )
    archive = materialize_massive_adaptive_target_archive_canary_v1(
        root=tmp_path,
        artifact_id=f"cross-experiment-{receipt_field}",
        decision_roots=decision_roots,
        source_targets=source_targets,
        committed_at_ms=25_000,
    )
    generic = parse_massive_adaptive_target_archive_v1(
        root=tmp_path, loaded_source=archive.loaded_source
    )
    assert generic.runtime_target_roots is None
    assert generic.runtime_source_targets is None
    assert not generic.runtime_roots_replayed
    assert not generic.development_training_authorized

    replacement_receipt = semantic_sha256((receipt_field, "experiment-b"))
    changed_target = source_targets[0].targets
    if receipt_field in {
        "fill_source_receipt_sha256",
        "terminal_authority_receipt_sha256",
        "economic_coverage_receipt_sha256",
    }:
        changed_target = replace(
            changed_target,
            **{receipt_field: replacement_receipt},
            semantic_receipt_sha256="0" * 64,
        )
        changed_target = replace(
            changed_target,
            semantic_receipt_sha256=semantic_sha256(
                changed_target.semantic_unsigned()
            ),
        )
        changed_target.validate()
    changed_source = replace(
        source_targets[0],
        targets=changed_target,
        target_receipt_sha256=changed_target.semantic_receipt_sha256,
        **{receipt_field: replacement_receipt},
        semantic_receipt_sha256="0" * 64,
    )
    changed_source = replace(
        changed_source,
        semantic_receipt_sha256=semantic_sha256(changed_source.semantic_unsigned()),
    )
    changed_source.validate()
    substituted = replace(
        archive,
        runtime_source_targets=(changed_source, source_targets[1]),
    )
    with pytest.raises(
        MassiveAdaptiveTargetArchiveV1Error,
        match="runtime target inventory differs",
    ):
        substituted.validate()


def test_canary_trainer_owns_forward_and_exact_resume(tmp_path) -> None:
    (
        session_authority,
        split_plan,
        features,
        contexts,
        origins,
        source_targets,
        tensor,
    ) = _fixture(tmp_path)
    model_spec = replace(
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
        token_dimension=16,
        fast_window_sessions=2,
        maximum_context_sessions=2,
        maximum_intraday_intervals=4,
        market_latent_count=4,
        attention_heads=4,
        dropout_probability=0.0,
    )
    config = replace(
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1,
        seed=5,
        scheduler_total_updates=8,
    )
    common = {
        "root": tmp_path,
        "decision_tensor": tensor,
        "features": features,
        "context_origins": contexts,
        "action_origins": origins,
        "source_targets": source_targets,
        "session_authority": session_authority,
        "split_plan": split_plan,
        "fold_index": 0,
        "split_role": "training",
        "model_spec": model_spec,
        "config": config,
    }
    uninterrupted = train_and_publish_massive_adaptive_alpha_canary_v1(
        **common,
        artifact_id="uninterrupted",
        updates=2,
        committed_at_ms=30_000,
    )
    first = train_and_publish_massive_adaptive_alpha_canary_v1(
        **common,
        artifact_id="interrupted-one",
        updates=1,
        committed_at_ms=31_000,
    )
    resumed = train_and_publish_massive_adaptive_alpha_canary_v1(
        **common,
        artifact_id="interrupted-two",
        updates=1,
        resume_checkpoint=first,
        committed_at_ms=32_000,
    )

    assert uninterrupted.runtime_checkpoint_replayed
    assert uninterrupted.runtime_state is not None
    assert resumed.runtime_state is not None
    assert not uninterrupted.development_training_authorized
    assert not uninterrupted.profitability_reporting_authorized
    assert not uninterrupted.reinforcement_learning_authorized
    assert uninterrupted.state_receipt_sha256 == resumed.state_receipt_sha256
    assert (
        uninterrupted.model_state_receipt_sha256
        == resumed.model_state_receipt_sha256
    )
    assert (
        uninterrupted.optimizer_state_receipt_sha256
        == resumed.optimizer_state_receipt_sha256
    )
    assert uninterrupted.runtime_state.loss_trace == resumed.runtime_state.loss_trace
    assert uninterrupted.target_archive_receipt_sha256
    assert uninterrupted.target_root_inventory_sha256
    assert uninterrupted.target_experiment_inventory_sha256

    generic = parse_massive_adaptive_checkpoint_v1(
        root=tmp_path, loaded_source=uninterrupted.loaded_source
    )
    assert generic.runtime_state is None
    assert not generic.runtime_checkpoint_replayed
    assert not generic.development_training_authorized

    historical_parameters = inspect.signature(
        train_and_publish_massive_adaptive_alpha_v1
    ).parameters
    assert "archive_freeze" in historical_parameters
    assert "decision_clocks" in historical_parameters
    assert "context_identity_authority" in historical_parameters
    assert "target_archive" in historical_parameters
    assert "target_source_runtimes" in historical_parameters
    assert "context_origins" not in historical_parameters

    bare_target_call = {
        **common,
        "source_targets": tuple(_targets(origin) for origin in origins),
    }
    with pytest.raises(
        MassiveAdaptiveTrainingAuthorityV1Error,
        match="source-target wrappers",
    ):
        train_and_publish_massive_adaptive_alpha_canary_v1(
            **bare_target_call,  # type: ignore[arg-type]
            artifact_id="bare-targets-must-fail",
            updates=1,
            committed_at_ms=33_000,
        )
