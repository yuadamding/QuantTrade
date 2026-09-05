from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    MassiveAdaptiveRLProfitabilityReportAuthorityV1Error,
    authorize_massive_adaptive_rl_profitability_report_authority_v1,
    materialize_massive_adaptive_rl_profitability_report_authority_v1,
    parse_massive_adaptive_rl_profitability_report_authority_v1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    run_massive_adaptive_rl_experiment_v2,
    verify_massive_adaptive_rl_experiment_v2,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2,
    MassiveAdaptiveRLExperimentStageV2,
    MassiveAdaptiveRLExperimentStateV2Error,
    advance_massive_adaptive_rl_experiment_state_v2,
    block_massive_adaptive_rl_experiment_state_v2,
    fail_massive_adaptive_rl_experiment_state_v2,
    load_massive_adaptive_rl_experiment_states_v2,
    publish_massive_adaptive_rl_development_report_state_v3,
    register_massive_adaptive_rl_experiment_state_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows import (
    massive_adaptive_rl_runtime_source_graph_authority_v1 as graph_module,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    authorize_massive_adaptive_rl_source_bundle_v1,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
    materialize_massive_adaptive_rl_source_bundle_v1,
)


_GLOBAL_SOURCE_PATHS = {
    "session-authority": "authorities/session-authority.json",
    "condition-authority": "authorities/condition-authority.json",
    "persisted-partition-inventory": "authorities/persisted-partition-inventory.json",
    "identity-authority": "authorities/identity-authority.json",
    "economic-event-archive": "authorities/economic-event-archive.json",
    "daily-input-authority": "authorities/daily-input-authority.json",
    "fill-source-authority": "authorities/fill-source-authority.json",
    "split-plan": "authorities/adaptive-split-plan.json",
    "development-origin-feature-inventory": (
        "authorities/development-origin-feature-inventory.json"
    ),
    "development-origin-action-inventory": (
        "authorities/development-origin-action-inventory.json"
    ),
}
_FOLD_SOURCE_PATHS = {
    "training-window-inventory": "training-window-inventory.json",
    "supervised-checkpoint-inventory": "supervised-checkpoint-inventory.json",
    "calibration-inventory": "calibration-inventory.json",
    "fit-forecast-archive-inventory": "fit-forecast-archive-inventory.json",
    "decision-root-inventory": "decision-root-inventory.json",
    "context-origin-inventory": "context-origin-inventory.json",
    "validation-origin-feature-inventory": ("validation-origin-feature-inventory.json"),
    "validation-origin-action-inventory": ("validation-origin-action-inventory.json"),
}


@dataclass(frozen=True)
class _SyntheticRuntimeSource:
    semantic_receipt_sha256: str
    source_transport_qualified: bool = True
    daily_input_data_qualified: bool = True
    source_data_qualified: bool = True
    source_paths_replayed: bool = True
    candidate_source_data_qualified: bool = True
    source_geometry_replayed: bool = True

    def validate(self) -> None:
        assert len(self.semantic_receipt_sha256) == 64


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _authorized_source_bundle(root, manifest, monkeypatch):
    root.mkdir()
    runtime_sources = {}
    relative_paths = {
        (role, None): relative_path
        for role, relative_path in _GLOBAL_SOURCE_PATHS.items()
    }
    for fold_index in range(4):
        relative_paths.update(
            {
                (role, fold_index): f"folds/fold-{fold_index}/{name}"
                for role, name in _FOLD_SOURCE_PATHS.items()
            }
        )
    for key, relative_path in relative_paths.items():
        receipt = _digest({"role": key[0], "fold_index": key[1]})
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_file_bytes({"semantic_receipt_sha256": receipt})
        )
        runtime_sources[key] = bind_massive_adaptive_rl_source_authority_v1(
            role=key[0],
            fold_index=key[1],
            authority=_SyntheticRuntimeSource(receipt),
            source_data_qualified=True,
            runtime_source_replayed=True,
        )
    materialize_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
        runtime_sources=runtime_sources,
    )
    generic = load_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
    )
    source_bundle = authorize_massive_adaptive_rl_source_bundle_v1(
        source_bundle=generic,
        runtime_sources=runtime_sources,
    )
    monkeypatch.setattr(
        graph_module,
        "_DOMAIN_RUNTIME_TYPES",
        {role: _SyntheticRuntimeSource for role in graph_module._DOMAIN_RUNTIME_TYPES},
    )
    monkeypatch.setattr(
        graph_module,
        "_DIRECT_DOMAIN_SPECIFICATIONS",
        {role: None for role in graph_module._DIRECT_DOMAIN_SPECIFICATIONS},
    )
    monkeypatch.setattr(
        graph_module,
        "_validate_runtime_graph_contract",
        lambda **_kwargs: (
            (("synthetic-runtime-coverage",),),
            (("synthetic-runtime-edge", "test", "test"),),
        ),
    )
    materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=root,
        manifest=manifest,
        source_bundle=source_bundle,
        runtime_sources=runtime_sources,
    )
    generic_graph = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=root,
        manifest=manifest,
        source_bundle=generic,
    )
    runtime_graph = authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
        authority=generic_graph,
        source_bundle=generic,
        runtime_sources=runtime_sources,
    )
    return source_bundle, runtime_graph


def _report_inputs(
    *,
    daily_mean: float,
    cost_ladder_monotone: bool = True,
) -> tuple[object, tuple[object, ...]]:
    folds = []
    authorities = []
    for fold_index in range(4):
        rows = tuple(
            daily_mean + (0.0001 if index % 2 == 0 else -0.0001) for index in range(126)
        )
        dates = tuple(f"F{fold_index}-{index:03d}" for index in range(126))
        terminal_return = __import__("math").expm1(sum(rows))
        rollout_authority_receipt = _digest(("rollout-authority", fold_index))
        rollout_receipt = _digest(("rollout", fold_index))
        trace_receipt = _digest(("trace", fold_index))
        transitions = tuple(
            SimpleNamespace(
                validate=lambda: None,
                economic_step=SimpleNamespace(
                    strategy_net_log_return=value,
                    strategy_posttrade_book=SimpleNamespace(
                        marked_equity=10_000_000.0 * __import__("math").exp(sum(rows))
                    ),
                ),
                strategy_liquidation_adjusted_equity=(
                    10_000_000.0 * __import__("math").exp(sum(rows))
                ),
                terminated=index == 125,
                truncated=False,
            )
            for index, value in enumerate(rows)
        )
        rollout = SimpleNamespace(
            fold_index=fold_index,
            semantic_receipt_sha256=rollout_receipt,
            policy_trace=SimpleNamespace(
                semantic_receipt_sha256=trace_receipt,
                decision_session_dates=dates,
                terminal_liquidation_adjusted_return=terminal_return,
            ),
            transitions=transitions,
            transition_inventory_sha256=_digest(("transition-inventory", fold_index)),
            source_data_qualified=True,
        )
        authority = SimpleNamespace(
            validate=lambda: None,
            fold_index=fold_index,
            semantic_receipt_sha256=rollout_authority_receipt,
            runtime_rollout=rollout,
            runtime_rollout_replayed=True,
            outer_evaluation_authorized=True,
            source_data_qualified=True,
        )
        authorities.append(authority)
        cost_fold = SimpleNamespace(
            primary_trace_receipt_sha256=trace_receipt,
            primary_strategy_active_log_returns=(0.0004,) * 126,
            primary_incremental_rl_log_returns=(0.0003,) * 126,
            primary_ppo_minus_fixed_control_log_returns=(0.0002,) * 126,
            maximum_drawdown=0.10,
        )
        authenticated_v2 = SimpleNamespace(
            cost_fold=cost_fold,
            outer_rollout_authority_receipt_sha256=rollout_authority_receipt,
            outer_rollout_receipt_sha256=rollout_receipt,
        )
        folds.append(
            SimpleNamespace(
                fold_index=fold_index,
                authenticated_fold_v3=SimpleNamespace(
                    authenticated_fold_v2=authenticated_v2
                ),
                source_data_qualified=True,
            )
        )
    evidence_v1 = SimpleNamespace(mean_high_cost_terminal_return=0.01)
    evidence = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("outer-evidence-v4"),
        authenticated_folds=tuple(folds),
        evidence_v3=SimpleNamespace(
            evidence_v2=SimpleNamespace(evidence_v1=evidence_v1)
        ),
        mean_high_cost_ppo_minus_fixed_control_log_return=0.0001,
        passed_gate_names=("cost-ladder-monotone",) if cost_ladder_monotone else (),
        failed_gate_names=() if cost_ladder_monotone else ("cost-ladder-monotone",),
        source_data_qualified=True,
    )
    outer_authority = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("outer-evidence-authority-v4"),
        runtime_evidence=evidence,
        runtime_folds=tuple(folds),
        runtime_evidence_replayed=True,
        source_data_qualified=True,
        outer_development_conclusion_authorized=cost_ladder_monotone,
    )
    return outer_authority, tuple(authorities)


def test_profitability_report_is_create_only_and_replay_authorized(tmp_path) -> None:
    outer_authority, rollout_authorities = _report_inputs(daily_mean=0.001)
    authority = materialize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        artifact_id="positive-development-report",
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
        committed_at_ms=1,
    )
    assert authority.runtime_report_replayed
    assert authority.development_profitability_reporting_authorized
    assert authority.report.primary_net_log_return_lcb95 > 0.0
    assert authority.report.net_sharpe_ratio > 0.0
    assert not authority.live_trading_authorized
    assert not authority.lockbox_access_authorized

    generic = parse_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    assert generic.runtime_report is None
    assert not generic.runtime_report_replayed
    assert not generic.development_profitability_reporting_authorized
    replayed = authorize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        authority=generic,
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
    )
    assert replayed.semantic_receipt_sha256 == authority.semantic_receipt_sha256
    assert replayed.development_profitability_reporting_authorized


def test_absolute_loss_remains_diagnostic_and_blocks_reporting(tmp_path) -> None:
    outer_authority, rollout_authorities = _report_inputs(daily_mean=-0.001)
    authority = materialize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        artifact_id="negative-development-report",
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
        committed_at_ms=1,
    )
    assert authority.runtime_report_replayed
    assert not authority.development_profitability_reporting_authorized
    assert "primary-net-log-return-lcb-positive" in authority.report.failed_gate_names
    assert not authority.live_trading_authorized


def test_nonmonotone_cost_ladder_completes_as_negative_report(tmp_path) -> None:
    outer_authority, rollout_authorities = _report_inputs(
        daily_mean=0.001,
        cost_ladder_monotone=False,
    )
    authority = materialize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        artifact_id="nonmonotone-development-report",
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
        committed_at_ms=1,
    )

    assert authority.runtime_report_replayed
    assert authority.report.primary_net_log_return_lcb95 > 0.0
    assert authority.report.failed_gate_names == ("cost-ladder-monotone",)
    assert not authority.development_profitability_reporting_authorized
    generic = parse_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        loaded_source=authority.loaded_source,
    )
    replayed = authorize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path,
        authority=generic,
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
    )
    assert replayed.runtime_report_replayed
    assert replayed.report.failed_gate_names == ("cost-ladder-monotone",)
    assert not replayed.development_profitability_reporting_authorized


def test_profitability_report_rejects_nonreconciling_daily_economics(tmp_path) -> None:
    outer_authority, rollout_authorities = _report_inputs(daily_mean=0.001)
    rollout_authorities[
        0
    ].runtime_rollout.policy_trace.terminal_liquidation_adjusted_return = 0.0
    with pytest.raises(
        MassiveAdaptiveRLProfitabilityReportAuthorityV1Error,
        match="do not reconcile",
    ):
        materialize_massive_adaptive_rl_profitability_report_authority_v1(
            root=tmp_path,
            artifact_id="nonreconciling-development-report",
            outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
            ppo_outer_rollout_authorities=(  # type: ignore[arg-type]
                rollout_authorities
            ),
            committed_at_ms=1,
        )


@pytest.mark.parametrize(
    ("daily_mean", "authorized"),
    ((0.001, True), (-0.001, False)),
)
def test_terminal_state_is_derived_from_manifest_bound_report_authority(
    tmp_path, monkeypatch, daily_mean: float, authorized: bool
) -> None:
    experiment_id = f"manifest-bound-report-{str(authorized).lower()}"
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=experiment_id
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    source_bundle, runtime_graph = _authorized_source_bundle(
        tmp_path / "source", manifest, monkeypatch
    )
    outer_authority, rollout_authorities = _report_inputs(daily_mean=daily_mean)
    outer_authority.runtime_evidence.passed_gate_names = tuple(
        gate
        for gate in MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3
        if gate != "primary-net-log-return-lcb-positive"
    )
    (tmp_path / "reports").mkdir()
    authority = materialize_massive_adaptive_rl_profitability_report_authority_v1(
        root=tmp_path / "reports",
        artifact_id=experiment_id,
        outer_evidence_authority_v4=outer_authority,  # type: ignore[arg-type]
        ppo_outer_rollout_authorities=rollout_authorities,  # type: ignore[arg-type]
        committed_at_ms=1,
    )

    artifact_root = tmp_path / "states"
    state = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    for stage in MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[1:-1]:
        if stage is MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED:
            stage_receipt = source_bundle.semantic_receipt_sha256
        elif (
            stage is MassiveAdaptiveRLExperimentStageV2.FOUR_FOLD_V4_EVIDENCE_COMPLETED
        ):
            stage_receipt = authority.report.outer_evidence_authority_v4_receipt_sha256
        else:
            stage_receipt = _digest(stage.value)
        state = advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=state,
            stage=stage,
            stage_artifact_receipt_sha256=stage_receipt,
        )
    evidence_state = state
    generic_source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=tmp_path / "source",
        manifest=manifest.base_manifest,
    )
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="source bundle is not replay authorized",
    ):
        publish_massive_adaptive_rl_development_report_state_v3(
            artifact_root=tmp_path / "unqualified-source-states",
            previous=evidence_state,
            manifest=manifest,
            report_authority=authority,
            source_bundle=generic_source_bundle,
            runtime_source_graph_authority=runtime_graph,
        )

    generic_runtime_graph = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=tmp_path / "source",
        manifest=manifest,
        source_bundle=generic_source_bundle,
    )
    generic_runtime_graph.validate()
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="runtime source graph is not replay authorized",
    ):
        publish_massive_adaptive_rl_development_report_state_v3(
            artifact_root=artifact_root,
            previous=evidence_state,
            manifest=manifest,
            report_authority=authority,
            source_bundle=source_bundle,
            runtime_source_graph_authority=generic_runtime_graph,
        )

    if authorized:
        other_outer_authority, other_rollout_authorities = _report_inputs(
            daily_mean=daily_mean
        )
        other_outer_authority.runtime_evidence.passed_gate_names = tuple(
            gate
            for gate in MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3
            if gate != "primary-net-log-return-lcb-positive"
        )
        (tmp_path / "other-reports").mkdir()
        other_authority = (
            materialize_massive_adaptive_rl_profitability_report_authority_v1(
                root=tmp_path / "other-reports",
                artifact_id="another-experiment",
                outer_evidence_authority_v4=other_outer_authority,  # type: ignore[arg-type]
                ppo_outer_rollout_authorities=other_rollout_authorities,  # type: ignore[arg-type]
                committed_at_ms=1,
            )
        )
        with pytest.raises(
            MassiveAdaptiveRLExperimentStateV2Error,
            match="belongs to another experiment",
        ):
            publish_massive_adaptive_rl_development_report_state_v3(
                artifact_root=tmp_path / "cross-experiment-report-states",
                previous=evidence_state,
                manifest=manifest,
                report_authority=other_authority,
                source_bundle=source_bundle,
                runtime_source_graph_authority=runtime_graph,
            )

    state = block_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        previous=state,
        blocked_stage=MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED,
        blocker_code="report-storage-temporarily-unavailable",
        blocker_evidence_receipt_sha256=_digest("report-storage-blocker"),
    )
    published = publish_massive_adaptive_rl_development_report_state_v3(
        artifact_root=artifact_root,
        previous=state,
        manifest=manifest,
        report_authority=authority,
        source_bundle=source_bundle,
        runtime_source_graph_authority=runtime_graph,
    )
    assert published.execution_complete
    assert published.development_profitability_reporting_authorized is authorized
    assert published.failed_gate_names == authority.report.failed_gate_names
    assert ("primary-net-log-return-lcb-positive" in published.failed_gate_names) is (
        not authorized
    )
    assert (
        published.profitability_report_authority_receipt_sha256
        == authority.semantic_receipt_sha256
    )
    assert (
        published.profitability_report_receipt_sha256
        == authority.report.semantic_receipt_sha256
    )
    assert (
        published.source_bundle_receipt_sha256 == source_bundle.semantic_receipt_sha256
    )
    assert published.source_data_qualified
    assert (
        published.runtime_source_graph_authority_receipt_sha256
        == runtime_graph.runtime_authority_receipt_sha256
    )
    assert (
        published.last_completed_stage
        is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
    )

    for mutation in (
        lambda: advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=published,
            stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            stage_artifact_receipt_sha256=_digest("late-advance"),
        ),
        lambda: block_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=published,
            blocked_stage=MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
            blocker_code="late-blocker",
            blocker_evidence_receipt_sha256=_digest("late-blocker"),
        ),
        lambda: fail_massive_adaptive_rl_experiment_state_v2(
            artifact_root=artifact_root,
            previous=published,
            failed_stage=MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED,
            failure_code="late-failure",
            failure_evidence_receipt_sha256=_digest("late-failure"),
        ),
    ):
        with pytest.raises(MassiveAdaptiveRLExperimentStateV2Error, match="terminal"):
            mutation()

    before_resume = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )
    resumed = run_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path / "source-is-not-mounted",
        artifact_root=artifact_root,
        device="cpu",
        resume=True,
    )
    verified = verify_massive_adaptive_rl_experiment_v2(
        manifest_path=manifest_path,
        source_root=tmp_path / "source-is-not-mounted",
        artifact_root=artifact_root,
    )
    assert resumed.semantic_receipt_sha256 == verified.semantic_receipt_sha256
    assert resumed.execution_complete
    assert resumed.source_data_qualified
    assert resumed.ledger_replayed
    assert not resumed.completion_authority_replayed
    assert not resumed.full_verification_complete
    assert resumed.source_bundle_receipt_sha256 == source_bundle.semantic_receipt_sha256
    assert (
        load_massive_adaptive_rl_experiment_states_v2(
            artifact_root=artifact_root,
            experiment_id=manifest.experiment_id,
        )
        == before_resume
    )

    other_manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=manifest.experiment_id,
        execution_device_specification="cuda:0",
    )
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="another manifest",
    ):
        publish_massive_adaptive_rl_development_report_state_v3(
            artifact_root=tmp_path / "cross-manifest-states",
            previous=state,
            manifest=other_manifest,
            report_authority=authority,
            source_bundle=source_bundle,
            runtime_source_graph_authority=runtime_graph,
        )

    mismatched_root = tmp_path / "mismatched-evidence-states"
    mismatched = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=mismatched_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    for stage in MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[1:-1]:
        stage_receipt = (
            source_bundle.semantic_receipt_sha256
            if stage is MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED
            else _digest((stage.value, "other-evidence"))
        )
        mismatched = advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=mismatched_root,
            previous=mismatched,
            stage=stage,
            stage_artifact_receipt_sha256=stage_receipt,
        )
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="does not descend",
    ):
        publish_massive_adaptive_rl_development_report_state_v3(
            artifact_root=mismatched_root,
            previous=mismatched,
            manifest=manifest,
            report_authority=authority,
            source_bundle=source_bundle,
            runtime_source_graph_authority=runtime_graph,
        )

    mismatched_source_root = tmp_path / "mismatched-source-states"
    mismatched_source = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=mismatched_source_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    for stage in MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[1:-1]:
        stage_receipt = (
            authority.report.outer_evidence_authority_v4_receipt_sha256
            if stage
            is MassiveAdaptiveRLExperimentStageV2.FOUR_FOLD_V4_EVIDENCE_COMPLETED
            else _digest((stage.value, "substituted-source"))
        )
        mismatched_source = advance_massive_adaptive_rl_experiment_state_v2(
            artifact_root=mismatched_source_root,
            previous=mismatched_source,
            stage=stage,
            stage_artifact_receipt_sha256=stage_receipt,
        )
    with pytest.raises(
        MassiveAdaptiveRLExperimentStateV2Error,
        match="differs from the source replay stage",
    ):
        publish_massive_adaptive_rl_development_report_state_v3(
            artifact_root=mismatched_source_root,
            previous=mismatched_source,
            manifest=manifest,
            report_authority=authority,
            source_bundle=source_bundle,
            runtime_source_graph_authority=runtime_graph,
        )

    assert evidence_state.last_completed_stage_artifact_receipt_sha256 == (
        authority.report.outer_evidence_authority_v4_receipt_sha256
    )
