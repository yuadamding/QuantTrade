from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest

from rl_quant.evaluation import (
    massive_adaptive_outer_access_commitment_v2 as access_module,
)
from rl_quant.evaluation import (
    massive_adaptive_rl_profitability_report_authority_v2 as report_module,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
    MassiveAdaptiveOuterAccessCommitmentV2Error,
    run_or_resume_massive_adaptive_outer_access_commitment_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_authority_v2 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV2Error,
    MassiveAdaptiveRLOuterRolloutComputationV2,
    execute_massive_adaptive_rl_outer_rollout_v2,
    run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_validation_release_authority_v1 import (
    run_or_resume_massive_adaptive_rl_delayed_validation_release_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.workflows import massive_adaptive_rl_writer_guard_v5 as writer_guard
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    build_massive_adaptive_rl_experiment_manifest_v5,
)
from rl_quant.workflows.massive_adaptive_rl_walk_forward_policy_schedule_v1 import (
    walk_forward_policy_schedule_relative_path_v1,
)


def _digest(label: object) -> str:
    return semantic_sha256(("v5-outer-component", label))


def _dates() -> tuple[str, ...]:
    return tuple(f"session-{index:03d}" for index in range(126))


def _generic_outer_access() -> MassiveAdaptiveOuterAccessCommitmentV2:
    dates = _dates()
    provisional = MassiveAdaptiveOuterAccessCommitmentV2(
        experiment_id="v5-outer-component",
        manifest_v5_receipt_sha256=_digest("manifest"),
        scientific_protocol_projection_sha256=_digest("protocol"),
        manifest_v5_registration_receipt_sha256=_digest("registration"),
        execution_implementation_registration_receipt_sha256=_digest("implementation"),
        scientific_execution_fingerprint_sha256=_digest("fingerprint"),
        fold_index=0,
        policy_schedule_receipt_sha256=_digest("schedule"),
        policy_schedule_source_receipt_sha256=_digest("schedule-source"),
        policy_schedule_commit_receipt_sha256=_digest("schedule-commit"),
        policy_schedule_committed_at_ms=10,
        frozen_ppo_policy_receipt_sha256=_digest("ppo"),
        frozen_ppo_source_receipt_sha256=_digest("ppo-source"),
        frozen_ppo_commit_receipt_sha256=_digest("ppo-commit"),
        frozen_ppo_committed_at_ms=8,
        frozen_fc06_control_receipt_sha256=_digest("fc06"),
        frozen_fc06_source_receipt_sha256=_digest("fc06-source"),
        frozen_fc06_commit_receipt_sha256=_digest("fc06-commit"),
        frozen_fc06_committed_at_ms=9,
        outer_fold_receipt_sha256=_digest("outer-fold"),
        outer_decision_session_dates=dates,
        outer_decision_inventory_sha256=semantic_sha256(dates),
        policy_validation_eligible=True,
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def _outer_computation() -> MassiveAdaptiveRLOuterRolloutComputationV2:
    dates = _dates()
    zeroes = (0.0,) * len(dates)
    actions = tuple(_digest(("action", index)) for index in range(len(dates)))
    provisional = MassiveAdaptiveRLOuterRolloutComputationV2(
        fold_index=0,
        outer_access_commitment_receipt_sha256=_digest("access"),
        frozen_ppo_policy_receipt_sha256=_digest("ppo"),
        frozen_fc06_control_receipt_sha256=_digest("fc06"),
        decision_session_dates=dates,
        ppo_action_evidence_receipts=actions,
        ppo_action_inventory_sha256=semantic_sha256(actions),
        ppo_primary_trace_receipt_sha256=_digest("primary-trace"),
        ppo_low_cost_trace_receipt_sha256=_digest("low-trace"),
        ppo_high_cost_trace_receipt_sha256=_digest("high-trace"),
        fixed_control_trace_receipt_sha256=_digest("fixed-trace"),
        fixed_control_low_cost_trace_receipt_sha256=_digest("fixed-low-trace"),
        fixed_control_high_cost_trace_receipt_sha256=_digest("fixed-high-trace"),
        ppo_primary_transition_inventory_sha256=_digest("primary-transitions"),
        ppo_low_cost_transition_inventory_sha256=_digest("low-transitions"),
        ppo_high_cost_transition_inventory_sha256=_digest("high-transitions"),
        fixed_control_transition_inventory_sha256=_digest("fixed-transitions"),
        fixed_control_low_cost_transition_inventory_sha256=_digest(
            "fixed-low-transitions"
        ),
        fixed_control_high_cost_transition_inventory_sha256=_digest(
            "fixed-high-transitions"
        ),
        decision_target_inventory_sha256=_digest("targets"),
        fixed_control_decision_target_inventory_sha256=_digest("fixed-targets"),
        strategy_net_log_returns=zeroes,
        neutral_net_log_returns=zeroes,
        benchmark_net_log_returns=zeroes,
        fixed_control_net_log_returns=zeroes,
        active_log_returns=zeroes,
        incremental_rl_log_returns=zeroes,
        ppo_minus_fixed_control_log_returns=zeroes,
        primary_terminal_liquidation_adjusted_return=0.0,
        low_cost_terminal_liquidation_adjusted_return=0.0,
        high_cost_terminal_liquidation_adjusted_return=0.0,
        fixed_control_terminal_liquidation_adjusted_return=0.0,
        fixed_control_low_cost_terminal_liquidation_adjusted_return=0.0,
        fixed_control_high_cost_terminal_liquidation_adjusted_return=0.0,
        ppo_cost_ladder_monotone=True,
        fixed_control_cost_ladder_monotone=True,
        maximum_drawdown=0.0,
        environment_source_inventory_sha256=_digest("environment"),
        source_data_qualified=True,
        semantic_receipt_sha256="0" * 64,
    )
    return replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )


def test_outer_public_surfaces_do_not_accept_economic_outcomes() -> None:
    access_parameters = set(
        inspect.signature(
            run_or_resume_massive_adaptive_outer_access_commitment_v2
        ).parameters
    )
    assert access_parameters == {
        "root",
        "manifest",
        "manifest_registration",
        "execution_registration",
        "policy_schedule",
        "frozen_policy",
        "frozen_control",
        "allow_materialize",
    }
    rollout_parameters = set(
        inspect.signature(
            run_or_resume_massive_adaptive_rl_outer_rollout_authority_v2
        ).parameters
    )
    assert rollout_parameters == {
        "root",
        "manifest",
        "manifest_registration",
        "outer_access",
        "allow_materialize",
    }
    assert set(
        inspect.signature(execute_massive_adaptive_rl_outer_rollout_v2).parameters
    ) == {"outer_access"}
    release_parameters = set(
        inspect.signature(
            run_or_resume_massive_adaptive_rl_delayed_validation_release_v1
        ).parameters
    )
    assert release_parameters == {
        "root",
        "manifest",
        "manifest_registration",
        "execution_registration",
        "initial_inputs",
        "predecessor_outer_fold_seal",
        "allow_materialize",
    }


def test_outer_environment_is_not_a_public_precommit_input() -> None:
    access = _generic_outer_access()
    access.validate()
    assert access.runtime_commitment_replayed is False
    assert access.outer_input_access_authorized is False
    assert "_authorize_massive_adaptive_outer_access_environment_v2" not in (
        access_module.__all__
    )
    with pytest.raises(
        MassiveAdaptiveOuterAccessCommitmentV2Error,
        match="has not been commitment-replayed",
    ):
        _ = access.runtime_environment_bundle


def test_outer_cost_ladder_nonmonotonicity_is_reportable_evidence() -> None:
    computation = _outer_computation()
    computation.validate()
    changed = replace(
        computation,
        high_cost_terminal_liquidation_adjusted_return=0.01,
        ppo_cost_ladder_monotone=False,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    changed.validate()
    assert changed.ppo_cost_ladder_monotone is False


def test_fixed_control_cost_ladder_nonmonotonicity_is_reportable_evidence() -> None:
    computation = _outer_computation()
    changed = replace(
        computation,
        fixed_control_high_cost_terminal_liquidation_adjusted_return=0.01,
        fixed_control_cost_ladder_monotone=False,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    changed.validate()
    assert changed.fixed_control_cost_ladder_monotone is False


def test_nonmonotone_cost_ladder_reaches_a_failed_report_gate() -> None:
    computation = _outer_computation()
    changed = replace(
        computation,
        high_cost_terminal_liquidation_adjusted_return=0.01,
        ppo_cost_ladder_monotone=False,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    seal = SimpleNamespace(
        fold_index=0,
        semantic_receipt_sha256=_digest("seal"),
        outer_rollout_authority_receipt_sha256=_digest("rollout"),
        rollout_authority=SimpleNamespace(rollout=changed),
        source_data_qualified=True,
        validate=lambda: None,
    )

    fold_report = report_module._fold_report(seal)

    assert fold_report.ppo_cost_ladder_monotone is False
    assert not report_module._cost_ladder_monotonicity_gate((fold_report,) * 4)


def test_outer_cost_ladder_observation_cannot_be_forged() -> None:
    computation = _outer_computation()
    changed = replace(
        computation,
        high_cost_terminal_liquidation_adjusted_return=0.01,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(
        MassiveAdaptiveRLOuterRolloutAuthorityV2Error,
        match="computation differs",
    ):
        changed.validate()


def test_policy_schedule_has_a_fold_zero_prefix() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v5(
        experiment_id="v5-outer-schedule-path"
    )
    assert walk_forward_policy_schedule_relative_path_v1(
        manifest=manifest, through_fold_index=0
    ).endswith("/prefix-through-fold-0.json")


def test_writer_guard_recognizes_only_canonical_outer_paths() -> None:
    capability = SimpleNamespace(
        writer_role="prequential-outer-execution",
        allowed_fold_indices=(0, 1, 2, 3),
    )
    assert writer_guard._prequential_scoped_path_authorized_v1(
        parts=(
            "adaptive-rl",
            "experiment",
            "outer-fold-seal-authority-v1",
            "fold-0.json",
        ),
        capability=capability,
    )
    assert writer_guard._prequential_scoped_path_authorized_v1(
        parts=(
            "adaptive-rl",
            "experiment",
            "profitability-report-authority-v2",
            "report.json",
        ),
        capability=capability,
    )
    assert not writer_guard._prequential_scoped_path_authorized_v1(
        parts=(
            "adaptive-rl",
            "experiment",
            "outer-fold-seal-v1",
            "fold-0.json",
        ),
        capability=capability,
    )
