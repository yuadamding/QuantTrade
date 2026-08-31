from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    MassiveAdaptiveRLProfitabilityReportAuthorityV1Error,
    authorize_massive_adaptive_rl_profitability_report_authority_v1,
    materialize_massive_adaptive_rl_profitability_report_authority_v1,
    parse_massive_adaptive_rl_profitability_report_authority_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


def _digest(value: object) -> str:
    return semantic_sha256(value)


def _report_inputs(*, daily_mean: float) -> tuple[object, tuple[object, ...]]:
    folds = []
    authorities = []
    for fold_index in range(4):
        rows = tuple(
            daily_mean + (0.0001 if index % 2 == 0 else -0.0001)
            for index in range(126)
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
            transition_inventory_sha256=_digest(
                ("transition-inventory", fold_index)
            ),
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
        passed_gate_names=("cost-ladder-monotone",),
        failed_gate_names=(),
        source_data_qualified=True,
    )
    outer_authority = SimpleNamespace(
        validate=lambda: None,
        semantic_receipt_sha256=_digest("outer-evidence-authority-v4"),
        runtime_evidence=evidence,
        runtime_folds=tuple(folds),
        runtime_evidence_replayed=True,
        source_data_qualified=True,
        outer_development_conclusion_authorized=True,
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


def test_profitability_report_rejects_nonreconciling_daily_economics(tmp_path) -> None:
    outer_authority, rollout_authorities = _report_inputs(daily_mean=0.001)
    rollout_authorities[0].runtime_rollout.policy_trace.terminal_liquidation_adjusted_return = 0.0
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
