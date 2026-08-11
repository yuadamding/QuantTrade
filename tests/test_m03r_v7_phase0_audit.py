from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation.m03r_alpha_head_diagnostics import (
    M03R_ALPHA_HEAD_DIAGNOSTIC_UNAVAILABLE_SCHEMA,
    M03RAlphaHeadDiagnosticError,
    M03RAlphaHeadDiagnosticInput,
    build_unavailable_m03r_alpha_head_diagnostics,
    evaluate_m03r_alpha_head_diagnostics,
)
from rl_quant.evaluation.m03r_cost_ladder_evaluator import (
    M03RCostLadderInput,
    evaluate_m03r_cost_ladder,
)
from rl_quant.evaluation.m03r_projection_attribution import (
    M03RProjectionAttributionInput,
    evaluate_m03r_projection_attribution,
)
from rl_quant.evaluation.m03r_setting9_risk_audit import (
    M03R_SETTING9_ID,
    M03RSetting9RiskAuditInput,
    evaluate_m03r_setting9_risk_audit,
)
from rl_quant.evaluation.top2000_m03r_v7_dev import tensor_sha256
from rl_quant.evaluation.m03r_v7_trace_audit import (
    M03RV7ForensicTrace,
    M03RV7FrozenCheckpointIdentity,
    build_m03r_v7_forensic_trace,
    compare_m03r_v7_forensic_traces,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.workflows import top2000_m03r_v7_forensic_audit as audit_workflow
from rl_quant.workflows import (
    top2000_m03r_v7_forensic_audit_worker as audit_worker,
)
from rl_quant.workflows.top2000_m03r_v7_forensic_audit import (
    M03RV7ForensicWorkflowError,
    _write_immutable_json,
    _write_immutable_npz,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_dev import (
    Top2000M03RV7Seed17TrainingPlan,
)


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _dates() -> tuple[str, ...]:
    return tuple(f"2025-01-{index + 1:02d}" for index in range(31)) + tuple(
        f"2025-02-{index + 1:02d}" for index in range(28)
    ) + tuple(f"2025-03-{index + 1:02d}" for index in range(4))


def _identity(setting_index: int = 0) -> M03RV7FrozenCheckpointIdentity:
    return M03RV7FrozenCheckpointIdentity(
        setting_index=setting_index,
        setting_id=M03R_TOP2000_DEV_SETTING_IDS[setting_index],
        fold_index=0,
        seed=17,
        checkpoint_file_sha256=_digest(f"checkpoint-{setting_index}"),
        model_state_sha256=_digest(f"model-{setting_index}"),
        alpha_core_optimizer_state_sha256=_digest(f"optimizer-{setting_index}"),
        overlay_optimizer_state_sha256=None,
        factor_calibration_receipt_sha256=_digest("calibration"),
    )


@pytest.mark.parametrize("setting_index", [0, 6])
def test_forensic_plan_resolves_seed17_id_through_bound_runtime_setting(
    setting_index: int,
) -> None:
    plan = Top2000M03RV7Seed17TrainingPlan(
        setting_index=setting_index,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[setting_index],
        runtime_setting_id=M03R_TOP2000_DEV_SETTING_IDS[setting_index],
        cache_path="/mnt/audit-package/cache.pt",
        cache_sha256=_digest("cache"),
        output_root=f"/mnt/output/setting-{setting_index:02d}",
    )

    resolved = audit_workflow._resolve_plan_development_setting(plan)

    assert resolved.setting_index == setting_index
    assert resolved.setting_id == plan.runtime_setting_id
    assert resolved.residual_alpha_head_mode == (
        "none" if setting_index == 6 else "mean-and-downside"
    )


def _forensic(setting_index: int = 0, shift: float = 0.0) -> M03RV7ForensicTrace:
    rows = 63
    benchmark = np.tile(np.array([0.0, 0.5, 0.5]), (rows, 1))
    policy = benchmark.copy()
    policy[:, 1] += shift
    policy[:, 2] -= shift
    policy_turnover = np.full(rows, 0.01)
    benchmark_turnover = np.full(rows, 0.002)
    policy_gross = np.linspace(-0.001, 0.002, rows)
    benchmark_gross = np.linspace(-0.0005, 0.001, rows)
    return M03RV7ForensicTrace(
        checkpoint=_identity(setting_index),
        score_dates=_dates(),
        decision_weights=benchmark,
        requested_weights=policy,
        post_hazard_weights=policy,
        post_projection_weights=policy,
        executed_weights=policy,
        benchmark_weights=benchmark,
        policy_gross_returns=policy_gross,
        policy_net_returns_20bp=policy_gross - 0.002 * policy_turnover,
        benchmark_gross_returns=benchmark_gross,
        benchmark_net_returns_20bp=benchmark_gross - 0.002 * benchmark_turnover,
        policy_total_one_way_turnover=policy_turnover,
        benchmark_total_one_way_turnover=benchmark_turnover,
        requested_action_trace_sha256=_digest(f"action-{setting_index}-{shift}"),
    )


def test_trace_pairwise_gate_rejects_identical_causal_setting_paths() -> None:
    left = _forensic(0)
    right = _forensic(1)
    result = compare_m03r_v7_forensic_traces(left, right)
    assert result["executed_weight_trace_equal"] is True
    assert result["causal_distinctness_gate_passed"] is False
    assert result["fraction_of_dates_with_identical_executed_weights"] == 1.0

    distinct = compare_m03r_v7_forensic_traces(left, _forensic(1, 0.01))
    assert distinct["causal_distinctness_gate_passed"] is True
    assert distinct["maximum_absolute_executed_weight_difference"] == pytest.approx(0.01)


def test_forensic_builder_extracts_exact_runtime_stages() -> None:
    rows = []
    intent = SimpleNamespace(
        entry_scores=torch.tensor([[0.0, 1.0, -1.0]]),
        target_logits=None,
        gate=None,
        raw_hazard_residual=torch.zeros((1, 3), dtype=torch.bfloat16),
        hazard_residual=torch.zeros((1, 3)),
        exact_hold_decision_st=None,
        active_risk_scale=torch.tensor([0.01]),
        signal_confidence=torch.tensor([0.5]),
        auxiliary_alpha_mean=torch.zeros((1, 3, 4)),
        exit_action_v6=None,
    )
    for _ in range(63):
        repaired = torch.tensor([[0.2, 0.4, 0.4]])
        requested_delta = torch.tensor([[0.0, 0.01, -0.01]])
        constructed_delta = torch.tensor([[0.0, 0.005, -0.005]])
        rows.append(
            SimpleNamespace(
                risk_repaired_weights=repaired,
                requested_delta=requested_delta,
                constructed_delta=constructed_delta,
                decision_weights=repaired,
                post_cost_weights=repaired + constructed_delta,
                holding_return=torch.tensor([0.001]),
                net_return=torch.tensor([0.00098]),
                turnover_by_cause={
                    cause: torch.tensor(
                        [0.01 if cause is TurnoverCause.DISCRETIONARY else 0.0]
                    )
                    for cause in TURNOVER_CAUSES
                },
                raw_intent=intent,
            )
        )
    trace = SimpleNamespace(transitions=tuple(rows))
    benchmark_weights = torch.tensor([[0.2, 0.4, 0.4]]).repeat(63, 1)
    benchmark_turnover = torch.full((63,), 0.002)
    benchmark_gross = torch.full((63,), 0.0005)
    result = build_m03r_v7_forensic_trace(
        trace,  # type: ignore[arg-type]
        checkpoint=_identity(),
        score_dates=_dates(),
        score_transition_start=0,
        score_transition_stop_exclusive=63,
        benchmark_weights=benchmark_weights,
        benchmark_gross_returns=benchmark_gross,
        benchmark_net_returns_20bp=benchmark_gross - 0.002 * benchmark_turnover,
        benchmark_total_one_way_turnover=benchmark_turnover,
    )
    assert result.requested_weights[0].tolist() == pytest.approx([0.2, 0.41, 0.39])
    assert result.post_projection_weights[0].tolist() == pytest.approx(
        [0.2, 0.405, 0.395]
    )
    array_sha256s = result.array_sha256s()
    assert array_sha256s["executed_weights"]
    assert result.receipt["arrays"] == array_sha256s
    assert "array_sha256" not in result.receipt


def test_tensor_sha256_hashes_bfloat16_storage_without_value_conversion() -> None:
    value = torch.tensor(
        [[1.0, -2.5], [0.0, 3.25]],
        dtype=torch.bfloat16,
    )

    assert tensor_sha256(value) == (
        "1142e809a8b869454b7788cfae7d9ce970a165377d3512dc965222d56d638784"
    )


def test_cost_ladder_separates_policy_and_c1_costs() -> None:
    trace = _forensic()
    result = evaluate_m03r_cost_ladder(
        M03RCostLadderInput(
            setting_index=0,
            setting_id=M03R_TOP2000_DEV_SETTING_IDS[0],
            fold_index=0,
            score_dates=trace.score_dates,
            policy_net_returns_20bp=trace.policy_net_returns_20bp,
            benchmark_net_returns_20bp=trace.benchmark_net_returns_20bp,
            policy_total_one_way_turnover=trace.policy_total_one_way_turnover,
            benchmark_total_one_way_turnover=trace.benchmark_total_one_way_turnover,
        )
    )
    assert result["annualized_policy_transaction_cost_at_20bp"] == pytest.approx(
        252 * 0.002 * 0.01
    )
    assert result["annualized_benchmark_transaction_cost_at_20bp"] == pytest.approx(
        252 * 0.002 * 0.002
    )
    assert result["cost_ladder"]["0"]["annualized_net_active_return"] == pytest.approx(
        result["annualized_gross_active_return"]
    )


def _alpha_input() -> M03RAlphaHeadDiagnosticInput:
    rows, assets, horizons = 63, 21, 4
    target = np.zeros((rows, assets, horizons), dtype=np.float64)
    prediction = np.zeros_like(target)
    base = np.linspace(-0.02, 0.02, assets - 1)
    for date in range(rows):
        for horizon in range(horizons):
            target[date, 1:, horizon] = base * (horizon + 1)
            prediction[date, 1:, horizon] = target[date, 1:, horizon]
    valid = np.ones_like(target, dtype=np.bool_)
    valid[:, 0] = False
    confidence = np.tile(np.linspace(0.0, 1.0, assets), (rows, 1))
    sector = np.tile(np.arange(assets) % 3, (rows, 1))
    return M03RAlphaHeadDiagnosticInput(
        setting_index=0,
        setting_id=M03R_TOP2000_DEV_SETTING_IDS[0],
        fold_index=0,
        score_dates=_dates(),
        action_ids=("CASH", *(f"S{index}" for index in range(1, assets))),
        predictions=prediction,
        targets=target,
        valid=valid,
        confidence=confidence,
        breakdowns={"sector": sector},
    )


def test_alpha_head_diagnostics_are_date_balanced_and_cash_excluded() -> None:
    value = _alpha_input()
    result = evaluate_m03r_alpha_head_diagnostics(value)
    assert result["horizons"]["21"]["spearman_rank_ic"] == pytest.approx(1.0)
    assert result["horizons"]["30"]["positive_rank_ic_date_count"] == 63
    assert result["horizons"]["30"]["top_minus_bottom_decile_residual_return"] > 0
    assert result["confidence_diagnostics"]["status"] == "available"
    assert set(result["conditional_breakdowns"]["sector"]) == {"0", "1", "2"}

    bad = value.valid.copy()
    bad[:, 0, 0] = True
    with pytest.raises(M03RAlphaHeadDiagnosticError, match="CASH"):
        replace(value, valid=bad)


def test_no_alpha_head_setting_emits_typed_unavailable_evidence_without_fake_arrays() -> None:
    raw_intent = SimpleNamespace(auxiliary_alpha_mean=None)
    trace = SimpleNamespace(
        transitions=tuple(SimpleNamespace(raw_intent=raw_intent) for _ in range(2))
    )
    sequence = SimpleNamespace(num_assets=3)
    assert (
        audit_workflow._alpha_arrays(
            trace,
            sequence,
            start=0,
            stop=2,
            expected_available=False,
        )
        is None
    )
    receipt = build_unavailable_m03r_alpha_head_diagnostics(
        setting_index=6,
        setting_id=M03R_TOP2000_DEV_SETTING_IDS[6],
        fold_index=0,
        score_dates=_dates(),
        action_ids=("CASH", "S1", "S2"),
        residual_alpha_head_mode="none",
    )
    assert receipt["schema"] == M03R_ALPHA_HEAD_DIAGNOSTIC_UNAVAILABLE_SCHEMA
    assert receipt["status"] == "unavailable"
    assert receipt["alpha_heads_present"] is False
    assert receipt["array_sha256"] == {
        "predictions": None,
        "targets": None,
        "valid": None,
        "confidence": None,
        "breakdowns": {},
    }
    assert all(row["status"] == "unavailable" for row in receipt["horizons"].values())


def test_alpha_head_presence_must_match_the_frozen_setting_route() -> None:
    missing = SimpleNamespace(
        transitions=(
            SimpleNamespace(
                raw_intent=SimpleNamespace(auxiliary_alpha_mean=None)
            ),
        )
    )
    sequence = SimpleNamespace(num_assets=3)
    with pytest.raises(M03RV7ForensicWorkflowError, match="omitted"):
        audit_workflow._alpha_arrays(
            missing,
            sequence,
            start=0,
            stop=1,
            expected_available=True,
        )

    unexpected = SimpleNamespace(
        transitions=(
            SimpleNamespace(
                raw_intent=SimpleNamespace(
                    auxiliary_alpha_mean=torch.zeros(1, 3, 4)
                )
            ),
        )
    )
    with pytest.raises(M03RV7ForensicWorkflowError, match="unexpectedly emitted"):
        audit_workflow._alpha_arrays(
            unexpected,
            sequence,
            start=0,
            stop=1,
            expected_available=False,
        )


def test_projection_attribution_reports_stage_loss_and_te() -> None:
    trace = _forensic(0, 0.02)
    projected = trace.requested_weights.copy()
    projected[:, 1:] = 0.5 * projected[:, 1:] + 0.5 * trace.benchmark_weights[:, 1:]
    projected[:, 0] = 1.0 - projected[:, 1:].sum(1)
    covariance = np.tile(np.eye(3)[None, :, :] * 0.0001, (63, 1, 1))
    alpha = np.tile(np.array([0.0, 1.0, -1.0]), (63, 1))
    result = evaluate_m03r_projection_attribution(
        M03RProjectionAttributionInput(
            setting_index=0,
            setting_id=M03R_TOP2000_DEV_SETTING_IDS[0],
            fold_index=0,
            score_dates=_dates(),
            benchmark_weights=trace.benchmark_weights,
            requested_weights=trace.requested_weights,
            post_hazard_weights=trace.post_hazard_weights,
            post_projection_weights=projected,
            executed_weights=projected,
            alpha_scores=alpha,
            covariance=covariance,
        )
    )
    assert result["mean_projection_retention_ratio"] == pytest.approx(0.5)
    assert result["stage_transitions"]["post_hazard_to_post_projection"][
        "fraction_of_decisions_changed"
    ] == 1.0
    assert result["stages"]["requested"]["mean_ex_ante_annual_tracking_error"] > 0

    no_alpha = evaluate_m03r_projection_attribution(
        M03RProjectionAttributionInput(
            setting_index=6,
            setting_id=M03R_TOP2000_DEV_SETTING_IDS[6],
            fold_index=0,
            score_dates=_dates(),
            benchmark_weights=trace.benchmark_weights,
            requested_weights=trace.requested_weights,
            post_hazard_weights=trace.post_hazard_weights,
            post_projection_weights=projected,
            executed_weights=projected,
            alpha_scores=None,
            covariance=None,
        )
    )
    assert all(
        row["alpha_attribution_status"]
        == "unavailable-alpha-scores-not-supplied"
        for row in no_alpha["stages"].values()
    )
    assert no_alpha["requested_versus_executed_score_rank_correlation"] is None


def test_setting9_audit_does_not_treat_high_te_route_as_clean_ablation() -> None:
    rows = 63
    result = evaluate_m03r_setting9_risk_audit(
        M03RSetting9RiskAuditInput(
            fold_index=0,
            initial_policy_weights=np.array([0.0, 0.9, 0.1]),
            initial_benchmark_weights=np.array([0.0, 0.5, 0.5]),
            requested_annual_tracking_error=np.full(rows, 0.04),
            post_projection_annual_tracking_error=np.full(rows, 0.04),
            realized_active_returns=np.tile(np.array([-0.02, 0.02, 0.0]), 21),
            reported_total_one_way_turnover=np.zeros(rows),
            startup_turnover=0.4,
            startup_turnover_in_reported_mean=False,
            benchmark_anchoring_enabled=True,
            tracking_error_control_enabled=True,
            active_beta_control_enabled=True,
        )
    )
    assert result["causal_interpretation_authorized"] is False
    assert "reported-turnover-excludes-startup" in result["diagnosis"]
    assert "realized-tracking-error-exceeded-ex-ante-control" in result["diagnosis"]
    assert result["setting_id"] == M03R_SETTING9_ID


def test_forensic_writer_is_no_clobber_and_json_retry_is_validation_only(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    first = _write_immutable_json(receipt_path, {"schema": "fixture", "value": 1})
    second = _write_immutable_json(receipt_path, {"schema": "fixture", "value": 1})
    assert first == second
    with pytest.raises(M03RV7ForensicWorkflowError, match="collision"):
        _write_immutable_json(receipt_path, {"schema": "fixture", "value": 2})

    arrays_path = tmp_path / "arrays.npz"
    assert _write_immutable_npz(arrays_path, {"x": np.arange(3)})
    with pytest.raises(M03RV7ForensicWorkflowError, match="already exists"):
        _write_immutable_npz(arrays_path, {"x": np.arange(3)})


def test_setting_worker_loads_cache_once_and_runs_six_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = _digest("identity")
    plan = SimpleNamespace(
        setting_index=0,
        setting_id=M03R_TOP2000_DEV_SETTING_IDS[0],
        runtime_setting_id="runtime-setting-00",
        receipt_sha256=_digest("plan-receipt"),
    )
    cache = object()
    cache_loads: list[Path] = []
    observed_folds: list[tuple[int, object, object]] = []

    monkeypatch.setattr(audit_workflow, "_load_plan", lambda *_args: plan)

    def load_cache(path: str | Path, **_kwargs: object) -> object:
        cache_loads.append(Path(path))
        return cache

    def run_fold(**kwargs: object) -> dict[str, object]:
        fold_index = int(kwargs["fold_index"])  # type: ignore[arg-type]
        observed_folds.append(
            (fold_index, kwargs["_prepared_plan"], kwargs["_verified_cache"])
        )
        return {"receipt_file_sha256": _digest(f"fold-{fold_index}")}

    monkeypatch.setattr(
        audit_workflow,
        "load_verified_top2000_hold30_development_cache",
        load_cache,
    )
    monkeypatch.setattr(
        audit_workflow,
        "run_m03r_v7_seed17_forensic_fold",
        run_fold,
    )
    setting_root = tmp_path / "setting"
    setting_root.mkdir()
    result = audit_workflow.run_m03r_v7_seed17_forensic_setting(
        setting_root=setting_root,
        cache_path=tmp_path / "cache.pt",
        expected_cache_sha256=digest,
        expected_training_plan_file_sha256=digest,
        evaluation_source_inventory_sha256=digest,
        source_training_archive_sha256=digest,
        output_root=tmp_path / "audit",
        device="cuda:0",
    )

    assert cache_loads == [tmp_path / "cache.pt"]
    assert observed_folds == [(index, plan, cache) for index in range(6)]
    assert result["fold_count"] == 6
    assert result["cache_load_count"] == 1
    receipt = json.loads((tmp_path / "audit/forensic-setting-receipt.json").read_text())
    assert receipt["fold_receipt_file_sha256"] == {
        f"fold-{index:02d}": _digest(f"fold-{index}") for index in range(6)
    }


def test_indexed_worker_uses_explicit_local_to_scientific_setting_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "setting_inputs": [
            {"completion_index": 7, "setting_index": 8},
            {"completion_index": 8, "setting_index": 7},
        ]
    }
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "0")
    selected = audit_worker._resolve_setting(manifest, (8, 7))
    assert selected == {
        "completion_index": 7,
        "setting_index": 8,
        "local_completion_index": 0,
    }

    monkeypatch.setenv("JOB_COMPLETION_INDEX", "1")
    selected = audit_worker._resolve_setting(manifest, (8, 7))
    assert selected["setting_index"] == 7
    assert selected["completion_index"] == 8

    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    with pytest.raises(M03RV7ForensicWorkflowError, match="outside"):
        audit_worker._resolve_setting(manifest, (8, 7))
