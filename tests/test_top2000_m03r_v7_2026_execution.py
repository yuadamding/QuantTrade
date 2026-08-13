from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rl_quant.envs.hold30 import AGE_BIN_COUNT, TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation import top2000_m03r_v7_2026_execution as execution
from rl_quant.evaluation import top2000_m03r_v7_2026_cohort_survival as cohort_adapter
from rl_quant.evaluation import top2000_m03r_v7_2026_trace_telemetry as trace_adapter
from rl_quant.evaluation.top2000_m03r_v7_2026 import (
    Top2000M03RV72026Telemetry,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_checkpoint import (
    Top2000M03RV72026CheckpointLoadReceipt,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_cohort_survival import (
    Top2000M03RV72026CohortTrajectories,
    Top2000M03RV72026CohortTrajectoryReceipt,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_execution import (
    Top2000M03RV72026CudaExecutionProof,
    Top2000M03RV72026ExecutionSession,
    load_top2000_m03r_v7_seed17_2026_execution_artifact,
    run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_factor_calibration import (
    fit_top2000_m03r_v7_2026_pre_score_factor_calibration,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    compose_top2000_m03r_v7_2026_retrospective_data,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_trace_telemetry import (
    Top2000M03RV72026TraceEvaluationInputs,
    Top2000M03RV72026TraceTelemetryReceipt,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_SETTING_IDS,
    runtime_setting_id,
)
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    Top2000M03RV7DevelopmentPolicy,
    render_top2000_m03r_v7_development_folds,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _weekdays(count: int, stop: dt.date) -> tuple[str, ...]:
    result: list[str] = []
    current = stop
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current -= dt.timedelta(days=1)
    return tuple(reversed(result))


def _date_range(start: dt.date, stop: dt.date) -> tuple[str, ...]:
    result: list[str] = []
    current = start
    while current <= stop:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += dt.timedelta(days=1)
    return tuple(result)


def _bars(rows: int, assets: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((rows, assets, 5), dtype=torch.float64)
    closes = 100.0 + torch.arange(rows, dtype=torch.float64) * 0.005
    stocks = torch.arange(1, assets, dtype=torch.float64)
    values[:, 1:, 0] = closes[:, None] + stocks[None] * 0.01
    values[:, 1:, 1] = values[:, 1:, 0] + 1.0
    values[:, 1:, 2] = values[:, 1:, 0] - 1.0
    values[:, 1:, 3] = values[:, 1:, 0]
    values[:, 1:, 4] = 1_000_000.0 + stocks[None] * 100.0
    return values, torch.ones((rows, assets), dtype=torch.bool)


def _data() -> tuple[Top2000VerifiedDevelopmentCache, Any]:
    actions = ("CASH", "A1", "A2", "A3", "A4", "A5")
    dates = _weekdays(1001, dt.date(2025, 12, 29))
    bars, available = _bars(len(dates), len(actions))
    cache = Top2000VerifiedDevelopmentCache(
        daily_ohlcv=bars,
        availability=available,
        exchange_dates=dates,
        action_ids=actions,
        cache_sha256=_digest("pre-cache-file"),
        cache_identity=_digest("pre-cache-identity"),
        search_identity=_digest("search"),
        action_hash=_axis_digest(actions),
        bar_seconds=300,
        acknowledgement=DEVELOPMENT_ACK,
        development_only=True,
        bars_only=True,
    )
    score_dates = _date_range(dt.date(2026, 1, 2), dt.date(2026, 6, 23))
    raw_dates = (dates[-1], *score_dates)
    raw_bars, raw_available = _bars(len(raw_dates), len(actions))
    raw_bars[0] = bars[-1]
    source = Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=_digest("base"),
        search_identity=cache.search_identity,
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test_partition_inventory_sha256=_digest("partition-inventory"),
        manifest_sha256=_digest("manifest"),
        universe_sha256=_digest("universe"),
        training_completion_receipt_sha256=_digest("completion"),
        evaluation_contract_sha256=_digest("contract"),
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )
    retrospective = compose_top2000_m03r_v7_2026_retrospective_data(
        cache,
        retrospective_daily_ohlcv=raw_bars,
        retrospective_availability=raw_available,
        retrospective_exchange_dates=raw_dates,
        retrospective_action_ids=actions,
        source_evidence=source,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    return cache, replace(
        retrospective,
        cache_file_sha256=_digest("retrospective-cache-file"),
    )


class _Policy(torch.nn.Module):
    state_provider_compatibility_id = (
        Top2000M03RV7DevelopmentPolicy.state_provider_compatibility_id
    )
    token_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("episode_factor_loadings", torch.empty((0, 0)))
        self.register_buffer("episode_factor_constraint_pinv", torch.empty((0, 0)))

    def bind_episode_factor_loadings(self, loadings: torch.Tensor) -> None:
        self.episode_factor_loadings = loadings.detach().clone()
        constraints = torch.cat(
            (torch.ones((loadings.shape[0], 1), dtype=loadings.dtype), loadings),
            dim=1,
        )
        self.episode_factor_constraint_pinv = torch.linalg.pinv(constraints)


def _inputs(retrospective: Any, view: Any, checkpoint_hash: str) -> Any:
    rows = len(retrospective.score_return_dates)
    assets = len(retrospective.action_ids)
    zeros = np.zeros(rows, dtype=np.float64)
    turnover = {
        cause.value: zeros.copy() for cause in TURNOVER_CAUSES
    }
    forced = {
        cause.value: np.zeros((1, rows, AGE_BIN_COUNT), dtype=np.float64)
        for cause in (
            TurnoverCause.MEMBERSHIP_FORCED,
            TurnoverCause.AVAILABILITY_FORCED,
            TurnoverCause.RISK_FORCED,
            TurnoverCause.TERMINAL,
        )
    }
    telemetry = Top2000M03RV72026Telemetry(
        requested_to_executed_projection_distance=np.zeros((1, rows)),
        age_notional_at_risk=np.zeros((1, rows, AGE_BIN_COUNT)),
        discretionary_exit_notional_by_age=np.zeros((1, rows, AGE_BIN_COUNT)),
        forced_exit_notional_by_cause_and_age=forced,
        action_counts_by_type={
            name: np.zeros((1, rows)) for name in ("HOLD", "CONTINUOUS", "EXIT")
        },
        continuous_hazard=np.zeros((1, rows, assets)),
        continuous_hazard_observed=np.zeros((1, rows, assets), dtype=np.bool_),
    )
    provisional = object.__new__(Top2000M03RV72026TraceEvaluationInputs)
    for name, value in {
        "score_dates": retrospective.score_return_dates,
        "portfolio_gross_returns": zeros.copy(),
        "benchmark_gross_returns": zeros.copy(),
        "portfolio_net_returns_20bp": zeros.copy(),
        "benchmark_net_returns_20bp": zeros.copy(),
        "portfolio_turnover_by_cause": turnover,
        "benchmark_turnover_by_cause": {
            key: value.copy() for key, value in turnover.items()
        },
        "telemetry": telemetry,
        "construction_to_fill_safety_projection_distance": zeros.copy(),
    }.items():
        object.__setattr__(provisional, name, value)
    hashes = tuple(
        sorted(
            (name, trace_adapter._array_sha256(value))
            for name, value in trace_adapter._result_arrays(provisional).items()
        )
    )
    receipt = Top2000M03RV72026TraceTelemetryReceipt(
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        runtime_setting_id=runtime_setting_id(M03R_SEED17_TOP2000_SETTING_IDS[0]),
        checkpoint_sha256=checkpoint_hash,
        checkpoint_fold_index=5,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        trace_axis_id=view.receipt.execution_axis_id,
        scored_array_sha256s=hashes,
        score_transition_start=view.receipt.local_score_transition_start,
        score_transition_stop_exclusive=(
            view.receipt.local_score_transition_stop_exclusive
        ),
        completed_transition_rows=view.receipt.executed_transition_rows,
        scored_transition_rows=rows,
        action_count=assets,
        economic_execution_receipt_sha256=view.receipt.receipt_sha256,
        economic_execution_start=view.receipt.economic_execution_start,
        global_score_transition_start=(
            retrospective.identity.score_transition_start
        ),
        global_score_transition_stop_exclusive=(
            retrospective.identity.score_transition_stop_exclusive
        ),
    )
    return Top2000M03RV72026TraceEvaluationInputs(
        score_dates=retrospective.score_return_dates,
        portfolio_gross_returns=zeros.copy(),
        benchmark_gross_returns=zeros.copy(),
        portfolio_net_returns_20bp=zeros.copy(),
        benchmark_net_returns_20bp=zeros.copy(),
        portfolio_turnover_by_cause=turnover,
        benchmark_turnover_by_cause={
            key: value.copy() for key, value in turnover.items()
        },
        telemetry=telemetry,
        construction_to_fill_safety_projection_distance=zeros.copy(),
        receipt=receipt,
    )


def _cuda_proof(transitions: int) -> Top2000M03RV72026CudaExecutionProof:
    gib = 1024**3
    return Top2000M03RV72026CudaExecutionProof(
        device="cuda:0",
        cuda_visible_device="GPU-test",
        visible_cuda_device_count=1,
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_total_memory_bytes=80 * gib,
        compute_capability=(9, 0),
        startup_allocated_bytes=2 * gib,
        startup_reserved_bytes=3 * gib,
        startup_free_bytes=70 * gib,
        peak_allocated_bytes=4 * gib,
        peak_reserved_bytes=5 * gib,
        final_allocated_bytes=2 * gib,
        final_reserved_bytes=3 * gib,
        final_free_bytes=70 * gib,
        allocator_oom_count_delta=0,
        allocator_retry_count_delta=0,
        completed_transition_rows=transitions,
    )


def _cohorts(retrospective: Any, view: Any, checkpoint_hash: str) -> Any:
    origins = len(retrospective.score_return_dates)
    entry = np.zeros(origins, dtype=np.float64)
    events = np.zeros((origins, AGE_BIN_COUNT), dtype=np.float64)
    forced_causes = (
        TurnoverCause.MEMBERSHIP_FORCED,
        TurnoverCause.AVAILABILITY_FORCED,
        TurnoverCause.RISK_FORCED,
        TurnoverCause.TERMINAL,
    )
    forced = {
        cause.value: np.zeros_like(events) for cause in forced_causes
    }
    terminal = np.zeros_like(events)
    receipt = Top2000M03RV72026CohortTrajectoryReceipt(
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        checkpoint_sha256=checkpoint_hash,
        checkpoint_fold_index=5,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        economic_execution_receipt_sha256=view.receipt.receipt_sha256,
        score_dates_sha256=cohort_adapter._sha256(
            list(retrospective.score_return_dates)
        ),
        entry_units_sha256=cohort_adapter._array_sha256("entry_units", entry),
        discretionary_event_units_by_age_sha256=cohort_adapter._array_sha256(
            "discretionary_event_units_by_age", events
        ),
        forced_censor_units_by_cause_and_age_sha256=tuple(
            (
                cause.value,
                cohort_adapter._array_sha256(
                    f"forced_censor/{cause.value}", forced[cause.value]
                ),
            )
            for cause in forced_causes
        ),
        terminal_censor_units_by_age_sha256=cohort_adapter._array_sha256(
            "terminal_censor_units_by_age", terminal
        ),
        origin_rows=origins,
    )
    return Top2000M03RV72026CohortTrajectories(
        origin_dates=retrospective.score_return_dates,
        entry_units=entry,
        discretionary_event_units_by_age=events,
        forced_censor_units_by_cause_and_age=forced,
        terminal_censor_units_by_age=terminal,
        receipt=receipt,
    )


def test_single_checkpoint_is_one_no_grad_no_reset_pass_and_round_trips(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, retrospective = _data()
    calibration = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(
        retrospective
    )
    cuda_start = execution._CudaStartup(
        device=torch.device("cuda:0"),
        cuda_visible_device="GPU-test",
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_total_memory_bytes=80 * 1024**3,
        compute_capability=(9, 0),
        startup_allocated_bytes=0,
        startup_reserved_bytes=0,
        startup_free_bytes=70 * 1024**3,
        startup_allocator_oom_count=0,
        startup_allocator_retry_count=0,
    )
    versioned = (
        ("decision_state", retrospective.sequence.decision_state),
        ("asset_returns", retrospective.sequence.asset_returns),
        ("decision_available", retrospective.sequence.decision_available),
        ("benchmark_weights", retrospective.sequence.benchmark_weights),
        ("benchmark_net_returns", retrospective.sequence.benchmark_net_returns),
        ("factor_loadings", calibration.loadings),
    )
    session = Top2000M03RV72026ExecutionSession(
        retrospective=retrospective,
        pre2026_cache=cache,
        factor_calibration=calibration,
        device=torch.device("cuda:0"),
        retrospective_cache_file_sha256=retrospective.cache_file_sha256,
        pre2026_cache_file_sha256=cache.cache_sha256,
        chronology_receipt_sha256=retrospective.identity.receipt_sha256,
        evaluation_plan_receipt_sha256=_digest("evaluation-plan"),
        execution_source_inventory_sha256=_digest("source-inventory"),
        _qualified_cuda=cuda_start,
        _tensor_versions=tuple((name, value._version) for name, value in versioned),
    )
    fold = render_top2000_m03r_v7_development_folds(1001)[5]
    policy = _Policy()
    checkpoint_receipt = Top2000M03RV72026CheckpointLoadReceipt(
        frozen_checkpoint_binding_sha256=_digest("binding"),
        training_plan_file_sha256=_digest("plan-file"),
        training_plan_receipt_sha256=_digest("plan"),
        fold_receipt_sha256=fold.receipt_sha256,
        model_file_sha256=_digest("model-file"),
        model_state_sha256=_digest("model-state"),
        setting_index=0,
        setting_id=M03R_SEED17_TOP2000_SETTING_IDS[0],
        runtime_setting_id=runtime_setting_id(M03R_SEED17_TOP2000_SETTING_IDS[0]),
        training_fold_index=5,
        checkpoint_role="headline",
    )
    loaded = SimpleNamespace(
        policy=policy,
        training_plan=SimpleNamespace(cache_sha256=cache.cache_sha256),
        training_fold=fold,
        receipt=checkpoint_receipt,
    )
    calls: list[tuple[str, bool, int]] = []

    class _Runtime:
        def canonical_pass(self, actor: Any, sequence: Any, roles: Any) -> Any:
            calls.append(("canonical", torch.is_grad_enabled(), sequence.n_positions))
            transitions = tuple(object() for _ in range(sequence.n_positions - 1))
            boundaries = tuple(
                SimpleNamespace(position_index=index)
                for index in range(sequence.n_positions)
            )
            trace = Hold30CanonicalTrace(
                boundary_states=boundaries,  # type: ignore[arg-type]
                decision_states=tuple(),
                pending_intents=tuple(),
                transitions=transitions,  # type: ignore[arg-type]
            )
            return trace, tuple()

    monkeypatch.setattr(execution, "_begin_checkpoint_cuda", lambda value: value)
    monkeypatch.setattr(
        execution,
        "load_top2000_m03r_v7_seed17_2026_checkpoint",
        lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(
        execution,
        "build_top2000_m03r_v7_validation_runtime",
        lambda *args, **kwargs: _Runtime(),
    )
    monkeypatch.setattr(
        execution,
        "model_state_sha256",
        lambda value: checkpoint_receipt.model_state_sha256,
    )
    monkeypatch.setattr(
        execution,
        "_finish_single_cuda",
        lambda value, *, completed_transition_rows: _cuda_proof(
            completed_transition_rows
        ),
    )

    def _adapt(trace: Any, data: Any, **kwargs: Any) -> Any:
        view = kwargs["economic_execution_view"]
        calls.append(("adapt", torch.is_grad_enabled(), len(trace.transitions)))
        return _inputs(data, view, checkpoint_receipt.model_file_sha256)

    monkeypatch.setattr(execution, "adapt_top2000_m03r_v7_2026_trace", _adapt)
    monkeypatch.setattr(
        execution,
        "build_top2000_m03r_v7_2026_cohort_trajectories",
        lambda trace, data, view, **kwargs: _cohorts(
            data, view, checkpoint_receipt.model_file_sha256
        ),
    )
    output = tmp_path / "setting-00-fold-05.json"
    binding = run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session(
        session,
        SimpleNamespace(),
        training_output_root=tmp_path,
        output_path=output,
    )

    assert calls[0][0:2] == ("canonical", False)
    assert calls[1][0] == "adapt"
    assert len([value for value in calls if value[0] == "canonical"]) == 1
    assert calls[0][2] == retrospective.sequence.n_positions - 93
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    assert output.stat().st_mode & 0o222 == 0

    restored = load_top2000_m03r_v7_seed17_2026_execution_artifact(
        output,
        expected_file_sha256=binding.artifact_file_sha256,
        expected_evaluation_plan_receipt_sha256=_digest("evaluation-plan"),
        expected_execution_source_inventory_sha256=_digest("source-inventory"),
    )
    assert restored.execution_receipt.receipt_sha256 == (
        binding.execution_receipt_sha256
    )
    assert restored.execution_receipt.economic_execution_start == 93
    assert restored.evaluation_inputs.score_dates == retrospective.score_return_dates
    assert restored.cohort_trajectories.origin_dates == (
        retrospective.score_return_dates
    )
    assert restored.execution_receipt.policy_model_state_sha256_before == (
        checkpoint_receipt.model_state_sha256
    )
    assert restored.execution_receipt.policy_model_state_sha256_after == (
        checkpoint_receipt.model_state_sha256
    )
    assert restored.execution_receipt.elapsed_wall_seconds > 0.0
    assert not restored.execution_receipt.policy_state_changed
    with pytest.raises(FileExistsError):
        run_top2000_m03r_v7_seed17_2026_single_checkpoint_from_session(
            session,
            SimpleNamespace(),
            training_output_root=tmp_path,
            output_path=output,
        )


def test_artifact_loader_requires_plan_source_and_exact_file_hash(
    tmp_path: Any,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(execution.Top2000M03RV72026ExecutionError, match="SHA-256"):
        load_top2000_m03r_v7_seed17_2026_execution_artifact(
            path,
            expected_file_sha256=_digest("wrong"),
            expected_evaluation_plan_receipt_sha256=_digest("plan"),
            expected_execution_source_inventory_sha256=_digest("source"),
        )
    with pytest.raises(execution.Top2000M03RV72026ExecutionError, match="inventory"):
        load_top2000_m03r_v7_seed17_2026_execution_artifact(
            path,
            expected_file_sha256=digest,
            expected_evaluation_plan_receipt_sha256=_digest("plan"),
            expected_execution_source_inventory_sha256=_digest("source"),
        )


def test_cuda_startup_rejects_multi_process_and_multi_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(
        execution.Top2000M03RV72026ExecutionError,
        match="WORLD_SIZE",
    ):
        execution._start_single_cuda("cuda:0")

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(
        execution.Top2000M03RV72026ExecutionError,
        match="exactly one visible CUDA",
    ):
        execution._start_single_cuda("cuda:0")


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_name": "NVIDIA A100-SXM4-40GB"},
        {"gpu_total_memory_bytes": 40 * 1024**3},
        {"compute_capability": (8, 0)},
    ],
)
def test_cuda_proof_rejects_non_h100_execution_shape(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(
        execution.Top2000M03RV72026ExecutionError,
        match="one clean inference process/GPU",
    ):
        replace(_cuda_proof(377), **changes)


def test_cuda_startup_rejects_non_h100_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            total_memory=40 * 1024**3,
            major=8,
            minor=0,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (35 * 1024**3, 40 * 1024**3),
    )
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda _device: {})
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA A100-SXM4-40GB",
    )
    with pytest.raises(
        execution.Top2000M03RV72026ExecutionError,
        match="exact H100 80GB",
    ):
        execution._start_single_cuda("cuda:0")
