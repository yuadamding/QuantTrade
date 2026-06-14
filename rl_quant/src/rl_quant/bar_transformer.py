"""Interval-neutral aliases for the causal transformer allocator."""

from rl_quant.hourly_transformer import (  # noqa: F401
    CausalTransformerQNetwork,
    CudaVramReservation,
    HourlyDataSplit as BarDataSplit,
    HourlyEnvConfig as BarEnvConfig,
    HourlyEvaluationResult as BarEvaluationResult,
    HourlyTransformerTrainingConfig as BarTransformerTrainingConfig,
    VectorizedHourlyAllocationEnv as VectorizedBarAllocationEnv,
    action_index,
    build_hourly_splits as build_bar_splits,
    evaluate_hourly_policy as evaluate_bar_policy,
    train_hourly_transformer_dqn as train_bar_transformer_dqn,
)
