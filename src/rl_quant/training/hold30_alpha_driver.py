"""Non-authorizing end-to-end synthetic driver for Hold-30 alpha v3.

The production experiment is deliberately not executable while its scientific
training plan is unresolved.  This module closes a narrower software gap: it
connects the registered daily policy, delayed-fill action/runtime, cohort
accounting, v3 objective, optimizer, validation telemetry, and content-bound
checkpoint artifacts on a deterministic synthetic chronology.

The synthetic path is impossible to mistake for production.  Its entry point
contains ``qualification`` in the name, every receipt records
``qualification_only=true`` and ``launch_authorized=false``, and the fixture
objective configs are rejected by the executable-plan preflight.  A06 uses
the receipt-bound, disjoint two-optimizer contract; it does not waive any
production data, image, distributed-parity, or launch-approval requirement.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, Final

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_quant.envs.hold30 import CohortLedger, TurnoverCause
from rl_quant.execution.hold30_sleeves import Hold30SleeveSnapshot
from rl_quant.models.daily_policy import (
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    Hold30Intent,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
    resolve_hold30_alpha_setting,
)
from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS, sha256_payload
from rl_quant.training.hold30 import Hold30LossContract
from rl_quant.training.hold30_alpha import (
    HOLD30_ALPHA_ANNUALIZATION,
    Hold30A06OptimizerSpecReceipt,
    Hold30A06OptimizerStateReceipt,
    Hold30AlphaBatch,
    Hold30AlphaGlobalMetrics,
    Hold30AlphaObjectiveConfig,
    Hold30AlphaTrainingError,
    Hold30AlphaValidationMetrics,
    bind_hold30_alpha_global_moments,
    build_hold30_a06_optimizer_spec_receipt,
    build_hold30_a06_optimizer_state_receipt,
    derive_hold30_a06_overlay_coefficients,
    derive_hold30_alpha_coefficients,
    drawdown_detached_log_weights,
    hold30_a06_overlay_surrogate,
    hold30_a06_overlay_two_pass_objective,
    hold30_alpha_evaluation_point_id,
    hold30_alpha_surrogate,
    hold30_alpha_two_pass_objective,
    partition_hold30_a06_parameters,
    train_hold30_a06_two_optimizer_update,
    train_hold30_alpha_two_pass_update,
)
from rl_quant.training.hold30_alpha_pilot_plan import (
    HOLD30_ALPHA_PILOT_PROFILE,
    build_hold30_alpha_pilot_training_plan,
    qualify_hold30_alpha_pilot_training_plan,
)
from rl_quant.training.hold30_alpha_plan import (
    Hold30AlphaTrainingPlan,
    Hold30AlphaTrainingPlanError,
)
from rl_quant.training.hold30_runtime import (
    Hold30ChronologicalRuntime,
    Hold30RuntimeState,
    Hold30Sequence,
    Hold30Transition,
)

HOLD30_ALPHA_SYNTHETIC_DRIVER_SCHEMA: Final[str] = (
    "rl-quant.hold30-alpha-v3.synthetic-driver-v1"
)
HOLD30_ALPHA_SYNTHETIC_METRICS_SCHEMA: Final[str] = (
    "rl-quant.hold30-alpha-v3.synthetic-metrics-v1"
)
HOLD30_ALPHA_SYNTHETIC_CHECKPOINT_SCHEMA: Final[str] = (
    "rl-quant.hold30-alpha-v3.synthetic-checkpoint-v1"
)
HOLD30_ALPHA_CPU_DISTRIBUTED_PARITY_SCHEMA: Final[str] = (
    "rl-quant.hold30-alpha-v3.full-policy-cpu-two-rank-parity-v1"
)
HOLD30_ALPHA_CPU_RESTART_PARITY_SCHEMA: Final[str] = (
    "rl-quant.hold30-alpha-v3.full-policy-cpu-restart-parity-v1"
)
HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS: Final[tuple[str, ...]] = (
    HOLD30_ALPHA_MECH8_IDS
)
# Sixty-three scored rows are the smallest chronology that actually exercises
# every frozen 5/21/30/63-day auxiliary target rather than merely routing past
# an empty branch.
HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS: Final[int] = 64
HOLD30_ALPHA_CPU_QUALIFICATION_LOCAL_PATHS_PER_RANK: Final[int] = 1
HOLD30_ALPHA_SYNTHETIC_UPDATES: Final[int] = 1
HOLD30_ALPHA_SYNTHETIC_POSITIONS: Final[int] = 64
HOLD30_ALPHA_SYNTHETIC_ASSETS: Final[int] = 6
HOLD30_ALPHA_SYNTHETIC_BATCH: Final[int] = 1
HOLD30_ALPHA_SYNTHETIC_AXIS_ID: Final[str] = hashlib.sha256(
    b"hold30-alpha-v3-synthetic-axis-v1"
).hexdigest()
HOLD30_ALPHA_SYNTHETIC_A06_BINDING: Final[str] = hashlib.sha256(
    b"qualification-only-a06-provisional-optimizer-spec-binding"
).hexdigest()
HOLD30_ALPHA_SYNTHETIC_A06_SHARPE_EPSILON: Final[float] = (
    HOLD30_ALPHA_PILOT_PROFILE.total_sharpe_epsilon
)
HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "dataset_sequence_receipt_sha256",
    "daily_policy_state_provider_binding_sha256",
    "evaluation_panel_id",
    "data_binding_receipt_id",
    "residual_labels_id",
    "training_objective_inputs_id",
    "inner_validation_objective_inputs_id",
    "fold_assignments_sha256",
)
HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS: Final[tuple[str, ...]] = (
    "typed_real_data_qualification_receipt_binding_unimplemented",
    "real_data_global_path_identity_binding_unimplemented",
)


class Hold30AlphaDriverError(RuntimeError):
    """A synthetic run or its artifacts violate the v3 driver contract."""


def _optional_digest(name: str, value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30AlphaDriverError(f"{name} must be a lowercase SHA-256 digest")


def _require_digest(name: str, value: str) -> None:
    _optional_digest(name, value)
    if value is None:
        raise Hold30AlphaDriverError(f"{name} is required")


@dataclass(frozen=True, slots=True)
class Hold30AlphaCpuDistributedParityReceipt:
    """Typed CPU-only evidence for full-policy one-rank/two-rank parity."""

    setting_ids: tuple[str, ...]
    seed: int
    positions: int
    setting_evidence_sha256: tuple[tuple[str, str], ...]
    world_size: int = 2
    local_paths_per_rank: int = HOLD30_ALPHA_CPU_QUALIFICATION_LOCAL_PATHS_PER_RANK
    one_rank_reference: str = "serial-batch1-canonical-path0-then-path1"
    backend: str = "gloo"
    device: str = "cpu"
    process_start_method: str = "spawn"
    gradient_reduction: str = "SUM"
    objective_absolute_tolerance: float = 1e-12
    exact_gradient_parity: bool = True
    exact_parameter_parity: bool = True
    exact_optimizer_state_parity: bool = True
    qualification_only: bool = True
    gpu_consumed: bool = False
    h100_parity_claimed: bool = False
    launch_authorized: bool = False
    schema: str = HOLD30_ALPHA_CPU_DISTRIBUTED_PARITY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HOLD30_ALPHA_CPU_DISTRIBUTED_PARITY_SCHEMA
            or self.setting_ids != HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS
            or self.seed not in HOLD30_SEEDS
            or self.positions != HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS
            or self.world_size != 2
            or self.local_paths_per_rank
            != HOLD30_ALPHA_CPU_QUALIFICATION_LOCAL_PATHS_PER_RANK
            or self.one_rank_reference
            != "serial-batch1-canonical-path0-then-path1"
            or self.backend != "gloo"
            or self.device != "cpu"
            or self.process_start_method != "spawn"
            or self.gradient_reduction != "SUM"
            or self.objective_absolute_tolerance != 1e-12
            or not (
                self.exact_gradient_parity
                and self.exact_parameter_parity
                and self.exact_optimizer_state_parity
                and self.qualification_only
            )
            or self.gpu_consumed
            or self.h100_parity_claimed
            or self.launch_authorized
            or tuple(name for name, _digest in self.setting_evidence_sha256)
            != self.setting_ids
        ):
            raise Hold30AlphaDriverError("invalid CPU distributed-parity receipt")
        for setting_id, digest in self.setting_evidence_sha256:
            if setting_id not in self.setting_ids:
                raise Hold30AlphaDriverError("unknown parity receipt setting")
            _require_digest("setting_evidence_sha256", digest)

    def manifest_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_id(self) -> str:
        return sha256_payload(self.manifest_payload())


@dataclass(frozen=True, slots=True)
class Hold30AlphaCpuRestartParityReceipt:
    """Typed CPU-only evidence for exact two-update save/reload parity."""

    setting_ids: tuple[str, ...]
    seed: int
    positions: int
    setting_evidence_sha256: tuple[tuple[str, str], ...]
    updates: int = 2
    checkpoint_update: int = 1
    serialization: str = "torch-save/weights-only-load"
    exact_model_state_parity: bool = True
    exact_optimizer_state_parity: bool = True
    exact_update_receipt_parity: bool = True
    qualification_only: bool = True
    gpu_consumed: bool = False
    h100_parity_claimed: bool = False
    launch_authorized: bool = False
    schema: str = HOLD30_ALPHA_CPU_RESTART_PARITY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HOLD30_ALPHA_CPU_RESTART_PARITY_SCHEMA
            or self.setting_ids != HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS
            or self.seed not in HOLD30_SEEDS
            or self.positions != HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS
            or self.updates != 2
            or self.checkpoint_update != 1
            or self.serialization != "torch-save/weights-only-load"
            or not (
                self.exact_model_state_parity
                and self.exact_optimizer_state_parity
                and self.exact_update_receipt_parity
                and self.qualification_only
            )
            or self.gpu_consumed
            or self.h100_parity_claimed
            or self.launch_authorized
            or tuple(name for name, _digest in self.setting_evidence_sha256)
            != self.setting_ids
        ):
            raise Hold30AlphaDriverError("invalid CPU restart-parity receipt")
        for setting_id, digest in self.setting_evidence_sha256:
            if setting_id not in self.setting_ids:
                raise Hold30AlphaDriverError("unknown restart receipt setting")
            _require_digest("setting_evidence_sha256", digest)

    def manifest_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_id(self) -> str:
        return sha256_payload(self.manifest_payload())


@dataclass(frozen=True, slots=True)
class Hold30AlphaProductionPreflightBindings:
    """Non-authorizing receipt inventory required after scientific planning.

    These are content identities, not data tensors.  Passing this preflight
    still does not grant permission to render or launch a Kubernetes Job.
    """

    dataset_sequence_receipt_sha256: str | None = None
    daily_policy_state_provider_binding_sha256: str | None = None
    evaluation_panel_id: str | None = None
    data_binding_receipt_id: str | None = None
    residual_labels_id: str | None = None
    training_objective_inputs_id: str | None = None
    inner_validation_objective_inputs_id: str | None = None
    fold_assignments_sha256: str | None = None
    container_image_digest: str | None = None
    ddp_pass_b_gradient_parity_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS:
            _optional_digest(name, getattr(self, name))
        _optional_digest(
            "ddp_pass_b_gradient_parity_receipt_sha256",
            self.ddp_pass_b_gradient_parity_receipt_sha256,
        )
        if self.container_image_digest is not None:
            if not self.container_image_digest.startswith("sha256:"):
                raise Hold30AlphaDriverError(
                    "container_image_digest must be an immutable sha256 digest"
                )
            _optional_digest(
                "container_image_digest",
                self.container_image_digest.removeprefix("sha256:"),
            )

    @property
    def missing_fields(self) -> tuple[str, ...]:
        result = [
            name
            for name in HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS
            if getattr(self, name) is None
        ]
        if self.container_image_digest is None:
            result.append("container_image_digest")
        if self.ddp_pass_b_gradient_parity_receipt_sha256 is None:
            result.append("ddp_pass_b_gradient_parity_receipt_sha256")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class Hold30AlphaSyntheticRoute:
    """Exact model/runtime/objective route for one registered v3 setting."""

    setting_id: str
    mechanism: str
    objective_kind: str
    supervised_alpha: bool
    uncertainty: bool
    separate_overlay: bool
    direct_sharpe: bool
    runnable_in_synthetic_driver: bool
    blocker: str | None


def resolve_hold30_alpha_synthetic_route(
    setting_id: str,
) -> Hold30AlphaSyntheticRoute:
    """Resolve all eight IDs without aliases or generation crossover."""

    setting = resolve_hold30_alpha_setting(setting_id)
    separate_overlay = setting.sharpe_mode == "separate-total-risk-overlay"
    return Hold30AlphaSyntheticRoute(
        setting_id=setting.setting_id,
        mechanism="H0" if setting.mechanism == "legacy-scalar-gate" else "H2",
        objective_kind=(
            "absolute-net-log-return"
            if setting.objective_mode == "absolute-net-log-return"
            else "v3-global-two-pass"
        ),
        supervised_alpha=setting.supervised_residual_alpha_heads,
        uncertainty=setting.uncertainty_downside_heads,
        separate_overlay=separate_overlay,
        direct_sharpe=setting.sharpe_mode == "direct-two-pass-gradient",
        runnable_in_synthetic_driver=True,
        blocker=None,
    )


def require_hold30_alpha_executable_plan(
    plan: Hold30AlphaTrainingPlan,
    bindings: Hold30AlphaProductionPreflightBindings | None = None,
) -> None:
    """Fail closed before production can allocate resources or write output.

    The typed plan is the only accepted executable scientific configuration.
    Qualification-only math configs, missing thresholds, and missing decision
    receipts all fail through the authoritative validators.
    """

    if not isinstance(plan, Hold30AlphaTrainingPlan):
        raise Hold30AlphaDriverError(
            "production preflight requires a typed Hold30AlphaTrainingPlan"
        )
    blockers: list[str] = []
    try:
        qualification = qualify_hold30_alpha_pilot_training_plan(plan)
    except ValueError as exc:
        blockers.append(f"pilot-training-plan: {exc}")
    else:
        blockers.extend(qualification.remaining_implementation_blockers)
    try:
        plan.require_resolved()
    except Hold30AlphaTrainingPlanError as exc:
        blockers.append(f"scientific-plan: {exc}")
    blockers.extend(HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS)
    inventory = (
        Hold30AlphaProductionPreflightBindings() if bindings is None else bindings
    )
    if not isinstance(inventory, Hold30AlphaProductionPreflightBindings):
        raise Hold30AlphaDriverError(
            "production preflight requires typed receipt bindings"
        )
    blockers.extend(f"missing-binding:{name}" for name in inventory.missing_fields)
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        raise Hold30AlphaDriverError(
            "v3 production preflight remains non-executable: " + "; ".join(blockers)
        )


def _synthetic_pilot_plan(
    a06_optimizer_spec_receipt_sha256: str = HOLD30_ALPHA_SYNTHETIC_A06_BINDING,
) -> Hold30AlphaTrainingPlan:
    """Build the exact pilot profile with an explicit A06 receipt binding."""

    return build_hold30_alpha_pilot_training_plan(
        a06_optimizer_spec_receipt_sha256=(
            a06_optimizer_spec_receipt_sha256
        ),
    )


def build_hold30_alpha_synthetic_objective_config(
    setting_id: str,
    *,
    a06_optimizer_spec_receipt_sha256: str = (
        HOLD30_ALPHA_SYNTHETIC_A06_BINDING
    ),
) -> Hold30AlphaObjectiveConfig | None:
    """Return explicit non-scientific coefficients for software qualification.

    M00/M01 use the legacy absolute objective contract and therefore return
    ``None``.  The A06 row is bound to the caller-supplied typed pre-update
    optimizer receipt; the default digest is only a provisional fixture used
    to construct the model before the concrete optimizers exist.
    """

    route = resolve_hold30_alpha_synthetic_route(setting_id)
    if route.objective_kind == "absolute-net-log-return":
        return None
    plan = _synthetic_pilot_plan(a06_optimizer_spec_receipt_sha256)
    config = next(
        config
        for config in plan.objective_configs
        if config.setting_id == route.setting_id
    )
    return replace(
        config,
        qualification_math_test_only=True,
        total_sharpe_epsilon=(
            HOLD30_ALPHA_SYNTHETIC_A06_SHARPE_EPSILON
            if route.separate_overlay
            else config.total_sharpe_epsilon
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synthetic_policy_config(
    route: Hold30AlphaSyntheticRoute,
    objective: Hold30AlphaObjectiveConfig | None,
) -> DailyCrossSectionConfig:
    return DailyCrossSectionConfig(
        context_dim=4,
        bar_feature_dim=5,
        raw_policy_dim=8,
        raw_policy_layers=1,
        raw_policy_heads=1,
        raw_block_seconds=2,
        session_seconds=4,
        news_raw_dim=1,
        max_news=1,
        news_embed_dim=4,
        token_dim=8,
        temporal_layers=1,
        temporal_heads=1,
        daily_lookback=63,
        max_days=64,
        alloc_layers=1,
        alloc_heads=1,
        feedforward_dim=16,
        dropout=0.0,
        temperature=1.0,
        max_stock_weight=0.01,
        grad_checkpoint=False,
        raw_norm="level",
        raw_recent_days=0,
        raw_stock_chunk=0,
        hold30_setting=route.setting_id,
        alpha_downside_penalty_kappa=(
            None if objective is None else objective.downside_penalty_kappa
        ),
        alpha_active_log_scale_bounds=(
            None if objective is None else objective.active_log_scale_bounds
        ),
        alpha_uncertainty_log_scale_bounds=(
            None if objective is None else objective.uncertainty_log_scale_bounds
        ),
    )


def _synthetic_sequence(
    *,
    cost_rate: float = 0.002,
    positions: int = HOLD30_ALPHA_SYNTHETIC_POSITIONS,
    batch: int = HOLD30_ALPHA_SYNTHETIC_BATCH,
) -> Hold30Sequence:
    if (
        isinstance(positions, bool)
        or not isinstance(positions, int)
        or positions < 3
        or isinstance(batch, bool)
        or not isinstance(batch, int)
        or batch < 1
    ):
        raise Hold30AlphaDriverError(
            "synthetic positions/batch must be integers with positions>=3 and batch>=1"
        )
    assets = HOLD30_ALPHA_SYNTHETIC_ASSETS
    width = 8
    dtype = torch.float32
    position = torch.arange(positions, dtype=dtype).view(positions, 1, 1, 1)
    path = torch.arange(batch, dtype=dtype).view(1, batch, 1, 1)
    asset = torch.arange(assets, dtype=dtype).view(1, 1, assets, 1)
    channel = torch.arange(width, dtype=dtype).view(1, 1, 1, width)
    decision_state = torch.sin(
        position * 0.071
        + path * 0.137
        + asset * 0.19
        + channel * 0.11
        + path * asset * 0.017
    )

    return_position = torch.arange(positions - 1, dtype=dtype).view(-1, 1, 1)
    return_path = torch.arange(batch, dtype=dtype).view(1, batch, 1)
    return_asset = torch.arange(assets, dtype=dtype).view(1, 1, -1)
    asset_returns = 0.0007 * torch.sin(
        return_position * 0.17
        + return_path * 0.31
        + return_asset * 0.83
        + return_path * return_asset * 0.023
    )
    # CASH earns the synthetic point-in-time risk-free return.
    asset_returns[..., 0] = 0.00005 + 0.00001 * torch.cos(
        return_position[:, 0, 0].view(-1, 1) * 0.13
        + torch.arange(batch, dtype=dtype).view(1, -1) * 0.29
    )

    initial_weights = torch.full((batch, assets), 0.01, dtype=dtype)
    if batch > 1:
        path_tilt = torch.arange(batch, dtype=dtype).view(-1, 1) * 0.0002
        initial_weights[:, 1:] += path_tilt
    initial_weights[:, 0] = 1.0 - 0.01 * (assets - 1)
    initial_weights[:, 0] = 1.0 - initial_weights[:, 1:].sum(-1)
    masks = torch.ones((positions, batch, assets), dtype=torch.bool)
    benchmark_weights = initial_weights.unsqueeze(0).expand(positions, -1, -1).clone()
    benchmark_net_returns = (benchmark_weights[:-1] * asset_returns).sum(-1)
    risk_asset_caps = torch.ones_like(benchmark_weights)
    risk_asset_caps[..., 1:] = 0.01
    risk_gross_max = torch.ones((positions, batch), dtype=dtype)
    return Hold30Sequence(
        decision_state=decision_state,
        asset_returns=asset_returns,
        decision_available=masks.clone(),
        fill_membership=masks.clone(),
        fill_availability=masks.clone(),
        benchmark_weights=benchmark_weights,
        risk_asset_caps=risk_asset_caps,
        risk_gross_max=risk_gross_max,
        benchmark_net_returns=benchmark_net_returns,
        initial_ledger=CohortLedger.from_weights(
            initial_weights,
            cash_index=0,
            initial_age=30,
            track_initial_units=True,
        ),
        cost_rate=cost_rate,
        axis_id=HOLD30_ALPHA_SYNTHETIC_AXIS_ID,
    )


def _fixture_alpha_targets(
    sequence: Hold30Sequence,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = sequence.asset_returns
    benchmark = sequence.benchmark_net_returns
    dates, batch, assets = returns.shape
    targets = torch.zeros(
        (dates, batch, assets, len(HOLD30_ALPHA_HORIZONS)),
        dtype=returns.dtype,
        device=returns.device,
    )
    valid = torch.zeros_like(targets, dtype=torch.bool)
    for index, horizon in enumerate(HOLD30_ALPHA_HORIZONS):
        for origin in range(dates):
            stop = origin + horizon
            if stop > dates:
                continue
            stock_log = torch.log1p(returns[origin:stop]).sum(0)
            benchmark_log = torch.log1p(benchmark[origin:stop]).sum(0)
            targets[origin, ..., index] = stock_log - benchmark_log.unsqueeze(-1)
            valid[origin, :, 1:, index] = True
    return targets, valid


def _select_synthetic_paths(
    sequence: Hold30Sequence,
    path_indices: torch.Tensor,
) -> Hold30Sequence:
    """Select immutable batch paths without regenerating their identities."""

    if (
        not isinstance(path_indices, torch.Tensor)
        or path_indices.dtype != torch.int64
        or path_indices.ndim != 1
        or path_indices.numel() < 1
        or bool((path_indices < 0).any())
        or bool((path_indices >= sequence.batch_size).any())
    ):
        raise Hold30AlphaDriverError("invalid synthetic path selection")

    def select_position(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(1, path_indices.to(device=value.device))

    ledger = sequence.initial_ledger
    ledger_indices = path_indices.to(device=ledger.economic_value.device)
    return replace(
        sequence,
        decision_state=select_position(sequence.decision_state),
        asset_returns=select_position(sequence.asset_returns),
        decision_available=select_position(sequence.decision_available),
        fill_membership=select_position(sequence.fill_membership),
        fill_availability=select_position(sequence.fill_availability),
        benchmark_weights=select_position(sequence.benchmark_weights),
        risk_asset_caps=select_position(sequence.risk_asset_caps),
        risk_gross_max=select_position(sequence.risk_gross_max),
        benchmark_net_returns=select_position(sequence.benchmark_net_returns),
        initial_ledger=CohortLedger(
            economic_value=ledger.economic_value.index_select(0, ledger_indices),
            retention_units=ledger.retention_units.index_select(0, ledger_indices),
            cash_index=ledger.cash_index,
        ),
        initial_equity=(
            None
            if sequence.initial_equity is None
            else sequence.initial_equity.index_select(
                0,
                path_indices.to(device=sequence.initial_equity.device),
            )
        ),
    )


def _slice_synthetic_chronology(
    sequence: Hold30Sequence,
    *,
    start: int,
    positions: int,
) -> Hold30Sequence:
    """Take one contiguous scored chunk while retaining the global axis ID."""

    stop = start + positions
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start < 0
        or isinstance(positions, bool)
        or not isinstance(positions, int)
        or positions < 2
        or stop > sequence.n_positions
    ):
        raise Hold30AlphaDriverError("invalid synthetic chronology slice")
    return replace(
        sequence,
        decision_state=sequence.decision_state[start:stop].clone(),
        asset_returns=sequence.asset_returns[start : stop - 1].clone(),
        decision_available=sequence.decision_available[start:stop].clone(),
        fill_membership=sequence.fill_membership[start:stop].clone(),
        fill_availability=sequence.fill_availability[start:stop].clone(),
        benchmark_weights=sequence.benchmark_weights[start:stop].clone(),
        risk_asset_caps=sequence.risk_asset_caps[start:stop].clone(),
        risk_gross_max=sequence.risk_gross_max[start:stop].clone(),
        benchmark_net_returns=sequence.benchmark_net_returns[
            start : stop - 1
        ].clone(),
        track_entry_units=(
            None
            if sequence.track_entry_units is None
            else sequence.track_entry_units[start : stop - 1].clone()
        ),
    )


def _next_stochastic_restart_chunk(
    sequence: Hold30Sequence,
    state: _CpuFullPolicyTrainingState,
) -> Hold30Sequence:
    """Sample a restart-sensitive continuation without changing economics."""

    chunk = _slice_synthetic_chronology(
        sequence,
        start=state.sampler_cursor,
        positions=HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS,
    )
    # Each persisted RNG family participates in the next actor observation.
    # Omitting any one from a checkpoint therefore breaks the continuation.
    jitter_scale = 1e-3 * (
        random.random()
        + float(np.random.random())
        + float(torch.rand(()))
        + float(torch.rand((), generator=state.sampler_generator))
    )
    pattern = torch.cos(
        torch.arange(
            chunk.decision_state.numel(),
            dtype=chunk.decision_state.dtype,
            device=chunk.decision_state.device,
        ).reshape_as(chunk.decision_state)
        * 0.017
    )
    return replace(
        chunk,
        decision_state=chunk.decision_state + jitter_scale * pattern,
    )


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _semantic_torch_sha256(value: Any) -> str:
    """Hash tensor-bearing optimizer/checkpoint objects independent of pickle."""

    def materialize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes_sha256": hashlib.sha256(
                    tensor.numpy().tobytes()
                ).hexdigest(),
            }
        if isinstance(item, Mapping):
            return [
                {
                    "key_type": type(key).__qualname__,
                    "key": repr(key),
                    "value": materialize(candidate),
                }
                for key, candidate in sorted(
                    item.items(),
                    key=lambda row: (type(row[0]).__qualname__, repr(row[0])),
                )
            ]
        if isinstance(item, (list, tuple)):
            return [materialize(candidate) for candidate in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise Hold30AlphaDriverError(
            f"unsupported semantic checkpoint value {type(item).__qualname__}"
        )

    return sha256_payload(materialize(value))


def _core_tensor_exact_sha256(value: torch.Tensor) -> str:
    """Match the canonical tensor hash used by the v3 objective runtime."""

    material = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(material.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(material.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(material.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _core_named_parameter_sha256(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    gradients: bool,
) -> str:
    return sha256_payload(
        [
            {
                "name": name,
                "sha256": (
                    None
                    if (value := parameter.grad if gradients else parameter) is None
                    else _core_tensor_exact_sha256(value)
                ),
            }
            for name, parameter in named_parameters
        ]
    )


def _canonical_path_gradient_sum(
    objectives: Sequence[torch.Tensor],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    """Assign the exact rank0-gradient + rank1-gradient binary SUM."""

    parameters = tuple(parameter for _name, parameter in named_parameters)
    if len(objectives) != 2 or not parameters:
        raise Hold30AlphaDriverError(
            "canonical gradient SUM requires two paths and trainable parameters"
        )
    accumulated: list[torch.Tensor | None] = [None] * len(parameters)
    for objective in objectives:
        gradients = torch.autograd.grad(
            -objective,
            parameters,
            allow_unused=True,
        )
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            accumulated[index] = (
                gradient.detach().clone()
                if accumulated[index] is None
                else accumulated[index] + gradient.detach()
            )
    for parameter, gradient in zip(parameters, accumulated, strict=True):
        parameter.grad = gradient


def _canonical_path_objective_sum(objectives: Sequence[torch.Tensor]) -> float:
    """Match rank-ordered float32-to-float64 objective reduction exactly."""

    if len(objectives) != 2:
        raise Hold30AlphaDriverError(
            "canonical objective SUM requires exactly two paths"
        )
    total = objectives[0].detach().to(device="cpu", dtype=torch.float64)
    total = total + objectives[1].detach().to(device="cpu", dtype=torch.float64)
    return float(total)


def _evaluation_point_id(policy: torch.nn.Module) -> str:
    return hold30_alpha_evaluation_point_id(policy)


def _trace_batch(
    transitions: Sequence[Hold30Transition],
    sequence: Hold30Sequence,
    route: Hold30AlphaSyntheticRoute,
    *,
    evaluation_point_id: str,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    stream_id: str = "primary",
    global_path_ids: torch.Tensor | None = None,
    origin_offset: int = 0,
    detach_inputs: bool = False,
) -> Hold30AlphaBatch:
    policy_return = torch.stack([row.net_return for row in transitions])
    benchmark = torch.stack([row.benchmark_net_return for row in transitions])
    turnover = torch.stack(
        [row.discretionary_accounting.turnover for row in transitions]
    )
    early = torch.stack(
        [row.discretionary_accounting.early_exit_notional for row in transitions]
    )
    dates, batch = policy_return.shape
    risk_free = sequence.asset_returns[:, :, sequence.cash_index]
    # A distinct, varying PIT market series makes beta numerically defined.
    phase = torch.arange(dates, dtype=policy_return.dtype).view(-1, 1)
    market = benchmark + 0.0004 * torch.sin(phase * 0.23 + 0.2)
    if (
        isinstance(origin_offset, bool)
        or not isinstance(origin_offset, int)
        or origin_offset < 0
    ):
        raise Hold30AlphaDriverError("origin_offset must be a nonnegative integer")
    origins = (
        torch.arange(dates, dtype=torch.int64).add(origin_offset).view(-1, 1)
    ).expand(dates, batch)
    path_ids = (
        torch.arange(batch, dtype=torch.int64)
        if global_path_ids is None
        else global_path_ids
    )
    if (
        not isinstance(path_ids, torch.Tensor)
        or path_ids.dtype != torch.int64
        or tuple(path_ids.shape) != (batch,)
        or len(set(path_ids.tolist())) != batch
        or bool((path_ids < 0).any())
    ):
        raise Hold30AlphaDriverError(
            "synthetic global_path_ids must be unique nonnegative int64 [batch]"
        )
    global_paths = path_ids.view(1, -1).expand(dates, batch)
    auxiliary_prediction: torch.Tensor | None = None
    downside: torch.Tensor | None = None
    auxiliary_target: torch.Tensor | None = None
    auxiliary_valid: torch.Tensor | None = None
    if route.supervised_alpha:
        auxiliary_prediction = torch.stack(
            [
                row.raw_intent.auxiliary_alpha_mean
                for row in transitions
                if row.raw_intent.auxiliary_alpha_mean is not None
            ]
        )
        if auxiliary_prediction.shape[0] != dates:
            raise Hold30AlphaDriverError(
                "alpha trace omitted an auxiliary prediction row"
            )
        auxiliary_target = targets
        auxiliary_valid = target_valid
        if route.uncertainty:
            downside = torch.stack(
                [
                    row.raw_intent.alpha_downside_30d
                    for row in transitions
                    if row.raw_intent.alpha_downside_30d is not None
                ]
            )
            if downside.shape[0] != dates:
                raise Hold30AlphaDriverError("uncertainty trace omitted a downside row")
    if detach_inputs:
        policy_return = policy_return.detach()
        turnover = turnover.detach()
        early = early.detach()
        auxiliary_prediction = (
            None if auxiliary_prediction is None else auxiliary_prediction.detach()
        )
        downside = None if downside is None else downside.detach()
    source_id = hashlib.sha256(
        f"{sequence.axis_id}:{route.setting_id}".encode()
    ).hexdigest()
    return Hold30AlphaBatch(
        binding_kind="qualification-math-fixture",
        source_axis_id=sequence.axis_id,
        objective_inputs_id=source_id,
        role="qualification-math-fixture",
        stream_id=stream_id,
        origin_row_ids=origins.reshape(-1),
        global_path_ids=global_paths.reshape(-1),
        evaluation_point_id=evaluation_point_id,
        policy_net_return=policy_return.reshape(-1),
        benchmark_net_return=benchmark.reshape(-1).detach(),
        market_return=market.reshape(-1).detach(),
        risk_free_return=risk_free.reshape(-1).detach(),
        discretionary_turnover=turnover.reshape(-1),
        early_exit_mass=early.reshape(-1),
        valid=torch.ones(dates * batch, dtype=torch.bool),
        auxiliary_prediction=(
            None
            if auxiliary_prediction is None
            else auxiliary_prediction.reshape(
                dates * batch,
                sequence.num_assets,
                len(HOLD30_ALPHA_HORIZONS),
            )
        ),
        auxiliary_target=(
            None
            if auxiliary_target is None
            else auxiliary_target.reshape(
                dates * batch,
                sequence.num_assets,
                len(HOLD30_ALPHA_HORIZONS),
            ).detach()
        ),
        auxiliary_valid=(
            None
            if auxiliary_valid is None
            else auxiliary_valid.reshape(
                dates * batch,
                sequence.num_assets,
                len(HOLD30_ALPHA_HORIZONS),
            )
        ),
        downside_30d=(
            None
            if downside is None
            else downside.reshape(dates * batch, sequence.num_assets)
        ),
    )


def _absolute_contract(route: Hold30AlphaSyntheticRoute) -> Hold30LossContract:
    if route.setting_id == "hold30a-m00-legacy-absolute":
        return Hold30LossContract(
            "H0",
            lambda_turn=0.0,
            lambda_early=0.0,
            gate_entropy_coef=1e-5,
            gate_budget_coef=1e-3,
        )
    if route.setting_id == "hold30a-m01-persistent-absolute":
        return Hold30LossContract(
            "H2",
            lambda_turn=1.0,
            lambda_early=0.002,
            gate_entropy_coef=0.0,
            gate_budget_coef=0.0,
        )
    raise Hold30AlphaDriverError("absolute contract requested for an active setting")


def _absolute_objective(
    pass_a: Sequence[Hold30Transition],
    pass_b: Sequence[Hold30Transition],
    contract: Hold30LossContract,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(pass_a) != len(pass_b) or not pass_a:
        raise Hold30AlphaDriverError("absolute Pass A/B traces do not align")
    for left, right in zip(pass_a, pass_b, strict=True):
        for name in ("net_return", "benchmark_net_return"):
            if not torch.equal(
                getattr(left, name).detach().cpu(),
                getattr(right, name).detach().cpu(),
            ):
                raise Hold30AlphaDriverError(
                    f"absolute Pass A/B {name} differs at one evaluation point"
                )
        for name in ("turnover", "early_exit_notional"):
            left_value = getattr(left.discretionary_accounting, name)
            right_value = getattr(right.discretionary_accounting, name)
            if not torch.equal(
                left_value.detach().cpu(), right_value.detach().cpu()
            ):
                raise Hold30AlphaDriverError(
                    f"absolute Pass A/B {name} differs at one evaluation point"
                )
        left_gate = left.raw_intent.gate
        right_gate = right.raw_intent.gate
        if (left_gate is None) != (right_gate is None) or (
            left_gate is not None
            and right_gate is not None
            and not torch.equal(
                left_gate.detach().cpu(), right_gate.detach().cpu()
            )
        ):
            raise Hold30AlphaDriverError(
                "absolute Pass A/B gate differs at one evaluation point"
            )
    returns = torch.stack([row.net_return for row in pass_b]).reshape(-1)
    value = torch.log1p(returns).mean()
    turnover_a = torch.stack(
        [row.discretionary_accounting.turnover for row in pass_a]
    ).mean()
    turnover_b = torch.stack(
        [row.discretionary_accounting.turnover for row in pass_b]
    ).mean()
    early_a = torch.stack(
        [row.discretionary_accounting.early_exit_notional for row in pass_a]
    ).mean()
    early_b = torch.stack(
        [row.discretionary_accounting.early_exit_notional for row in pass_b]
    ).mean()
    gate_mean = returns.new_zeros(())
    if contract.mechanism == "H2":
        excess = (turnover_b - float(contract.target_turnover)).clamp_min(0.0)
        value = value - float(contract.lambda_turn) * excess.square()
        value = value - float(contract.lambda_early) * early_b
    else:
        gates_a = torch.stack(
            [row.raw_intent.gate for row in pass_a if row.raw_intent.gate is not None]
        ).reshape(-1)
        gates_b = torch.stack(
            [row.raw_intent.gate for row in pass_b if row.raw_intent.gate is not None]
        ).reshape(-1)
        if gates_a.numel() != returns.numel() or gates_b.numel() != returns.numel():
            raise Hold30AlphaDriverError("legacy absolute trace omitted gate rows")
        gate_mean = gates_a.mean()
        if float(gate_mean) > 12.0 / 252.0:
            value = value - float(contract.gate_budget_coef) * gates_b.mean()
        entropy = -(
            gates_b.clamp(1e-8, 1.0 - 1e-8) * torch.log(gates_b.clamp(1e-8, 1.0 - 1e-8))
            + (1.0 - gates_b).clamp(1e-8, 1.0)
            * torch.log((1.0 - gates_b).clamp(1e-8, 1.0))
        ).mean()
        value = value + float(contract.gate_entropy_coef) * entropy
    return value, {
        "mean_turnover": float(turnover_a),
        "mean_early_exit": float(early_a),
        "mean_gate": float(gate_mean),
    }


def _absolute_global_rows(
    pass_a: Sequence[Hold30Transition],
    *,
    contract: Hold30LossContract,
    global_path_ids: torch.Tensor,
    origin_offset: int,
    group: dist.ProcessGroup | None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, float]]:
    """Collect an exact, sorted Pass-A row inventory for legacy objectives."""

    local: list[dict[str, Any]] = []
    for offset, transition in enumerate(pass_a):
        batch = int(transition.net_return.shape[0])
        if tuple(global_path_ids.shape) != (batch,):
            raise Hold30AlphaDriverError(
                "legacy global path IDs do not match the transition batch"
            )
        for column, path_id in enumerate(global_path_ids.tolist()):
            gate = transition.raw_intent.gate
            if contract.mechanism == "H0":
                if gate is None:
                    raise Hold30AlphaDriverError(
                        "legacy absolute trace omitted its gate"
                    )
                gate_value = float(gate.detach()[column])
                clipped = min(max(gate_value, 1e-8), 1.0 - 1e-8)
                gate_entropy = -(
                    clipped * math.log(clipped)
                    + (1.0 - clipped) * math.log(1.0 - clipped)
                )
            else:
                gate_value = 0.0
                gate_entropy = 0.0
            row = {
                "origin_row_id": origin_offset + offset,
                "global_path_id": int(path_id),
                "log_return": float(
                    torch.log1p(transition.net_return.detach()[column])
                ),
                "turnover": float(
                    transition.discretionary_accounting.turnover.detach()[column]
                ),
                "early_exit": float(
                    transition.discretionary_accounting.early_exit_notional.detach()[
                        column
                    ]
                ),
                "gate": gate_value,
                "gate_entropy": gate_entropy,
            }
            row["content_sha256"] = sha256_payload(row)
            local.append(row)
    world_size = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    gathered: list[object] = [None] * world_size
    if world_size == 1:
        gathered[0] = local
    else:
        dist.all_gather_object(gathered, local, group=group)
    global_rows: list[dict[str, Any]] = []
    for shard in gathered:
        if not isinstance(shard, list) or any(
            not isinstance(row, dict) for row in shard
        ):
            raise Hold30AlphaDriverError(
                "legacy distributed Pass-A row inventory is malformed"
            )
        global_rows.extend(shard)
    global_rows.sort(
        key=lambda row: (row["origin_row_id"], row["global_path_id"])
    )
    identities = tuple(
        (row["origin_row_id"], row["global_path_id"]) for row in global_rows
    )
    if len(set(identities)) != len(identities):
        raise Hold30AlphaDriverError("legacy Pass-A contains duplicate global rows")
    if not global_rows:
        raise Hold30AlphaDriverError("legacy Pass-A contains no global rows")
    return tuple(global_rows), _absolute_moments_from_rows(global_rows)


def _absolute_moments_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not rows:
        raise Hold30AlphaDriverError("legacy Pass-A contains no global rows")
    count = float(len(rows))
    moments = {
        "count": count,
        "mean_log_return": math.fsum(row["log_return"] for row in rows)
        / count,
        "mean_turnover": math.fsum(row["turnover"] for row in rows)
        / count,
        "mean_early_exit": math.fsum(row["early_exit"] for row in rows)
        / count,
        "mean_gate": math.fsum(row["gate"] for row in rows) / count,
        "mean_gate_entropy": math.fsum(
            row["gate_entropy"] for row in rows
        )
        / count,
    }
    return moments


def _train_absolute_global_update(
    state: _CpuFullPolicyTrainingState,
    pass_a: Sequence[Hold30Transition],
    pass_b: Sequence[Hold30Transition],
    *,
    global_path_ids: torch.Tensor,
    origin_offset: int,
    group: dist.ProcessGroup | None,
) -> dict[str, object]:
    """Apply the exact global H0/H2 objective over distinct path shards."""

    if state.optimizer is None:
        raise AssertionError("absolute CPU parity optimizer is absent")
    if len(pass_a) != len(pass_b) or not pass_a:
        raise Hold30AlphaDriverError("absolute Pass A/B traces do not align")
    for left, right in zip(pass_a, pass_b, strict=True):
        for name in ("net_return", "benchmark_net_return"):
            if not torch.equal(
                getattr(left, name).detach().cpu(),
                getattr(right, name).detach().cpu(),
            ):
                raise Hold30AlphaDriverError(
                    f"absolute Pass A/B {name} differs at one evaluation point"
                )
        for name in ("turnover", "early_exit_notional"):
            left_value = getattr(left.discretionary_accounting, name)
            right_value = getattr(right.discretionary_accounting, name)
            if not torch.equal(
                left_value.detach().cpu(), right_value.detach().cpu()
            ):
                raise Hold30AlphaDriverError(
                    f"absolute Pass A/B {name} differs at one evaluation point"
                )
        left_gate = left.raw_intent.gate
        right_gate = right.raw_intent.gate
        if (left_gate is None) != (right_gate is None) or (
            left_gate is not None
            and right_gate is not None
            and not torch.equal(
                left_gate.detach().cpu(), right_gate.detach().cpu()
            )
        ):
            raise Hold30AlphaDriverError(
                "absolute Pass A/B gate differs at one evaluation point"
            )
    contract = _absolute_contract(state.route)
    rows, moments = _absolute_global_rows(
        pass_a,
        contract=contract,
        global_path_ids=global_path_ids,
        origin_offset=origin_offset,
        group=group,
    )
    count = moments["count"]
    excess = max(moments["mean_turnover"] - float(contract.target_turnover), 0.0)
    gate_budget_active = moments["mean_gate"] > 12.0 / 252.0
    objective = moments["mean_log_return"]
    if contract.mechanism == "H2":
        objective -= float(contract.lambda_turn) * excess**2
        objective -= float(contract.lambda_early) * moments["mean_early_exit"]
    else:
        if gate_budget_active:
            objective -= float(contract.gate_budget_coef) * moments["mean_gate"]
        objective += float(contract.gate_entropy_coef) * moments[
            "mean_gate_entropy"
        ]

    state.optimizer.zero_grad(set_to_none=True)
    # Accumulate one path at a time.  The single-rank order (path 0 then path
    # 1) is deliberately the same binary addition performed by the two-rank
    # Gloo SUM, avoiding batch-reduction-order artifacts in an exact receipt.
    surrogate: torch.Tensor | None = None
    for column in range(int(global_path_ids.numel())):
        contribution = _absolute_surrogate_contribution(
            pass_b,
            column=column,
            contract=contract,
            count=count,
            excess=excess,
            gate_budget_active=gate_budget_active,
        )
        surrogate = contribution if surrogate is None else surrogate + contribution

    if surrogate is None:
        raise AssertionError("absolute global surrogate has no local path")
    (-surrogate).backward()

    parameters = tuple(
        parameter for parameter in state.policy.parameters() if parameter.requires_grad
    )
    world_size = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if world_size > 1:
        for parameter in parameters:
            used = torch.tensor(
                0 if parameter.grad is None else 1,
                dtype=torch.int64,
                device=parameter.device,
            )
            dist.all_reduce(used, op=dist.ReduceOp.SUM, group=group)
            if int(used.item()) == 0:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
    gradient_sha256 = _semantic_torch_sha256(
        {
            name: parameter.grad
            for name, parameter in state.policy.named_parameters()
            if parameter.requires_grad
        }
    )
    state.optimizer.step()
    return {
        "objective": objective,
        "gradient_sha256": gradient_sha256,
        "parameter_sha256": _state_dict_sha256(state.policy.state_dict()),
        "row_identity_sha256": sha256_payload(
            [
                [row["origin_row_id"], row["global_path_id"]]
                for row in rows
            ]
        ),
        "pass_a_content_sha256": sha256_payload(
            [row["content_sha256"] for row in rows]
        ),
        "moments_sha256": sha256_payload(moments),
        "optimizer_steps": 1,
        "distributed_world_size": world_size,
        "gradient_reduction": "SUM",
    }


def _absolute_surrogate_contribution(
    pass_b: Sequence[Hold30Transition],
    *,
    column: int,
    contract: Hold30LossContract,
    count: float,
    excess: float,
    gate_budget_active: bool,
) -> torch.Tensor:
    contribution = torch.stack(
        [torch.log1p(row.net_return[column]) for row in pass_b]
    ).sum() / count
    if contract.mechanism == "H2":
        turnover = torch.stack(
            [row.discretionary_accounting.turnover[column] for row in pass_b]
        ).sum() / count
        early = torch.stack(
            [
                row.discretionary_accounting.early_exit_notional[column]
                for row in pass_b
            ]
        ).sum() / count
        contribution = contribution - (
            2.0 * float(contract.lambda_turn) * excess * turnover
        )
        return contribution - float(contract.lambda_early) * early
    gates = torch.stack(
        [
            row.raw_intent.gate[column]
            for row in pass_b
            if row.raw_intent.gate is not None
        ]
    )
    if gates.numel() != len(pass_b):
        raise Hold30AlphaDriverError("legacy Pass B omitted gate rows")
    if gate_budget_active:
        contribution = contribution - float(contract.gate_budget_coef) * (
            gates.sum() / count
        )
    clipped = gates.clamp(1e-8, 1.0 - 1e-8)
    entropy = -(
        clipped * torch.log(clipped)
        + (1.0 - clipped) * torch.log(1.0 - clipped)
    ).sum() / count
    return contribution + float(contract.gate_entropy_coef) * entropy


def _train_absolute_serial_shard_update(
    state: _CpuFullPolicyTrainingState,
    pass_a_shards: Sequence[Sequence[Hold30Transition]],
    pass_b_shards: Sequence[Sequence[Hold30Transition]],
    *,
    global_path_ids: Sequence[int],
    origin_offset: int,
) -> dict[str, object]:
    """Exact path0-then-path1 reference for one-path-per-rank H0/H2."""

    if state.optimizer is None:
        raise AssertionError("absolute serial-shard optimizer is absent")
    if (
        len(pass_a_shards) != 2
        or len(pass_b_shards) != 2
        or tuple(global_path_ids) != (0, 1)
    ):
        raise Hold30AlphaDriverError(
            "absolute serial reference requires canonical paths (0, 1)"
        )
    contract = _absolute_contract(state.route)
    rows: list[dict[str, Any]] = []
    for pass_a, pass_b, path_id in zip(
        pass_a_shards,
        pass_b_shards,
        global_path_ids,
        strict=True,
    ):
        # This validates every result-moving Pass-A/Pass-B tensor before the
        # detached global moments are allowed to coefficient the gradient pass.
        _absolute_objective(pass_a, pass_b, contract)
        shard_rows, _ignored = _absolute_global_rows(
            pass_a,
            contract=contract,
            global_path_ids=torch.tensor([path_id], dtype=torch.int64),
            origin_offset=origin_offset,
            group=None,
        )
        rows.extend(shard_rows)
    rows.sort(key=lambda row: (row["origin_row_id"], row["global_path_id"]))
    moments = _absolute_moments_from_rows(rows)
    count = moments["count"]
    excess = max(moments["mean_turnover"] - float(contract.target_turnover), 0.0)
    gate_budget_active = moments["mean_gate"] > 12.0 / 252.0
    objective = moments["mean_log_return"]
    if contract.mechanism == "H2":
        objective -= float(contract.lambda_turn) * excess**2
        objective -= float(contract.lambda_early) * moments["mean_early_exit"]
    else:
        if gate_budget_active:
            objective -= float(contract.gate_budget_coef) * moments["mean_gate"]
        objective += float(contract.gate_entropy_coef) * moments[
            "mean_gate_entropy"
        ]

    parameters = tuple(
        parameter for parameter in state.policy.parameters() if parameter.requires_grad
    )
    state.optimizer.zero_grad(set_to_none=True)
    accumulated: list[torch.Tensor | None] = [None] * len(parameters)
    for pass_b in pass_b_shards:
        contribution = _absolute_surrogate_contribution(
            pass_b,
            column=0,
            contract=contract,
            count=count,
            excess=excess,
            gate_budget_active=gate_budget_active,
        )
        gradients = torch.autograd.grad(
            -contribution,
            parameters,
            allow_unused=True,
        )
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            accumulated[index] = (
                gradient.detach().clone()
                if accumulated[index] is None
                else accumulated[index] + gradient.detach()
            )
    for parameter, gradient in zip(parameters, accumulated, strict=True):
        parameter.grad = gradient
    gradient_sha256 = _semantic_torch_sha256(
        {
            name: parameter.grad
            for name, parameter in state.policy.named_parameters()
            if parameter.requires_grad
        }
    )
    state.optimizer.step()
    return {
        "objective": objective,
        "gradient_sha256": gradient_sha256,
        "parameter_sha256": _state_dict_sha256(state.policy.state_dict()),
        "row_identity_sha256": sha256_payload(
            [[row["origin_row_id"], row["global_path_id"]] for row in rows]
        ),
        "pass_a_content_sha256": sha256_payload(
            [row["content_sha256"] for row in rows]
        ),
        "moments_sha256": sha256_payload(moments),
        "optimizer_steps": 1,
        "distributed_world_size": 1,
        "gradient_reduction": "SUM",
        "serial_shard_count": 2,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise Hold30AlphaDriverError(f"payload already contains {field}")
    result = dict(payload)
    result[field] = sha256_payload(result)
    return result


def _json_normalize(value: Any) -> Any:
    """Apply the exact tuple-to-array conversion used by persisted JSON."""

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Hold30AlphaDriverError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Hold30AlphaDriverError(f"invalid driver JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Hold30AlphaDriverError(f"driver JSON artifact must be an object: {path}")
    return value


def _validate_self_hash(
    payload: Mapping[str, Any],
    field: str,
) -> None:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise Hold30AlphaDriverError(f"artifact lacks valid {field}")
    unsigned = dict(payload)
    del unsigned[field]
    if sha256_payload(unsigned) != claimed:
        raise Hold30AlphaDriverError(f"artifact {field} mismatch")


def _maximum_drawdown(returns: torch.Tensor) -> float:
    wealth = torch.cat((returns.new_ones(1), torch.cumprod(1.0 + returns, 0)))
    peak = torch.cummax(wealth, 0).values
    return float(((peak - wealth) / peak.clamp_min(1e-18)).max())


def _median_sale_age(transitions: Sequence[Hold30Transition]) -> float:
    sold = torch.stack(
        [
            row.discretionary_accounting.sold_units_by_age.sum((0, 1))
            for row in transitions
        ]
    ).sum(0)
    total = float(sold.sum())
    if total <= 0:
        return 0.0
    index = int(torch.searchsorted(sold.cumsum(0), sold.new_tensor(total * 0.5)))
    return float(index)


def _validation_metrics(
    transitions_20: Sequence[Hold30Transition],
    transitions_40: Sequence[Hold30Transition],
    sequence: Hold30Sequence,
) -> Hold30AlphaValidationMetrics:
    policy20 = torch.stack([row.net_return for row in transitions_20]).reshape(-1)
    policy40 = torch.stack([row.net_return for row in transitions_40]).reshape(-1)
    benchmark = torch.stack(
        [row.benchmark_net_return for row in transitions_20]
    ).reshape(-1)
    active20 = torch.log1p(policy20) - torch.log1p(benchmark)
    active40 = torch.log1p(policy40) - torch.log1p(benchmark)
    tracking_error = float(
        active20.std(unbiased=True) * math.sqrt(HOLD30_ALPHA_ANNUALIZATION)
    )
    risk_free = sequence.asset_returns[:, :, sequence.cash_index].reshape(-1)
    phase = torch.arange(policy20.numel(), dtype=policy20.dtype)
    market = benchmark + 0.0004 * torch.sin(phase * 0.23 + 0.2)
    policy_excess = policy20 - risk_free
    market_excess = market - risk_free
    centered_market = market_excess - market_excess.mean()
    beta = float(
        ((policy_excess - policy_excess.mean()) * centered_market).sum()
        / centered_market.square().sum().clamp_min(1e-18)
    )
    active_std = active20.std(unbiased=True)
    information_ratio = (
        float(active20.mean() / active_std * math.sqrt(HOLD30_ALPHA_ANNUALIZATION))
        if float(active_std) > 0
        else 0.0
    )
    total_excess = policy20 - risk_free
    total_std = total_excess.std(unbiased=True)
    total_sharpe = (
        float(total_excess.mean() / total_std * math.sqrt(HOLD30_ALPHA_ANNUALIZATION))
        if float(total_std) > 0
        else 0.0
    )
    forced_causes = (
        TurnoverCause.MEMBERSHIP_FORCED,
        TurnoverCause.AVAILABILITY_FORCED,
        TurnoverCause.RISK_FORCED,
    )
    forced = sum(
        float(row.turnover_by_cause[cause].sum())
        for row in transitions_20
        for cause in forced_causes
    )
    total_turnover = sum(
        float(amount.sum())
        for row in transitions_20
        for amount in row.turnover_by_cause.values()
    )
    trace_payload = {
        "policy20": policy20.detach().cpu().tolist(),
        "policy40": policy40.detach().cpu().tolist(),
        "benchmark": benchmark.detach().cpu().tolist(),
    }
    return Hold30AlphaValidationMetrics(
        update=HOLD30_ALPHA_SYNTHETIC_UPDATES,
        coverage_complete=True,
        active_return_20bp=float(active20.sum()),
        active_return_40bp=float(active40.sum()),
        tracking_error=tracking_error,
        beta=beta,
        median_sale_age=_median_sale_age(transitions_20),
        projection_distance=max(
            float(row.projection_distance.max()) for row in transitions_20
        ),
        forced_turnover_fraction=(
            forced / total_turnover if total_turnover > 0 else 0.0
        ),
        median_active_return_20bp=float(active20.sum()),
        information_ratio_20bp=information_ratio,
        total_sharpe_20bp=total_sharpe,
        max_drawdown_20bp=_maximum_drawdown(policy20),
        turnover_cost_20bp=sum(float(row.cost.sum()) for row in transitions_20),
        trace_sha256=sha256_payload(trace_payload),
    )


def _run_trace(
    policy: DailyCrossSectionPolicy,
    sequence: Hold30Sequence,
    runtime: Hold30ChronologicalRuntime,
    *,
    gradient: bool,
) -> tuple[Hold30Transition, ...]:
    context = torch.enable_grad() if gradient else torch.no_grad()
    with context:
        _terminal, transitions = runtime.run_to_terminal(policy, sequence)
    if len(transitions) != sequence.n_positions - 1:
        raise Hold30AlphaDriverError("synthetic chronology lost a scored transition")
    return transitions


def _run_trace_from_state(
    policy: DailyCrossSectionPolicy,
    sequence: Hold30Sequence,
    runtime: Hold30ChronologicalRuntime,
    *,
    gradient: bool,
    state: Hold30RuntimeState | None,
) -> tuple[Hold30RuntimeState, tuple[Hold30Transition, ...]]:
    context = torch.enable_grad() if gradient else torch.no_grad()
    with context:
        terminal, transitions = runtime.run_to_terminal(
            policy,
            sequence,
            None if state is None else state.detach(),
        )
    if len(transitions) != sequence.n_positions - 1:
        raise Hold30AlphaDriverError("synthetic chronology lost a scored transition")
    return terminal, transitions


@dataclass(frozen=True, slots=True)
class _CpuA06SelectiveDetachPolicy:
    """Preserve one identical forward while isolating A06 gradient owners."""

    policy: DailyCrossSectionPolicy
    owner: str

    def __post_init__(self) -> None:
        if self.owner not in {"alpha-core", "overlay"}:
            raise Hold30AlphaDriverError("unknown A06 selective-detach owner")

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        intent = self.policy.hold30_intent(
            state_t,
            prev_weights,
            available,
            age_summaries,
        )

        def detached(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach()

        if self.owner == "alpha-core":
            return replace(
                intent,
                total_risk_overlay=detached(intent.total_risk_overlay),
            )
        return Hold30Intent(
            entry_scores=detached(intent.entry_scores),
            target_logits=detached(intent.target_logits),
            gate=detached(intent.gate),
            hazard_residual=detached(intent.hazard_residual),
            raw_hazard_residual=detached(intent.raw_hazard_residual),
            exact_hold_probability=detached(intent.exact_hold_probability),
            exposure_residual=detached(intent.exposure_residual),
            alpha_mean_30d=detached(intent.alpha_mean_30d),
            alpha_downside_30d=detached(intent.alpha_downside_30d),
            active_risk_scale=detached(intent.active_risk_scale),
            signal_confidence=detached(intent.signal_confidence),
            total_risk_overlay=intent.total_risk_overlay,
            auxiliary_alpha_mean=detached(intent.auxiliary_alpha_mean),
        )


def _run_trace_with_frozen_partition(
    policy: DailyCrossSectionPolicy,
    sequence: Hold30Sequence,
    runtime: Hold30ChronologicalRuntime,
    frozen: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[Hold30Transition, ...]:
    """Build one differentiable A06 stream with the opposite owner frozen."""

    parameters = tuple(parameter for _name, parameter in frozen)
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise Hold30AlphaDriverError(
            "A06 synthetic gradient partition is empty or already frozen"
        )
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        return _run_trace(policy, sequence, runtime, gradient=True)
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


def _run_trace_from_state_with_frozen_partition(
    policy: DailyCrossSectionPolicy,
    sequence: Hold30Sequence,
    runtime: Hold30ChronologicalRuntime,
    frozen: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    state: Hold30RuntimeState | None,
) -> tuple[Hold30RuntimeState, tuple[Hold30Transition, ...]]:
    parameters = tuple(parameter for _name, parameter in frozen)
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise Hold30AlphaDriverError(
            "A06 synthetic gradient partition is empty or already frozen"
        )
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        # Keep the actor/attention forward on the same autograd kernel even
        # when opposite parameter partitions are frozen.  Otherwise a
        # no-input-grad CPU path can differ by a few bits across A06 streams.
        gradient_sequence = replace(
            sequence,
            decision_state=(
                sequence.decision_state.detach().clone().requires_grad_(True)
            ),
        )
        return _run_trace_from_state(
            policy,
            gradient_sequence,
            runtime,
            gradient=True,
            state=state,
        )
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


def _finite_gradient_norm(policy: torch.nn.Module) -> float:
    gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        raise Hold30AlphaDriverError("synthetic optimizer lacks finite model gradients")
    result = float(
        torch.sqrt(
            sum(gradient.detach().float().square().sum() for gradient in gradients)
        )
    )
    if result <= 0:
        raise Hold30AlphaDriverError("synthetic optimizer produced a zero gradient")
    return result


@dataclass(slots=True)
class _CpuFullPolicyTrainingState:
    route: Hold30AlphaSyntheticRoute
    objective: Hold30AlphaObjectiveConfig | None
    policy: DailyCrossSectionPolicy
    optimizer: torch.optim.Optimizer | None
    alpha_core_optimizer: torch.optim.Optimizer | None
    overlay_optimizer: torch.optim.Optimizer | None
    a06_partition: Any | None
    a06_optimizer_spec_receipt: Hold30A06OptimizerSpecReceipt | None
    a06_optimizer_state_receipt: Hold30A06OptimizerStateReceipt | None
    sampler_generator: torch.Generator
    update_index: int = 0
    sampler_cursor: int = 0
    runtime_state: Hold30RuntimeState | None = None
    a06_core_runtime_state: Hold30RuntimeState | None = None
    a06_executed_runtime_state: Hold30RuntimeState | None = None


def _normalized_continuation_state(
    terminal: Hold30RuntimeState,
) -> Hold30RuntimeState:
    if terminal.pending_intent is not None:
        raise Hold30AlphaDriverError(
            "a continuation checkpoint cannot contain a pending fill"
        )
    return replace(terminal.detach(), position_index=0, pending_intent=None)


def _combine_runtime_states(
    states: Sequence[Hold30RuntimeState],
) -> Hold30RuntimeState:
    """Combine independent batch-1 paths without mixing economic state."""

    if len(states) != 2 or any(
        state.position_index != 0 or state.pending_intent is not None
        for state in states
    ):
        raise Hold30AlphaDriverError(
            "serial reference requires two normalized fill-free runtime states"
        )
    cash_indices = {state.ledger.cash_index for state in states}
    if len(cash_indices) != 1:
        raise Hold30AlphaDriverError("serial runtime paths disagree on CASH")
    snapshots = tuple(state.sleeve_snapshot for state in states)
    combined_snapshot: Hold30SleeveSnapshot | None = None
    if any(snapshot is not None for snapshot in snapshots):
        if any(snapshot is None for snapshot in snapshots):
            raise Hold30AlphaDriverError(
                "serial runtime paths disagree on sleeve state"
            )
        left, right = snapshots
        assert left is not None and right is not None
        if (
            left.session_index != right.session_index
            or left.cash_index != right.cash_index
            or not torch.equal(
                left.last_review_session, right.last_review_session
            )
            or not torch.equal(left.review_count, right.review_count)
        ):
            raise Hold30AlphaDriverError(
                "serial runtime sleeve clocks do not align"
            )
        combined_snapshot = Hold30SleeveSnapshot(
            books=torch.cat((left.books, right.books), dim=0),
            session_index=left.session_index,
            last_review_session=left.last_review_session.clone(),
            review_count=left.review_count.clone(),
            cash_index=left.cash_index,
        )
    return Hold30RuntimeState(
        position_index=0,
        ledger=CohortLedger(
            economic_value=torch.cat(
                tuple(state.ledger.economic_value for state in states), dim=0
            ),
            retention_units=torch.cat(
                tuple(state.ledger.retention_units for state in states), dim=0
            ),
            cash_index=next(iter(cash_indices)),
        ),
        equity=torch.cat(tuple(state.equity for state in states), dim=0),
        pending_intent=None,
        sleeve_snapshot=combined_snapshot,
    )


def _runtime_state_payload(
    state: Hold30RuntimeState | None,
) -> dict[str, Any] | None:
    if state is None:
        return None
    if state.pending_intent is not None or state.position_index != 0:
        raise Hold30AlphaDriverError(
            "only normalized fill-free runtime states are checkpointable"
        )
    sleeve = state.sleeve_snapshot
    return {
        "position_index": state.position_index,
        "ledger": {
            "economic_value": state.ledger.economic_value.detach().cpu().clone(),
            "retention_units": state.ledger.retention_units.detach().cpu().clone(),
            "cash_index": state.ledger.cash_index,
        },
        "equity": state.equity.detach().cpu().clone(),
        "pending_intent": None,
        "sleeve_snapshot": (
            None
            if sleeve is None
            else {
                "books": sleeve.books.detach().cpu().clone(),
                "session_index": sleeve.session_index,
                "last_review_session": (
                    sleeve.last_review_session.detach().cpu().clone()
                ),
                "review_count": sleeve.review_count.detach().cpu().clone(),
                "cash_index": sleeve.cash_index,
            }
        ),
    }


def _runtime_state_from_payload(
    payload: Mapping[str, Any] | None,
) -> Hold30RuntimeState | None:
    if payload is None:
        return None
    ledger = payload.get("ledger")
    sleeve = payload.get("sleeve_snapshot")
    if (
        payload.get("position_index") != 0
        or payload.get("pending_intent") is not None
        or not isinstance(ledger, Mapping)
    ):
        raise Hold30AlphaDriverError("runtime continuation payload is malformed")
    sleeve_snapshot: Hold30SleeveSnapshot | None = None
    if sleeve is not None:
        if not isinstance(sleeve, Mapping):
            raise Hold30AlphaDriverError("sleeve continuation payload is malformed")
        sleeve_snapshot = Hold30SleeveSnapshot(
            books=sleeve["books"],
            session_index=int(sleeve["session_index"]),
            last_review_session=sleeve["last_review_session"],
            review_count=sleeve["review_count"],
            cash_index=int(sleeve["cash_index"]),
        )
    return Hold30RuntimeState(
        position_index=0,
        ledger=CohortLedger(
            economic_value=ledger["economic_value"],
            retention_units=ledger["retention_units"],
            cash_index=int(ledger["cash_index"]),
        ),
        equity=payload["equity"],
        pending_intent=None,
        sleeve_snapshot=sleeve_snapshot,
    )


def _cpu_continuation_state_sha256(
    state: _CpuFullPolicyTrainingState,
) -> str:
    return _semantic_torch_sha256(
        {
            "update_index": state.update_index,
            "sampler_cursor": state.sampler_cursor,
            "runtime_state": _runtime_state_payload(state.runtime_state),
            "a06_core_runtime_state": _runtime_state_payload(
                state.a06_core_runtime_state
            ),
            "a06_executed_runtime_state": _runtime_state_payload(
                state.a06_executed_runtime_state
            ),
            "sampler_generator_state": state.sampler_generator.get_state(),
        }
    )


def _runtime_path_sha256(
    state: Hold30RuntimeState | None,
    column: int,
) -> str | None:
    payload = _runtime_state_payload(state)
    if payload is None:
        return None
    ledger = payload["ledger"]
    sleeve = payload["sleeve_snapshot"]
    assert isinstance(ledger, dict)

    def row(value: torch.Tensor) -> torch.Tensor:
        return value.clone() if value.ndim == 0 else value[column : column + 1].clone()

    path_payload: dict[str, Any] = {
        "position_index": 0,
        "ledger": {
            "economic_value": row(ledger["economic_value"]),
            "retention_units": row(ledger["retention_units"]),
            "cash_index": ledger["cash_index"],
        },
        "equity": row(payload["equity"]),
        "pending_intent": None,
        "sleeve_snapshot": None,
    }
    if sleeve is not None:
        assert isinstance(sleeve, dict)
        path_payload["sleeve_snapshot"] = {
            "books": row(sleeve["books"]),
            "session_index": sleeve["session_index"],
            "last_review_session": sleeve["last_review_session"].clone(),
            "review_count": sleeve["review_count"].clone(),
            "cash_index": sleeve["cash_index"],
        }
    return _semantic_torch_sha256(path_payload)


def _global_cpu_continuation_state_sha256(
    state: _CpuFullPolicyTrainingState,
    *,
    global_path_ids: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> str:
    local = [
        {
            "global_path_id": int(path_id),
            "runtime": _runtime_path_sha256(state.runtime_state, column),
            "a06_core": _runtime_path_sha256(
                state.a06_core_runtime_state, column
            ),
            "a06_executed": _runtime_path_sha256(
                state.a06_executed_runtime_state, column
            ),
        }
        for column, path_id in enumerate(global_path_ids.tolist())
    ]
    world_size = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    gathered: list[object] = [None] * world_size
    if world_size == 1:
        gathered[0] = local
    else:
        dist.all_gather_object(gathered, local, group=group)
    rows: list[dict[str, Any]] = []
    for shard in gathered:
        if not isinstance(shard, list) or any(
            not isinstance(item, dict) for item in shard
        ):
            raise Hold30AlphaDriverError(
                "distributed continuation inventory is malformed"
            )
        rows.extend(shard)
    rows.sort(key=lambda item: item["global_path_id"])
    if len({item["global_path_id"] for item in rows}) != len(rows):
        raise Hold30AlphaDriverError(
            "distributed continuation inventory has duplicate paths"
        )
    return sha256_payload(
        {
            "update_index": state.update_index,
            "sampler_cursor": state.sampler_cursor,
            "paths": rows,
        }
    )


def _new_cpu_full_policy_state(
    setting_id: str,
    *,
    seed: int,
) -> _CpuFullPolicyTrainingState:
    if setting_id not in HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS:
        raise Hold30AlphaDriverError("unknown CPU full-policy qualification setting")
    _seed_everything(seed)
    route = resolve_hold30_alpha_synthetic_route(setting_id)
    provisional = build_hold30_alpha_synthetic_objective_config(setting_id)
    policy = DailyCrossSectionPolicy(_synthetic_policy_config(route, provisional))
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(
        seed + HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS.index(setting_id) + 10_000
    )
    if not route.separate_overlay:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-5,
        )
        if provisional is not None:
            provisional.require_resolved()
        return _CpuFullPolicyTrainingState(
            route=route,
            objective=provisional,
            policy=policy,
            optimizer=optimizer,
            alpha_core_optimizer=None,
            overlay_optimizer=None,
            a06_partition=None,
            a06_optimizer_spec_receipt=None,
            a06_optimizer_state_receipt=None,
            sampler_generator=sampler_generator,
        )

    partition = partition_hold30_a06_parameters(policy, provisional)
    alpha_core_optimizer = torch.optim.AdamW(
        (parameter for _name, parameter in partition.alpha_core),
        lr=1e-4,
        weight_decay=1e-4,
        eps=1e-5,
    )
    overlay_optimizer = torch.optim.AdamW(
        (parameter for _name, parameter in partition.overlay),
        lr=1e-4,
        weight_decay=1e-4,
        eps=1e-5,
    )
    spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    objective = build_hold30_alpha_synthetic_objective_config(
        setting_id,
        a06_optimizer_spec_receipt_sha256=spec.receipt_id,
    )
    if objective is None:
        raise AssertionError("A06 objective disappeared")
    objective.require_resolved()
    state_receipt = build_hold30_a06_optimizer_state_receipt(
        policy,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        spec,
        update_index=0,
        parent_state_receipt_sha256=None,
    )
    return _CpuFullPolicyTrainingState(
        route=route,
        objective=objective,
        policy=policy,
        optimizer=None,
        alpha_core_optimizer=alpha_core_optimizer,
        overlay_optimizer=overlay_optimizer,
        a06_partition=partition,
        a06_optimizer_spec_receipt=spec,
        a06_optimizer_state_receipt=state_receipt,
        sampler_generator=sampler_generator,
    )


def _finalize_cpu_full_policy_update(
    state: _CpuFullPolicyTrainingState,
    result: Mapping[str, object],
    *,
    initial_parameter_sha256: str,
    origin_offset: int,
    scored_rows: int,
    horizon_valid_counts: tuple[int, ...],
    global_path_ids: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> dict[str, object]:
    state.update_index += 1
    state.sampler_cursor += scored_rows
    final_parameter_sha256 = _state_dict_sha256(state.policy.state_dict())
    if initial_parameter_sha256 == final_parameter_sha256:
        raise Hold30AlphaDriverError("CPU qualification update was a no-op")
    gradient_norm = _finite_gradient_norm(state.policy)
    route_sha256 = sha256_payload(asdict(state.route))
    objective_contract_sha256 = sha256_payload(
        asdict(_absolute_contract(state.route))
        if state.objective is None
        else asdict(state.objective)
    )
    finalized = dict(result)
    finalized["initial_parameter_sha256"] = initial_parameter_sha256
    finalized["parameter_sha256"] = final_parameter_sha256
    finalized["gradient_norm"] = gradient_norm
    finalized["route_sha256"] = route_sha256
    finalized["objective_contract_sha256"] = objective_contract_sha256
    finalized["horizon_valid_counts"] = horizon_valid_counts
    finalized["continuation_state_sha256"] = (
        _global_cpu_continuation_state_sha256(
            state,
            global_path_ids=global_path_ids,
            group=group,
        )
    )
    finalized["qualification_update_receipt_sha256"] = sha256_payload(
        {
            "setting_id": state.route.setting_id,
            "update_index": state.update_index,
            "origin_start": origin_offset,
            "origin_stop": state.sampler_cursor,
            "route_sha256": route_sha256,
            "objective_contract_sha256": objective_contract_sha256,
            "initial_parameter_sha256": initial_parameter_sha256,
            "parameter_sha256": final_parameter_sha256,
            "gradient_sha256": (
                sha256_payload(
                    {
                        "alpha_core": finalized[
                            "alpha_core_gradient_sha256"
                        ],
                        "overlay": finalized["overlay_gradient_sha256"],
                    }
                )
                if state.route.separate_overlay
                else finalized["gradient_sha256"]
            ),
            "continuation_state_sha256": finalized[
                "continuation_state_sha256"
            ],
        }
    )
    return finalized


def _train_active_serial_shard_update(
    state: _CpuFullPolicyTrainingState,
    pass_a_batches: Sequence[Hold30AlphaBatch],
    pass_b_batches: Sequence[Hold30AlphaBatch],
) -> dict[str, object]:
    """Exact canonical-path SUM for the seven single-optimizer objectives."""

    if state.optimizer is None or state.objective is None:
        raise AssertionError("serial active update state is incomplete")
    binding = bind_hold30_alpha_global_moments(
        pass_a_batches,
        device="cpu",
    )
    _objective, metrics = hold30_alpha_two_pass_objective(
        pass_a_batches,
        pass_b_batches,
        state.objective,
        global_moments=binding,
    )
    coefficients = derive_hold30_alpha_coefficients(
        binding.moments,
        state.objective,
    )
    contributions = tuple(
        hold30_alpha_surrogate(batch, coefficients, state.objective)
        for batch in pass_b_batches
    )
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in state.policy.named_parameters()
        if parameter.requires_grad
    )
    state.optimizer.zero_grad(set_to_none=True)
    _canonical_path_gradient_sum(contributions, named_parameters)
    gradient_sha256 = _core_named_parameter_sha256(
        named_parameters,
        gradients=True,
    )
    state.optimizer.step()
    parameter_sha256 = _core_named_parameter_sha256(
        named_parameters,
        gradients=False,
    )
    return {
        "objective": _canonical_path_objective_sum(contributions),
        "global_metrics": metrics,
        "global_moment_receipt": binding.manifest_payload(),
        "global_moment_receipt_sha256": binding.receipt_id,
        "row_identity_sha256": binding.row_identity_sha256,
        "pass_a_content_sha256": binding.pass_a_content_sha256,
        "moments_sha256": binding.moments_sha256,
        "initial_parameter_sha256": "serial-reference-finalizer-binds-this",
        "gradient_sha256": gradient_sha256,
        "parameter_sha256": parameter_sha256,
        "optimizer_steps": 1,
        "distributed_world_size": 1,
        "gradient_reduction": "SUM",
        "serial_shard_count": 2,
    }


def _a06_drawdown_weights_by_batch(
    binding: Any,
    batches: Sequence[Hold30AlphaBatch],
    objective: Hold30AlphaObjectiveConfig,
) -> tuple[torch.Tensor, ...]:
    by_path: dict[int, list[Any]] = {}
    for row in binding.rows:
        if row.valid:
            by_path.setdefault(row.global_path_id, []).append(row)
    weights_by_row: dict[tuple[int, int], float] = {}
    for path_id, rows in by_path.items():
        ordered = sorted(rows, key=lambda row: row.origin_row_id)
        returns = torch.tensor(
            [row.policy_net_return for row in ordered],
            dtype=torch.float64,
        )
        weights, _drawdown = drawdown_detached_log_weights(
            returns,
            drawdown_limit=float(objective.drawdown_limit),
            lambda_drawdown=float(objective.lambda_drawdown),
        )
        for row, weight in zip(ordered, weights.tolist(), strict=True):
            weights_by_row[(row.origin_row_id, path_id)] = float(weight)
    result: list[torch.Tensor] = []
    for batch in batches:
        keys = zip(
            batch.origin_row_ids.detach().cpu().tolist(),
            batch.global_path_ids.detach().cpu().tolist(),
            strict=True,
        )
        result.append(
            batch.policy_net_return.new_tensor(
                [
                    weights_by_row.get((int(origin), int(path_id)), 0.0)
                    for origin, path_id in keys
                ]
            )
        )
    return tuple(result)


def _train_a06_serial_shard_update(
    state: _CpuFullPolicyTrainingState,
    core_a_batches: Sequence[Hold30AlphaBatch],
    core_b_batches: Sequence[Hold30AlphaBatch],
    executed_a_batches: Sequence[Hold30AlphaBatch],
    overlay_b_batches: Sequence[Hold30AlphaBatch],
) -> dict[str, object]:
    """Exact canonical path SUM with one disjoint A06 optimizer step each."""

    if (
        state.objective is None
        or state.alpha_core_optimizer is None
        or state.overlay_optimizer is None
        or state.a06_partition is None
        or state.a06_optimizer_spec_receipt is None
        or state.a06_optimizer_state_receipt is None
    ):
        raise AssertionError("serial A06 optimizer state is incomplete")
    actual_spec = build_hold30_a06_optimizer_spec_receipt(
        state.a06_partition,
        state.alpha_core_optimizer,
        state.overlay_optimizer,
    )
    if actual_spec != state.a06_optimizer_spec_receipt:
        raise Hold30AlphaDriverError("serial A06 optimizer spec changed")
    current_state = build_hold30_a06_optimizer_state_receipt(
        state.policy,
        state.a06_partition,
        state.alpha_core_optimizer,
        state.overlay_optimizer,
        state.a06_optimizer_spec_receipt,
        update_index=state.a06_optimizer_state_receipt.update_index,
        parent_state_receipt_sha256=(
            state.a06_optimizer_state_receipt.parent_state_receipt_sha256
        ),
    )
    if current_state != state.a06_optimizer_state_receipt:
        raise Hold30AlphaDriverError("serial A06 initial state receipt differs")
    core_binding = bind_hold30_alpha_global_moments(
        core_a_batches,
        device="cpu",
    )
    executed_binding = bind_hold30_alpha_global_moments(
        executed_a_batches,
        device="cpu",
    )
    if core_binding.receipt_id == executed_binding.receipt_id:
        raise Hold30AlphaDriverError("serial A06 streams are not distinct")
    _core_objective, core_metrics = hold30_alpha_two_pass_objective(
        core_a_batches,
        core_b_batches,
        state.objective,
        global_moments=core_binding,
    )
    _overlay_objective, executed_metrics = hold30_a06_overlay_two_pass_objective(
        executed_a_batches,
        overlay_b_batches,
        state.objective,
        global_moments=executed_binding,
    )
    core_coefficients = derive_hold30_alpha_coefficients(
        core_binding.moments,
        state.objective,
    )
    core_contributions = tuple(
        hold30_alpha_surrogate(batch, core_coefficients, state.objective)
        for batch in core_b_batches
    )
    overlay_coefficients = derive_hold30_a06_overlay_coefficients(
        executed_binding.moments,
        state.objective,
    )
    drawdown_weights = _a06_drawdown_weights_by_batch(
        executed_binding,
        overlay_b_batches,
        state.objective,
    )
    overlay_contributions = tuple(
        hold30_a06_overlay_surrogate(batch, overlay_coefficients, weights)
        for batch, weights in zip(
            overlay_b_batches,
            drawdown_weights,
            strict=True,
        )
    )
    state.alpha_core_optimizer.zero_grad(set_to_none=True)
    state.overlay_optimizer.zero_grad(set_to_none=True)
    _canonical_path_gradient_sum(
        core_contributions,
        state.a06_partition.alpha_core,
    )
    if any(
        parameter.grad is not None
        for _name, parameter in state.a06_partition.overlay
    ):
        raise Hold30AlphaDriverError(
            "serial A06 core gradient leaked into overlay"
        )
    _canonical_path_gradient_sum(
        overlay_contributions,
        state.a06_partition.overlay,
    )
    alpha_core_gradient_sha256 = _core_named_parameter_sha256(
        state.a06_partition.alpha_core,
        gradients=True,
    )
    overlay_gradient_sha256 = _core_named_parameter_sha256(
        state.a06_partition.overlay,
        gradients=True,
    )
    state.alpha_core_optimizer.step()
    state.overlay_optimizer.step()
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in state.policy.named_parameters()
        if parameter.requires_grad
    )
    parameter_sha256 = _core_named_parameter_sha256(
        named_parameters,
        gradients=False,
    )
    post_state = build_hold30_a06_optimizer_state_receipt(
        state.policy,
        state.a06_partition,
        state.alpha_core_optimizer,
        state.overlay_optimizer,
        state.a06_optimizer_spec_receipt,
        update_index=state.a06_optimizer_state_receipt.update_index + 1,
        parent_state_receipt_sha256=state.a06_optimizer_state_receipt.receipt_id,
    )
    return {
        "alpha_core_objective": _canonical_path_objective_sum(
            core_contributions
        ),
        "overlay_objective": _canonical_path_objective_sum(
            overlay_contributions
        ),
        "alpha_core_global_metrics": core_metrics,
        "executed_global_metrics": executed_metrics,
        "alpha_core_global_moment_receipt": core_binding.manifest_payload(),
        "alpha_core_global_moment_receipt_sha256": core_binding.receipt_id,
        "executed_global_moment_receipt": executed_binding.manifest_payload(),
        "executed_global_moment_receipt_sha256": executed_binding.receipt_id,
        "optimizer_spec_receipt_sha256": state.a06_optimizer_spec_receipt.receipt_id,
        "pre_update_optimizer_state_receipt_sha256": (
            state.a06_optimizer_state_receipt.receipt_id
        ),
        "post_update_optimizer_state_receipt": post_state.manifest_payload(),
        "post_update_optimizer_state_receipt_sha256": post_state.receipt_id,
        "pre_update_evaluation_point_id": (
            state.a06_optimizer_state_receipt.evaluation_point_id
        ),
        "post_update_evaluation_point_id": post_state.evaluation_point_id,
        "initial_parameter_sha256": "serial-reference-finalizer-binds-this",
        "alpha_core_gradient_sha256": alpha_core_gradient_sha256,
        "overlay_gradient_sha256": overlay_gradient_sha256,
        "parameter_sha256": parameter_sha256,
        "alpha_core_optimizer_steps": 1,
        "overlay_optimizer_steps": 1,
        "distributed_world_size": 1,
        "gradient_reduction": "SUM",
        "gradient_isolation_verified": True,
        "three_stream_contract_verified": True,
        "serial_shard_count": 2,
    }


def _cpu_full_policy_update_inner(
    state: _CpuFullPolicyTrainingState,
    sequence: Hold30Sequence,
    *,
    global_path_ids: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> dict[str, object]:
    targets, target_valid = _fixture_alpha_targets(sequence)
    horizon_counts_tensor = torch.tensor(
        [
            int(target_valid[..., index].sum())
            for index in range(len(HOLD30_ALPHA_HORIZONS))
        ],
        dtype=torch.int64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(horizon_counts_tensor, op=dist.ReduceOp.SUM, group=group)
    horizon_valid_counts = tuple(int(value) for value in horizon_counts_tensor)
    if any(count <= 0 for count in horizon_valid_counts):
        raise Hold30AlphaDriverError(
            "CPU qualification must exercise every auxiliary horizon"
        )
    evaluation_point = hold30_alpha_evaluation_point_id(state.policy)
    origin_offset = state.sampler_cursor
    initial_parameter_sha256 = _state_dict_sha256(state.policy.state_dict())
    result: dict[str, object]
    if state.objective is None:
        runtime = Hold30ChronologicalRuntime(state.route.mechanism)
        terminal, pass_a = _run_trace_from_state(
            state.policy,
            sequence,
            runtime,
            gradient=False,
            state=state.runtime_state,
        )
        _ignored, pass_b = _run_trace_from_state(
            state.policy,
            sequence,
            runtime,
            gradient=True,
            state=state.runtime_state,
        )
        result = _train_absolute_global_update(
            state,
            pass_a,
            pass_b,
            global_path_ids=global_path_ids,
            origin_offset=origin_offset,
            group=group,
        )
        state.runtime_state = _normalized_continuation_state(terminal)
    elif not state.route.separate_overlay:
        if state.optimizer is None:
            raise AssertionError("canonical CPU parity optimizer is absent")
        runtime = Hold30ChronologicalRuntime(state.route.mechanism)
        terminal, pass_a = _run_trace_from_state(
            state.policy,
            sequence,
            runtime,
            gradient=False,
            state=state.runtime_state,
        )
        _ignored, pass_b = _run_trace_from_state(
            state.policy,
            sequence,
            runtime,
            gradient=True,
            state=state.runtime_state,
        )
        batch_a = _trace_batch(
            pass_a,
            sequence,
            state.route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            global_path_ids=global_path_ids,
            origin_offset=origin_offset,
        )
        batch_b = _trace_batch(
            pass_b,
            sequence,
            state.route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            global_path_ids=global_path_ids,
            origin_offset=origin_offset,
        )
        result = train_hold30_alpha_two_pass_update(
            state.policy,
            state.optimizer,
            (batch_a,),
            (batch_b,),
            state.objective,
            group=group,
            moment_device="cpu",
        )
        state.runtime_state = _normalized_continuation_state(terminal)
    else:
        if (
            state.alpha_core_optimizer is None
            or state.overlay_optimizer is None
            or state.a06_partition is None
            or state.a06_optimizer_spec_receipt is None
            or state.a06_optimizer_state_receipt is None
        ):
            raise AssertionError("A06 CPU parity state is incomplete")
        core_runtime = Hold30ChronologicalRuntime(
            state.route.mechanism,
            alpha_total_risk_step=0.0,
        )
        executed_runtime = Hold30ChronologicalRuntime(
            state.route.mechanism,
            alpha_total_risk_step=state.objective.a06_total_risk_step,
        )
        core_policy = _CpuA06SelectiveDetachPolicy(state.policy, "alpha-core")
        overlay_policy = _CpuA06SelectiveDetachPolicy(state.policy, "overlay")
        core_terminal, core_pass_a = _run_trace_from_state(
            core_policy,  # type: ignore[arg-type]
            sequence,
            core_runtime,
            gradient=True,
            state=state.a06_core_runtime_state,
        )
        _ignored, core_pass_b = _run_trace_from_state(
            core_policy,  # type: ignore[arg-type]
            sequence,
            core_runtime,
            gradient=True,
            state=state.a06_core_runtime_state,
        )
        executed_terminal, executed_pass_a = _run_trace_from_state(
            overlay_policy,  # type: ignore[arg-type]
            sequence,
            executed_runtime,
            gradient=True,
            state=state.a06_executed_runtime_state,
        )
        _ignored, overlay_pass_b = _run_trace_from_state(
            overlay_policy,  # type: ignore[arg-type]
            sequence,
            executed_runtime,
            gradient=True,
            state=state.a06_executed_runtime_state,
        )

        def batch(
            transitions: Sequence[Hold30Transition],
            stream_id: str,
            *,
            detach_inputs: bool,
        ) -> Hold30AlphaBatch:
            return _trace_batch(
                transitions,
                sequence,
                state.route,
                evaluation_point_id=evaluation_point,
                targets=targets,
                target_valid=target_valid,
                stream_id=stream_id,
                global_path_ids=global_path_ids,
                origin_offset=origin_offset,
                detach_inputs=detach_inputs,
            )

        result = train_hold30_a06_two_optimizer_update(
            state.policy,
            state.alpha_core_optimizer,
            state.overlay_optimizer,
            (
                batch(
                    core_pass_a,
                    "a06-alpha-core",
                    detach_inputs=True,
                ),
            ),
            (
                batch(
                    core_pass_b,
                    "a06-alpha-core",
                    detach_inputs=False,
                ),
            ),
            (
                batch(
                    executed_pass_a,
                    "a06-executed-overlay",
                    detach_inputs=True,
                ),
            ),
            (
                batch(
                    overlay_pass_b,
                    "a06-executed-overlay",
                    detach_inputs=False,
                ),
            ),
            state.objective,
            optimizer_spec_receipt=state.a06_optimizer_spec_receipt,
            optimizer_state_receipt=state.a06_optimizer_state_receipt,
            group=group,
            moment_device="cpu",
        )
        post_payload = result["post_update_optimizer_state_receipt"]
        if not isinstance(post_payload, dict):
            raise Hold30AlphaDriverError("A06 CPU update omitted its state receipt")
        state.a06_optimizer_state_receipt = Hold30A06OptimizerStateReceipt(
            **post_payload
        )
        state.a06_core_runtime_state = _normalized_continuation_state(
            core_terminal
        )
        state.a06_executed_runtime_state = _normalized_continuation_state(
            executed_terminal
        )

    return _finalize_cpu_full_policy_update(
        state,
        result,
        initial_parameter_sha256=initial_parameter_sha256,
        origin_offset=origin_offset,
        scored_rows=sequence.n_positions - 1,
        horizon_valid_counts=horizon_valid_counts,
        global_path_ids=global_path_ids,
        group=group,
    )


def _cpu_full_policy_serial_shard_update_inner(
    state: _CpuFullPolicyTrainingState,
    sequences: Sequence[Hold30Sequence],
    *,
    global_path_ids: Sequence[int],
) -> dict[str, object]:
    """One-rank exact reference for two one-path-per-rank shards."""

    if (
        len(sequences) != 2
        or tuple(global_path_ids) != (0, 1)
        or any(sequence.batch_size != 1 for sequence in sequences)
        or len({sequence.n_positions for sequence in sequences}) != 1
        or state.update_index != 0
        or state.sampler_cursor != 0
        or state.runtime_state is not None
        or state.a06_core_runtime_state is not None
        or state.a06_executed_runtime_state is not None
    ):
        raise Hold30AlphaDriverError(
            "serial parity reference requires fresh canonical batch-1 paths (0, 1)"
        )
    targets_and_valid = tuple(_fixture_alpha_targets(sequence) for sequence in sequences)
    horizon_valid_counts = tuple(
        sum(
            int(valid[..., index].sum())
            for _targets, valid in targets_and_valid
        )
        for index in range(len(HOLD30_ALPHA_HORIZONS))
    )
    if any(count <= 0 for count in horizon_valid_counts):
        raise Hold30AlphaDriverError(
            "serial CPU qualification must exercise every auxiliary horizon"
        )
    evaluation_point = hold30_alpha_evaluation_point_id(state.policy)
    initial_parameter_sha256 = _state_dict_sha256(state.policy.state_dict())
    origin_offset = state.sampler_cursor
    path_tensors = tuple(
        torch.tensor([path_id], dtype=torch.int64)
        for path_id in global_path_ids
    )
    result: dict[str, object]

    if state.objective is None:
        runtime = Hold30ChronologicalRuntime(state.route.mechanism)
        terminals: list[Hold30RuntimeState] = []
        pass_a_shards: list[tuple[Hold30Transition, ...]] = []
        pass_b_shards: list[tuple[Hold30Transition, ...]] = []
        for sequence in sequences:
            terminal, pass_a = _run_trace_from_state(
                state.policy,
                sequence,
                runtime,
                gradient=False,
                state=None,
            )
            _ignored, pass_b = _run_trace_from_state(
                state.policy,
                sequence,
                runtime,
                gradient=True,
                state=None,
            )
            terminals.append(_normalized_continuation_state(terminal))
            pass_a_shards.append(pass_a)
            pass_b_shards.append(pass_b)
        if hold30_alpha_evaluation_point_id(state.policy) != evaluation_point:
            raise Hold30AlphaDriverError(
                "serial path collection mutated the model evaluation point"
            )
        result = _train_absolute_serial_shard_update(
            state,
            pass_a_shards,
            pass_b_shards,
            global_path_ids=global_path_ids,
            origin_offset=origin_offset,
        )
        state.runtime_state = _combine_runtime_states(terminals)
    elif not state.route.separate_overlay:
        if state.optimizer is None:
            raise AssertionError("serial active optimizer is absent")
        runtime = Hold30ChronologicalRuntime(state.route.mechanism)
        terminals = []
        pass_a_batches: list[Hold30AlphaBatch] = []
        pass_b_batches: list[Hold30AlphaBatch] = []
        for sequence, path_ids, (targets, target_valid) in zip(
            sequences,
            path_tensors,
            targets_and_valid,
            strict=True,
        ):
            terminal, pass_a = _run_trace_from_state(
                state.policy,
                sequence,
                runtime,
                gradient=False,
                state=None,
            )
            _ignored, pass_b = _run_trace_from_state(
                state.policy,
                sequence,
                runtime,
                gradient=True,
                state=None,
            )
            terminals.append(_normalized_continuation_state(terminal))
            pass_a_batches.append(
                _trace_batch(
                    pass_a,
                    sequence,
                    state.route,
                    evaluation_point_id=evaluation_point,
                    targets=targets,
                    target_valid=target_valid,
                    global_path_ids=path_ids,
                    origin_offset=origin_offset,
                )
            )
            pass_b_batches.append(
                _trace_batch(
                    pass_b,
                    sequence,
                    state.route,
                    evaluation_point_id=evaluation_point,
                    targets=targets,
                    target_valid=target_valid,
                    global_path_ids=path_ids,
                    origin_offset=origin_offset,
                )
            )
        if hold30_alpha_evaluation_point_id(state.policy) != evaluation_point:
            raise Hold30AlphaDriverError(
                "serial path collection mutated the model evaluation point"
            )
        result = _train_active_serial_shard_update(
            state,
            tuple(pass_a_batches),
            tuple(pass_b_batches),
        )
        state.runtime_state = _combine_runtime_states(terminals)
    else:
        if (
            state.alpha_core_optimizer is None
            or state.overlay_optimizer is None
            or state.a06_partition is None
            or state.a06_optimizer_spec_receipt is None
            or state.a06_optimizer_state_receipt is None
        ):
            raise AssertionError("serial A06 state is incomplete")
        core_runtime = Hold30ChronologicalRuntime(
            state.route.mechanism,
            alpha_total_risk_step=0.0,
        )
        executed_runtime = Hold30ChronologicalRuntime(
            state.route.mechanism,
            alpha_total_risk_step=state.objective.a06_total_risk_step,
        )
        core_policy = _CpuA06SelectiveDetachPolicy(state.policy, "alpha-core")
        overlay_policy = _CpuA06SelectiveDetachPolicy(state.policy, "overlay")
        core_terminals: list[Hold30RuntimeState] = []
        executed_terminals: list[Hold30RuntimeState] = []
        core_a_batches: list[Hold30AlphaBatch] = []
        core_b_batches: list[Hold30AlphaBatch] = []
        executed_a_batches: list[Hold30AlphaBatch] = []
        overlay_b_batches: list[Hold30AlphaBatch] = []
        for sequence, path_ids, (targets, target_valid) in zip(
            sequences,
            path_tensors,
            targets_and_valid,
            strict=True,
        ):
            core_terminal, core_a = _run_trace_from_state(
                core_policy,  # type: ignore[arg-type]
                sequence,
                core_runtime,
                gradient=True,
                state=None,
            )
            _ignored, core_b = _run_trace_from_state(
                core_policy,  # type: ignore[arg-type]
                sequence,
                core_runtime,
                gradient=True,
                state=None,
            )
            executed_terminal, executed_a = _run_trace_from_state(
                overlay_policy,  # type: ignore[arg-type]
                sequence,
                executed_runtime,
                gradient=True,
                state=None,
            )
            _ignored, overlay_b = _run_trace_from_state(
                overlay_policy,  # type: ignore[arg-type]
                sequence,
                executed_runtime,
                gradient=True,
                state=None,
            )
            core_terminals.append(_normalized_continuation_state(core_terminal))
            executed_terminals.append(
                _normalized_continuation_state(executed_terminal)
            )

            def batch(
                transitions: Sequence[Hold30Transition],
                stream_id: str,
                *,
                detach_inputs: bool,
                sequence_bound: Hold30Sequence = sequence,
                path_ids_bound: torch.Tensor = path_ids,
                targets_bound: torch.Tensor = targets,
                target_valid_bound: torch.Tensor = target_valid,
            ) -> Hold30AlphaBatch:
                return _trace_batch(
                    transitions,
                    sequence_bound,
                    state.route,
                    evaluation_point_id=evaluation_point,
                    targets=targets_bound,
                    target_valid=target_valid_bound,
                    stream_id=stream_id,
                    global_path_ids=path_ids_bound,
                    origin_offset=origin_offset,
                    detach_inputs=detach_inputs,
                )

            core_a_batches.append(
                batch(core_a, "a06-alpha-core", detach_inputs=True)
            )
            core_b_batches.append(
                batch(core_b, "a06-alpha-core", detach_inputs=False)
            )
            executed_a_batches.append(
                batch(executed_a, "a06-executed-overlay", detach_inputs=True)
            )
            overlay_b_batches.append(
                batch(overlay_b, "a06-executed-overlay", detach_inputs=False)
            )
        if hold30_alpha_evaluation_point_id(state.policy) != evaluation_point:
            raise Hold30AlphaDriverError(
                "serial A06 path collection mutated the evaluation point"
            )
        result = _train_a06_serial_shard_update(
            state,
            tuple(core_a_batches),
            tuple(core_b_batches),
            tuple(executed_a_batches),
            tuple(overlay_b_batches),
        )
        post_payload = result["post_update_optimizer_state_receipt"]
        if not isinstance(post_payload, dict):
            raise Hold30AlphaDriverError(
                "serial A06 update omitted its state receipt"
            )
        state.a06_optimizer_state_receipt = Hold30A06OptimizerStateReceipt(
            **post_payload
        )
        state.a06_core_runtime_state = _combine_runtime_states(core_terminals)
        state.a06_executed_runtime_state = _combine_runtime_states(
            executed_terminals
        )

    return _finalize_cpu_full_policy_update(
        state,
        result,
        initial_parameter_sha256=initial_parameter_sha256,
        origin_offset=origin_offset,
        scored_rows=sequences[0].n_positions - 1,
        horizon_valid_counts=horizon_valid_counts,
        global_path_ids=torch.tensor(global_path_ids, dtype=torch.int64),
        group=None,
    )


def _cpu_full_policy_serial_shard_update(
    state: _CpuFullPolicyTrainingState,
    sequences: Sequence[Hold30Sequence],
    *,
    global_path_ids: Sequence[int],
) -> dict[str, object]:
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        return _cpu_full_policy_serial_shard_update_inner(
            state,
            sequences,
            global_path_ids=global_path_ids,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def _cpu_full_policy_update(
    state: _CpuFullPolicyTrainingState,
    sequence: Hold30Sequence,
    *,
    global_path_ids: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> dict[str, object]:
    """Use one CPU kernel mode for bit-exact detached/gradient replay."""

    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        return _cpu_full_policy_update_inner(
            state,
            sequence,
            global_path_ids=global_path_ids,
            group=group,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def _cpu_update_evidence(
    state: _CpuFullPolicyTrainingState,
    result: Mapping[str, object],
) -> dict[str, Any]:
    if state.route.separate_overlay:
        if state.alpha_core_optimizer is None or state.overlay_optimizer is None:
            raise AssertionError("A06 evidence lacks optimizers")
        core_moments = result["alpha_core_global_moment_receipt"]
        executed_moments = result["executed_global_moment_receipt"]
        if not isinstance(core_moments, dict) or not isinstance(
            executed_moments, dict
        ):
            raise Hold30AlphaDriverError("A06 evidence lacks moment receipts")
        gradient_sha256 = sha256_payload(
            {
                "alpha_core": result["alpha_core_gradient_sha256"],
                "overlay": result["overlay_gradient_sha256"],
            }
        )
        optimizer_state_sha256 = _semantic_torch_sha256(
            {
                "alpha_core": state.alpha_core_optimizer.state_dict(),
                "overlay": state.overlay_optimizer.state_dict(),
            }
        )
        objective = (
            float(result["alpha_core_objective"]),
            float(result["overlay_objective"]),
        )
        moment_identity = {
            "alpha_core": {
                name: core_moments[name]
                for name in (
                    "row_identity_sha256",
                    "pass_a_content_sha256",
                    "moments_sha256",
                )
            },
            "executed": {
                name: executed_moments[name]
                for name in (
                    "row_identity_sha256",
                    "pass_a_content_sha256",
                    "moments_sha256",
                )
            },
        }
        update_receipt_sha256 = result[
            "post_update_optimizer_state_receipt_sha256"
        ]
    elif state.objective is None:
        if state.optimizer is None:
            raise AssertionError("absolute evidence lacks optimizer")
        gradient_sha256 = result["gradient_sha256"]
        optimizer_state_sha256 = _semantic_torch_sha256(
            state.optimizer.state_dict()
        )
        objective = (float(result["objective"]),)
        moment_identity = {
            name: result[name]
            for name in (
                "row_identity_sha256",
                "pass_a_content_sha256",
                "moments_sha256",
            )
        }
        update_receipt_sha256 = result[
            "qualification_update_receipt_sha256"
        ]
    else:
        if state.optimizer is None:
            raise AssertionError("active evidence lacks optimizer")
        gradient_sha256 = result["gradient_sha256"]
        optimizer_state_sha256 = _semantic_torch_sha256(
            state.optimizer.state_dict()
        )
        objective = (float(result["objective"]),)
        moment_identity = {
            name: result[name]
            for name in (
                "row_identity_sha256",
                "pass_a_content_sha256",
                "moments_sha256",
            )
        }
        update_receipt_sha256 = result[
            "qualification_update_receipt_sha256"
        ]
    return {
        "setting_id": state.route.setting_id,
        "route_sha256": result["route_sha256"],
        "objective_contract_sha256": result["objective_contract_sha256"],
        "initial_parameter_sha256": result["initial_parameter_sha256"],
        "model_state_sha256": _state_dict_sha256(state.policy.state_dict()),
        "parameter_sha256": result["parameter_sha256"],
        "gradient_sha256": gradient_sha256,
        "optimizer_state_sha256": optimizer_state_sha256,
        "update_receipt_sha256": result[
            "qualification_update_receipt_sha256"
        ],
        "a06_optimizer_state_receipt_sha256": (
            update_receipt_sha256 if state.route.separate_overlay else None
        ),
        "continuation_state_sha256": result["continuation_state_sha256"],
        "gradient_norm": result["gradient_norm"],
        "horizon_valid_counts": result["horizon_valid_counts"],
        "objective": objective,
        "moment_identity": moment_identity,
    }


def _cpu_distributed_full_policy_worker(
    rank: int,
    init_file: str,
    queue: Any,
    seed: int,
    positions: int,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.set_num_threads(1)
        full_sequence = _synthetic_sequence(positions=positions, batch=2)
        local_sequence = _select_synthetic_paths(
            full_sequence,
            torch.tensor([rank], dtype=torch.int64),
        )
        evidence: dict[str, Any] = {}
        for setting_id in HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS:
            state = _new_cpu_full_policy_state(setting_id, seed=seed)
            result = _cpu_full_policy_update(
                state,
                local_sequence,
                global_path_ids=torch.tensor([rank], dtype=torch.int64),
                group=dist.group.WORLD,
            )
            if rank == 0:
                evidence[setting_id] = _cpu_update_evidence(state, result)
        if rank == 0:
            queue.put(evidence)
    finally:
        dist.destroy_process_group()


def _compare_cpu_update_evidence(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    expected_copy = dict(expected)
    actual_copy = dict(actual)
    expected_objective = tuple(expected_copy.pop("objective"))
    actual_objective = tuple(actual_copy.pop("objective"))
    if expected_copy != actual_copy:
        differing = sorted(
            key
            for key in set(expected_copy) | set(actual_copy)
            if expected_copy.get(key) != actual_copy.get(key)
        )
        raise Hold30AlphaDriverError(
            "full-policy CPU one-rank/two-rank evidence differs for "
            f"{expected.get('setting_id')}: "
            + ", ".join(differing)
        )
    if len(expected_objective) != len(actual_objective) or any(
        abs(float(left) - float(right)) > 1e-12
        for left, right in zip(expected_objective, actual_objective, strict=True)
    ):
        raise Hold30AlphaDriverError(
            "full-policy CPU distributed objectives exceed the frozen tolerance "
            f"for {expected.get('setting_id')}: "
            f"one-rank={expected_objective}, two-rank={actual_objective}"
        )


def qualify_hold30_alpha_full_policy_cpu_two_rank_parity(
    *,
    seed: int = HOLD30_SEEDS[0],
) -> Hold30AlphaCpuDistributedParityReceipt:
    """Issue non-GPU evidence for exact full-policy Gloo parity."""

    if seed not in HOLD30_SEEDS:
        raise Hold30AlphaDriverError(f"seed must be one of {HOLD30_SEEDS}")
    if not dist.is_available() or not dist.is_gloo_available():
        raise Hold30AlphaDriverError("CPU two-rank qualification requires Gloo")
    old_threads = torch.get_num_threads()
    processes: list[Any] = []
    try:
        torch.set_num_threads(1)
        full_sequence = _synthetic_sequence(
            positions=HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS,
            batch=2,
        )
        serial_sequences = tuple(
            _select_synthetic_paths(
                full_sequence,
                torch.tensor([path_id], dtype=torch.int64),
            )
            for path_id in (0, 1)
        )
        one_rank: dict[str, Any] = {}
        for setting_id in HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS:
            state = _new_cpu_full_policy_state(setting_id, seed=seed)
            result = _cpu_full_policy_serial_shard_update(
                state,
                serial_sequences,
                global_path_ids=(0, 1),
            )
            one_rank[setting_id] = _cpu_update_evidence(state, result)

        context = mp.get_context("spawn")
        queue = context.Queue()
        with tempfile.TemporaryDirectory(
            prefix="hold30-alpha-full-policy-gloo-"
        ) as temporary:
            init_file = str(Path(temporary) / "init")
            processes = [
                context.Process(
                    target=_cpu_distributed_full_policy_worker,
                    args=(
                        rank,
                        init_file,
                        queue,
                        seed,
                        HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS,
                    ),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=180)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    raise Hold30AlphaDriverError(
                        "CPU distributed qualification timed out"
                    )
                if process.exitcode != 0:
                    raise Hold30AlphaDriverError(
                        "CPU distributed qualification worker failed"
                    )
            try:
                two_rank = queue.get(timeout=10)
            except Empty as exc:
                raise Hold30AlphaDriverError(
                    "CPU distributed qualification emitted no rank-zero evidence"
                ) from exc
        evidence: list[tuple[str, str]] = []
        for setting_id in HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS:
            expected = one_rank[setting_id]
            actual = two_rank[setting_id]
            _compare_cpu_update_evidence(expected, actual)
            evidence.append(
                (
                    setting_id,
                    sha256_payload(
                        {
                            "one_rank": expected,
                            "two_rank": actual,
                        }
                    ),
                )
            )
        return Hold30AlphaCpuDistributedParityReceipt(
            setting_ids=HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS,
            seed=seed,
            positions=HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS,
            setting_evidence_sha256=tuple(evidence),
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        torch.set_num_threads(old_threads)


def _rng_checkpoint_payload(
    state: _CpuFullPolicyTrainingState,
) -> dict[str, Any]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python": {
            "version": int(python_state[0]),
            "state": torch.tensor(python_state[1], dtype=torch.int64),
            "gauss_next": python_state[2],
        },
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "sampler": state.sampler_generator.get_state().clone(),
    }


def _restore_rng_checkpoint(
    state: _CpuFullPolicyTrainingState,
    payload: Mapping[str, Any],
) -> None:
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    torch_cpu = payload.get("torch_cpu")
    sampler = payload.get("sampler")
    if (
        not isinstance(python_state, Mapping)
        or not isinstance(numpy_state, Mapping)
        or not isinstance(torch_cpu, torch.Tensor)
        or not isinstance(sampler, torch.Tensor)
    ):
        raise Hold30AlphaDriverError("checkpoint RNG payload is incomplete")
    python_words = python_state.get("state")
    numpy_keys = numpy_state.get("keys")
    if not isinstance(python_words, torch.Tensor) or not isinstance(
        numpy_keys, torch.Tensor
    ):
        raise Hold30AlphaDriverError("checkpoint RNG vectors are malformed")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_words.tolist()),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.detach().cpu())
    state.sampler_generator.set_state(sampler.detach().cpu())


def _cpu_checkpoint_payload(state: _CpuFullPolicyTrainingState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "setting_id": state.route.setting_id,
        "update_index": state.update_index,
        "sampler_cursor": state.sampler_cursor,
        "runtime_state": _runtime_state_payload(state.runtime_state),
        "a06_core_runtime_state": _runtime_state_payload(
            state.a06_core_runtime_state
        ),
        "a06_executed_runtime_state": _runtime_state_payload(
            state.a06_executed_runtime_state
        ),
        "rng": _rng_checkpoint_payload(state),
        "continuation_state_sha256": _cpu_continuation_state_sha256(state),
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in state.policy.state_dict().items()
        },
    }
    if state.route.separate_overlay:
        if (
            state.alpha_core_optimizer is None
            or state.overlay_optimizer is None
            or state.a06_optimizer_spec_receipt is None
            or state.a06_optimizer_state_receipt is None
        ):
            raise AssertionError("A06 checkpoint state is incomplete")
        payload.update(
            {
                "alpha_core_optimizer_state": (
                    state.alpha_core_optimizer.state_dict()
                ),
                "overlay_optimizer_state": state.overlay_optimizer.state_dict(),
                "optimizer_spec_receipt": (
                    state.a06_optimizer_spec_receipt.manifest_payload()
                ),
                "optimizer_state_receipt": (
                    state.a06_optimizer_state_receipt.manifest_payload()
                ),
            }
        )
    else:
        if state.optimizer is None:
            raise AssertionError("canonical checkpoint optimizer is absent")
        payload["optimizer_state"] = state.optimizer.state_dict()
    return payload


def _reload_cpu_checkpoint(
    setting_id: str,
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
) -> _CpuFullPolicyTrainingState:
    state = _new_cpu_full_policy_state(setting_id, seed=seed)
    if checkpoint.get("setting_id") != setting_id or not isinstance(
        checkpoint.get("model_state"), dict
    ):
        raise Hold30AlphaDriverError("CPU restart checkpoint identity is invalid")
    state.policy.load_state_dict(checkpoint["model_state"], strict=True)
    update_index = checkpoint.get("update_index")
    sampler_cursor = checkpoint.get("sampler_cursor")
    if (
        isinstance(update_index, bool)
        or not isinstance(update_index, int)
        or update_index < 1
        or isinstance(sampler_cursor, bool)
        or not isinstance(sampler_cursor, int)
        or sampler_cursor < 1
    ):
        raise Hold30AlphaDriverError("CPU restart checkpoint cursor is invalid")
    state.update_index = update_index
    state.sampler_cursor = sampler_cursor
    state.runtime_state = _runtime_state_from_payload(
        checkpoint.get("runtime_state")
    )
    state.a06_core_runtime_state = _runtime_state_from_payload(
        checkpoint.get("a06_core_runtime_state")
    )
    state.a06_executed_runtime_state = _runtime_state_from_payload(
        checkpoint.get("a06_executed_runtime_state")
    )
    if state.route.separate_overlay:
        if (
            state.alpha_core_optimizer is None
            or state.overlay_optimizer is None
            or state.a06_partition is None
            or state.a06_optimizer_spec_receipt is None
        ):
            raise AssertionError("A06 reload state is incomplete")
        state.alpha_core_optimizer.load_state_dict(
            checkpoint["alpha_core_optimizer_state"]
        )
        state.overlay_optimizer.load_state_dict(
            checkpoint["overlay_optimizer_state"]
        )
        if checkpoint.get("optimizer_spec_receipt") != (
            state.a06_optimizer_spec_receipt.manifest_payload()
        ):
            raise Hold30AlphaDriverError("A06 optimizer spec changed across reload")
        raw_receipt = checkpoint.get("optimizer_state_receipt")
        if not isinstance(raw_receipt, dict):
            raise Hold30AlphaDriverError("A06 reload lacks a state receipt")
        state.a06_optimizer_state_receipt = Hold30A06OptimizerStateReceipt(
            **raw_receipt
        )
        rebuilt = build_hold30_a06_optimizer_state_receipt(
            state.policy,
            state.a06_partition,
            state.alpha_core_optimizer,
            state.overlay_optimizer,
            state.a06_optimizer_spec_receipt,
            update_index=state.a06_optimizer_state_receipt.update_index,
            parent_state_receipt_sha256=(
                state.a06_optimizer_state_receipt.parent_state_receipt_sha256
            ),
        )
        if rebuilt != state.a06_optimizer_state_receipt:
            raise Hold30AlphaDriverError(
                "A06 optimizer/model state differs after checkpoint reload"
            )
    else:
        if state.optimizer is None:
            raise AssertionError("canonical reload optimizer is absent")
        state.optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng = checkpoint.get("rng")
    if not isinstance(rng, Mapping):
        raise Hold30AlphaDriverError("CPU restart checkpoint lacks RNG state")
    _restore_rng_checkpoint(state, rng)
    if checkpoint.get("continuation_state_sha256") != (
        _cpu_continuation_state_sha256(state)
    ):
        raise Hold30AlphaDriverError(
            "CPU restart continuation state differs after reload"
        )
    return state


def qualify_hold30_alpha_full_policy_cpu_restart_parity(
    *,
    seed: int = HOLD30_SEEDS[0],
) -> Hold30AlphaCpuRestartParityReceipt:
    """Issue non-GPU evidence for exact two-update save/reload parity."""

    if seed not in HOLD30_SEEDS:
        raise Hold30AlphaDriverError(f"seed must be one of {HOLD30_SEEDS}")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        sequence = _synthetic_sequence(
            positions=2 * (HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS - 1) + 1,
            batch=1,
        )
        path_ids = torch.tensor([0], dtype=torch.int64)
        evidence: list[tuple[str, str]] = []
        for setting_id in HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS:
            uninterrupted = _new_cpu_full_policy_state(setting_id, seed=seed)
            first_chunk = _next_stochastic_restart_chunk(
                sequence,
                uninterrupted,
            )
            first_result = _cpu_full_policy_update(
                uninterrupted,
                first_chunk,
                global_path_ids=path_ids,
            )
            first_evidence = _cpu_update_evidence(uninterrupted, first_result)
            checkpoint = _cpu_checkpoint_payload(uninterrupted)
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            serialized = stream.getvalue()
            checkpoint_sha256 = hashlib.sha256(serialized).hexdigest()
            stream.seek(0)
            loaded = torch.load(stream, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise Hold30AlphaDriverError(
                    "CPU restart checkpoint did not reload as an object"
                )

            second_chunk = _next_stochastic_restart_chunk(
                sequence,
                uninterrupted,
            )
            second_uninterrupted = _cpu_full_policy_update(
                uninterrupted,
                second_chunk,
                global_path_ids=path_ids,
            )
            uninterrupted_evidence = _cpu_update_evidence(
                uninterrupted,
                second_uninterrupted,
            )
            restarted = _reload_cpu_checkpoint(
                setting_id,
                loaded,
                seed=seed,
            )
            restarted_second_chunk = _next_stochastic_restart_chunk(
                sequence,
                restarted,
            )
            if not torch.equal(
                second_chunk.decision_state,
                restarted_second_chunk.decision_state,
            ):
                raise Hold30AlphaDriverError(
                    "CPU restart sampler/RNG continuation differs"
                )
            second_restarted = _cpu_full_policy_update(
                restarted,
                restarted_second_chunk,
                global_path_ids=path_ids,
            )
            restarted_evidence = _cpu_update_evidence(
                restarted,
                second_restarted,
            )
            if uninterrupted_evidence != restarted_evidence:
                differing = sorted(
                    key
                    for key in set(uninterrupted_evidence)
                    | set(restarted_evidence)
                    if uninterrupted_evidence.get(key)
                    != restarted_evidence.get(key)
                )
                raise Hold30AlphaDriverError(
                    "full-policy CPU restart evidence differs: "
                    + ", ".join(differing)
                )
            evidence.append(
                (
                    setting_id,
                    sha256_payload(
                        {
                            "checkpoint_sha256": checkpoint_sha256,
                            "first_update": first_evidence,
                            "uninterrupted_second_update": (
                                uninterrupted_evidence
                            ),
                            "restarted_second_update": restarted_evidence,
                        }
                    ),
                )
            )
        return Hold30AlphaCpuRestartParityReceipt(
            setting_ids=HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS,
            seed=seed,
            positions=HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS,
            setting_evidence_sha256=tuple(evidence),
        )
    finally:
        torch.set_num_threads(old_threads)


def run_hold30_alpha_synthetic_qualification(
    setting_id: str,
    output_dir: str | Path,
    *,
    seed: int = HOLD30_SEEDS[0],
) -> dict[str, Any]:
    """Run one deterministic, one-update, CPU synthetic qualification trial.

    This function neither accepts nor emits executable approval.  It validates
    all result-moving fixture fields and initializes any A06 optimizer receipt
    before touching ``output_dir``.
    """

    route = resolve_hold30_alpha_synthetic_route(setting_id)
    if isinstance(seed, bool) or seed not in HOLD30_SEEDS:
        raise Hold30AlphaDriverError(f"seed must be one of {HOLD30_SEEDS}")
    destination = Path(output_dir)
    if destination.exists():
        raise Hold30AlphaDriverError(
            f"refusing to overwrite existing synthetic run: {destination}"
        )

    _seed_everything(seed)
    provisional_objective = build_hold30_alpha_synthetic_objective_config(
        route.setting_id
    )
    model_config = _synthetic_policy_config(route, provisional_objective)
    policy = DailyCrossSectionPolicy(model_config)
    initial_model_state = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    initial_state_sha256 = _state_dict_sha256(initial_model_state)
    evaluation_point = _evaluation_point_id(policy)

    optimizer: torch.optim.Optimizer | None = None
    alpha_core_optimizer: torch.optim.Optimizer | None = None
    overlay_optimizer: torch.optim.Optimizer | None = None
    a06_partition = None
    a06_optimizer_spec_receipt = None
    a06_initial_state_receipt = None
    if route.separate_overlay:
        if provisional_objective is None:
            raise AssertionError("A06 requires its v3 objective config")
        a06_partition = partition_hold30_a06_parameters(
            policy,
            provisional_objective,
        )
        alpha_core_optimizer = torch.optim.AdamW(
            (parameter for _name, parameter in a06_partition.alpha_core),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-5,
        )
        overlay_optimizer = torch.optim.AdamW(
            (parameter for _name, parameter in a06_partition.overlay),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-5,
        )
        a06_optimizer_spec_receipt = build_hold30_a06_optimizer_spec_receipt(
            a06_partition,
            alpha_core_optimizer,
            overlay_optimizer,
        )
        a06_initial_state_receipt = build_hold30_a06_optimizer_state_receipt(
            policy,
            a06_partition,
            alpha_core_optimizer,
            overlay_optimizer,
            a06_optimizer_spec_receipt,
            update_index=0,
            parent_state_receipt_sha256=None,
        )
        if a06_initial_state_receipt.evaluation_point_id != evaluation_point:
            raise AssertionError("A06 state receipt evaluation point drifted")
        pilot_plan = _synthetic_pilot_plan(a06_optimizer_spec_receipt.receipt_id)
        objective = build_hold30_alpha_synthetic_objective_config(
            route.setting_id,
            a06_optimizer_spec_receipt_sha256=(
                a06_optimizer_spec_receipt.receipt_id
            ),
        )
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            eps=1e-5,
        )
        pilot_plan = _synthetic_pilot_plan()
        objective = provisional_objective
    if objective is not None:
        try:
            objective.require_resolved()
        except Hold30AlphaTrainingError as exc:
            raise Hold30AlphaDriverError(
                f"{route.setting_id} cannot enter the synthetic optimizer: {exc}"
            ) from exc
    checkpoint_contract = pilot_plan.checkpoint_contract
    if not checkpoint_contract.result_moving_thresholds_complete:
        raise AssertionError("fixture checkpoint contract must be complete")

    sequence = _synthetic_sequence()
    targets, target_valid = _fixture_alpha_targets(sequence)
    runtime = Hold30ChronologicalRuntime(
        route.mechanism,
        alpha_total_risk_step=(
            objective.a06_total_risk_step
            if route.separate_overlay and objective is not None
            else None
        ),
    )
    core_runtime = (
        Hold30ChronologicalRuntime(route.mechanism, alpha_total_risk_step=0.0)
        if route.separate_overlay
        else runtime
    )
    pass_a = _run_trace(policy, sequence, core_runtime, gradient=False)
    executed_pass_a = (
        _run_trace(policy, sequence, runtime, gradient=False)
        if route.separate_overlay
        else None
    )
    objective_metrics: Hold30AlphaGlobalMetrics | None = None
    absolute_metrics: dict[str, float] | None = None
    optimizer_update_evidence: dict[str, Any] | None = None
    a06_post_receipt_sha256: str | None = None
    if objective is None:
        if optimizer is None:
            raise AssertionError("legacy objective requires one optimizer")
        pass_b = _run_trace(policy, sequence, runtime, gradient=True)
        absolute_contract = _absolute_contract(route)
        value, absolute_metrics = _absolute_objective(
            pass_a,
            pass_b,
            absolute_contract,
        )
        objective_contract_payload: Mapping[str, Any] = {
            "kind": "legacy-absolute-qualification",
            "contract": asdict(absolute_contract),
        }
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise Hold30AlphaDriverError(
                "synthetic objective is not one finite scalar"
            )
        optimizer.zero_grad(set_to_none=True)
        (-value).backward()
        gradient_norm = _finite_gradient_norm(policy)
        optimizer.step()
        objective_payload: float | dict[str, float] = float(value.detach())
    elif route.separate_overlay:
        if (
            a06_partition is None
            or alpha_core_optimizer is None
            or overlay_optimizer is None
            or a06_optimizer_spec_receipt is None
            or a06_initial_state_receipt is None
            or executed_pass_a is None
        ):
            raise AssertionError("A06 optimizer partition was not initialized")
        alpha_core_pass_b = _run_trace_with_frozen_partition(
            policy,
            sequence,
            core_runtime,
            a06_partition.overlay,
        )
        overlay_pass_b = _run_trace_with_frozen_partition(
            policy,
            sequence,
            runtime,
            a06_partition.alpha_core,
        )
        batch_a = _trace_batch(
            pass_a,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            stream_id="a06-alpha-core",
        )
        alpha_core_batch_b = _trace_batch(
            alpha_core_pass_b,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            stream_id="a06-alpha-core",
        )
        executed_batch_a = _trace_batch(
            executed_pass_a,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            stream_id="a06-executed-overlay",
        )
        overlay_batch_b = _trace_batch(
            overlay_pass_b,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
            stream_id="a06-executed-overlay",
        )
        update_result = train_hold30_a06_two_optimizer_update(
            policy,
            alpha_core_optimizer,
            overlay_optimizer,
            (batch_a,),
            (alpha_core_batch_b,),
            (executed_batch_a,),
            (overlay_batch_b,),
            objective,
            optimizer_spec_receipt=a06_optimizer_spec_receipt,
            optimizer_state_receipt=a06_initial_state_receipt,
        )
        objective_metrics = update_result["executed_global_metrics"]
        if not isinstance(objective_metrics, Hold30AlphaGlobalMetrics):
            raise Hold30AlphaDriverError("A06 update omitted typed global metrics")
        objective_payload = {
            "alpha_core": float(update_result["alpha_core_objective"]),
            "overlay": float(update_result["overlay_objective"]),
        }
        if not all(math.isfinite(value) for value in objective_payload.values()):
            raise Hold30AlphaDriverError("A06 objectives must both be finite")
        a06_post_receipt_sha256 = str(
            update_result["post_update_optimizer_state_receipt_sha256"]
        )
        optimizer_update_evidence = {
            "gradient_isolation_verified": update_result[
                "gradient_isolation_verified"
            ],
            "three_stream_contract_verified": update_result[
                "three_stream_contract_verified"
            ],
            "gradient_reduction": update_result["gradient_reduction"],
            "distributed_world_size": update_result["distributed_world_size"],
            "alpha_core_optimizer_steps": update_result[
                "alpha_core_optimizer_steps"
            ],
            "overlay_optimizer_steps": update_result["overlay_optimizer_steps"],
            "alpha_core_gradient_sha256": update_result[
                "alpha_core_gradient_sha256"
            ],
            "overlay_gradient_sha256": update_result["overlay_gradient_sha256"],
            "optimizer_spec_receipt": (
                a06_optimizer_spec_receipt.manifest_payload()
            ),
            "optimizer_spec_receipt_sha256": (
                a06_optimizer_spec_receipt.receipt_id
            ),
            "initial_optimizer_state_receipt": (
                a06_initial_state_receipt.manifest_payload()
            ),
            "initial_optimizer_state_receipt_sha256": (
                a06_initial_state_receipt.receipt_id
            ),
            "pre_update_evaluation_point_id": update_result[
                "pre_update_evaluation_point_id"
            ],
            "post_update_evaluation_point_id": update_result[
                "post_update_evaluation_point_id"
            ],
            "alpha_core_global_moment_receipt_sha256": update_result[
                "alpha_core_global_moment_receipt_sha256"
            ],
            "executed_global_moment_receipt_sha256": update_result[
                "executed_global_moment_receipt_sha256"
            ],
            "alpha_core_global_metrics": asdict(
                update_result["alpha_core_global_metrics"]
            ),
            "post_update_optimizer_state_receipt": update_result[
                "post_update_optimizer_state_receipt"
            ],
            "post_update_optimizer_state_receipt_sha256": (
                a06_post_receipt_sha256
            ),
        }
        objective_contract_payload = {
            "kind": "v3-a06-disjoint-two-optimizer-qualification",
            "config": asdict(objective),
        }
        gradient_norm = _finite_gradient_norm(policy)
    else:
        if optimizer is None:
            raise AssertionError("v3 objective requires one optimizer")
        pass_b = _run_trace(policy, sequence, runtime, gradient=True)
        batch_a = _trace_batch(
            pass_a,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
        )
        batch_b = _trace_batch(
            pass_b,
            sequence,
            route,
            evaluation_point_id=evaluation_point,
            targets=targets,
            target_valid=target_valid,
        )
        value, objective_metrics = hold30_alpha_two_pass_objective(
            (batch_a,),
            (batch_b,),
            objective,
        )
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise Hold30AlphaDriverError(
                "synthetic objective is not one finite scalar"
            )
        optimizer.zero_grad(set_to_none=True)
        (-value).backward()
        gradient_norm = _finite_gradient_norm(policy)
        optimizer.step()
        objective_payload = float(value.detach())
        objective_contract_payload = {
            "kind": "v3-global-two-pass-qualification",
            "config": asdict(objective),
        }
    final_state_sha256 = _state_dict_sha256(policy.state_dict())
    if final_state_sha256 == initial_state_sha256:
        raise Hold30AlphaDriverError("optimizer step did not change the model state")

    policy.eval()
    final20 = _run_trace(policy, sequence, runtime, gradient=False)
    final40 = _run_trace(
        policy,
        replace(sequence, cost_rate=0.004),
        runtime,
        gradient=False,
    )
    validation = _validation_metrics(final20, final40, sequence)
    eligibility_gates_pass = validation.eligible(
        route.setting_id,
        contract=checkpoint_contract,
    )
    checkpoint_eligible = bool(
        eligibility_gates_pass
        and validation.update >= checkpoint_contract.minimum_updates
    )
    metrics_payload = _with_self_hash(
        {
            "schema": HOLD30_ALPHA_SYNTHETIC_METRICS_SCHEMA,
            "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
            "setting_id": route.setting_id,
            "qualification_only": True,
            "objective": objective_payload,
            "gradient_norm": gradient_norm,
            "optimizer_update_evidence": optimizer_update_evidence,
            "global_objective_metrics": (
                None if objective_metrics is None else asdict(objective_metrics)
            ),
            "absolute_objective_metrics": absolute_metrics,
            "validation": asdict(validation),
            "checkpoint_gate_eligibility": eligibility_gates_pass,
            "minimum_update_satisfied": (
                validation.update >= checkpoint_contract.minimum_updates
            ),
            "checkpoint_eligible": checkpoint_eligible,
        },
        "metrics_sha256",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.stage-",
            dir=destination.parent,
        )
    )
    try:
        initial_path = stage / "initial-checkpoint.pt"
        final_path = stage / "update-000001.pt"
        metrics_path = stage / "metrics.json"
        torch.save(
            {
                "schema": HOLD30_ALPHA_SYNTHETIC_CHECKPOINT_SCHEMA,
                "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
                "setting_id": route.setting_id,
                "update": 0,
                "qualification_only": True,
                "model_state": initial_model_state,
            },
            initial_path,
        )
        final_checkpoint: dict[str, Any] = {
            "schema": HOLD30_ALPHA_SYNTHETIC_CHECKPOINT_SCHEMA,
            "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
            "setting_id": route.setting_id,
            "update": HOLD30_ALPHA_SYNTHETIC_UPDATES,
            "qualification_only": True,
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in policy.state_dict().items()
            },
        }
        if route.separate_overlay:
            if (
                alpha_core_optimizer is None
                or overlay_optimizer is None
                or optimizer_update_evidence is None
            ):
                raise AssertionError("A06 checkpoint lacks optimizer evidence")
            final_checkpoint.update(
                {
                    "alpha_core_optimizer_state": (
                        alpha_core_optimizer.state_dict()
                    ),
                    "overlay_optimizer_state": overlay_optimizer.state_dict(),
                    "optimizer_spec_receipt": (
                        optimizer_update_evidence["optimizer_spec_receipt"]
                    ),
                    "initial_optimizer_state_receipt": (
                        optimizer_update_evidence[
                            "initial_optimizer_state_receipt"
                        ]
                    ),
                    "post_update_optimizer_state_receipt": (
                        optimizer_update_evidence[
                            "post_update_optimizer_state_receipt"
                        ]
                    ),
                }
            )
        else:
            if optimizer is None:
                raise AssertionError("single-optimizer checkpoint lacks optimizer")
            final_checkpoint["optimizer_state"] = optimizer.state_dict()
        torch.save(final_checkpoint, final_path)
        metrics_path.write_text(
            json.dumps(metrics_payload, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        file_hashes = {
            path.name: _file_sha256(path)
            for path in (initial_path, final_path, metrics_path)
        }
        receipt = _with_self_hash(
            {
                "schema": HOLD30_ALPHA_SYNTHETIC_DRIVER_SCHEMA,
                "schema_version": 1,
                "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
                "setting_id": route.setting_id,
                "seed": seed,
                "updates": HOLD30_ALPHA_SYNTHETIC_UPDATES,
                "scored_sessions": HOLD30_ALPHA_SYNTHETIC_POSITIONS - 1,
                "route": asdict(route),
                "qualification_only": True,
                "launch_authorized": False,
                "production_data_consumed": False,
                "gpu_consumed": False,
                "checkpoint_contract": asdict(checkpoint_contract),
                "pilot_training_plan_receipt_sha256": pilot_plan.receipt_id,
                "a06_optimizer_spec_receipt_sha256": (
                    None
                    if a06_optimizer_spec_receipt is None
                    else a06_optimizer_spec_receipt.receipt_id
                ),
                "a06_initial_optimizer_state_receipt_sha256": (
                    None
                    if a06_initial_state_receipt is None
                    else a06_initial_state_receipt.receipt_id
                ),
                "a06_post_update_optimizer_state_receipt_sha256": (
                    a06_post_receipt_sha256
                ),
                "objective_contract": dict(objective_contract_payload),
                "source_axis_id": sequence.axis_id,
                "initial_model_state_sha256": initial_state_sha256,
                "final_model_state_sha256": final_state_sha256,
                "file_sha256s": file_hashes,
                "metrics_sha256": metrics_payload["metrics_sha256"],
            },
            "receipt_sha256",
        )
        (stage / "run-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_hold30_alpha_synthetic_run(destination)


def verify_hold30_alpha_synthetic_run(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify exact artifact closure and return the authenticated receipt."""

    root = Path(output_dir)
    expected = {
        "initial-checkpoint.pt",
        "update-000001.pt",
        "metrics.json",
        "run-receipt.json",
    }
    if not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise Hold30AlphaDriverError("synthetic run artifact set is partial or unknown")
    if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise Hold30AlphaDriverError("synthetic artifacts must be regular files")
    receipt = _read_json(root / "run-receipt.json")
    _validate_self_hash(receipt, "receipt_sha256")
    expected_receipt_fields = {
        "schema",
        "schema_version",
        "protocol_generation",
        "setting_id",
        "seed",
        "updates",
        "scored_sessions",
        "route",
        "qualification_only",
        "launch_authorized",
        "production_data_consumed",
        "gpu_consumed",
        "checkpoint_contract",
        "pilot_training_plan_receipt_sha256",
        "a06_optimizer_spec_receipt_sha256",
        "a06_initial_optimizer_state_receipt_sha256",
        "a06_post_update_optimizer_state_receipt_sha256",
        "objective_contract",
        "source_axis_id",
        "initial_model_state_sha256",
        "final_model_state_sha256",
        "file_sha256s",
        "metrics_sha256",
        "receipt_sha256",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema") != HOLD30_ALPHA_SYNTHETIC_DRIVER_SCHEMA
        or receipt.get("schema_version") != 1
        or receipt.get("protocol_generation") != HOLD30_ALPHA_PROTOCOL_GENERATION
        or receipt.get("setting_id") not in HOLD30_ALPHA_MECH8_IDS
        or receipt.get("seed") not in HOLD30_SEEDS
        or receipt.get("updates") != HOLD30_ALPHA_SYNTHETIC_UPDATES
        or receipt.get("scored_sessions") != HOLD30_ALPHA_SYNTHETIC_POSITIONS - 1
        or receipt.get("qualification_only") is not True
        or receipt.get("launch_authorized") is not False
        or receipt.get("production_data_consumed") is not False
        or receipt.get("gpu_consumed") is not False
        or receipt.get("source_axis_id") != HOLD30_ALPHA_SYNTHETIC_AXIS_ID
    ):
        raise Hold30AlphaDriverError("synthetic receipt identity/authority is invalid")
    route = resolve_hold30_alpha_synthetic_route(receipt["setting_id"])
    a06_optimizer_spec_receipt_sha256 = receipt.get(
        "a06_optimizer_spec_receipt_sha256"
    )
    a06_initial_receipt_sha256 = receipt.get(
        "a06_initial_optimizer_state_receipt_sha256"
    )
    a06_post_receipt_sha256 = receipt.get(
        "a06_post_update_optimizer_state_receipt_sha256"
    )
    if route.separate_overlay:
        try:
            _optional_digest(
                "a06_optimizer_spec_receipt_sha256",
                a06_optimizer_spec_receipt_sha256,
            )
            _optional_digest(
                "a06_initial_optimizer_state_receipt_sha256",
                a06_initial_receipt_sha256,
            )
            _optional_digest(
                "a06_post_update_optimizer_state_receipt_sha256",
                a06_post_receipt_sha256,
            )
        except Hold30AlphaDriverError as exc:
            raise Hold30AlphaDriverError("A06 receipt binding is invalid") from exc
        if (
            a06_optimizer_spec_receipt_sha256 is None
            or a06_initial_receipt_sha256 is None
            or a06_post_receipt_sha256 is None
            or a06_initial_receipt_sha256 == a06_post_receipt_sha256
        ):
            raise Hold30AlphaDriverError("A06 optimizer receipt chain is incomplete")
        pilot_plan = _synthetic_pilot_plan(a06_optimizer_spec_receipt_sha256)
        objective = build_hold30_alpha_synthetic_objective_config(
            route.setting_id,
            a06_optimizer_spec_receipt_sha256=(
                a06_optimizer_spec_receipt_sha256
            ),
        )
    else:
        if (
            a06_optimizer_spec_receipt_sha256 is not None
            or a06_initial_receipt_sha256 is not None
            or a06_post_receipt_sha256 is not None
        ):
            raise Hold30AlphaDriverError(
                "non-A06 receipt contains A06 optimizer-state evidence"
            )
        pilot_plan = _synthetic_pilot_plan()
        objective = build_hold30_alpha_synthetic_objective_config(route.setting_id)
    expected_objective = (
        {
            "kind": "legacy-absolute-qualification",
            "contract": asdict(_absolute_contract(route)),
        }
        if objective is None
        else {
            "kind": (
                "v3-a06-disjoint-two-optimizer-qualification"
                if route.separate_overlay
                else "v3-global-two-pass-qualification"
            ),
            "config": asdict(objective),
        }
    )
    if (
        receipt.get("route") != _json_normalize(asdict(route))
        or receipt.get("checkpoint_contract")
        != _json_normalize(asdict(pilot_plan.checkpoint_contract))
        or receipt.get("pilot_training_plan_receipt_sha256") != pilot_plan.receipt_id
        or receipt.get("objective_contract") != _json_normalize(expected_objective)
    ):
        raise Hold30AlphaDriverError("synthetic receipt contract binding is invalid")
    file_hashes = receipt.get("file_sha256s")
    if not isinstance(file_hashes, dict) or set(file_hashes) != expected - {
        "run-receipt.json"
    }:
        raise Hold30AlphaDriverError("synthetic receipt file inventory is invalid")
    for name, claimed in file_hashes.items():
        if _file_sha256(root / name) != claimed:
            raise Hold30AlphaDriverError(f"synthetic artifact hash mismatch: {name}")
    metrics = _read_json(root / "metrics.json")
    _validate_self_hash(metrics, "metrics_sha256")
    expected_metrics_fields = {
        "schema",
        "protocol_generation",
        "setting_id",
        "qualification_only",
        "objective",
        "gradient_norm",
        "optimizer_update_evidence",
        "global_objective_metrics",
        "absolute_objective_metrics",
        "validation",
        "checkpoint_gate_eligibility",
        "minimum_update_satisfied",
        "checkpoint_eligible",
        "metrics_sha256",
    }
    if (
        set(metrics) != expected_metrics_fields
        or metrics.get("schema") != HOLD30_ALPHA_SYNTHETIC_METRICS_SCHEMA
        or metrics.get("protocol_generation") != HOLD30_ALPHA_PROTOCOL_GENERATION
        or metrics.get("setting_id") != receipt["setting_id"]
        or metrics.get("qualification_only") is not True
        or metrics.get("metrics_sha256") != receipt.get("metrics_sha256")
    ):
        raise Hold30AlphaDriverError("metrics identity differs from run receipt")
    optimizer_evidence = metrics.get("optimizer_update_evidence")
    if route.separate_overlay:
        expected_optimizer_fields = {
            "gradient_isolation_verified",
            "three_stream_contract_verified",
            "gradient_reduction",
            "distributed_world_size",
            "alpha_core_optimizer_steps",
            "overlay_optimizer_steps",
            "alpha_core_gradient_sha256",
            "overlay_gradient_sha256",
            "optimizer_spec_receipt",
            "optimizer_spec_receipt_sha256",
            "initial_optimizer_state_receipt",
            "initial_optimizer_state_receipt_sha256",
            "pre_update_evaluation_point_id",
            "post_update_evaluation_point_id",
            "alpha_core_global_moment_receipt_sha256",
            "executed_global_moment_receipt_sha256",
            "alpha_core_global_metrics",
            "post_update_optimizer_state_receipt",
            "post_update_optimizer_state_receipt_sha256",
        }
        if (
            not isinstance(metrics.get("objective"), dict)
            or set(metrics["objective"]) != {"alpha_core", "overlay"}
            or not isinstance(optimizer_evidence, dict)
            or set(optimizer_evidence) != expected_optimizer_fields
            or optimizer_evidence.get("gradient_isolation_verified") is not True
            or optimizer_evidence.get("three_stream_contract_verified") is not True
            or optimizer_evidence.get("gradient_reduction") != "SUM"
            or optimizer_evidence.get("distributed_world_size") != 1
            or optimizer_evidence.get("alpha_core_optimizer_steps") != 1
            or optimizer_evidence.get("overlay_optimizer_steps") != 1
            or optimizer_evidence.get(
                "optimizer_spec_receipt_sha256"
            )
            != a06_optimizer_spec_receipt_sha256
            or optimizer_evidence.get(
                "initial_optimizer_state_receipt_sha256"
            )
            != a06_initial_receipt_sha256
            or optimizer_evidence.get(
                "post_update_optimizer_state_receipt_sha256"
            )
            != a06_post_receipt_sha256
            or sha256_payload(
                optimizer_evidence.get("optimizer_spec_receipt")
            )
            != a06_optimizer_spec_receipt_sha256
            or sha256_payload(
                optimizer_evidence.get("initial_optimizer_state_receipt")
            )
            != a06_initial_receipt_sha256
            or sha256_payload(
                optimizer_evidence.get("post_update_optimizer_state_receipt")
            )
            != a06_post_receipt_sha256
        ):
            raise Hold30AlphaDriverError("A06 optimizer update evidence is invalid")
        if not isinstance(
            optimizer_evidence["initial_optimizer_state_receipt"], dict
        ) or not isinstance(
            optimizer_evidence["post_update_optimizer_state_receipt"], dict
        ):
            raise Hold30AlphaDriverError("A06 state receipt payload is invalid")
        initial_state_receipt = optimizer_evidence[
            "initial_optimizer_state_receipt"
        ]
        post_state_receipt = optimizer_evidence[
            "post_update_optimizer_state_receipt"
        ]
        if (
            initial_state_receipt.get("optimizer_spec_receipt_sha256")
            != a06_optimizer_spec_receipt_sha256
            or initial_state_receipt.get("update_index") != 0
            or initial_state_receipt.get("parent_state_receipt_sha256") is not None
            or initial_state_receipt.get("evaluation_point_id")
            != optimizer_evidence.get("pre_update_evaluation_point_id")
            or post_state_receipt.get("optimizer_spec_receipt_sha256")
            != a06_optimizer_spec_receipt_sha256
            or post_state_receipt.get("update_index") != 1
            or post_state_receipt.get("parent_state_receipt_sha256")
            != a06_initial_receipt_sha256
            or post_state_receipt.get("evaluation_point_id")
            != optimizer_evidence.get("post_update_evaluation_point_id")
            or optimizer_evidence.get("pre_update_evaluation_point_id")
            == optimizer_evidence.get("post_update_evaluation_point_id")
            or optimizer_evidence.get(
                "alpha_core_global_moment_receipt_sha256"
            )
            == optimizer_evidence.get(
                "executed_global_moment_receipt_sha256"
            )
            or not isinstance(
                optimizer_evidence.get("alpha_core_global_metrics"), dict
            )
        ):
            raise Hold30AlphaDriverError("A06 receipt or stream chain is invalid")
    elif not isinstance(metrics.get("objective"), float) or optimizer_evidence is not None:
        raise Hold30AlphaDriverError("single-optimizer metrics contain invalid evidence")
    try:
        validation = Hold30AlphaValidationMetrics(**metrics["validation"])
    except (TypeError, Hold30AlphaTrainingError) as exc:
        raise Hold30AlphaDriverError(
            "synthetic validation metrics are invalid"
        ) from exc
    if metrics.get("checkpoint_eligible") is not False:
        raise Hold30AlphaDriverError(
            "one-update synthetic checkpoint must never become selection eligible"
        )
    try:
        initial = torch.load(
            root / "initial-checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        final = torch.load(
            root / "update-000001.pt",
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Hold30AlphaDriverError(
            "synthetic checkpoint cannot be loaded safely"
        ) from exc
    expected_initial_checkpoint_fields = {
            "schema",
            "protocol_generation",
            "setting_id",
            "update",
            "qualification_only",
            "model_state",
        }
    expected_final_checkpoint_fields = {
            "schema",
            "protocol_generation",
            "setting_id",
            "update",
            "qualification_only",
            "model_state",
        }
    if route.separate_overlay:
        expected_final_checkpoint_fields.update(
            {
                "alpha_core_optimizer_state",
                "overlay_optimizer_state",
                "optimizer_spec_receipt",
                "initial_optimizer_state_receipt",
                "post_update_optimizer_state_receipt",
            }
        )
    else:
        expected_final_checkpoint_fields.add("optimizer_state")
    for payload, update, fields in zip(
        (initial, final),
        (0, HOLD30_ALPHA_SYNTHETIC_UPDATES),
        (
            expected_initial_checkpoint_fields,
            expected_final_checkpoint_fields,
        ),
        strict=True,
    ):
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema") != HOLD30_ALPHA_SYNTHETIC_CHECKPOINT_SCHEMA
            or payload.get("protocol_generation") != HOLD30_ALPHA_PROTOCOL_GENERATION
            or payload.get("setting_id") != receipt["setting_id"]
            or payload.get("update") != update
            or payload.get("qualification_only") is not True
            or not isinstance(payload.get("model_state"), dict)
        ):
            raise Hold30AlphaDriverError("synthetic checkpoint identity is invalid")
    if route.separate_overlay and (
        final.get("optimizer_spec_receipt")
        != optimizer_evidence["optimizer_spec_receipt"]
        or final.get("initial_optimizer_state_receipt")
        != optimizer_evidence["initial_optimizer_state_receipt"]
        or final.get("post_update_optimizer_state_receipt")
        != optimizer_evidence["post_update_optimizer_state_receipt"]
    ):
        raise Hold30AlphaDriverError("A06 checkpoint optimizer receipts drifted")
    if (
        _state_dict_sha256(initial["model_state"])
        != receipt["initial_model_state_sha256"]
    ):
        raise Hold30AlphaDriverError("initial model-state digest mismatch")
    if _state_dict_sha256(final["model_state"]) != receipt["final_model_state_sha256"]:
        raise Hold30AlphaDriverError("final model-state digest mismatch")
    if route.separate_overlay:
        if objective is None:
            raise AssertionError("A06 verifier lacks its objective config")

        def audit_a06_state(
            model_state: Mapping[str, torch.Tensor],
            *,
            update_index: int,
            parent_state_receipt_sha256: str | None,
            core_optimizer_state: Mapping[str, Any] | None = None,
            overlay_optimizer_state: Mapping[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            audit_policy = DailyCrossSectionPolicy(
                _synthetic_policy_config(route, objective)
            )
            audit_policy.load_state_dict(model_state, strict=True)
            audit_partition = partition_hold30_a06_parameters(
                audit_policy,
                objective,
            )
            audit_core_optimizer = torch.optim.AdamW(
                (parameter for _name, parameter in audit_partition.alpha_core),
                lr=1e-4,
                weight_decay=1e-4,
                eps=1e-5,
            )
            audit_overlay_optimizer = torch.optim.AdamW(
                (parameter for _name, parameter in audit_partition.overlay),
                lr=1e-4,
                weight_decay=1e-4,
                eps=1e-5,
            )
            if core_optimizer_state is not None:
                audit_core_optimizer.load_state_dict(core_optimizer_state)
            if overlay_optimizer_state is not None:
                audit_overlay_optimizer.load_state_dict(overlay_optimizer_state)
            audit_spec = build_hold30_a06_optimizer_spec_receipt(
                audit_partition,
                audit_core_optimizer,
                audit_overlay_optimizer,
            )
            audit_state = build_hold30_a06_optimizer_state_receipt(
                audit_policy,
                audit_partition,
                audit_core_optimizer,
                audit_overlay_optimizer,
                audit_spec,
                update_index=update_index,
                parent_state_receipt_sha256=parent_state_receipt_sha256,
            )
            return audit_spec.manifest_payload(), audit_state.manifest_payload()

        initial_spec, audited_initial_state = audit_a06_state(
            initial["model_state"],
            update_index=0,
            parent_state_receipt_sha256=None,
        )
        final_spec, audited_final_state = audit_a06_state(
            final["model_state"],
            update_index=1,
            parent_state_receipt_sha256=a06_initial_receipt_sha256,
            core_optimizer_state=final["alpha_core_optimizer_state"],
            overlay_optimizer_state=final["overlay_optimizer_state"],
        )
        if (
            initial_spec != optimizer_evidence["optimizer_spec_receipt"]
            or final_spec != optimizer_evidence["optimizer_spec_receipt"]
            or audited_initial_state
            != optimizer_evidence["initial_optimizer_state_receipt"]
            or audited_final_state
            != optimizer_evidence["post_update_optimizer_state_receipt"]
        ):
            raise Hold30AlphaDriverError(
                "A06 checkpoint cannot reconstruct its spec/state receipt chain"
            )
    if (
        receipt["initial_model_state_sha256"] == receipt["final_model_state_sha256"]
        or validation.update != HOLD30_ALPHA_SYNTHETIC_UPDATES
        or metrics.get("minimum_update_satisfied") is not False
    ):
        raise Hold30AlphaDriverError(
            "synthetic one-update checkpoint semantics drifted"
        )
    return receipt


__all__ = [
    "HOLD30_ALPHA_CPU_DISTRIBUTED_PARITY_SCHEMA",
    "HOLD30_ALPHA_CPU_QUALIFICATION_POSITIONS",
    "HOLD30_ALPHA_CPU_QUALIFICATION_SETTINGS",
    "HOLD30_ALPHA_CPU_RESTART_PARITY_SCHEMA",
    "HOLD30_ALPHA_PRODUCTION_IMPLEMENTATION_BLOCKERS",
    "HOLD30_ALPHA_REAL_DATA_ADAPTER_REQUIRED_FIELDS",
    "HOLD30_ALPHA_SYNTHETIC_A06_BINDING",
    "HOLD30_ALPHA_SYNTHETIC_A06_SHARPE_EPSILON",
    "HOLD30_ALPHA_SYNTHETIC_AXIS_ID",
    "HOLD30_ALPHA_SYNTHETIC_CHECKPOINT_SCHEMA",
    "HOLD30_ALPHA_SYNTHETIC_DRIVER_SCHEMA",
    "HOLD30_ALPHA_SYNTHETIC_METRICS_SCHEMA",
    "Hold30AlphaCpuDistributedParityReceipt",
    "Hold30AlphaCpuRestartParityReceipt",
    "Hold30AlphaDriverError",
    "Hold30AlphaProductionPreflightBindings",
    "Hold30AlphaSyntheticRoute",
    "build_hold30_alpha_synthetic_objective_config",
    "qualify_hold30_alpha_full_policy_cpu_restart_parity",
    "qualify_hold30_alpha_full_policy_cpu_two_rank_parity",
    "require_hold30_alpha_executable_plan",
    "resolve_hold30_alpha_synthetic_route",
    "run_hold30_alpha_synthetic_qualification",
    "verify_hold30_alpha_synthetic_run",
]
