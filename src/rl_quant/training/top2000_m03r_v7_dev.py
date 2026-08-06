"""Executable TOP2000 development trainer for the disjoint M03R-v7 panel.

The canonical M03R-v7 protocol remains the point-in-time Active-300 design.
This module is a deliberately separate, nonpromotable compatibility runner for
the currently available pre-2026, future-selected TOP2000 daily-OHLCV cache.
It never falls back to synthetic data and never emits promotion evidence.

Two ranks see the complete cross-section and the same chronological episode.
They split loss-bearing decision origins and SUM gradients once per optimizer
step.  This is pooled two-H100 trajectory training; the stock axis is not
sharded and no legacy recurrent-PPO code is used.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist
from torch import nn

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.execution.hold30 import (
    Hold30BuiltAction,
    build_alpha_hold30_action,
)
from rl_quant.models.daily_policy import (
    DailyCrossSectionConfig,
    DailyCrossSectionPolicy,
    Hold30Intent,
)
from rl_quant.models.hold30_alpha_m03r_v7_top2000_dev import (
    resolve_m03r_top2000_dev_model_route,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
    M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
    M03R_TOP2000_DEV_SETTING_IDS,
    M03RTop2000DevSetting,
    resolve_m03r_top2000_dev_setting,
)
from rl_quant.training.hold30 import (
    Hold30LossContract,
    Hold30OriginReplay,
    Hold30ReplayGeometry,
    detach_tree,
    sequence_coefficients,
)
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalReplayAdapter,
    Hold30ChronologicalRuntime,
    Hold30Policy,
    Hold30Sequence,
)

TOP2000_M03R_V7_DEV_TRAINING_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-two-rank-training-v1"
)
TOP2000_M03R_V7_DEV_FOLD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-six-fold-geometry-v1"
)
TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS = 1001
TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS = 378
TOP2000_M03R_V7_DEV_WARMUP_DECISIONS = 251
TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS = 63
TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS = 63
TOP2000_M03R_V7_DEV_PURGE_DECISIONS = 30
TOP2000_M03R_V7_DEV_FOLD_ADVANCE = 93
TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START = 408
TOP2000_M03R_V7_DEV_SEEDS = (17, 29, 43, 71, 101)
TOP2000_M03R_V7_DEV_FOLD_COUNT = 6
TOP2000_M03R_V7_DEV_COST_RATE = 0.002
TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT = 0.01
TOP2000_M03R_V7_DEV_ACTIVE_RISK_MAX = 0.04
TOP2000_M03R_V7_DEV_FIXED_ACTIVE_RISK = 0.02
TOP2000_M03R_V7_DEV_ALPHA_HORIZONS = (5, 21, 30, 63)
TOP2000_M03R_V7_DEV_ALPHA_HORIZON_WEIGHTS = (0.10, 0.20, 0.50, 0.20)
TOP2000_M03R_V7_DEV_ALPHA_HORIZON_SCALES = tuple(
    0.02 * math.sqrt(horizon)
    for horizon in TOP2000_M03R_V7_DEV_ALPHA_HORIZONS
)
TOP2000_M03R_V7_DEV_AUXILIARY_WEIGHT = 1.0e-4
TOP2000_M03R_V7_DEV_UNCERTAINTY_WEIGHT = 5.0e-5
TOP2000_M03R_V7_DEV_TOTAL_RISK_OVERLAY_STEP = 0.05
TOP2000_M03R_V7_DEV_SHARPE_WEIGHT = 5.0e-5


class Top2000M03RV7DevelopmentError(RuntimeError):
    """A development-only training invariant is absent or inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV7DevelopmentError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def top2000_m03r_v7_persistence_penalty(
    discretionary_sold_value_by_age: torch.Tensor,
    *,
    coefficient_basis_points: float,
    warmup_multiplier: float,
    valid_decision_session_count: int,
) -> torch.Tensor:
    """Return the proportional NAV/session young-sale penalty.

    Mature sales have exactly zero value and zero gradient in this term.  No
    sold-notional denominator exists, so adding mature sales cannot dilute a
    young-position exit.
    """

    sold = discretionary_sold_value_by_age
    if (
        not isinstance(sold, torch.Tensor)
        or sold.ndim != 1
        or sold.numel() != 61
        or not sold.is_floating_point()
        or bool((sold < 0).any())
        or not bool(torch.isfinite(sold).all())
    ):
        raise Top2000M03RV7DevelopmentError(
            "discretionary sold notional must be finite nonnegative [61]"
        )
    if (
        isinstance(coefficient_basis_points, bool)
        or not math.isfinite(float(coefficient_basis_points))
        or coefficient_basis_points < 0.0
        or isinstance(warmup_multiplier, bool)
        or not math.isfinite(float(warmup_multiplier))
        or not 0.0 <= warmup_multiplier <= 1.0
        or isinstance(valid_decision_session_count, bool)
        or not isinstance(valid_decision_session_count, int)
        or valid_decision_session_count <= 0
    ):
        raise Top2000M03RV7DevelopmentError(
            "persistence coefficient, warmup, or valid-session count is invalid"
        )
    ages = torch.arange(61, device=sold.device, dtype=sold.dtype)
    weights = ((30.0 - ages).clamp_min(0.0) / 30.0).square()
    weighted_notional = (sold * weights).sum()
    return weighted_notional * (
        float(coefficient_basis_points)
        * 1.0e-4
        * float(warmup_multiplier)
        / float(valid_decision_session_count)
    )


def _auxiliary_alpha_loss(
    replay: Hold30OriginReplay,
    sequence: Hold30Sequence,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    prediction = replay.auxiliary_alpha_mean
    if prediction is None:
        return None, None
    if tuple(prediction.shape[:2]) != (
        sequence.batch_size,
        sequence.num_assets,
    ) or prediction.shape[-1] != len(TOP2000_M03R_V7_DEV_ALPHA_HORIZONS):
        raise Top2000M03RV7DevelopmentError(
            "origin auxiliary prediction is not [batch,asset,4]"
        )
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    uncertainty_loss: torch.Tensor | None = None
    transition_count = int(sequence.asset_returns.shape[0])
    for index, (horizon, horizon_weight, scale) in enumerate(
        zip(
            TOP2000_M03R_V7_DEV_ALPHA_HORIZONS,
            TOP2000_M03R_V7_DEV_ALPHA_HORIZON_WEIGHTS,
            TOP2000_M03R_V7_DEV_ALPHA_HORIZON_SCALES,
            strict=True,
        )
    ):
        first = replay.origin + 1
        stop = first + horizon
        if stop > transition_count:
            continue
        stock_log = torch.log1p(
            sequence.asset_returns[first:stop].clamp_min(-0.999999)
        ).sum(0)
        benchmark_log = torch.log1p(
            sequence.benchmark_net_returns[first:stop].clamp_min(-0.999999)
        ).sum(0)
        target = stock_log - benchmark_log.unsqueeze(-1)
        valid = sequence.decision_available[first : stop + 1].all(0)
        valid = valid.clone()
        valid[:, 0] = False
        if not bool(valid.any()):
            continue
        error = prediction[..., index] - target.detach()
        losses.append(error.div(float(scale)).square().masked_select(valid).mean())
        weights.append(float(horizon_weight))
        if horizon == 30 and replay.alpha_downside_30d is not None:
            downside = replay.alpha_downside_30d.clamp_min(1.0e-6)
            uncertainty_loss = (
                0.5 * error.div(downside).square() + torch.log(downside)
            ).masked_select(valid).mean()
    if not losses:
        raise Top2000M03RV7DevelopmentError(
            "alpha-head origin has no causally supported future target"
        )
    normalizer = sum(weights)
    auxiliary = losses[0].new_zeros(())
    for loss, weight in zip(losses, weights, strict=True):
        auxiliary = auxiliary + loss * (weight / normalizer)
    return auxiliary, uncertainty_loss


@dataclass(frozen=True, slots=True)
class Top2000M03RV7DevelopmentFold:
    """Six expanding development folds supported by the 1001-state cache.

    The geometry is intentionally not the canonical PIT-300 outer protocol.
    Each validation period has 63 decisions, preceded by a 30-decision purge.
    Earlier development validation rows may enter later expanding folds.
    """

    fold_index: int
    training_state_start: int
    training_state_stop_exclusive: int
    purge_start: int
    purge_stop_exclusive: int
    validation_decision_start: int
    validation_decision_stop_exclusive: int
    holding_support_stop_exclusive: int
    schema: str = TOP2000_M03R_V7_DEV_FOLD_SCHEMA

    def __post_init__(self) -> None:
        expected_validation_start = (
            TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
            + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * self.fold_index
        )
        if (
            self.schema != TOP2000_M03R_V7_DEV_FOLD_SCHEMA
            or not 0 <= self.fold_index < TOP2000_M03R_V7_DEV_FOLD_COUNT
            or self.training_state_start != 0
            or self.training_state_stop_exclusive
            != expected_validation_start - TOP2000_M03R_V7_DEV_PURGE_DECISIONS
            or self.purge_start != self.training_state_stop_exclusive
            or self.purge_stop_exclusive != expected_validation_start
            or self.validation_decision_start != expected_validation_start
            or self.validation_decision_stop_exclusive
            != expected_validation_start
            + TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
            or self.holding_support_stop_exclusive
            != self.validation_decision_stop_exclusive
            + TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS
            or self.training_state_stop_exclusive
            < TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS
            or self.holding_support_stop_exclusive
            > TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
        ):
            raise Top2000M03RV7DevelopmentError(
                "TOP2000 development fold geometry drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def render_top2000_m03r_v7_development_folds(
    state_rows: int,
) -> tuple[Top2000M03RV7DevelopmentFold, ...]:
    """Return the sole six-fold compatibility geometry for this cache."""

    if state_rows != TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS:
        raise Top2000M03RV7DevelopmentError(
            "TOP2000 v7 development folds require exactly 1001 cache states"
        )
    return tuple(
        Top2000M03RV7DevelopmentFold(
            fold_index=index,
            training_state_start=0,
            training_state_stop_exclusive=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
                - TOP2000_M03R_V7_DEV_PURGE_DECISIONS
            ),
            purge_start=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
                - TOP2000_M03R_V7_DEV_PURGE_DECISIONS
            ),
            purge_stop_exclusive=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
            ),
            validation_decision_start=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
            ),
            validation_decision_stop_exclusive=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
                + TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
            ),
            holding_support_stop_exclusive=(
                TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                + TOP2000_M03R_V7_DEV_FOLD_ADVANCE * index
                + TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
                + TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS
            ),
        )
        for index in range(TOP2000_M03R_V7_DEV_FOLD_COUNT)
    )


def _source_v6_setting_id(setting: M03RTop2000DevSetting) -> str:
    route = resolve_m03r_top2000_dev_model_route(setting.setting_id)
    if route.source_v6_model_setting_id is not None:
        return route.source_v6_model_setting_id
    if setting.active_risk_budget_mode == M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE:
        # A12 changes execution sizing only.  Its alpha/hazard architecture is
        # exactly canonical; the wrapper below replaces the risk scale with a
        # bound 2% constant after the native model emits its raw intent.
        return "M03R-soft-persistence-active-alpha-hold30"
    raise Top2000M03RV7DevelopmentError(
        f"no truthful native policy capability for {setting.setting_id}"
    )


def _detached_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach()


def _route_hold30_intent_gradients(
    intent: Hold30Intent,
    *,
    gradient_route: str,
) -> Hold30Intent:
    """Make the A06 alpha-core and overlay autograd graphs disjoint.

    The overlay head already consumes a detached market summary in the native
    policy.  This boundary additionally detaches the overlay from the alpha
    pass and every alpha/hazard/risk tensor from the overlay pass.  Therefore
    the two optimizers cannot update one another's parameter set through the
    economic ledger even though they share the same chronological execution.
    """

    if gradient_route == "alpha-core":
        return replace(
            intent,
            total_risk_overlay=_detached_tensor(intent.total_risk_overlay),
        )
    if gradient_route != "total-risk-overlay":
        raise Top2000M03RV7DevelopmentError(
            f"unknown development gradient route {gradient_route!r}"
        )
    if intent.total_risk_overlay is None:
        raise Top2000M03RV7DevelopmentError(
            "the total-risk-overlay route requires the registered A06 head"
        )
    return Hold30Intent(
        entry_scores=_detached_tensor(intent.entry_scores),
        target_logits=_detached_tensor(intent.target_logits),
        gate=_detached_tensor(intent.gate),
        hazard_residual=_detached_tensor(intent.hazard_residual),
        raw_hazard_residual=_detached_tensor(intent.raw_hazard_residual),
        exact_hold_probability=_detached_tensor(intent.exact_hold_probability),
        exact_hold_logit=_detached_tensor(intent.exact_hold_logit),
        exact_hold_soft_probability=_detached_tensor(
            intent.exact_hold_soft_probability
        ),
        exact_hold_decision_st=_detached_tensor(intent.exact_hold_decision_st),
        exposure_residual=_detached_tensor(intent.exposure_residual),
        alpha_mean_30d=_detached_tensor(intent.alpha_mean_30d),
        alpha_downside_30d=_detached_tensor(intent.alpha_downside_30d),
        active_risk_scale=_detached_tensor(intent.active_risk_scale),
        signal_confidence=_detached_tensor(intent.signal_confidence),
        uncalibrated_signal_confidence_logit=_detached_tensor(
            intent.uncalibrated_signal_confidence_logit
        ),
        benchmark_derisk_request=_detached_tensor(
            intent.benchmark_derisk_request
        ),
        # This is the sole gradient-bearing field on the overlay route.
        total_risk_overlay=intent.total_risk_overlay,
        auxiliary_alpha_mean=_detached_tensor(intent.auxiliary_alpha_mean),
        exit_action_v6=(
            None
            if intent.exit_action_v6 is None
            else intent.exit_action_v6.clone(detach=True)
        ),
    )


def top2000_m03r_v7_factor_neutral_executed_weights(
    requested_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    factor_loadings: torch.Tensor,
    trade_mask: torch.Tensor,
    risk_asset_caps: torch.Tensor,
    risk_gross_max: torch.Tensor,
    *,
    cash_index: int = 0,
    factor_constraint_pinv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project executed active weights into the calibrated factor nullspace.

    This is deliberately an execution-book operation, not score
    residualization.  It first applies the exact orthogonal nullspace
    projection for the simplex and calibrated factor equalities.  A single
    benchmark-radial scale then repairs long-only, availability, per-name cap,
    and gross-risk constraints.  Radial repair preserves all equalities and is
    differentiable almost everywhere with respect to the requested book.

    The TOP2000 compatibility cache has no qualified sector manifest, so this
    helper neutralizes the four bound pre-episode price/volume controls only.
    Its distinct development identity must not be presented as canonical PIT
    factor/sector projection evidence.
    """

    if (
        not isinstance(requested_weights, torch.Tensor)
        or requested_weights.ndim != 2
        or not requested_weights.is_floating_point()
        or not bool(torch.isfinite(requested_weights).all())
    ):
        raise Top2000M03RV7DevelopmentError(
            "requested_weights must be finite floating [batch,asset]"
        )
    batch, assets = requested_weights.shape
    expected = (batch, assets)
    for name, value in (
        ("benchmark_weights", benchmark_weights),
        ("risk_asset_caps", risk_asset_caps),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise Top2000M03RV7DevelopmentError(
                f"{name} must be finite floating {expected}"
            )
    if (
        not isinstance(trade_mask, torch.Tensor)
        or tuple(trade_mask.shape) != expected
        or trade_mask.dtype != torch.bool
        or not isinstance(factor_loadings, torch.Tensor)
        or factor_loadings.ndim != 2
        or factor_loadings.shape[0] != assets
        or factor_loadings.shape[1] < 1
        or factor_loadings.requires_grad
        or not factor_loadings.is_floating_point()
        or not bool(torch.isfinite(factor_loadings).all())
        or not isinstance(risk_gross_max, torch.Tensor)
        or tuple(risk_gross_max.shape) != (batch,)
        or not risk_gross_max.is_floating_point()
        or not bool(torch.isfinite(risk_gross_max).all())
        or not 0 <= cash_index < assets
    ):
        raise Top2000M03RV7DevelopmentError(
            "factor projection inputs are unbound or misaligned"
        )
    work_dtype = (
        torch.float64
        if requested_weights.dtype == torch.float64
        else torch.float32
    )
    requested = requested_weights.to(dtype=work_dtype)
    benchmark = benchmark_weights.to(dtype=work_dtype)
    loadings = factor_loadings.to(
        device=requested.device,
        dtype=work_dtype,
    )
    # CASH carries no factor identity.  Refuse a malformed calibration rather
    # than allowing the factor solver to redefine the risk-free coordinate.
    if not bool(torch.allclose(loadings[cash_index], torch.zeros_like(loadings[cash_index]))):
        raise Top2000M03RV7DevelopmentError(
            "CASH factor loadings must be exactly zero"
        )
    constraints = torch.cat(
        [
            torch.ones((assets, 1), device=requested.device, dtype=work_dtype),
            loadings,
        ],
        dim=1,
    )
    # The calibration tensors are detached; pinv is consequently a constant
    # linear map for autograd while remaining robust to redundant controls.
    if factor_constraint_pinv is None:
        constraint_pinv = torch.linalg.pinv(constraints)
    else:
        if (
            tuple(factor_constraint_pinv.shape)
            != (constraints.shape[1], assets)
            or factor_constraint_pinv.requires_grad
            or not factor_constraint_pinv.is_floating_point()
            or not bool(torch.isfinite(factor_constraint_pinv).all())
        ):
            raise Top2000M03RV7DevelopmentError(
                "cached factor-constraint pseudoinverse is invalid"
            )
        constraint_pinv = factor_constraint_pinv.to(
            device=requested.device,
            dtype=work_dtype,
        )
    risky = torch.ones(expected, dtype=torch.bool, device=requested.device)
    risky[:, cash_index] = False
    available = trade_mask.to(device=requested.device).clone()
    available[:, cash_index] = True
    caps = torch.minimum(
        risk_asset_caps.to(dtype=work_dtype).clamp_min(0.0),
        requested.new_tensor(TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT),
    )
    upper = torch.where(risky & available, caps, torch.zeros_like(caps))
    upper[:, cash_index] = 1.0
    tolerance = 5.0e-6
    if bool((benchmark < -tolerance).any()) or bool(
        (benchmark - upper > tolerance).any()
    ):
        raise Top2000M03RV7DevelopmentError(
            "the fill-time benchmark is not feasible under availability/caps"
        )
    benchmark_gross = torch.where(risky, benchmark, torch.zeros_like(benchmark)).sum(-1)
    if bool((benchmark_gross - risk_gross_max.to(work_dtype) > tolerance).any()):
        raise Top2000M03RV7DevelopmentError(
            "the fill-time benchmark exceeds the gross-risk ceiling"
        )

    # Project only in the fill-tradable subspace.  A global pseudoinverse can
    # introduce active weights on a masked name even when both the requested
    # and benchmark books are zero there; the subsequent radial feasibility
    # repair would then scale the entire active book to zero.  The small
    # per-row Gram system keeps unavailable coordinates fixed while enforcing
    # the simplex and factor equalities on the executable coordinates.
    active = torch.where(available, requested - benchmark, torch.zeros_like(requested))
    if bool(available.all()):
        neutral_active = active - (active @ constraints) @ constraint_pinv
    else:
        masked_constraints = constraints.unsqueeze(0) * available.to(
            dtype=work_dtype
        ).unsqueeze(-1)
        moments = torch.einsum("ba,bak->bk", active, masked_constraints)
        gram = torch.einsum(
            "bak,bal->bkl",
            masked_constraints,
            masked_constraints,
        )
        gram_pinv = torch.linalg.pinv(gram, hermitian=True)
        coefficients = torch.bmm(moments.unsqueeze(1), gram_pinv).squeeze(1)
        correction = torch.einsum(
            "bk,bak->ba",
            coefficients,
            masked_constraints,
        )
        neutral_active = torch.where(
            available,
            active - correction,
            torch.zeros_like(active),
        )

    positive_ratio = torch.where(
        neutral_active > 0,
        (upper - benchmark).clamp_min(0.0)
        / neutral_active.clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(neutral_active, torch.inf),
    )
    negative_ratio = torch.where(
        neutral_active < 0,
        benchmark.clamp_min(0.0)
        / (-neutral_active).clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(neutral_active, torch.inf),
    )
    scale = torch.minimum(positive_ratio.amin(-1), negative_ratio.amin(-1))
    active_gross_delta = torch.where(
        risky,
        neutral_active,
        torch.zeros_like(neutral_active),
    ).sum(-1)
    gross_ratio = torch.where(
        active_gross_delta > 0,
        (risk_gross_max.to(work_dtype) - benchmark_gross).clamp_min(0.0)
        / active_gross_delta.clamp_min(torch.finfo(work_dtype).tiny),
        torch.full_like(active_gross_delta, torch.inf),
    )
    scale = torch.minimum(scale, gross_ratio).clamp(0.0, 1.0)
    projected = benchmark + scale.unsqueeze(-1) * neutral_active
    projected = projected.to(dtype=requested_weights.dtype)
    exposure = (
        (projected - benchmark_weights)
        @ factor_loadings.to(device=projected.device, dtype=projected.dtype)
    )
    if (
        not bool(
            torch.allclose(
                projected.sum(-1),
                torch.ones(batch, device=projected.device, dtype=projected.dtype),
                atol=2.0e-5,
                rtol=2.0e-5,
            )
        )
        or float(exposure.detach().abs().max()) > 2.0e-4
    ):
        raise Top2000M03RV7DevelopmentError(
            "executed active-weight factor projection failed reconciliation"
        )
    return cast(torch.Tensor, projected)


class Top2000M03RV7ActionPolicy(Protocol):
    """Execution-bearing surface shared by seed and ensemble policies."""

    setting: M03RTop2000DevSetting
    episode_factor_loadings: torch.Tensor
    episode_factor_constraint_pinv: torch.Tensor


class Top2000M03RV7ActionBuilder:
    """Generation-local differentiable execution builder."""

    def __init__(
        self,
        policy: Top2000M03RV7ActionPolicy,
    ) -> None:
        self.policy = policy

    def __call__(
        self,
        intent: Hold30Intent,
        repaired_ledger: CohortLedger,
        benchmark_weights: torch.Tensor,
        trade_mask: torch.Tensor,
        risk_asset_caps: torch.Tensor,
        risk_gross_max: torch.Tensor,
    ) -> Hold30BuiltAction:
        built = build_alpha_hold30_action(
            repaired_ledger.weights,
            repaired_ledger.economic_value,
            cast(torch.Tensor, intent.entry_scores),
            cast(torch.Tensor, intent.hazard_residual),
            cast(torch.Tensor, intent.active_risk_scale),
            benchmark_weights,
            trade_mask,
            risk_asset_caps,
            risk_gross_max,
            exact_hold_probability=(
                intent.exact_hold_decision_st
                if intent.exact_hold_decision_st is not None
                else intent.exact_hold_probability
            ),
            exit_action_v6=intent.exit_action_v6,
            total_risk_overlay=intent.total_risk_overlay,
            total_risk_step=(
                TOP2000_M03R_V7_DEV_TOTAL_RISK_OVERLAY_STEP
                if intent.total_risk_overlay is not None
                else None
            ),
            cash_index=repaired_ledger.cash_index,
        )
        if not self.policy.setting.factor_sector_neutral_projection:
            return built
        loadings = self.policy.episode_factor_loadings
        if loadings.shape[0] != built.target_weights.shape[1]:
            raise Top2000M03RV7DevelopmentError(
                "factor loadings are not bound to the current execution axis"
            )
        target = top2000_m03r_v7_factor_neutral_executed_weights(
            built.target_weights,
            benchmark_weights,
            loadings,
            trade_mask,
            risk_asset_caps,
            risk_gross_max,
            cash_index=repaired_ledger.cash_index,
            factor_constraint_pinv=(
                self.policy.episode_factor_constraint_pinv
            ),
        )
        constructed = target - repaired_ledger.weights
        risky = torch.ones_like(target, dtype=torch.bool)
        risky[:, repaired_ledger.cash_index] = False
        return replace(
            built,
            target_weights=target,
            # Preserve the actor's pre-projection request separately; the
            # constructed delta is the exact executable factor-neutral book.
            constructed_delta=constructed,
            constructed_turnover=0.5 * constructed.abs().sum(-1),
            desired_risky_exposure=torch.where(
                risky,
                target,
                torch.zeros_like(target),
            ).sum(-1),
        )


def top2000_m03r_v7_policy_config(
    setting_id: str,
    *,
    token_dim: int = 128,
    raw_stock_chunk: int = 512,
) -> DailyCrossSectionConfig:
    """Build the compact daily-OHLCV policy backing one development row."""

    setting = resolve_m03r_top2000_dev_setting(setting_id)
    source_id = _source_v6_setting_id(setting)
    uses_downside = setting.residual_alpha_head_mode == "mean-and-downside"
    uses_confidence = (
        setting.active_risk_budget_mode
        in {
            M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE,
            M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE,
        }
    )
    fixed_hazard = (
        0.0
        if setting.exit_hazard_mode == "fixed-structural-30-session-prior"
        else None
    )
    slow = setting.learned_temporal_context_trading_sessions
    return DailyCrossSectionConfig(
        context_dim=1,
        bar_feature_dim=5,
        raw_policy_dim=64,
        raw_policy_layers=2,
        raw_policy_heads=4,
        # The cache row is one daily OHLCV token aggregated from its bound
        # 300-second source.  These are token counts, not a claim of raw-second
        # input at this development boundary.
        raw_block_seconds=1,
        session_seconds=1,
        news_raw_dim=1,
        max_news=1,
        news_embed_dim=8,
        token_dim=token_dim,
        temporal_layers=2,
        temporal_heads=4,
        daily_lookback=slow,
        temporal_attention_lookback=slow,
        max_days=TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
        alloc_layers=2,
        alloc_heads=4,
        feedforward_dim=512,
        dropout=0.0,
        temperature=1.0,
        max_stock_weight=TOP2000_M03R_V7_DEV_MAX_STOCK_WEIGHT,
        grad_checkpoint=True,
        raw_norm="level",
        raw_recent_days=42,
        encode_aggregated_daily_ohlcv_all_days=True,
        raw_stock_chunk=raw_stock_chunk,
        hold30_setting=source_id,
        alpha_downside_penalty_kappa=0.25 if uses_downside else None,
        alpha_active_log_scale_bounds=None,
        alpha_uncertainty_log_scale_bounds=(math.log(0.01), 0.0)
        if uses_downside
        else None,
        hold30_mechanism_generation="m03r-v3",
        hold30_fast_raw_context_sessions=42,
        hold30_slow_context_sessions=slow,
        hold30_hazard_bound_mode="smooth_tanh",
        hold30_exact_hold_mixture=False,
        hold30_exact_hold_logit_bias=None,
        hold30_fixed_hazard_residual=fixed_hazard,
        alpha_m03r_v6_confidence_stage=(
            "v6-training-uncalibrated" if uses_confidence else None
        ),
    )


class Top2000M03RV7DevelopmentPolicy(nn.Module):
    """Outer development identity plus a generation-qualified native policy."""

    episode_factor_loadings: torch.Tensor
    episode_factor_constraint_pinv: torch.Tensor
    state_provider_compatibility_id = (
        "top2000-m03r-v7-daily-ohlcv-causal-episode-encoder-v1"
    )

    def __init__(
        self,
        setting_id: str,
        *,
        token_dim: int = 128,
        raw_stock_chunk: int = 512,
    ) -> None:
        super().__init__()
        self.setting = resolve_m03r_top2000_dev_setting(setting_id)
        self.core = DailyCrossSectionPolicy(
            top2000_m03r_v7_policy_config(
                setting_id,
                token_dim=token_dim,
                raw_stock_chunk=raw_stock_chunk,
            )
        )
        self.register_buffer(
            "episode_factor_loadings",
            torch.empty((0, 0), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "episode_factor_constraint_pinv",
            torch.empty((0, 0), dtype=torch.float32),
            persistent=False,
        )
        self._gradient_route = "alpha-core"

    @property
    def token_dim(self) -> int:
        return int(self.core.token_dim)

    def bind_episode_factor_loadings(self, loadings: torch.Tensor) -> None:
        if (
            not isinstance(loadings, torch.Tensor)
            or loadings.ndim != 2
            or loadings.shape[0] < 2
            or loadings.shape[1] < 1
            or not loadings.is_floating_point()
            or not bool(torch.isfinite(loadings).all())
        ):
            raise Top2000M03RV7DevelopmentError(
                "episode factor loadings must be finite floating [asset,factor]"
            )
        bound = loadings.detach().to(
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        constraints = torch.cat(
            [
                torch.ones(
                    (bound.shape[0], 1),
                    device=bound.device,
                    dtype=bound.dtype,
                ),
                bound,
            ],
            dim=1,
        )
        self.episode_factor_loadings = bound
        self.episode_factor_constraint_pinv = torch.linalg.pinv(
            constraints
        ).detach()

    def set_gradient_route(self, route: str) -> None:
        if route not in {"alpha-core", "total-risk-overlay"}:
            raise Top2000M03RV7DevelopmentError(
                f"unknown development gradient route {route!r}"
            )
        if route == "total-risk-overlay" and not self.total_risk_overlay_parameters():
            raise Top2000M03RV7DevelopmentError(
                "only A06 has a total-risk-overlay gradient route"
            )
        self._gradient_route = route

    def total_risk_overlay_parameters(self) -> tuple[nn.Parameter, ...]:
        head = self.core.alpha_head
        overlay = None if head is None else head.total_risk_head
        return () if overlay is None else tuple(overlay.parameters())

    def alpha_core_parameters(self) -> tuple[nn.Parameter, ...]:
        overlay_ids = {
            id(parameter) for parameter in self.total_risk_overlay_parameters()
        }
        return tuple(
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in overlay_ids
        )

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.core.encode_episode(*args, **kwargs),  # type: ignore[no-untyped-call]
        )

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        # Episode encoding runs under BF16 autocast, but the chronological
        # runtime invokes this decision head after the encoder scope closes.
        # Re-enter autocast for the complete allocator/alpha-head path so its
        # BF16 state can be consumed by FP32 parameters without promoting the
        # large per-decision activation surface to FP32.
        use_bfloat16_autocast = (
            state_t.device.type == "cuda" or state_t.dtype == torch.bfloat16
        )
        with torch.autocast(
            device_type=state_t.device.type,
            dtype=torch.bfloat16,
            enabled=use_bfloat16_autocast,
        ):
            intent = self.core.hold30_intent(
                state_t,
                prev_weights,
                available,
                age_summaries,
            )
        if intent.entry_scores is None:
            raise Top2000M03RV7DevelopmentError(
                "TOP2000 v7 development rows require H2 entry scores"
            )
        risk_scale = intent.active_risk_scale
        if (
            self.setting.active_risk_budget_mode
            == M03R_V7_FIXED_2PCT_ACTIVE_RISK_BUDGET_MODE
        ):
            risk_scale = prev_weights.new_full(
                (prev_weights.shape[0],), TOP2000_M03R_V7_DEV_FIXED_ACTIVE_RISK
            )
        elif (
            self.setting.active_risk_budget_mode
            == M03R_V7_CALIBRATED_ACTIVE_RISK_BUDGET_MODE
        ):
            raw = intent.uncalibrated_signal_confidence_logit
            if raw is None:
                raise Top2000M03RV7DevelopmentError(
                    "confidence-budgeted route omitted its raw confidence logit"
                )
            # This cache-only development panel has no promotion-calibrated
            # confidence artifact.  Its distinct identity binds the raw
            # sigmoid development sizing; it must never be called calibrated
            # PIT-300 evidence.
            risk_scale = TOP2000_M03R_V7_DEV_ACTIVE_RISK_MAX * torch.sigmoid(raw)
        if risk_scale is None:
            raise Top2000M03RV7DevelopmentError(
                "TOP2000 v7 development route omitted active-risk scale"
            )
        routed = replace(
            intent,
            # Factor control belongs to the executed active book.  Keeping
            # scores untouched makes A10 differ only at the execution repair.
            entry_scores=intent.entry_scores,
            active_risk_scale=risk_scale,
            signal_confidence=(
                None
                if intent.uncalibrated_signal_confidence_logit is None
                else torch.sigmoid(intent.uncalibrated_signal_confidence_logit)
            ),
        )
        return _route_hold30_intent_gradients(
            routed,
            gradient_route=self._gradient_route,
        )


@dataclass(frozen=True, slots=True)
class Top2000M03RV7DecisionInputs:
    """Causal daily-OHLCV tensors owned by one training episode."""

    daily_ohlcv: torch.Tensor  # [batch,time,asset,5]
    availability: torch.Tensor  # [batch,time,asset]
    past_returns: torch.Tensor  # [batch,time,asset]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.daily_ohlcv, torch.Tensor)
            or self.daily_ohlcv.ndim != 4
            or self.daily_ohlcv.shape[-1] != 5
            or not self.daily_ohlcv.is_floating_point()
            or not bool(torch.isfinite(self.daily_ohlcv).all())
        ):
            raise Top2000M03RV7DevelopmentError(
                "daily_ohlcv must be finite floating [batch,time,asset,5]"
            )
        expected = self.daily_ohlcv.shape[:3]
        if (
            not isinstance(self.availability, torch.Tensor)
            or tuple(self.availability.shape) != expected
            or self.availability.dtype != torch.bool
            or not isinstance(self.past_returns, torch.Tensor)
            or tuple(self.past_returns.shape) != expected
            or self.past_returns.dtype != self.daily_ohlcv.dtype
            or self.past_returns.device != self.daily_ohlcv.device
            or not bool(torch.isfinite(self.past_returns).all())
        ):
            raise Top2000M03RV7DevelopmentError(
                "availability and past_returns must align with daily_ohlcv"
            )


class Top2000M03RV7DecisionStateProvider:
    """Differentiably recompute the causal daily/temporal encoder per replay."""

    trains_upstream_encoder = True

    def __init__(self, inputs: Top2000M03RV7DecisionInputs) -> None:
        self.inputs = inputs

    def _states(
        self,
        policy: Hold30Policy,
    ) -> torch.Tensor:
        if (
            getattr(policy, "state_provider_compatibility_id", None)
            != Top2000M03RV7DevelopmentPolicy.state_provider_compatibility_id
            or not callable(getattr(policy, "encode_episode", None))
        ):
            raise Top2000M03RV7DevelopmentError(
                "TOP2000 state provider requires a generation-qualified "
                "development episode encoder"
            )
        bars = self.inputs.daily_ohlcv.unsqueeze(-2)
        bar_mask = self.inputs.availability.unsqueeze(-1)
        volume_context = (
            torch.log1p(self.inputs.daily_ohlcv[..., 4].clamp_min(0.0)) / 20.0
        ).unsqueeze(-1)
        available_float = self.inputs.availability.to(volume_context.dtype)
        market_context = (
            (volume_context.squeeze(-1) * available_float).sum(-1)
            / available_float.sum(-1).clamp_min(1.0)
        ).unsqueeze(-1)
        batch, time, assets = self.inputs.availability.shape
        news_raw = bars.new_zeros((batch, time, assets, 1, 1))
        news_mask = torch.zeros(
            (batch, time, assets, 1),
            dtype=torch.bool,
            device=bars.device,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if bars.device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            encoder = cast(Any, policy)
            return cast(torch.Tensor, encoder.encode_episode(
                market_context,
                volume_context,
                bars,
                bar_mask,
                news_raw,
                news_mask,
                self.inputs.availability,
                self.inputs.past_returns,
                self.inputs.availability,
            ))

    def canonical_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
    ) -> torch.Tensor:
        del sequence
        return self._states(policy).transpose(0, 1)[:-1]

    def replay_origin_state(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origin: int,
    ) -> torch.Tensor:
        del sequence
        return self._states(policy)[:, origin]

    def replay_origin_states(
        self,
        policy: Hold30Policy,
        sequence: Hold30Sequence,
        origins: torch.Tensor,
    ) -> torch.Tensor:
        del sequence
        states = self._states(policy)
        indexes = origins.to(device=states.device, dtype=torch.long)
        return states.index_select(1, indexes).transpose(0, 1)


def top2000_m03r_v7_decision_inputs(
    sequence: Hold30Sequence,
) -> Top2000M03RV7DecisionInputs:
    """Extract causal raw inputs before replacing the runtime state placeholder."""

    raw = sequence.decision_state.transpose(0, 1).contiguous()
    if raw.shape[-1] != 5:
        raise Top2000M03RV7DevelopmentError(
            "TOP2000 adapter must supply daily OHLCV as its decision placeholder"
        )
    returns = sequence.asset_returns.transpose(0, 1)
    past = torch.zeros_like(raw[..., 0])
    past[:, 1:] = returns
    return Top2000M03RV7DecisionInputs(
        daily_ohlcv=raw,
        availability=sequence.decision_available.transpose(0, 1).contiguous(),
        past_returns=past,
    )


def bind_top2000_m03r_v7_runtime_sequence(
    sequence: Hold30Sequence,
    policy: Top2000M03RV7DevelopmentPolicy,
) -> tuple[Hold30Sequence, Top2000M03RV7DecisionStateProvider]:
    inputs = top2000_m03r_v7_decision_inputs(sequence)
    placeholder = torch.zeros(
        (
            sequence.n_positions,
            sequence.batch_size,
            sequence.num_assets,
            policy.token_dim,
        ),
        dtype=sequence.asset_returns.dtype,
        device=sequence.asset_returns.device,
    )
    bound = replace(sequence, decision_state=placeholder)
    return bound, Top2000M03RV7DecisionStateProvider(inputs)


def top2000_m03r_v7_total_excess_sharpe(
    active_log_return: torch.Tensor,
    benchmark_simple_return: torch.Tensor,
    cash_simple_return: torch.Tensor,
) -> torch.Tensor:
    """Investor-facing total simple-return Sharpe over cash.

    The alpha core is parameterized through benchmark-relative log utility.
    Convert that variable exactly back to the policy's simple return before
    subtracting the contemporaneous cash return.  This prevents A07 from
    accidentally optimizing active-return IR under a Sharpe label.
    """

    if (
        active_log_return.ndim != 1
        or benchmark_simple_return.shape != active_log_return.shape
        or cash_simple_return.shape != active_log_return.shape
        or not active_log_return.is_floating_point()
        or not benchmark_simple_return.is_floating_point()
        or not cash_simple_return.is_floating_point()
        or not bool(torch.isfinite(active_log_return).all())
        or not bool(torch.isfinite(benchmark_simple_return).all())
        or not bool(torch.isfinite(cash_simple_return).all())
        or bool((benchmark_simple_return <= -1).any())
    ):
        raise Top2000M03RV7DevelopmentError(
            "Sharpe inputs must be aligned finite daily vectors"
        )
    policy_simple_return = torch.expm1(
        active_log_return + torch.log1p(benchmark_simple_return)
    )
    excess = policy_simple_return - cash_simple_return
    return excess.mean() / excess.std(unbiased=False).clamp_min(1.0e-3)


def top2000_m03r_v7_direct_sharpe_vjp_coefficients(
    active_log_return: torch.Tensor,
    benchmark_simple_return: torch.Tensor,
    cash_simple_return: torch.Tensor,
    moment_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detached two-pass coefficients for the complete effective batch.

    Pass one owns the full chronological moment calculation.  The returned
    coefficients are detached and can be multiplied by gradient-bearing
    replay utilities in pass two without averaging per-microbatch Sharpes.
    """

    rows = torch.as_tensor(moment_rows, dtype=torch.long, device="cpu")
    if rows.ndim != 1 or rows.numel() < 2 or len(set(rows.tolist())) != rows.numel():
        raise Top2000M03RV7DevelopmentError(
            "Sharpe moment rows must be a unique one-dimensional inventory"
        )
    variable = active_log_return.detach().to(torch.float64).requires_grad_(True)
    benchmark = benchmark_simple_return.detach().to(torch.float64)
    cash = cash_simple_return.detach().to(torch.float64)
    indexes = rows.to(device=variable.device)
    sharpe = top2000_m03r_v7_total_excess_sharpe(
        variable.index_select(0, indexes),
        benchmark.index_select(0, indexes),
        cash.index_select(0, indexes),
    )
    gradient = torch.autograd.grad(sharpe, variable)[0]
    return gradient.detach(), sharpe.detach()


def _objective_row_coefficients(
    trace: Hold30CanonicalTrace,
    setting: M03RTop2000DevSetting,
    *,
    cash_simple_returns: torch.Tensor,
    moment_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached VJP coefficients for active risk and Sharpe terms."""

    active = torch.stack(
        [transition.utility.mean() for transition in trace.transitions]
    ).detach().to(dtype=torch.float64)
    benchmark = torch.stack(
        [transition.benchmark_net_return.mean() for transition in trace.transitions]
    ).detach().to(dtype=torch.float64)
    cash = cash_simple_returns.detach().to(dtype=torch.float64)
    if cash.shape != active.shape:
        raise Top2000M03RV7DevelopmentError(
            "cash-return chronology must align with canonical transitions"
        )
    indexes = torch.as_tensor(moment_rows, dtype=torch.long, device=active.device)
    if indexes.ndim != 1 or indexes.numel() < 2:
        raise Top2000M03RV7DevelopmentError(
            "objective moments require at least two owned utility rows"
        )
    selected_active = active.index_select(0, indexes)
    selected_benchmark = benchmark.index_select(0, indexes)
    selected_cash = cash.index_select(0, indexes)
    market = torch.log1p(selected_benchmark)
    variable = active.clone().requires_grad_(True)
    selected_variable = variable.index_select(0, indexes)
    objective = selected_variable.mean()
    annual_te = selected_variable.std(unbiased=False) * math.sqrt(252.0)
    objective = objective - 0.50 * torch.relu(annual_te - 0.06).square()
    centered_market = market - market.mean()
    beta = (
        (
            (selected_variable - selected_variable.mean())
            * centered_market
        ).mean()
        / centered_market.square().mean().clamp_min(1.0e-12)
    )
    objective = objective - 0.01 * beta.square()
    active_sharpe = selected_variable.mean() / selected_variable.std(
        unbiased=False
    ).clamp_min(1.0e-3)
    if setting.sharpe_mode == "direct-full-batch-two-pass-gradient":
        total_sharpe = top2000_m03r_v7_total_excess_sharpe(
            selected_variable,
            selected_benchmark,
            selected_cash,
        )
        objective = objective + TOP2000_M03R_V7_DEV_SHARPE_WEIGHT * total_sharpe
    gradient = torch.autograd.grad(objective, variable)[0]
    coefficients = (gradient * indexes.numel()).detach().to(
        device=trace.transitions[0].utility.device,
        dtype=trace.transitions[0].utility.dtype,
    )
    reported_total_sharpe = top2000_m03r_v7_total_excess_sharpe(
        selected_active,
        selected_benchmark,
        selected_cash,
    )
    return coefficients, {
        "canonical_active_mean": float(selected_active.mean()),
        "canonical_annual_tracking_error": float(annual_te),
        "canonical_active_beta": float(beta),
        "canonical_active_sharpe": float(active_sharpe),
        "canonical_total_excess_sharpe": float(reported_total_sharpe),
        "objective_moment_row_count": int(indexes.numel()),
    }


def _all_reduce_gradients(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        used = torch.tensor(
            0 if parameter.grad is None else 1,
            dtype=torch.int64,
            device=parameter.device,
        )
        dist.all_reduce(used, op=dist.ReduceOp.SUM)
        if int(used.item()) == 0:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def _optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _validate_optimizer_partition(
    policy: Top2000M03RV7DevelopmentPolicy,
    optimizer: torch.optim.Optimizer,
    overlay_optimizer: torch.optim.Optimizer | None,
) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
    core = policy.alpha_core_parameters()
    overlay = policy.total_risk_overlay_parameters()
    core_ids = {id(parameter) for parameter in core}
    overlay_ids = {id(parameter) for parameter in overlay}
    if core_ids & overlay_ids:
        raise Top2000M03RV7DevelopmentError(
            "alpha-core and overlay parameter ownership overlaps"
        )
    observed_core = _optimizer_parameter_ids(optimizer)
    if observed_core != core_ids:
        raise Top2000M03RV7DevelopmentError(
            "primary optimizer must own exactly the alpha-core parameters"
        )
    if overlay:
        if overlay_optimizer is None:
            raise Top2000M03RV7DevelopmentError(
                "A06 requires a separate total-risk-overlay optimizer"
            )
        observed_overlay = _optimizer_parameter_ids(overlay_optimizer)
        if observed_overlay != overlay_ids or observed_core & observed_overlay:
            raise Top2000M03RV7DevelopmentError(
                "overlay optimizer must own only the isolated overlay head"
            )
    elif overlay_optimizer is not None:
        raise Top2000M03RV7DevelopmentError(
            "an overlay optimizer is forbidden outside A06"
        )
    return core, overlay


def build_top2000_m03r_v7_development_optimizers(
    policy: Top2000M03RV7DevelopmentPolicy,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Create the exact setting-qualified optimizer partition.

    A06 receives one AdamW state for the alpha core and a disjoint AdamW state
    for the total-risk overlay.  All other rows receive only the alpha-core
    optimizer.  Callers must checkpoint both returned states verbatim.
    """

    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
        or isinstance(weight_decay, bool)
        or not math.isfinite(float(weight_decay))
        or weight_decay < 0
    ):
        raise Top2000M03RV7DevelopmentError(
            "optimizer learning rate and weight decay are invalid"
        )
    core = torch.optim.AdamW(
        policy.alpha_core_parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    overlay_parameters = policy.total_risk_overlay_parameters()
    overlay = (
        None
        if not overlay_parameters
        else torch.optim.AdamW(
            overlay_parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
    )
    _validate_optimizer_partition(policy, core, overlay)
    return core, overlay


def _gradient_norm_and_clip(
    parameters: tuple[nn.Parameter, ...],
    grad_clip: float,
) -> float:
    populated = tuple(
        parameter for parameter in parameters if parameter.grad is not None
    )
    if not populated:
        return 0.0
    if grad_clip > 0:
        return float(torch.nn.utils.clip_grad_norm_(populated, grad_clip))
    gradients = []
    for parameter in populated:
        gradient = parameter.grad
        assert gradient is not None
        gradients.append(gradient.detach().norm())
    return float(
        torch.linalg.vector_norm(
            torch.stack(gradients)
        )
    )


def train_top2000_m03r_v7_development_update(
    policy: Top2000M03RV7DevelopmentPolicy,
    sequence: Hold30Sequence,
    state_provider: Top2000M03RV7DecisionStateProvider,
    optimizer: torch.optim.Optimizer,
    *,
    overlay_optimizer: torch.optim.Optimizer | None = None,
    completed_optimizer_steps: int,
    total_optimizer_steps: int,
    max_origin_batch: int,
    grad_clip: float,
    distributed_rank: int,
    distributed_world_size: int,
) -> dict[str, Any]:
    """One canonical sweep, origin replay, gradient SUM, and optimizer step."""

    if distributed_world_size not in {1, 2}:
        raise Top2000M03RV7DevelopmentError("world size must be one or two")
    if distributed_rank not in range(distributed_world_size):
        raise Top2000M03RV7DevelopmentError("distributed rank is out of range")
    distributed = distributed_world_size == 2
    if distributed and (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != distributed_rank
    ):
        raise Top2000M03RV7DevelopmentError(
            "two-rank development training requires a matching process group"
        )
    if not 0 <= completed_optimizer_steps < total_optimizer_steps:
        raise Top2000M03RV7DevelopmentError("optimizer progress is out of range")
    geometry = Hold30ReplayGeometry(
        warmup_decisions=TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
        label_support_decisions=TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS,
        max_origin_batch=max_origin_batch,
    )
    roles = geometry.roles(sequence.n_positions)
    setting = policy.setting
    core_parameters, overlay_parameters = _validate_optimizer_partition(
        policy,
        optimizer,
        overlay_optimizer,
    )
    policy.set_gradient_route("alpha-core")
    runtime = Hold30ChronologicalRuntime(
        "H2",
        action_builder=Top2000M03RV7ActionBuilder(policy),
        state_provider=state_provider,
        require_trainable_state_provider=True,
    )
    adapter = Hold30ChronologicalReplayAdapter(runtime)
    policy.train()
    with torch.no_grad():
        canonical_state, rows = adapter.canonical_pass(policy, sequence, roles)
    canonical_state = detach_tree(canonical_state)
    if not isinstance(canonical_state, Hold30CanonicalTrace):
        raise Top2000M03RV7DevelopmentError(
            "chronological adapter returned an unexpected canonical state"
        )
    loss_contract = Hold30LossContract(
        "H2",
        lambda_turn=0.25,
        lambda_early=0.0,
        gate_entropy_coef=0.0,
        gate_budget_coef=0.0,
        target_turnover=1.0 / 30.0,
    )
    sequence_terms = sequence_coefficients(rows, roles.anchors, loss_contract)
    moment_rows = torch.unique(roles.utility_rows.reshape(-1), sorted=True)
    cash_simple_returns = sequence.asset_returns[:, :, sequence.cash_index].mean(-1)
    row_coefficients, canonical_metrics = _objective_row_coefficients(
        canonical_state,
        setting,
        cash_simple_returns=cash_simple_returns,
        moment_rows=moment_rows,
    )
    warmup_steps = max(1, math.ceil(0.10 * total_optimizer_steps))
    warmup = min(1.0, completed_optimizer_steps / warmup_steps)
    age = torch.arange(
        61,
        device=sequence.asset_returns.device,
        dtype=sequence.asset_returns.dtype,
    )
    persistence_weights = ((30.0 - age).clamp_min(0.0) / 30.0).square()
    denominator = float(sequence_terms.anchor_count)
    local_anchors = roles.anchors[distributed_rank::distributed_world_size]
    if local_anchors.numel() == 0:
        raise Top2000M03RV7DevelopmentError(
            "every rank must own at least one loss-bearing origin"
        )
    optimizer.zero_grad(set_to_none=True)
    if overlay_optimizer is not None:
        overlay_optimizer.zero_grad(set_to_none=True)
    objective_total = 0.0
    weighted_early_total = 0.0
    auxiliary_total = 0.0
    uncertainty_total = 0.0
    replayed = 0
    for origin_batch in geometry.origin_batches(local_anchors):
        replays = adapter.replay_origins(
            policy,
            sequence,
            canonical_state,
            origin_batch,
            roles,
        )
        batch_value: torch.Tensor | None = None
        for replay in replays:
            replay.validate()
            coefficients = row_coefficients[
                replay.origin : replay.origin + replay.utility_rows.numel()
            ]
            value = (coefficients * replay.utility_rows).sum()
            value = value - (
                sequence_terms.turnover_coefficient
                * replay.discretionary_turnover.reshape(())
            )
            sold_by_age = replay.discretionary_sold_value_by_age
            if sold_by_age is None:
                raise Top2000M03RV7DevelopmentError(
                    "v7 replay omitted cause-typed sold notional by age"
                )
            weighted_early = (sold_by_age * persistence_weights).sum()
            value = value / denominator
            persistence_penalty = top2000_m03r_v7_persistence_penalty(
                sold_by_age,
                coefficient_basis_points=(
                    setting.persistence_coefficient_basis_points
                ),
                warmup_multiplier=warmup,
                valid_decision_session_count=sequence_terms.anchor_count,
            )
            value = value - persistence_penalty
            auxiliary_loss, uncertainty_loss = _auxiliary_alpha_loss(
                replay,
                sequence,
            )
            if auxiliary_loss is not None:
                value = value - (
                    TOP2000_M03R_V7_DEV_AUXILIARY_WEIGHT
                    * auxiliary_loss
                    / denominator
                )
                auxiliary_total += float(auxiliary_loss.detach())
            if uncertainty_loss is not None:
                value = value - (
                    TOP2000_M03R_V7_DEV_UNCERTAINTY_WEIGHT
                    * uncertainty_loss
                    / denominator
                )
                uncertainty_total += float(uncertainty_loss.detach())
            batch_value = value if batch_value is None else batch_value + value
            objective_total += float(value.detach())
            weighted_early_total += float(weighted_early.detach())
            replayed += 1
        if batch_value is None:
            raise Top2000M03RV7DevelopmentError("origin replay batch was empty")
        (-batch_value).backward()  # type: ignore[no-untyped-call]
    if any(parameter.grad is not None for parameter in overlay_parameters):
        raise Top2000M03RV7DevelopmentError(
            "alpha-core pass leaked gradients into the A06 overlay"
        )

    overlay_objective_total = 0.0
    overlay_replayed = 0
    overlay_sharpe = canonical_metrics["canonical_total_excess_sharpe"]
    if overlay_parameters:
        active = torch.stack(
            [transition.utility.mean() for transition in canonical_state.transitions]
        )
        benchmark = torch.stack(
            [
                transition.benchmark_net_return.mean()
                for transition in canonical_state.transitions
            ]
        )
        overlay_gradient, overlay_sharpe_tensor = (
            top2000_m03r_v7_direct_sharpe_vjp_coefficients(
                active,
                benchmark,
                cash_simple_returns,
                moment_rows,
            )
        )
        overlay_sharpe = float(overlay_sharpe_tensor)
        overlay_coefficients = (
            overlay_gradient
            * float(moment_rows.numel())
            * TOP2000_M03R_V7_DEV_SHARPE_WEIGHT
        ).to(device=active.device, dtype=active.dtype)
        core_gradients_before = {
            id(parameter): (
                None if parameter.grad is None else parameter.grad.detach().clone()
            )
            for parameter in core_parameters
        }
        policy.set_gradient_route("total-risk-overlay")
        try:
            for origin_batch in geometry.origin_batches(local_anchors):
                replays = adapter.replay_origins(
                    policy,
                    sequence,
                    canonical_state,
                    origin_batch,
                    roles,
                )
                batch_value = None
                for replay in replays:
                    coefficients = overlay_coefficients[
                        replay.origin : replay.origin + replay.utility_rows.numel()
                    ]
                    value = (coefficients * replay.utility_rows).sum() / denominator
                    batch_value = (
                        value if batch_value is None else batch_value + value
                    )
                    overlay_objective_total += float(value.detach())
                    overlay_replayed += 1
                if batch_value is None:
                    raise Top2000M03RV7DevelopmentError(
                        "A06 overlay replay batch was empty"
                    )
                (-batch_value).backward()  # type: ignore[no-untyped-call]
        finally:
            policy.set_gradient_route("alpha-core")
        for parameter in core_parameters:
            before = core_gradients_before[id(parameter)]
            after = parameter.grad
            if before is None:
                if after is not None:
                    raise Top2000M03RV7DevelopmentError(
                        "A06 overlay pass created an alpha-core gradient"
                    )
            elif after is None or not torch.equal(after, before):
                raise Top2000M03RV7DevelopmentError(
                    "A06 overlay pass changed an alpha-core gradient"
                )
        if not any(parameter.grad is not None for parameter in overlay_parameters):
            raise Top2000M03RV7DevelopmentError(
                "A06 overlay pass did not reach its isolated head"
            )

    if distributed:
        _all_reduce_gradients(core_parameters)
        _all_reduce_gradients(overlay_parameters)
    gradient_norm = _gradient_norm_and_clip(core_parameters, grad_clip)
    overlay_gradient_norm = _gradient_norm_and_clip(
        overlay_parameters,
        grad_clip,
    )
    optimizer.step()
    if overlay_optimizer is not None:
        overlay_optimizer.step()
    totals = torch.tensor(
        [
            objective_total,
            weighted_early_total,
            auxiliary_total,
            uncertainty_total,
            float(replayed),
        ],
        dtype=torch.float64,
        device=sequence.asset_returns.device,
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        **canonical_metrics,
        "objective": float(totals[0]),
        "weighted_early_exit_notional": float(totals[1]),
        "auxiliary_alpha_loss_sum": float(totals[2]),
        "uncertainty_loss_sum": float(totals[3]),
        "replayed_origins": int(totals[4]),
        "anchor_count": sequence_terms.anchor_count,
        "mean_discretionary_turnover": sequence_terms.mean_turnover,
        "turnover_vjp_coefficient": sequence_terms.turnover_coefficient,
        "persistence_coefficient_basis_points": (
            setting.persistence_coefficient_basis_points
        ),
        "persistence_warmup_multiplier": warmup,
        "gradient_norm_before_clip": gradient_norm,
        "overlay_gradient_norm_before_clip": overlay_gradient_norm,
        "overlay_total_excess_sharpe": overlay_sharpe,
        "overlay_objective": overlay_objective_total,
        "overlay_replayed_origins": overlay_replayed,
        "alpha_core_optimizer_steps": 1,
        "overlay_optimizer_steps": 1 if overlay_optimizer is not None else 0,
        "optimizer_steps": 1 + (1 if overlay_optimizer is not None else 0),
        "distributed_world_size": distributed_world_size,
        "origin_shard_policy": "strided-rank-mod-world-size",
        "development_only": True,
        "promotion_eligible": False,
    }


@dataclass(frozen=True, slots=True)
class Top2000M03RV7DevelopmentTrainingPlan:
    setting_index: int
    setting_id: str
    cache_path: str
    cache_sha256: str
    output_root: str
    total_optimizer_steps_per_fold_seed: int = 64
    max_origin_batch: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    token_dim: int = 128
    raw_stock_chunk: int = 512
    expected_world_size: int = 2
    activation_checkpointing: bool = True
    mixed_precision: str = "bfloat16"
    protocol_generation: str = M03R_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_TOP2000_DEV_DESIGN_ID
    protocol_sha256: str = M03R_TOP2000_DEV_PROTOCOL_SHA256
    schema: str = TOP2000_M03R_V7_DEV_TRAINING_SCHEMA
    development_only: bool = True
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        setting = resolve_m03r_top2000_dev_setting(self.setting_id)
        _require_digest("cache_sha256", self.cache_sha256)
        _require_digest("protocol_sha256", self.protocol_sha256)
        if (
            self.setting_index != setting.setting_index
            or self.setting_id != M03R_TOP2000_DEV_SETTING_IDS[self.setting_index]
            or not self.cache_path
            or not self.output_root
            or self.total_optimizer_steps_per_fold_seed <= 0
            or self.max_origin_batch <= 0
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.grad_clip <= 0
            or self.token_dim <= 0
            or self.raw_stock_chunk <= 0
            or self.expected_world_size != 2
            or not self.activation_checkpointing
            or self.mixed_precision != "bfloat16"
            or self.protocol_generation != M03R_TOP2000_DEV_PROTOCOL_GENERATION
            or self.design_id != M03R_TOP2000_DEV_DESIGN_ID
            or self.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256
            or self.schema != TOP2000_M03R_V7_DEV_TRAINING_SCHEMA
            or not self.development_only
            or self.promotion_eligible
        ):
            raise Top2000M03RV7DevelopmentError(
                "TOP2000 v7 development training plan drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))

    @property
    def episode_schedule_sha256(self) -> str:
        """Path- and setting-independent paired episode schedule identity."""

        return _sha256(
            {
                "schema": (
                    "rl-quant.top2000-dev.m03r-v7-paired-episode-schedule-v2"
                ),
                "protocol_sha256": self.protocol_sha256,
                "cache_sha256": self.cache_sha256,
                "required_state_rows": TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
                "episode_state_rows": TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
                "warmup_decisions": TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
                "label_support_decisions": (
                    TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS
                ),
                "validation_decisions": TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS,
                "purge_decisions": TOP2000_M03R_V7_DEV_PURGE_DECISIONS,
                "fold_advance": TOP2000_M03R_V7_DEV_FOLD_ADVANCE,
                "first_validation_start": (
                    TOP2000_M03R_V7_DEV_FIRST_VALIDATION_START
                ),
                "fold_count": TOP2000_M03R_V7_DEV_FOLD_COUNT,
                "paired_seeds": list(TOP2000_M03R_V7_DEV_SEEDS),
            }
        )


__all__ = [
    "TOP2000_M03R_V7_DEV_COST_RATE",
    "TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS",
    "TOP2000_M03R_V7_DEV_FOLD_COUNT",
    "TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS",
    "TOP2000_M03R_V7_DEV_SEEDS",
    "TOP2000_M03R_V7_DEV_TRAINING_SCHEMA",
    "Top2000M03RV7ActionBuilder",
    "Top2000M03RV7ActionPolicy",
    "Top2000M03RV7DecisionInputs",
    "Top2000M03RV7DecisionStateProvider",
    "Top2000M03RV7DevelopmentError",
    "Top2000M03RV7DevelopmentFold",
    "Top2000M03RV7DevelopmentPolicy",
    "Top2000M03RV7DevelopmentTrainingPlan",
    "bind_top2000_m03r_v7_runtime_sequence",
    "build_top2000_m03r_v7_development_optimizers",
    "render_top2000_m03r_v7_development_folds",
    "top2000_m03r_v7_decision_inputs",
    "top2000_m03r_v7_direct_sharpe_vjp_coefficients",
    "top2000_m03r_v7_factor_neutral_executed_weights",
    "top2000_m03r_v7_persistence_penalty",
    "top2000_m03r_v7_policy_config",
    "top2000_m03r_v7_total_excess_sharpe",
    "train_top2000_m03r_v7_development_update",
]
