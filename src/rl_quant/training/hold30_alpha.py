"""Global-moment training and checkpoint selection for Hold-30 alpha v3.

The additive alpha objective and every ratio/band term are computed from one
effective batch.  Microbatches contribute sufficient statistics and gradient
sums; they never compute or average local tracking-error, beta, volatility, or
Sharpe ratios.  a07's direct Sharpe branch uses the exact detached-coefficient
two-pass gradient of simple policy return minus the PIT CASH simple return.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

import torch
import torch.distributed as dist

from rl_quant.datasets.hold30 import Hold30DatasetSequence
from rl_quant.datasets.hold30_alpha import (
    Hold30AlphaDataBindingReceipt,
    Hold30AlphaEvaluationPanel,
    Hold30AlphaLabelDomain,
    Hold30ResidualAlphaLabels,
    bind_hold30_alpha_evaluation_panel,
    verify_hold30_residual_alpha_labels,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_BETA_TARGET,
    HOLD30_ALPHA_BETA_TOLERANCE,
    HOLD30_ALPHA_HORIZONS,
    HOLD30_ALPHA_TE_MAX_ANNUAL,
    HOLD30_ALPHA_TE_MIN_ANNUAL,
    HOLD30_ALPHA_TE_TARGET_ANNUAL,
    HOLD30_ALPHA_TRAIN_COST_BPS,
    HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
    HOLD30_ALPHA_VALIDATION_COSTS_BPS,
    Hold30AlphaCheckpointContract,
    resolve_hold30_alpha_setting,
)
from rl_quant.protocol.hold30_freeze import HOLD30_SEEDS

HOLD30_ALPHA_TARGET_TURNOVER: Final[float] = 1.0 / 30.0
HOLD30_ALPHA_MEDIAN_AGE_BAND: Final[tuple[float, float]] = (20.0, 40.0)
HOLD30_ALPHA_ANNUALIZATION: Final[float] = 252.0


class Hold30AlphaTrainingError(ValueError):
    """The v3 alpha objective is unresolved or receives invalid evidence."""


class Hold30AlphaUnresolvedCoefficientError(Hold30AlphaTrainingError):
    """A result-moving v3 coefficient was not frozen in the manifest."""


Hold30AlphaObjectiveRole = Literal["training", "inner-validation"]
Hold30AlphaBatchBindingKind = Literal[
    "receipt-bound", "qualification-math-fixture"
]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30AlphaTrainingError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _tensor_exact_sha256(value: torch.Tensor) -> str:
    """Hash exact detached tensor identity, including dtype and shape."""

    material = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(material.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(material.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(
        material.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Hold30AlphaObjectiveDomainBinding:
    """Exactly one permitted label domain and its non-outer objective role."""

    role: Hold30AlphaObjectiveRole
    domain: Hold30AlphaLabelDomain

    def __post_init__(self) -> None:
        expected_name = {
            "training": "train",
            "inner-validation": "validation",
        }.get(self.role)
        if expected_name is None:
            raise Hold30AlphaTrainingError(
                "objective role must be training or inner-validation; outer is forbidden"
            )
        if not isinstance(self.domain, Hold30AlphaLabelDomain):
            raise Hold30AlphaTrainingError(
                "objective domain binding requires a typed label domain"
            )
        if self.domain.name != expected_name:
            raise Hold30AlphaTrainingError(
                f"{self.role} objective requires label domain {expected_name!r}"
            )

    @property
    def binding_id(self) -> str:
        return _canonical_sha256(
            {
                "role": self.role,
                "domain": {
                    "name": self.domain.name,
                    "start": self.domain.start,
                    "stop": self.domain.stop,
                },
            }
        )


def _finite_nonnegative(name: str, value: float | None) -> float:
    if value is None:
        raise Hold30AlphaUnresolvedCoefficientError(
            f"{name} is an unresolved manifest coefficient"
        )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise Hold30AlphaTrainingError(
            f"{name} must be a finite non-negative scalar"
        )
    return float(value)


def _optional_finite_nonnegative(name: str, value: float | None) -> None:
    if value is not None:
        _finite_nonnegative(name, value)


def _finite_positive(name: str, value: float | None) -> float:
    result = _finite_nonnegative(name, value)
    if result <= 0:
        raise Hold30AlphaTrainingError(f"{name} must be strictly positive")
    return result


@dataclass(frozen=True, slots=True)
class Hold30AlphaObjectiveConfig:
    """Manifest-owned coefficients for one v3 alpha setting.

    The scientific band/targets supplied by the user have defaults.  Every
    result-moving multiplier remains ``None`` until explicitly frozen.
    """

    setting_id: str
    downside_penalty_kappa: float | None = None
    active_log_scale_bounds: tuple[float, float] | None = None
    uncertainty_log_scale_bounds: tuple[float, float] | None = None
    auxiliary_horizon_weights: tuple[float, float, float, float] | None = None
    auxiliary_horizon_scales: tuple[float, float, float, float] | None = None
    a06_total_risk_step: float | None = None
    alpha_core_parameter_selector: str | None = None
    overlay_parameter_selector: str | None = None
    stop_gradient_core_to_overlay: bool | None = None
    stop_gradient_overlay_to_core: bool | None = None
    separate_optimizer_spec_receipt_sha256: str | None = None
    lambda_te_floor: float | None = None
    lambda_te_ceiling: float | None = None
    lambda_beta: float | None = None
    lambda_turnover: float | None = None
    lambda_early_exit: float | None = None
    lambda_auxiliary_alpha: float | None = None
    lambda_uncertainty: float | None = None
    lambda_total_excess_mean: float | None = None
    lambda_total_sharpe_overlay: float | None = None
    total_sharpe_epsilon: float | None = None
    lambda_volatility_ratio: float | None = None
    target_volatility_ratio: float | None = None
    lambda_drawdown: float | None = None
    drawdown_limit: float | None = None
    lambda_direct_sharpe: float | None = None
    direct_sharpe_epsilon: float | None = None
    te_floor: float = HOLD30_ALPHA_TE_MIN_ANNUAL
    te_target: float = HOLD30_ALPHA_TE_TARGET_ANNUAL
    te_ceiling: float = HOLD30_ALPHA_TE_MAX_ANNUAL
    beta_target: float = HOLD30_ALPHA_BETA_TARGET
    beta_tolerance: float = HOLD30_ALPHA_BETA_TOLERANCE
    target_turnover: float = HOLD30_ALPHA_TARGET_TURNOVER
    train_cost_bps: int = HOLD30_ALPHA_TRAIN_COST_BPS
    qualification_math_test_only: bool = False

    def __post_init__(self) -> None:
        setting = resolve_hold30_alpha_setting(self.setting_id)
        if setting.objective_mode == "absolute-net-log-return":
            raise Hold30AlphaTrainingError(
                "absolute-return controls do not use the v3 alpha objective"
            )
        if not (
            float(self.te_floor)
            == HOLD30_ALPHA_TE_MIN_ANNUAL
            < float(self.te_target)
            == HOLD30_ALPHA_TE_TARGET_ANNUAL
            < float(self.te_ceiling)
            == HOLD30_ALPHA_TE_MAX_ANNUAL
        ):
            raise Hold30AlphaTrainingError("TE floor/target/ceiling must be .02/.04/.06")
        if (
            float(self.beta_target) != HOLD30_ALPHA_BETA_TARGET
            or float(self.beta_tolerance) != HOLD30_ALPHA_BETA_TOLERANCE
        ):
            raise Hold30AlphaTrainingError("beta target/tolerance must be 1.0/0.1")
        if float(self.target_turnover) != HOLD30_ALPHA_TARGET_TURNOVER:
            raise Hold30AlphaTrainingError("target turnover must be exactly 1/30")
        if self.train_cost_bps != HOLD30_ALPHA_TRAIN_COST_BPS:
            raise Hold30AlphaTrainingError("v3 training must use exactly 20 bp")
        for name in (
            "downside_penalty_kappa",
            "a06_total_risk_step",
            "lambda_te_floor",
            "lambda_te_ceiling",
            "lambda_beta",
            "lambda_turnover",
            "lambda_early_exit",
            "lambda_auxiliary_alpha",
            "lambda_uncertainty",
            "lambda_total_excess_mean",
            "lambda_total_sharpe_overlay",
            "total_sharpe_epsilon",
            "lambda_volatility_ratio",
            "target_volatility_ratio",
            "lambda_drawdown",
            "drawdown_limit",
            "lambda_direct_sharpe",
            "direct_sharpe_epsilon",
        ):
            _optional_finite_nonnegative(name, getattr(self, name))
        if self.active_log_scale_bounds is not None:
            lower, upper = self.active_log_scale_bounds
            if (
                any(isinstance(value, bool) for value in (lower, upper))
                or not all(math.isfinite(float(value)) for value in (lower, upper))
                or float(lower) >= float(upper)
            ):
                raise Hold30AlphaTrainingError(
                    "active_log_scale_bounds must be finite and strictly ordered"
                )
        if self.uncertainty_log_scale_bounds is not None:
            lower, upper = self.uncertainty_log_scale_bounds
            if (
                any(isinstance(value, bool) for value in (lower, upper))
                or not all(math.isfinite(float(value)) for value in (lower, upper))
                or float(lower) >= float(upper)
            ):
                raise Hold30AlphaTrainingError(
                    "uncertainty_log_scale_bounds must be finite and strictly ordered"
                )
        for name in ("auxiliary_horizon_weights", "auxiliary_horizon_scales"):
            values = getattr(self, name)
            if values is not None and (
                not isinstance(values, tuple)
                or len(values) != len(HOLD30_ALPHA_HORIZONS)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in values
                )
            ):
                raise Hold30AlphaTrainingError(
                    f"{name} must contain four finite strictly positive values"
                )
        if not isinstance(self.qualification_math_test_only, bool):
            raise Hold30AlphaTrainingError(
                "qualification_math_test_only must be boolean"
            )

    @property
    def te_floor_enabled(self) -> bool:
        return resolve_hold30_alpha_setting(self.setting_id).te_floor_annual is not None

    @property
    def uncertainty_enabled(self) -> bool:
        return resolve_hold30_alpha_setting(
            self.setting_id
        ).uncertainty_downside_heads

    @property
    def sharpe_overlay_enabled(self) -> bool:
        return (
            resolve_hold30_alpha_setting(self.setting_id).sharpe_mode
            == "separate-total-risk-overlay"
        )

    @property
    def direct_sharpe_enabled(self) -> bool:
        return (
            resolve_hold30_alpha_setting(self.setting_id).sharpe_mode
            == "direct-two-pass-gradient"
        )

    def require_resolved(self) -> None:
        setting = resolve_hold30_alpha_setting(self.setting_id)
        alpha_heads = setting.supervised_residual_alpha_heads
        required = {
            "lambda_te_ceiling": self.lambda_te_ceiling,
            "lambda_turnover": self.lambda_turnover,
            "lambda_early_exit": self.lambda_early_exit,
        }
        if self.te_floor_enabled:
            required["lambda_te_floor"] = self.lambda_te_floor
        elif self.lambda_te_floor not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "the registered no-floor ablation cannot carry a TE-floor coefficient"
            )
        if setting.beta_targeting:
            required["lambda_beta"] = self.lambda_beta
        elif self.lambda_beta not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "beta loss is exclusive to beta-targeted m03/a04-a07 settings"
            )
        if alpha_heads:
            required["lambda_auxiliary_alpha"] = self.lambda_auxiliary_alpha
            if self.active_log_scale_bounds is None:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "active_log_scale_bounds are unresolved action bounds"
                )
            if self.auxiliary_horizon_weights is None:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "auxiliary_horizon_weights are unresolved"
                )
            if self.auxiliary_horizon_scales is None:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "auxiliary_horizon_scales are unresolved"
                )
            weights = self.auxiliary_horizon_weights
            assert weights is not None
            if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise Hold30AlphaTrainingError(
                    "auxiliary_horizon_weights must sum to one"
                )
            primary = HOLD30_ALPHA_HORIZONS.index(30)
            if any(weights[primary] <= weight for index, weight in enumerate(weights) if index != primary):
                raise Hold30AlphaTrainingError(
                    "the primary 30d auxiliary weight must be uniquely largest"
                )
        elif self.lambda_auxiliary_alpha not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "m02 must not acquire supervised residual-alpha losses"
            )
        elif self.active_log_scale_bounds is not None:
            raise Hold30AlphaTrainingError(
                "m02 must not acquire alpha-head active-risk action bounds"
            )
        elif self.auxiliary_horizon_weights is not None or self.auxiliary_horizon_scales is not None:
            raise Hold30AlphaTrainingError(
                "m02 must not acquire auxiliary horizon weights or scales"
            )
        if self.uncertainty_enabled:
            required["lambda_uncertainty"] = self.lambda_uncertainty
            required["downside_penalty_kappa"] = self.downside_penalty_kappa
            if self.uncertainty_log_scale_bounds is None:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "uncertainty_log_scale_bounds are unresolved"
                )
        elif self.lambda_uncertainty not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "the registered no-uncertainty ablation cannot carry its loss"
            )
        elif self.downside_penalty_kappa not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "the registered no-uncertainty ablation cannot carry downside kappa"
            )
        elif self.uncertainty_log_scale_bounds is not None:
            raise Hold30AlphaTrainingError(
                "the registered no-uncertainty ablation cannot carry uncertainty bounds"
            )
        if self.sharpe_overlay_enabled:
            required.update(
                {
                    "lambda_total_excess_mean": self.lambda_total_excess_mean,
                    "lambda_total_sharpe_overlay": self.lambda_total_sharpe_overlay,
                    "total_sharpe_epsilon": self.total_sharpe_epsilon,
                    "lambda_volatility_ratio": self.lambda_volatility_ratio,
                    "target_volatility_ratio": self.target_volatility_ratio,
                    "lambda_drawdown": self.lambda_drawdown,
                    "drawdown_limit": self.drawdown_limit,
                    "a06_total_risk_step": self.a06_total_risk_step,
                }
            )
            if self.alpha_core_parameter_selector != "alpha-core-only":
                raise Hold30AlphaUnresolvedCoefficientError(
                    "a06 alpha_core_parameter_selector must be alpha-core-only"
                )
            if self.overlay_parameter_selector != "a06-overlay-only":
                raise Hold30AlphaUnresolvedCoefficientError(
                    "a06 overlay_parameter_selector must be a06-overlay-only"
                )
            if self.stop_gradient_core_to_overlay is not True:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "a06 must stop gradients from alpha core to overlay"
                )
            if self.stop_gradient_overlay_to_core is not True:
                raise Hold30AlphaUnresolvedCoefficientError(
                    "a06 must stop gradients from overlay to alpha core"
                )
            digest = self.separate_optimizer_spec_receipt_sha256
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise Hold30AlphaUnresolvedCoefficientError(
                    "a06 requires a lowercase SHA-256 immutable optimizer-spec receipt"
                )
        elif any(
            value not in (None, 0, 0.0)
            for value in (
                self.lambda_total_excess_mean,
                self.lambda_total_sharpe_overlay,
                self.total_sharpe_epsilon,
                self.lambda_volatility_ratio,
                self.target_volatility_ratio,
                self.lambda_drawdown,
                self.drawdown_limit,
                self.a06_total_risk_step,
                self.alpha_core_parameter_selector,
                self.overlay_parameter_selector,
                self.stop_gradient_core_to_overlay,
                self.stop_gradient_overlay_to_core,
                self.separate_optimizer_spec_receipt_sha256,
            )
        ):
            raise Hold30AlphaTrainingError(
                "total-risk/Sharpe overlay terms are exclusive to a06"
            )
        if self.direct_sharpe_enabled:
            required["lambda_direct_sharpe"] = self.lambda_direct_sharpe
            required["direct_sharpe_epsilon"] = self.direct_sharpe_epsilon
        elif self.lambda_direct_sharpe not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError("direct Sharpe is exclusive to a07")
        elif self.direct_sharpe_epsilon not in (None, 0, 0.0):
            raise Hold30AlphaTrainingError(
                "direct Sharpe epsilon is exclusive to a07"
            )
        validate_required = (
            _finite_nonnegative
            if self.qualification_math_test_only
            else _finite_positive
        )
        for name, value in required.items():
            validate_required(name, value)
        if self.direct_sharpe_enabled and float(self.direct_sharpe_epsilon) <= 0:
            raise Hold30AlphaTrainingError(
                "direct_sharpe_epsilon must be explicitly frozen above zero"
            )
        if self.sharpe_overlay_enabled and float(self.total_sharpe_epsilon) <= 0:
            raise Hold30AlphaTrainingError(
                "total_sharpe_epsilon must be explicitly frozen above zero"
            )
        if self.uncertainty_enabled and float(self.downside_penalty_kappa) <= 0:
            raise Hold30AlphaTrainingError(
                "downside_penalty_kappa must be explicitly frozen above zero"
            )
    def validate_batch(self, batch: Hold30AlphaBatch) -> None:
        """Enforce the registered A2-to-A3 supervision boundary."""

        setting = resolve_hold30_alpha_setting(self.setting_id)
        alpha_heads = setting.supervised_residual_alpha_heads
        has_auxiliary = batch.auxiliary_prediction is not None
        if alpha_heads != has_auxiliary:
            expected = "requires" if alpha_heads else "forbids"
            raise Hold30AlphaTrainingError(
                f"{self.setting_id} {expected} supervised residual-alpha tensors"
            )
        if setting.uncertainty_downside_heads != (batch.downside_30d is not None):
            expected = "requires" if setting.uncertainty_downside_heads else "forbids"
            raise Hold30AlphaTrainingError(
                f"{self.setting_id} {expected} downside/uncertainty tensors"
            )


@dataclass(frozen=True, slots=True)
class Hold30AlphaBatch:
    """One chronology or microbatch entering the global v3 objective."""

    binding_kind: Hold30AlphaBatchBindingKind
    source_axis_id: str
    objective_inputs_id: str
    role: Hold30AlphaObjectiveRole | Literal["qualification-math-fixture"]
    stream_id: str
    origin_row_ids: torch.Tensor
    global_path_ids: torch.Tensor
    evaluation_point_id: str
    policy_net_return: torch.Tensor
    benchmark_net_return: torch.Tensor
    market_return: torch.Tensor
    risk_free_return: torch.Tensor
    discretionary_turnover: torch.Tensor
    early_exit_mass: torch.Tensor
    valid: torch.Tensor | None = None
    auxiliary_prediction: torch.Tensor | None = None
    auxiliary_target: torch.Tensor | None = None
    auxiliary_valid: torch.Tensor | None = None
    downside_30d: torch.Tensor | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.policy_net_return.shape)
        if len(shape) != 1 or not shape:
            raise Hold30AlphaTrainingError("return tensors must be non-empty vectors")
        if self.binding_kind not in {
            "receipt-bound",
            "qualification-math-fixture",
        }:
            raise Hold30AlphaTrainingError("unknown batch binding kind")
        if not isinstance(self.source_axis_id, str) or not self.source_axis_id:
            raise Hold30AlphaTrainingError("source_axis_id must be non-empty")
        _require_sha256("objective_inputs_id", self.objective_inputs_id)
        _require_sha256("evaluation_point_id", self.evaluation_point_id)
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise Hold30AlphaTrainingError("stream_id must be non-empty")
        if self.binding_kind == "receipt-bound":
            if self.role not in {"training", "inner-validation"}:
                raise Hold30AlphaTrainingError(
                    "receipt-bound batches need a permitted objective role"
                )
        elif self.role != "qualification-math-fixture":
            raise Hold30AlphaTrainingError(
                "math fixtures must carry the qualification-math-fixture role"
            )
        for name in ("origin_row_ids", "global_path_ids"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.int64
                or tuple(value.shape) != shape
                or value.requires_grad
            ):
                raise Hold30AlphaTrainingError(
                    f"{name} must be detached int64 with shape {shape}"
                )
            if bool((value < 0).any()):
                raise Hold30AlphaTrainingError(f"{name} cannot be negative")
        identities = tuple(
            zip(
                self.origin_row_ids.detach().to(device="cpu").tolist(),
                self.global_path_ids.detach().to(device="cpu").tolist(),
                strict=True,
            )
        )
        if len(set(identities)) != len(identities):
            raise Hold30AlphaTrainingError(
                "origin/batch row identities must be unique"
            )
        for name in (
            "policy_net_return",
            "benchmark_net_return",
            "market_return",
            "risk_free_return",
            "discretionary_turnover",
            "early_exit_mass",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise Hold30AlphaTrainingError(f"{name} must be finite with shape {shape}")
        if bool((self.policy_net_return <= -1).any()) or bool(
            (self.benchmark_net_return <= -1).any()
        ) or bool((self.market_return <= -1).any()) or bool(
            (self.risk_free_return <= -1).any()
        ):
            raise Hold30AlphaTrainingError("simple returns must exceed -1")
        for name in ("benchmark_net_return", "market_return", "risk_free_return"):
            if getattr(self, name).requires_grad:
                raise Hold30AlphaTrainingError(
                    f"bound objective input {name} must be detached"
                )
        if bool((self.discretionary_turnover < 0).any()) or bool(
            (self.early_exit_mass < 0).any()
        ):
            raise Hold30AlphaTrainingError("turnover/early-exit mass cannot be negative")
        if self.valid is not None and (
            self.valid.dtype != torch.bool or tuple(self.valid.shape) != shape
        ):
            raise Hold30AlphaTrainingError("valid must be boolean with the return shape")
        auxiliary = (
            self.auxiliary_prediction,
            self.auxiliary_target,
            self.auxiliary_valid,
        )
        if any(value is not None for value in auxiliary):
            if not all(value is not None for value in auxiliary):
                raise Hold30AlphaTrainingError(
                    "auxiliary prediction, target, and mask must be supplied together"
                )
            assert self.auxiliary_prediction is not None
            assert self.auxiliary_target is not None
            assert self.auxiliary_valid is not None
            if (
                self.auxiliary_prediction.ndim != 3
                or self.auxiliary_prediction.shape[0] != shape[0]
                or self.auxiliary_prediction.shape[-1] != len(HOLD30_ALPHA_HORIZONS)
                or self.auxiliary_target.shape != self.auxiliary_prediction.shape
                or self.auxiliary_valid.shape != self.auxiliary_prediction.shape
                or self.auxiliary_valid.dtype != torch.bool
                or not self.auxiliary_prediction.is_floating_point()
                or not self.auxiliary_target.is_floating_point()
                or not bool(torch.isfinite(self.auxiliary_prediction).all())
                or not bool(torch.isfinite(self.auxiliary_target).all())
            ):
                raise Hold30AlphaTrainingError(
                    "auxiliary tensors must be finite [date,asset,4] with a bool mask"
                )
            if self.downside_30d is not None and (
                self.downside_30d.shape != self.auxiliary_prediction.shape[:-1]
                or not self.downside_30d.is_floating_point()
                or not bool(torch.isfinite(self.downside_30d).all())
                or bool((self.downside_30d < 0).any())
            ):
                raise Hold30AlphaTrainingError(
                    "downside_30d must be finite nonnegative [date,asset]"
                )
            if self.downside_30d is not None:
                horizon_30 = HOLD30_ALPHA_HORIZONS.index(30)
                valid_30 = self.auxiliary_valid[..., horizon_30]
                if bool((self.downside_30d.masked_select(valid_30) <= 0).any()):
                    raise Hold30AlphaTrainingError(
                        "downside_30d must be strictly positive on valid 30d cells"
                    )
        elif self.downside_30d is not None:
            raise Hold30AlphaTrainingError(
                "downside_30d cannot be supplied without bound auxiliary targets"
            )

    @property
    def mask(self) -> torch.Tensor:
        return (
            torch.ones_like(self.policy_net_return, dtype=torch.bool)
            if self.valid is None
            else self.valid
        )

    @property
    def active_log_return(self) -> torch.Tensor:
        return torch.log1p(self.policy_net_return) - torch.log1p(
            self.benchmark_net_return
        )

    @property
    def risk_free_excess_return(self) -> torch.Tensor:
        """User-defined total-risk convention: simple policy return minus RF."""

        return self.policy_net_return - self.risk_free_return


@dataclass(frozen=True, slots=True)
class Hold30AlphaBoundObjectiveInputs:
    """Receipt-bound score-row inputs; external series cannot be substituted."""

    source_axis_id: str
    binding_receipt_id: str
    objective_inputs_id: str
    domain_binding: Hold30AlphaObjectiveDomainBinding
    score_origin_rows: torch.Tensor
    global_path_ids: torch.Tensor
    benchmark_net_return: torch.Tensor
    market_return: torch.Tensor
    risk_free_return: torch.Tensor
    valid: torch.Tensor
    auxiliary_target: torch.Tensor
    auxiliary_valid: torch.Tensor
    auxiliary_censored: torch.Tensor

    def build_batch(
        self,
        setting_id: str,
        *,
        policy_net_return: torch.Tensor,
        discretionary_turnover: torch.Tensor,
        early_exit_mass: torch.Tensor,
        evaluation_point_id: str,
        stream_id: str = "primary",
        auxiliary_prediction: torch.Tensor | None = None,
        downside_30d: torch.Tensor | None = None,
    ) -> Hold30AlphaBatch:
        """Attach differentiable outputs to the immutable bound data panel."""

        setting = resolve_hold30_alpha_setting(setting_id)
        _require_sha256("evaluation_point_id", evaluation_point_id)
        score_shape = tuple(self.benchmark_net_return.shape)
        for name, value in (
            ("policy_net_return", policy_net_return),
            ("discretionary_turnover", discretionary_turnover),
            ("early_exit_mass", early_exit_mass),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != score_shape:
                raise Hold30AlphaTrainingError(
                    f"{name} must have bound score-row shape {score_shape}"
                )
        alpha_heads = setting.supervised_residual_alpha_heads
        auxiliary_shape = tuple(self.auxiliary_target.shape)
        if alpha_heads:
            if (
                auxiliary_prediction is None
                or tuple(auxiliary_prediction.shape) != auxiliary_shape
            ):
                raise Hold30AlphaTrainingError(
                    f"{setting_id} requires auxiliary_prediction shape {auxiliary_shape}"
                )
        elif auxiliary_prediction is not None or downside_30d is not None:
            raise Hold30AlphaTrainingError(
                "m02 cannot receive supervised alpha predictions or uncertainty"
            )
        downside_shape = auxiliary_shape[:-1]
        if setting.uncertainty_downside_heads:
            if downside_30d is None or tuple(downside_30d.shape) != downside_shape:
                raise Hold30AlphaTrainingError(
                    f"{setting_id} requires downside_30d shape {downside_shape}"
                )
        elif downside_30d is not None:
            raise Hold30AlphaTrainingError(
                f"{setting_id} forbids downside/uncertainty predictions"
            )

        device = policy_net_return.device
        dtype = policy_net_return.dtype

        def bound_float(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=device, dtype=dtype).detach().reshape(-1)

        rows, batch = score_shape
        assets = auxiliary_shape[2]
        horizons = auxiliary_shape[3]
        origin_row_ids = self.score_origin_rows.view(rows, 1).expand(
            rows, batch
        )
        if (
            self.global_path_ids.dtype != torch.int64
            or tuple(self.global_path_ids.shape) != (batch,)
            or len(set(self.global_path_ids.tolist())) != batch
        ):
            raise Hold30AlphaTrainingError(
                "bound objective inputs require globally unique path IDs"
            )
        global_path_ids = self.global_path_ids.view(1, batch).expand(rows, batch)
        return Hold30AlphaBatch(
            binding_kind="receipt-bound",
            source_axis_id=self.source_axis_id,
            objective_inputs_id=self.objective_inputs_id,
            role=self.domain_binding.role,
            stream_id=stream_id,
            origin_row_ids=origin_row_ids.to(device=device).reshape(-1),
            global_path_ids=global_path_ids.to(device=device).reshape(-1),
            evaluation_point_id=evaluation_point_id,
            policy_net_return=policy_net_return.reshape(-1),
            benchmark_net_return=bound_float(self.benchmark_net_return),
            market_return=bound_float(self.market_return),
            risk_free_return=bound_float(self.risk_free_return),
            discretionary_turnover=discretionary_turnover.reshape(-1),
            early_exit_mass=early_exit_mass.reshape(-1),
            valid=self.valid.to(device=device).reshape(-1),
            auxiliary_prediction=(
                None
                if auxiliary_prediction is None
                else auxiliary_prediction.reshape(rows * batch, assets, horizons)
            ),
            auxiliary_target=(
                None
                if auxiliary_prediction is None
                else self.auxiliary_target.to(device=device, dtype=dtype)
                .detach()
                .reshape(rows * batch, assets, horizons)
            ),
            auxiliary_valid=(
                None
                if auxiliary_prediction is None
                else self.auxiliary_valid.to(device=device).reshape(
                    rows * batch, assets, horizons
                )
            ),
            downside_30d=(
                None
                if downside_30d is None
                else downside_30d.reshape(rows * batch, assets)
            ),
        )


def bind_hold30_alpha_objective_inputs(
    sequence: Hold30DatasetSequence,
    labels: Hold30ResidualAlphaLabels,
    panel: Hold30AlphaEvaluationPanel,
    receipt: Hold30AlphaDataBindingReceipt,
    domain_binding: Hold30AlphaObjectiveDomainBinding,
) -> Hold30AlphaBoundObjectiveInputs:
    """Create score-row objective data only from recomputed typed receipts."""

    if not isinstance(receipt, Hold30AlphaDataBindingReceipt):
        raise Hold30AlphaTrainingError("a typed data-binding receipt is required")
    if not isinstance(domain_binding, Hold30AlphaObjectiveDomainBinding):
        raise Hold30AlphaTrainingError(
            "one typed allowed objective domain binding is required"
        )
    recomputed = bind_hold30_alpha_evaluation_panel(sequence, panel)
    if recomputed.receipt_id != receipt.receipt_id:
        raise Hold30AlphaTrainingError(
            "objective data-binding receipt does not match the sequence/panel"
        )
    if labels.source_axis_id != receipt.source_axis_id:
        raise Hold30AlphaTrainingError(
            "residual labels and objective binding do not share one source axis"
        )
    verify_hold30_residual_alpha_labels(sequence, labels)
    all_origins = sequence.roles.score_indices
    if not torch.equal(labels.origin_rows, all_origins):
        raise Hold30AlphaTrainingError(
            "residual labels must cover the exact bound score-origin rows"
        )
    if domain_binding.domain not in labels.domains:
        raise Hold30AlphaTrainingError(
            "allowed objective domain is not one exact bound label domain"
        )
    domain = domain_binding.domain
    selected = (all_origins >= domain.start) & (all_origins < domain.stop)
    if not bool(selected.any()):
        raise Hold30AlphaTrainingError(
            "allowed objective domain contains no score-origin rows"
        )
    label_indices = selected.nonzero(as_tuple=False).flatten()
    origins = all_origins.index_select(0, label_indices)
    if labels.auxiliary_only is not True or labels.actor_access is not False:
        raise Hold30AlphaTrainingError(
            "future residual labels must remain auxiliary-only and actor-invisible"
        )
    risk_free_valid = panel.risk_free_valid.index_select(0, origins)
    market_valid = panel.market_valid.index_select(0, origins)
    valid = risk_free_valid & market_valid
    if not bool(valid.all()):
        raise Hold30AlphaTrainingError(
            "bound PIT market/risk-free inputs must cover every score row"
        )
    # Label storage is [horizon, origin, batch, asset]. The objective consumes
    # [origin, batch, asset, horizon] without exposing targets to the actor.
    auxiliary_target = (
        labels.values.index_select(1, label_indices)
        .permute(1, 2, 3, 0)
        .contiguous()
    )
    auxiliary_valid = (
        labels.valid.index_select(1, label_indices)
        .permute(1, 2, 3, 0)
        .contiguous()
    )
    auxiliary_censored = (
        labels.censored.index_select(1, label_indices)
        .permute(1, 2, 3, 0)
        .contiguous()
    )
    if bool((auxiliary_valid & auxiliary_censored).any()):
        raise Hold30AlphaTrainingError("valid and censored alpha cells overlap")
    objective_inputs_id = _canonical_sha256(
        {
            "source_axis_id": sequence.axis_id,
            "data_binding_receipt_id": receipt.receipt_id,
            "labels_id": labels.labels_id,
            "domain_binding_id": domain_binding.binding_id,
            "score_origin_rows": origins.tolist(),
            "global_path_ids": list(receipt.global_path_ids),
        }
    )
    return Hold30AlphaBoundObjectiveInputs(
        source_axis_id=sequence.axis_id,
        binding_receipt_id=receipt.receipt_id,
        objective_inputs_id=objective_inputs_id,
        domain_binding=domain_binding,
        score_origin_rows=origins.clone(),
        global_path_ids=torch.tensor(receipt.global_path_ids, dtype=torch.int64),
        benchmark_net_return=sequence.c1_benchmark_net_returns.index_select(
            0, origins
        ).detach(),
        market_return=panel.market_total_returns.index_select(0, origins).detach(),
        risk_free_return=panel.risk_free_returns.index_select(0, origins).detach(),
        valid=valid.detach(),
        auxiliary_target=auxiliary_target.detach(),
        auxiliary_valid=auxiliary_valid.detach(),
        auxiliary_censored=auxiliary_censored.detach(),
    )


_MOMENT_FIELDS: Final[tuple[str, ...]] = (
    "count",
    "sum_active_log",
    "sum_active_log_sq",
    "sum_policy",
    "sum_policy_sq",
    "sum_benchmark",
    "sum_benchmark_sq",
    "sum_policy_excess",
    "sum_market_excess",
    "sum_market_excess_sq",
    "sum_policy_market_excess",
    "sum_risk_free_excess",
    "sum_risk_free_excess_sq",
    "sum_turnover",
    "sum_early_exit",
    "auxiliary_count",
    "auxiliary_squared_error",
    "auxiliary_count_h5",
    "auxiliary_count_h21",
    "auxiliary_count_h30",
    "auxiliary_count_h63",
    "uncertainty_count",
    "uncertainty_nll",
)


def _hold30_uncertainty_nll(
    batch: Hold30AlphaBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite 30d NLL and its mask without dividing on invalid cells."""

    assert batch.auxiliary_prediction is not None
    assert batch.auxiliary_target is not None
    assert batch.auxiliary_valid is not None
    assert batch.downside_30d is not None
    horizon = HOLD30_ALPHA_HORIZONS.index(30)
    valid = batch.auxiliary_valid[..., horizon]
    scale = torch.where(valid, batch.downside_30d, torch.ones_like(batch.downside_30d))
    error = (
        batch.auxiliary_prediction[..., horizon]
        - batch.auxiliary_target[..., horizon]
    )
    nll = 0.5 * (error / scale).square() + torch.log(scale)
    return nll, valid


@dataclass(frozen=True, slots=True)
class Hold30AlphaMomentSums:
    """Mergeable sufficient statistics for one effective training batch."""

    count: float = 0.0
    sum_active_log: float = 0.0
    sum_active_log_sq: float = 0.0
    sum_policy: float = 0.0
    sum_policy_sq: float = 0.0
    sum_benchmark: float = 0.0
    sum_benchmark_sq: float = 0.0
    sum_policy_excess: float = 0.0
    sum_market_excess: float = 0.0
    sum_market_excess_sq: float = 0.0
    sum_policy_market_excess: float = 0.0
    sum_risk_free_excess: float = 0.0
    sum_risk_free_excess_sq: float = 0.0
    sum_turnover: float = 0.0
    sum_early_exit: float = 0.0
    auxiliary_count: float = 0.0
    auxiliary_squared_error: float = 0.0
    auxiliary_count_h5: float = 0.0
    auxiliary_count_h21: float = 0.0
    auxiliary_count_h30: float = 0.0
    auxiliary_count_h63: float = 0.0
    uncertainty_count: float = 0.0
    uncertainty_nll: float = 0.0

    def __post_init__(self) -> None:
        for name in _MOMENT_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise Hold30AlphaTrainingError(f"moment {name} must be finite")
        count_fields = (
            "count",
            "auxiliary_count",
            "auxiliary_count_h5",
            "auxiliary_count_h21",
            "auxiliary_count_h30",
            "auxiliary_count_h63",
            "uncertainty_count",
        )
        if any(float(getattr(self, name)) < 0 for name in count_fields):
            raise Hold30AlphaTrainingError("moment counts cannot be negative")

    def __add__(self, other: Hold30AlphaMomentSums) -> Hold30AlphaMomentSums:
        if not isinstance(other, Hold30AlphaMomentSums):
            return NotImplemented
        return Hold30AlphaMomentSums(
            **{
                name: float(getattr(self, name)) + float(getattr(other, name))
                for name in _MOMENT_FIELDS
            }
        )

    @classmethod
    def from_batch(cls, batch: Hold30AlphaBatch) -> Hold30AlphaMomentSums:
        mask = batch.mask

        def total(value: torch.Tensor) -> float:
            return float(value.masked_select(mask).detach().to(torch.float64).sum().cpu())

        policy = batch.policy_net_return
        active_log = batch.active_log_return
        policy_excess = policy - batch.risk_free_return
        market_excess = batch.market_return - batch.risk_free_return
        risk_free_excess = batch.risk_free_excess_return
        values: dict[str, float] = {
            "count": float(mask.sum().item()),
            "sum_active_log": total(active_log),
            "sum_active_log_sq": total(active_log.square()),
            "sum_policy": total(policy),
            "sum_policy_sq": total(policy.square()),
            "sum_benchmark": total(batch.benchmark_net_return),
            "sum_benchmark_sq": total(batch.benchmark_net_return.square()),
            "sum_policy_excess": total(policy_excess),
            "sum_market_excess": total(market_excess),
            "sum_market_excess_sq": total(market_excess.square()),
            "sum_policy_market_excess": total(policy_excess * market_excess),
            "sum_risk_free_excess": total(risk_free_excess),
            "sum_risk_free_excess_sq": total(risk_free_excess.square()),
            "sum_turnover": total(batch.discretionary_turnover),
            "sum_early_exit": total(batch.early_exit_mass),
            "auxiliary_count": 0.0,
            "auxiliary_squared_error": 0.0,
            "auxiliary_count_h5": 0.0,
            "auxiliary_count_h21": 0.0,
            "auxiliary_count_h30": 0.0,
            "auxiliary_count_h63": 0.0,
            "uncertainty_count": 0.0,
            "uncertainty_nll": 0.0,
        }
        if batch.auxiliary_prediction is not None:
            assert batch.auxiliary_target is not None
            assert batch.auxiliary_valid is not None
            residual = batch.auxiliary_prediction - batch.auxiliary_target
            aux_mask = batch.auxiliary_valid
            values["auxiliary_count"] = float(aux_mask.sum().item())
            values["auxiliary_squared_error"] = float(
                residual.square().masked_select(aux_mask).detach().to(torch.float64).sum().cpu()
            )
            for horizon_index, horizon in enumerate(HOLD30_ALPHA_HORIZONS):
                values[f"auxiliary_count_h{horizon}"] = float(
                    aux_mask[..., horizon_index].sum().item()
                )
            if batch.downside_30d is not None:
                nll, uncertainty_mask = _hold30_uncertainty_nll(batch)
                values["uncertainty_count"] = float(uncertainty_mask.sum().item())
                values["uncertainty_nll"] = float(
                    nll.masked_select(uncertainty_mask)
                    .detach()
                    .to(torch.float64)
                    .sum()
                    .cpu()
                )
        return cls(**values)

    def packed(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [float(getattr(self, name)) for name in _MOMENT_FIELDS],
            dtype=torch.float64,
            device=device,
        )

    @classmethod
    def unpack(cls, value: torch.Tensor) -> Hold30AlphaMomentSums:
        if value.ndim != 1 or value.numel() != len(_MOMENT_FIELDS):
            raise Hold30AlphaTrainingError("packed moment vector has the wrong shape")
        material = value.detach().to(device="cpu", dtype=torch.float64).tolist()
        return cls(**dict(zip(_MOMENT_FIELDS, material, strict=True)))


@dataclass(frozen=True, slots=True)
class Hold30AlphaMomentRowBinding:
    """Exact Pass-A content bound to one origin/batch row identity."""

    origin_row_id: int
    global_path_id: int
    policy_net_return: float
    valid: bool
    pass_a_row_sha256: str

    def __post_init__(self) -> None:
        for name in ("origin_row_id", "global_path_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Hold30AlphaTrainingError(
                    f"{name} must be a nonnegative integer"
                )
        _require_sha256("pass_a_row_sha256", self.pass_a_row_sha256)
        if not math.isfinite(float(self.policy_net_return)) or float(
            self.policy_net_return
        ) <= -1:
            raise Hold30AlphaTrainingError(
                "bound Pass-A policy return must be finite and exceed -1"
            )
        if not isinstance(self.valid, bool):
            raise Hold30AlphaTrainingError("bound Pass-A validity must be boolean")

    def manifest_payload(self) -> dict[str, object]:
        return {
            "origin_row_id": self.origin_row_id,
            "global_path_id": self.global_path_id,
            "policy_net_return": float(self.policy_net_return),
            "valid": self.valid,
            "pass_a_row_sha256": self.pass_a_row_sha256,
        }


@dataclass(frozen=True, slots=True)
class Hold30AlphaGlobalMomentBinding:
    """Receipt-bound global Pass-A sufficient statistics.

    This is the only accepted external-moment type.  It binds the reduced
    values to one data receipt, objective role, evaluation point, and exact
    row/content inventory so a different Pass B cannot reuse them.
    """

    binding_kind: Hold30AlphaBatchBindingKind
    source_axis_id: str
    objective_inputs_id: str
    role: Hold30AlphaObjectiveRole | Literal["qualification-math-fixture"]
    stream_id: str
    evaluation_point_id: str
    world_size: int
    rows: tuple[Hold30AlphaMomentRowBinding, ...]
    moments: Hold30AlphaMomentSums

    def __post_init__(self) -> None:
        if self.binding_kind not in {
            "receipt-bound",
            "qualification-math-fixture",
        }:
            raise Hold30AlphaTrainingError("unknown global-moment binding kind")
        if not isinstance(self.source_axis_id, str) or not self.source_axis_id:
            raise Hold30AlphaTrainingError(
                "global moments require a non-empty source_axis_id"
            )
        _require_sha256("objective_inputs_id", self.objective_inputs_id)
        _require_sha256("evaluation_point_id", self.evaluation_point_id)
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise Hold30AlphaTrainingError("global moments require a stream_id")
        if self.binding_kind == "receipt-bound":
            if self.role not in {"training", "inner-validation"}:
                raise Hold30AlphaTrainingError(
                    "receipt-bound global moments need a permitted objective role"
                )
        elif self.role != "qualification-math-fixture":
            raise Hold30AlphaTrainingError(
                "math-fixture global moments need the qualification role"
            )
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size not in {1, 2}
        ):
            raise Hold30AlphaTrainingError(
                "global moments support only world_size one or two"
            )
        if not isinstance(self.rows, tuple) or not self.rows:
            raise Hold30AlphaTrainingError(
                "global moments require a non-empty typed row inventory"
            )
        if any(not isinstance(row, Hold30AlphaMomentRowBinding) for row in self.rows):
            raise Hold30AlphaTrainingError(
                "global-moment rows must be typed row bindings"
            )
        keys = tuple((row.origin_row_id, row.global_path_id) for row in self.rows)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise Hold30AlphaTrainingError(
                "global-moment row identities must be unique and sorted"
            )
        if not isinstance(self.moments, Hold30AlphaMomentSums):
            raise Hold30AlphaTrainingError(
                "global-moment binding requires typed sufficient statistics"
            )
        if self.moments.count < 2 or self.moments.count > len(self.rows):
            raise Hold30AlphaTrainingError(
                "global valid-row count must lie between two and inventory size"
            )
        if int(self.moments.count) != sum(row.valid for row in self.rows):
            raise Hold30AlphaTrainingError(
                "global moment count differs from the bound valid-row inventory"
            )

    @property
    def row_identity_sha256(self) -> str:
        return _canonical_sha256(
            [
                [row.origin_row_id, row.global_path_id]
                for row in self.rows
            ]
        )

    @property
    def pass_a_content_sha256(self) -> str:
        return _canonical_sha256(
            [row.manifest_payload() for row in self.rows]
        )

    @property
    def moments_sha256(self) -> str:
        return _canonical_sha256(
            {name: float(getattr(self.moments, name)) for name in _MOMENT_FIELDS}
        )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "binding_kind": self.binding_kind,
            "source_axis_id": self.source_axis_id,
            "objective_inputs_id": self.objective_inputs_id,
            "role": self.role,
            "stream_id": self.stream_id,
            "evaluation_point_id": self.evaluation_point_id,
            "world_size": self.world_size,
            "row_count": len(self.rows),
            "row_identity_sha256": self.row_identity_sha256,
            "pass_a_content_sha256": self.pass_a_content_sha256,
            "moments_sha256": self.moments_sha256,
            "rows": [row.manifest_payload() for row in self.rows],
        }

    @property
    def receipt_id(self) -> str:
        return _canonical_sha256(self.manifest_payload())


def aggregate_hold30_alpha_moments(
    batches: Iterable[Hold30AlphaBatch],
) -> Hold30AlphaMomentSums:
    """Merge raw sufficient statistics; never average microbatch ratios."""

    total = Hold30AlphaMomentSums()
    observed = False
    for batch in batches:
        observed = True
        total = total + Hold30AlphaMomentSums.from_batch(batch)
    if not observed or total.count < 2:
        raise Hold30AlphaTrainingError(
            "the effective batch needs at least two valid return observations"
        )
    return total


def distributed_sum_hold30_alpha_moments(
    local: Hold30AlphaMomentSums,
    *,
    group: dist.ProcessGroup | None = None,
    device: torch.device | str | None = None,
) -> Hold30AlphaMomentSums:
    """SUM-reduce sufficient statistics once across ranks."""

    if not dist.is_available() or not dist.is_initialized():
        raise Hold30AlphaTrainingError("distributed moments require an initialized group")
    if device is None:
        backend = dist.get_backend(group)
        device = torch.device("cpu") if backend == "gloo" else torch.device("cuda")
    packed = local.packed(device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    return Hold30AlphaMomentSums.unpack(packed)


@dataclass(frozen=True, slots=True)
class Hold30AlphaGlobalMetrics:
    count: int
    mean_active_log: float
    tracking_error: float
    beta: float
    policy_volatility: float
    risk_free_excess_mean: float
    risk_free_excess_volatility: float
    total_sharpe: float
    mean_turnover: float
    mean_early_exit: float


def _sample_variance(total: float, total_sq: float, count: float) -> float:
    if count < 2:
        raise Hold30AlphaTrainingError("sample variance requires at least two rows")
    numerator = total_sq - total * total / count
    tolerance = 1e-15 * max(1.0, abs(total_sq))
    if numerator < -tolerance:
        raise Hold30AlphaTrainingError("moment sums imply a negative variance")
    return max(numerator, 0.0) / (count - 1.0)


def hold30_alpha_global_metrics(
    moments: Hold30AlphaMomentSums,
    *,
    direct_sharpe_epsilon: float = 0.0,
) -> Hold30AlphaGlobalMetrics:
    n = float(moments.count)
    if n < 2:
        raise Hold30AlphaTrainingError("global metrics require at least two rows")
    active_variance = _sample_variance(
        moments.sum_active_log, moments.sum_active_log_sq, n
    )
    policy_variance = _sample_variance(moments.sum_policy, moments.sum_policy_sq, n)
    market_variance = _sample_variance(
        moments.sum_market_excess, moments.sum_market_excess_sq, n
    )
    covariance_numerator = (
        moments.sum_policy_market_excess
        - moments.sum_policy_excess * moments.sum_market_excess / n
    )
    market_ss = market_variance * (n - 1.0)
    if market_ss <= 0:
        raise Hold30AlphaTrainingError(
            "beta is undefined for a zero-variance PIT market series"
        )
    beta = covariance_numerator / market_ss
    policy_vol = math.sqrt(max(policy_variance, 0.0) * HOLD30_ALPHA_ANNUALIZATION)
    population_risk_free_variance = (
        moments.sum_risk_free_excess_sq
        - moments.sum_risk_free_excess**2 / n
    ) / n
    if direct_sharpe_epsilon < 0 or not math.isfinite(direct_sharpe_epsilon):
        raise Hold30AlphaTrainingError(
            "direct_sharpe_epsilon must be finite and nonnegative"
        )
    risk_free_std = math.sqrt(
        max(population_risk_free_variance, 0.0) + direct_sharpe_epsilon
    )
    risk_free_mean = moments.sum_risk_free_excess / n
    total_sharpe = (
        math.sqrt(HOLD30_ALPHA_ANNUALIZATION)
        * risk_free_mean
        / risk_free_std
        if risk_free_std > 0
        else 0.0
    )
    return Hold30AlphaGlobalMetrics(
        count=int(n),
        mean_active_log=moments.sum_active_log / n,
        tracking_error=math.sqrt(
            max(active_variance, 0.0) * HOLD30_ALPHA_ANNUALIZATION
        ),
        beta=beta,
        policy_volatility=policy_vol,
        risk_free_excess_mean=risk_free_mean,
        risk_free_excess_volatility=risk_free_std
        * math.sqrt(HOLD30_ALPHA_ANNUALIZATION),
        total_sharpe=total_sharpe,
        mean_turnover=moments.sum_turnover / n,
        mean_early_exit=moments.sum_early_exit / n,
    )


@dataclass(frozen=True, slots=True)
class Hold30AlphaDetachedCoefficients:
    count: int
    active_log_mean: float
    active_log_variance: float
    te_active_log_gradient_scale: float
    market_excess_mean: float
    market_excess_ss: float
    beta: float
    beta_objective_slope: float
    risk_free_excess_mean: float
    risk_free_excess_std: float
    direct_sharpe_weight: float
    turnover_objective_slope: float
    early_exit_weight: float
    auxiliary_weight: float
    auxiliary_weighted_denominator: float
    uncertainty_weight: float
    uncertainty_denominator: float


def derive_hold30_alpha_coefficients(
    moments: Hold30AlphaMomentSums,
    config: Hold30AlphaObjectiveConfig,
) -> Hold30AlphaDetachedCoefficients:
    """Pass A: derive all global coefficients from detached full-batch moments."""

    config.require_resolved()
    epsilon = (
        _finite_nonnegative(
            "direct_sharpe_epsilon", config.direct_sharpe_epsilon
        )
        if config.direct_sharpe_enabled
        else 0.0
    )
    if config.direct_sharpe_enabled and epsilon <= 0:
        raise Hold30AlphaTrainingError(
            "direct_sharpe_epsilon must be explicitly frozen above zero"
        )
    metrics = hold30_alpha_global_metrics(
        moments,
        direct_sharpe_epsilon=epsilon,
    )
    n = float(moments.count)
    active_variance = _sample_variance(
        moments.sum_active_log,
        moments.sum_active_log_sq,
        n,
    )
    tracking_error = math.sqrt(
        max(active_variance, 0.0) * HOLD30_ALPHA_ANNUALIZATION
    )
    te_gradient_scale = 0.0
    if tracking_error > 0:
        te_jacobian_scale = (
            HOLD30_ALPHA_ANNUALIZATION
            / ((n - 1.0) * tracking_error)
        )
        if config.te_floor_enabled and tracking_error < float(config.te_floor):
            te_gradient_scale += (
                2.0
                * _finite_nonnegative(
                    "lambda_te_floor", config.lambda_te_floor
                )
                * (float(config.te_floor) - tracking_error)
                * te_jacobian_scale
            )
        if tracking_error > float(config.te_ceiling):
            te_gradient_scale -= (
                2.0
                * _finite_nonnegative(
                    "lambda_te_ceiling", config.lambda_te_ceiling
                )
                * (tracking_error - float(config.te_ceiling))
                * te_jacobian_scale
            )

    beta_slope = 0.0
    setting = resolve_hold30_alpha_setting(config.setting_id)
    if setting.beta_targeting:
        beta_slope = (
            -2.0
            * _finite_nonnegative("lambda_beta", config.lambda_beta)
            * (metrics.beta - float(config.beta_target))
        )

    direct_weight = (
        _finite_nonnegative("lambda_direct_sharpe", config.lambda_direct_sharpe)
        if config.direct_sharpe_enabled
        else 0.0
    )
    turnover_lambda = _finite_nonnegative(
        "lambda_turnover", config.lambda_turnover
    )
    turnover_slope = -2.0 * turnover_lambda * max(
        metrics.mean_turnover - float(config.target_turnover), 0.0
    )
    market_ss = (
        moments.sum_market_excess_sq
        - moments.sum_market_excess**2 / n
    )
    population_risk_free_variance = max(
        (
            moments.sum_risk_free_excess_sq
            - moments.sum_risk_free_excess**2 / n
        )
        / n,
        0.0,
    )
    alpha_heads = setting.supervised_residual_alpha_heads
    auxiliary_denominator = 0.0
    if alpha_heads:
        assert config.auxiliary_horizon_weights is not None
        horizon_counts = (
            moments.auxiliary_count_h5,
            moments.auxiliary_count_h21,
            moments.auxiliary_count_h30,
            moments.auxiliary_count_h63,
        )
        auxiliary_denominator = sum(
            float(weight) * float(count)
            for weight, count in zip(
                config.auxiliary_horizon_weights,
                horizon_counts,
                strict=True,
            )
        )
        if auxiliary_denominator <= 0:
            raise Hold30AlphaTrainingError(
                "auxiliary objective has no globally valid weighted cells"
            )
    uncertainty_denominator = (
        float(moments.uncertainty_count)
        if config.uncertainty_enabled
        else 0.0
    )
    if config.uncertainty_enabled and uncertainty_denominator <= 0:
        raise Hold30AlphaTrainingError(
            "uncertainty objective has no globally valid 30d cells"
        )
    return Hold30AlphaDetachedCoefficients(
        count=int(n),
        active_log_mean=moments.sum_active_log / n,
        active_log_variance=active_variance,
        te_active_log_gradient_scale=te_gradient_scale,
        market_excess_mean=moments.sum_market_excess / n,
        market_excess_ss=market_ss,
        beta=metrics.beta,
        beta_objective_slope=beta_slope,
        risk_free_excess_mean=moments.sum_risk_free_excess / n,
        risk_free_excess_std=math.sqrt(
            population_risk_free_variance + epsilon
        ),
        direct_sharpe_weight=direct_weight,
        turnover_objective_slope=turnover_slope,
        early_exit_weight=_finite_nonnegative(
            "lambda_early_exit", config.lambda_early_exit
        ),
        auxiliary_weight=(
            _finite_nonnegative(
                "lambda_auxiliary_alpha", config.lambda_auxiliary_alpha
            )
            if alpha_heads
            else 0.0
        ),
        auxiliary_weighted_denominator=auxiliary_denominator,
        uncertainty_weight=(
            _finite_nonnegative("lambda_uncertainty", config.lambda_uncertainty)
            if config.uncertainty_enabled
            else 0.0
        ),
        uncertainty_denominator=uncertainty_denominator,
    )


def direct_sharpe_detached_weights(
    risk_free_excess_return: torch.Tensor,
    coefficients: Hold30AlphaDetachedCoefficients,
) -> torch.Tensor:
    """Exact simple-excess population-Sharpe derivative weights for a07."""

    if risk_free_excess_return.ndim != 1:
        raise Hold30AlphaTrainingError(
            "risk-free simple excess return must be a vector"
        )
    n = float(coefficients.count)
    std = float(coefficients.risk_free_excess_std)
    if std <= 0 or coefficients.direct_sharpe_weight == 0:
        return torch.zeros_like(risk_free_excess_return)
    mean = float(coefficients.risk_free_excess_mean)
    weights = math.sqrt(HOLD30_ALPHA_ANNUALIZATION) * (
        1.0 / (n * std)
        - mean
        * (risk_free_excess_return.detach() - mean)
        / (n * std**3)
    )
    return weights * float(coefficients.direct_sharpe_weight)


def hold30_alpha_surrogate(
    batch: Hold30AlphaBatch,
    coefficients: Hold30AlphaDetachedCoefficients,
    config: Hold30AlphaObjectiveConfig,
    *,
    drawdown_log_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pass B: exact detached-coefficient global objective contribution."""

    config.require_resolved()
    config.validate_batch(batch)
    mask = batch.mask
    n = float(coefficients.count)
    policy = batch.policy_net_return
    active_log = batch.active_log_return
    value = active_log.masked_select(mask).sum() / n

    active_log_gradient = float(
        coefficients.te_active_log_gradient_scale
    ) * (
        active_log.detach() - float(coefficients.active_log_mean)
    )
    value = value + (
        active_log * active_log_gradient.detach()
    ).masked_select(mask).sum()

    market_excess = (
        batch.market_return - batch.risk_free_return
    ).detach()
    beta_gradient = (
        float(coefficients.beta_objective_slope)
        * (market_excess - float(coefficients.market_excess_mean))
        / float(coefficients.market_excess_ss)
    )
    value = value + (
        policy * beta_gradient.detach()
    ).masked_select(mask).sum()

    if config.direct_sharpe_enabled:
        direct = direct_sharpe_detached_weights(
            batch.risk_free_excess_return,
            coefficients,
        )
        value = value + (
            batch.risk_free_excess_return * direct.detach()
        ).masked_select(mask).sum()
    if drawdown_log_weights is not None:
        if tuple(drawdown_log_weights.shape) != tuple(policy.shape):
            raise Hold30AlphaTrainingError("drawdown weights must match return rows")
        value = value + (
            torch.log1p(policy) * drawdown_log_weights.detach()
        ).masked_select(mask).sum()

    value = value + float(coefficients.turnover_objective_slope) * (
        batch.discretionary_turnover.masked_select(mask).sum() / n
    )
    value = value - float(coefficients.early_exit_weight) * (
        batch.early_exit_mass.masked_select(mask).sum() / n
    )
    if batch.auxiliary_prediction is not None:
        assert batch.auxiliary_target is not None
        assert batch.auxiliary_valid is not None
        if coefficients.auxiliary_weight:
            assert config.auxiliary_horizon_weights is not None
            assert config.auxiliary_horizon_scales is not None
            horizon_weights = batch.auxiliary_prediction.new_tensor(
                config.auxiliary_horizon_weights
            )
            horizon_scales = batch.auxiliary_prediction.new_tensor(
                config.auxiliary_horizon_scales
            )
            residual = batch.auxiliary_prediction - batch.auxiliary_target
            weighted_squared_error = (
                residual.div(horizon_scales).square() * horizon_weights
            )
            value = value - float(coefficients.auxiliary_weight) * (
                weighted_squared_error.masked_select(batch.auxiliary_valid).sum()
                / float(coefficients.auxiliary_weighted_denominator)
            )
        if config.uncertainty_enabled and batch.downside_30d is not None:
            nll, valid = _hold30_uncertainty_nll(batch)
            value = value - float(coefficients.uncertainty_weight) * (
                nll.masked_select(valid).sum()
                / float(coefficients.uncertainty_denominator)
            )
    return value


def drawdown_detached_log_weights(
    policy_net_return: torch.Tensor,
    *,
    drawdown_limit: float,
    lambda_drawdown: float,
) -> tuple[torch.Tensor, float]:
    """Exact subgradient of a full-chronology maximum log drawdown penalty."""

    if policy_net_return.ndim != 1 or policy_net_return.numel() == 0:
        raise Hold30AlphaTrainingError("drawdown chronology must be a non-empty vector")
    if bool((policy_net_return <= -1).any()):
        raise Hold30AlphaTrainingError("drawdown returns must exceed -1")
    limit = _finite_nonnegative("drawdown_limit", drawdown_limit)
    coefficient = _finite_nonnegative("lambda_drawdown", lambda_drawdown)
    cumulative = torch.cat(
        (
            policy_net_return.new_zeros(1),
            torch.log1p(policy_net_return.detach()).cumsum(0),
        )
    )
    running, peaks = torch.cummax(cumulative, dim=0)
    gaps = running - cumulative
    trough = int(torch.argmax(gaps))
    drawdown = float(gaps[trough])
    weights = torch.zeros_like(policy_net_return)
    if drawdown > limit:
        peak = int(peaks[trough])
        # Raising any log return after the peak through the trough reduces the
        # drawdown and therefore increases the maximization objective.
        weights[peak:trough] = 2.0 * coefficient * (drawdown - limit)
    return weights, drawdown


def _tensors_match_exactly(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().to(device="cpu"),
            right.detach().to(device="cpu"),
        )
    )


def _verify_hold30_alpha_pass_pair(
    pass_a: Hold30AlphaBatch,
    pass_b: Hold30AlphaBatch,
) -> None:
    """Prove Pass A and Pass B are the same rows at one evaluation point."""

    for name in (
        "binding_kind",
        "source_axis_id",
        "objective_inputs_id",
        "role",
        "stream_id",
        "evaluation_point_id",
    ):
        if getattr(pass_a, name) != getattr(pass_b, name):
            raise Hold30AlphaTrainingError(
                f"Pass A/B {name} identities do not match"
            )
    for name in (
        "origin_row_ids",
        "global_path_ids",
        "benchmark_net_return",
        "market_return",
        "risk_free_return",
        "valid",
        "auxiliary_target",
        "auxiliary_valid",
        # The evaluation-point receipt also binds the recomputed actor outputs.
        "policy_net_return",
        "discretionary_turnover",
        "early_exit_mass",
        "auxiliary_prediction",
        "downside_30d",
    ):
        if not _tensors_match_exactly(getattr(pass_a, name), getattr(pass_b, name)):
            raise Hold30AlphaTrainingError(
                f"Pass A/B tensor {name} does not match exactly"
            )


def _verify_hold30_alpha_effective_batch_identity(
    batches: Sequence[Hold30AlphaBatch],
) -> None:
    reference = batches[0]
    identities: list[tuple[int, int]] = []
    for batch in batches:
        for name in (
            "binding_kind",
            "source_axis_id",
            "objective_inputs_id",
            "role",
            "stream_id",
            "evaluation_point_id",
        ):
            if getattr(batch, name) != getattr(reference, name):
                raise Hold30AlphaTrainingError(
                    f"effective batch mixes incompatible {name} identities"
                )
        identities.extend(
            zip(
                batch.origin_row_ids.detach().to(device="cpu").tolist(),
                batch.global_path_ids.detach().to(device="cpu").tolist(),
                strict=True,
            )
        )
    if len(set(identities)) != len(identities):
        raise Hold30AlphaTrainingError(
            "effective batch repeats origin/batch row identities"
        )


def _hold30_alpha_batch_identity(
    batches: Sequence[Hold30AlphaBatch],
) -> tuple[str, str, str, str, str, str]:
    if not batches:
        raise Hold30AlphaTrainingError(
            "global-moment binding requires at least one local batch"
        )
    _verify_hold30_alpha_effective_batch_identity(batches)
    reference = batches[0]
    return (
        reference.binding_kind,
        reference.source_axis_id,
        reference.objective_inputs_id,
        reference.role,
        reference.stream_id,
        reference.evaluation_point_id,
    )


def _hold30_alpha_row_bindings(
    batches: Sequence[Hold30AlphaBatch],
) -> tuple[Hold30AlphaMomentRowBinding, ...]:
    """Hash all moment-driving values independently for every batch row."""

    _hold30_alpha_batch_identity(batches)
    result: list[Hold30AlphaMomentRowBinding] = []
    tensor_names = (
        "policy_net_return",
        "benchmark_net_return",
        "market_return",
        "risk_free_return",
        "discretionary_turnover",
        "early_exit_mass",
        "valid",
        "auxiliary_prediction",
        "auxiliary_target",
        "auxiliary_valid",
        "downside_30d",
    )
    for batch in batches:
        origins = batch.origin_row_ids.detach().to(device="cpu").tolist()
        path_ids = batch.global_path_ids.detach().to(device="cpu").tolist()
        for index, (origin, path_id) in enumerate(
            zip(origins, path_ids, strict=True)
        ):
            tensors: dict[str, str | None] = {}
            for name in tensor_names:
                value = getattr(batch, name)
                tensors[name] = (
                    None
                    if value is None
                    else _tensor_exact_sha256(value[index])
                )
            result.append(
                Hold30AlphaMomentRowBinding(
                    origin_row_id=int(origin),
                    global_path_id=int(path_id),
                    policy_net_return=float(
                        batch.policy_net_return[index].detach().to(device="cpu")
                    ),
                    valid=(
                        True
                        if batch.valid is None
                        else bool(batch.valid[index].detach().to(device="cpu"))
                    ),
                    pass_a_row_sha256=_canonical_sha256(tensors),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda row: (row.origin_row_id, row.global_path_id),
        )
    )


def _slice_hold30_alpha_batch_row(
    batch: Hold30AlphaBatch,
    index: int,
) -> Hold30AlphaBatch:
    updates: dict[str, torch.Tensor | None] = {}
    for name in (
        "origin_row_ids",
        "global_path_ids",
        "policy_net_return",
        "benchmark_net_return",
        "market_return",
        "risk_free_return",
        "discretionary_turnover",
        "early_exit_mass",
        "valid",
        "auxiliary_prediction",
        "auxiliary_target",
        "auxiliary_valid",
        "downside_30d",
    ):
        value = getattr(batch, name)
        updates[name] = None if value is None else value[index : index + 1]
    return replace(batch, **updates)


def _hold30_alpha_row_moments(
    batches: Sequence[Hold30AlphaBatch],
) -> tuple[tuple[Hold30AlphaMomentRowBinding, Hold30AlphaMomentSums], ...]:
    bindings = {
        (row.origin_row_id, row.global_path_id): row
        for row in _hold30_alpha_row_bindings(batches)
    }
    result: list[tuple[Hold30AlphaMomentRowBinding, Hold30AlphaMomentSums]] = []
    for batch in batches:
        origins = batch.origin_row_ids.detach().to(device="cpu").tolist()
        paths = batch.global_path_ids.detach().to(device="cpu").tolist()
        for index, (origin, path) in enumerate(zip(origins, paths, strict=True)):
            row = bindings[(int(origin), int(path))]
            result.append(
                (
                    row,
                    Hold30AlphaMomentSums.from_batch(
                        _slice_hold30_alpha_batch_row(batch, index)
                    ),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item[0].origin_row_id,
                item[0].global_path_id,
            ),
        )
    )


def _local_hold30_alpha_moments(
    batches: Sequence[Hold30AlphaBatch],
) -> Hold30AlphaMomentSums:
    """Aggregate a rank shard without requiring two rows on every rank."""

    if not batches:
        raise Hold30AlphaTrainingError(
            "global-moment binding requires at least one local batch"
        )
    total = Hold30AlphaMomentSums()
    for batch in batches:
        total = total + Hold30AlphaMomentSums.from_batch(batch)
    if total.count < 1:
        raise Hold30AlphaTrainingError(
            "every distributed rank must own at least one valid Pass-A row"
        )
    return total


def _hold30_alpha_distributed_world(
    group: dist.ProcessGroup | None,
) -> int:
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise Hold30AlphaTrainingError(
                "a process group was supplied before distributed initialization"
            )
        return 1
    world_size = dist.get_world_size(group)
    if world_size not in {1, 2}:
        raise Hold30AlphaTrainingError(
            "Hold30 alpha supports only world_size one or two"
        )
    return world_size


def bind_hold30_alpha_global_moments(
    pass_a_batches: Sequence[Hold30AlphaBatch],
    *,
    group: dist.ProcessGroup | None = None,
    device: torch.device | str | None = None,
) -> Hold30AlphaGlobalMomentBinding:
    """Bind rows and sum moments in one canonical global row order."""

    del device  # The canonical CPU fold is partition-independent.
    identity = _hold30_alpha_batch_identity(pass_a_batches)
    local_pairs = _hold30_alpha_row_moments(pass_a_batches)
    if not any(moment.count > 0 for _row, moment in local_pairs):
        raise Hold30AlphaTrainingError(
            "every distributed rank must own at least one valid Pass-A row"
        )
    world_size = _hold30_alpha_distributed_world(group)

    gathered: list[object]
    if world_size == 1:
        gathered = [(identity, local_pairs)]
    else:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (identity, local_pairs), group=group)

    all_pairs: list[tuple[Hold30AlphaMomentRowBinding, Hold30AlphaMomentSums]] = []
    for payload in gathered:
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise Hold30AlphaTrainingError(
                "distributed Pass-A identity payload is malformed"
            )
        shard_identity, shard_pairs = payload
        if shard_identity != identity:
            raise Hold30AlphaTrainingError(
                "distributed Pass-A ranks disagree on data/evaluation identity"
            )
        if not isinstance(shard_pairs, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], Hold30AlphaMomentRowBinding)
            or not isinstance(pair[1], Hold30AlphaMomentSums)
            for pair in shard_pairs
        ):
            raise Hold30AlphaTrainingError(
                "distributed Pass-A row/moment inventory is malformed"
            )
        all_pairs.extend(shard_pairs)

    ordered_pairs = tuple(
        sorted(
            all_pairs,
            key=lambda pair: (
                pair[0].origin_row_id,
                pair[0].global_path_id,
            ),
        )
    )
    ordered_rows = tuple(row for row, _moment in ordered_pairs)
    global_moments = Hold30AlphaMomentSums()
    for _row, moment in ordered_pairs:
        global_moments = global_moments + moment
    (
        binding_kind,
        source_axis_id,
        objective_inputs_id,
        role,
        stream_id,
        evaluation_point_id,
    ) = identity
    return Hold30AlphaGlobalMomentBinding(
        binding_kind=binding_kind,  # type: ignore[arg-type]
        source_axis_id=source_axis_id,
        objective_inputs_id=objective_inputs_id,
        role=role,  # type: ignore[arg-type]
        stream_id=stream_id,
        evaluation_point_id=evaluation_point_id,
        world_size=world_size,
        rows=ordered_rows,
        moments=global_moments,
    )


def _verify_hold30_alpha_global_binding(
    binding: Hold30AlphaGlobalMomentBinding,
    batches: Sequence[Hold30AlphaBatch],
    *,
    require_complete: bool,
) -> None:
    if not isinstance(binding, Hold30AlphaGlobalMomentBinding):
        raise Hold30AlphaTrainingError(
            "external global moments require a typed row/evaluation receipt"
        )
    if _hold30_alpha_batch_identity(batches) != (
        binding.binding_kind,
        binding.source_axis_id,
        binding.objective_inputs_id,
        binding.role,
        binding.stream_id,
        binding.evaluation_point_id,
    ):
        raise Hold30AlphaTrainingError(
            "global moments belong to another data or evaluation identity"
        )
    local_rows = _hold30_alpha_row_bindings(batches)
    expected = {
        (row.origin_row_id, row.global_path_id): row.pass_a_row_sha256
        for row in binding.rows
    }
    for row in local_rows:
        key = (row.origin_row_id, row.global_path_id)
        if expected.get(key) != row.pass_a_row_sha256:
            raise Hold30AlphaTrainingError(
                "global moments do not bind the exact local Pass-A/B row content"
            )
    if require_complete and local_rows != binding.rows:
        raise Hold30AlphaTrainingError(
            "global moments do not bind the complete Pass-A/B row inventory"
        )


def _verify_distributed_hold30_alpha_pass_b(
    binding: Hold30AlphaGlobalMomentBinding,
    pass_b_batches: Sequence[Hold30AlphaBatch],
    *,
    group: dist.ProcessGroup | None,
) -> None:
    """Collectively prove the Pass-B shard union equals the Pass-A receipt."""

    world_size = _hold30_alpha_distributed_world(group)
    if world_size != binding.world_size:
        raise Hold30AlphaTrainingError(
            "global-moment receipt world size differs from Pass B"
        )
    _verify_hold30_alpha_global_binding(
        binding,
        pass_b_batches,
        require_complete=world_size == 1,
    )
    if world_size == 1:
        return
    local_rows = _hold30_alpha_row_bindings(pass_b_batches)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_rows, group=group)
    global_rows: list[Hold30AlphaMomentRowBinding] = []
    for shard_rows in gathered:
        if not isinstance(shard_rows, tuple) or any(
            not isinstance(row, Hold30AlphaMomentRowBinding)
            for row in shard_rows
        ):
            raise Hold30AlphaTrainingError(
                "distributed Pass-B row inventory is malformed"
            )
        global_rows.extend(shard_rows)
    if tuple(
        sorted(
            global_rows,
            key=lambda row: (row.origin_row_id, row.global_path_id),
        )
    ) != binding.rows:
        raise Hold30AlphaTrainingError(
            "distributed Pass-B rows differ from the exact Pass-A receipt"
        )


@dataclass(frozen=True, slots=True)
class Hold30A06ParameterPartition:
    """Typed, exhaustive alpha-core versus A06-overlay ownership."""

    alpha_core: tuple[tuple[str, torch.nn.Parameter], ...]
    overlay: tuple[tuple[str, torch.nn.Parameter], ...]

    def __post_init__(self) -> None:
        if not self.alpha_core or not self.overlay:
            raise Hold30AlphaTrainingError(
                "A06 requires non-empty alpha-core and overlay parameter sets"
            )
        combined = (*self.alpha_core, *self.overlay)
        names = tuple(name for name, _parameter in combined)
        parameters = tuple(parameter for _name, parameter in combined)
        if len(set(names)) != len(names) or len({id(value) for value in parameters}) != len(
            parameters
        ):
            raise Hold30AlphaTrainingError(
                "A06 parameter ownership must be disjoint and unique"
            )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(parameter, torch.nn.Parameter)
            or not parameter.requires_grad
            for name, parameter in combined
        ):
            raise Hold30AlphaTrainingError(
                "A06 partitions require named trainable Parameters"
            )
        if any("total_risk_head." in name for name, _parameter in self.alpha_core):
            raise Hold30AlphaTrainingError(
                "A06 total-risk parameters leaked into the alpha core"
            )
        if any("total_risk_head." not in name for name, _parameter in self.overlay):
            raise Hold30AlphaTrainingError(
                "A06 overlay may own only total_risk_head parameters"
            )

    @property
    def alpha_core_names(self) -> tuple[str, ...]:
        return tuple(name for name, _parameter in self.alpha_core)

    @property
    def overlay_names(self) -> tuple[str, ...]:
        return tuple(name for name, _parameter in self.overlay)

    @property
    def alpha_core_parameter_names_sha256(self) -> str:
        return _canonical_sha256(list(self.alpha_core_names))

    @property
    def overlay_parameter_names_sha256(self) -> str:
        return _canonical_sha256(list(self.overlay_names))


def partition_hold30_a06_parameters(
    module: torch.nn.Module,
    config: Hold30AlphaObjectiveConfig,
) -> Hold30A06ParameterPartition:
    """Apply the frozen A06 selectors to every trainable model parameter."""

    if config.setting_id != "hold30a-a06-sharpe-overlay":
        raise Hold30AlphaTrainingError(
            "A06 parameter partition requires the registered A06 setting"
        )
    if (
        config.alpha_core_parameter_selector != "alpha-core-only"
        or config.overlay_parameter_selector != "a06-overlay-only"
        or config.stop_gradient_core_to_overlay is not True
        or config.stop_gradient_overlay_to_core is not True
    ):
        raise Hold30AlphaTrainingError(
            "A06 parameter selectors and bidirectional stop-gradient must be frozen"
        )
    named = tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    alpha_core = tuple(
        row for row in named if "total_risk_head." not in row[0]
    )
    overlay = tuple(
        row for row in named if "total_risk_head." in row[0]
    )
    partition = Hold30A06ParameterPartition(
        alpha_core=alpha_core,
        overlay=overlay,
    )
    if {id(parameter) for _name, parameter in named} != {
        id(parameter)
        for _name, parameter in (*partition.alpha_core, *partition.overlay)
    }:
        raise Hold30AlphaTrainingError(
            "A06 partition did not exhaust the trainable module"
        )
    return partition


def _optimizer_json_materialize(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_dtype": str(value.dtype),
            "tensor_shape": list(value.shape),
            "tensor_sha256": _tensor_exact_sha256(value),
        }
    if isinstance(value, dict):
        return {
            str(key): _optimizer_json_materialize(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_optimizer_json_materialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise Hold30AlphaTrainingError(
        f"unsupported optimizer value {type(value).__qualname__}"
    )


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    return _canonical_sha256(
        _optimizer_json_materialize(optimizer.state_dict())
    )


def _require_optimizer_subset(
    optimizer: torch.optim.Optimizer,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    *,
    owner: str,
) -> None:
    expected = tuple(parameter for _name, parameter in named_parameters)
    actual = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if (
        len({id(parameter) for parameter in actual}) != len(actual)
        or {id(parameter) for parameter in actual}
        != {id(parameter) for parameter in expected}
    ):
        raise Hold30AlphaTrainingError(
            f"{owner} optimizer does not exactly own its A06 parameter partition"
        )


@dataclass(frozen=True, slots=True)
class Hold30A06OptimizerSpecReceipt:
    """Immutable plan binding for disjoint ownership and optimizer settings."""

    alpha_core_partition_names: tuple[str, ...]
    overlay_partition_names: tuple[str, ...]
    alpha_core_optimizer_parameter_names: tuple[str, ...]
    overlay_optimizer_parameter_names: tuple[str, ...]
    alpha_core_optimizer_class: str
    overlay_optimizer_class: str
    alpha_core_optimizer_groups_sha256: str
    overlay_optimizer_groups_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "alpha_core_optimizer_groups_sha256",
            "overlay_optimizer_groups_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("alpha_core_optimizer_class", "overlay_optimizer_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise Hold30AlphaTrainingError(
                    f"{name} must be a non-empty class identity"
                )
        for name in (
            "alpha_core_partition_names",
            "overlay_partition_names",
            "alpha_core_optimizer_parameter_names",
            "overlay_optimizer_parameter_names",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise Hold30AlphaTrainingError(
                    f"{name} must contain unique ordered parameter names"
                )
        if set(self.alpha_core_partition_names) != set(
            self.alpha_core_optimizer_parameter_names
        ) or set(self.overlay_partition_names) != set(
            self.overlay_optimizer_parameter_names
        ):
            raise Hold30AlphaTrainingError(
                "optimizer-spec ordered names do not exhaust their partitions"
            )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "alpha_core_partition_names": list(self.alpha_core_partition_names),
            "overlay_partition_names": list(self.overlay_partition_names),
            "alpha_core_optimizer_parameter_names": list(
                self.alpha_core_optimizer_parameter_names
            ),
            "overlay_optimizer_parameter_names": list(
                self.overlay_optimizer_parameter_names
            ),
            "alpha_core_optimizer_class": self.alpha_core_optimizer_class,
            "overlay_optimizer_class": self.overlay_optimizer_class,
            "alpha_core_optimizer_groups_sha256": (
                self.alpha_core_optimizer_groups_sha256
            ),
            "overlay_optimizer_groups_sha256": (
                self.overlay_optimizer_groups_sha256
            ),
        }

    @property
    def receipt_id(self) -> str:
        return _canonical_sha256(self.manifest_payload())


def _optimizer_spec(
    optimizer: torch.optim.Optimizer,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[tuple[str, ...], str]:
    name_by_parameter = {
        id(parameter): name for name, parameter in named_parameters
    }
    ordered_names: list[str] = []
    groups: list[dict[str, object]] = []
    for group in optimizer.param_groups:
        names = tuple(name_by_parameter[id(parameter)] for parameter in group["params"])
        ordered_names.extend(names)
        groups.append(
            {
                "parameter_names": list(names),
                "options": _optimizer_json_materialize(
                    {key: value for key, value in group.items() if key != "params"}
                ),
            }
        )
    return tuple(ordered_names), _canonical_sha256(groups)


def build_hold30_a06_optimizer_spec_receipt(
    partition: Hold30A06ParameterPartition,
    alpha_core_optimizer: torch.optim.Optimizer,
    overlay_optimizer: torch.optim.Optimizer,
) -> Hold30A06OptimizerSpecReceipt:
    """Bind immutable optimizer topology, options, and ordered ownership."""

    _require_optimizer_subset(
        alpha_core_optimizer,
        partition.alpha_core,
        owner="alpha-core",
    )
    _require_optimizer_subset(
        overlay_optimizer,
        partition.overlay,
        owner="overlay",
    )
    alpha_names, alpha_groups_sha256 = _optimizer_spec(
        alpha_core_optimizer,
        partition.alpha_core,
    )
    overlay_names, overlay_groups_sha256 = _optimizer_spec(
        overlay_optimizer,
        partition.overlay,
    )
    return Hold30A06OptimizerSpecReceipt(
        alpha_core_partition_names=partition.alpha_core_names,
        overlay_partition_names=partition.overlay_names,
        alpha_core_optimizer_parameter_names=alpha_names,
        overlay_optimizer_parameter_names=overlay_names,
        alpha_core_optimizer_class=(
            f"{type(alpha_core_optimizer).__module__}."
            f"{type(alpha_core_optimizer).__qualname__}"
        ),
        overlay_optimizer_class=(
            f"{type(overlay_optimizer).__module__}."
            f"{type(overlay_optimizer).__qualname__}"
        ),
        alpha_core_optimizer_groups_sha256=alpha_groups_sha256,
        overlay_optimizer_groups_sha256=overlay_groups_sha256,
    )


def hold30_alpha_evaluation_point_id(module: torch.nn.Module) -> str:
    """Hash the complete module state at one model evaluation point."""

    if not isinstance(module, torch.nn.Module):
        raise Hold30AlphaTrainingError("evaluation point requires a torch Module")
    return _canonical_sha256(
        [
            {
                "name": name,
                "sha256": _tensor_exact_sha256(value),
            }
            for name, value in sorted(module.state_dict().items())
        ]
    )


@dataclass(frozen=True, slots=True)
class Hold30A06OptimizerStateReceipt:
    """Mutable parent-linked ledger entry for one exact optimizer state."""

    optimizer_spec_receipt_sha256: str
    update_index: int
    parent_state_receipt_sha256: str | None
    evaluation_point_id: str
    alpha_core_parameter_state_sha256: str
    overlay_parameter_state_sha256: str
    alpha_core_optimizer_state_sha256: str
    overlay_optimizer_state_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "optimizer_spec_receipt_sha256",
            "evaluation_point_id",
            "alpha_core_parameter_state_sha256",
            "overlay_parameter_state_sha256",
            "alpha_core_optimizer_state_sha256",
            "overlay_optimizer_state_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.update_index, bool)
            or not isinstance(self.update_index, int)
            or self.update_index < 0
        ):
            raise Hold30AlphaTrainingError(
                "optimizer-state update_index must be nonnegative"
            )
        if self.update_index == 0:
            if self.parent_state_receipt_sha256 is not None:
                raise Hold30AlphaTrainingError(
                    "initial optimizer state cannot have a parent receipt"
                )
        elif self.parent_state_receipt_sha256 is None:
            raise Hold30AlphaTrainingError(
                "post-initial optimizer state requires a parent receipt"
            )
        else:
            _require_sha256(
                "parent_state_receipt_sha256",
                self.parent_state_receipt_sha256,
            )

    def manifest_payload(self) -> dict[str, object]:
        return {
            "optimizer_spec_receipt_sha256": self.optimizer_spec_receipt_sha256,
            "update_index": self.update_index,
            "parent_state_receipt_sha256": self.parent_state_receipt_sha256,
            "evaluation_point_id": self.evaluation_point_id,
            "alpha_core_parameter_state_sha256": (
                self.alpha_core_parameter_state_sha256
            ),
            "overlay_parameter_state_sha256": self.overlay_parameter_state_sha256,
            "alpha_core_optimizer_state_sha256": (
                self.alpha_core_optimizer_state_sha256
            ),
            "overlay_optimizer_state_sha256": (
                self.overlay_optimizer_state_sha256
            ),
        }

    @property
    def receipt_id(self) -> str:
        return _canonical_sha256(self.manifest_payload())


def build_hold30_a06_optimizer_state_receipt(
    module: torch.nn.Module,
    partition: Hold30A06ParameterPartition,
    alpha_core_optimizer: torch.optim.Optimizer,
    overlay_optimizer: torch.optim.Optimizer,
    optimizer_spec_receipt: Hold30A06OptimizerSpecReceipt,
    *,
    update_index: int,
    parent_state_receipt_sha256: str | None,
) -> Hold30A06OptimizerStateReceipt:
    """Materialize one mutable state entry under an immutable spec."""

    actual_spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    if actual_spec != optimizer_spec_receipt:
        raise Hold30AlphaTrainingError(
            "A06 optimizers do not match the immutable optimizer spec"
        )
    return Hold30A06OptimizerStateReceipt(
        optimizer_spec_receipt_sha256=optimizer_spec_receipt.receipt_id,
        update_index=update_index,
        parent_state_receipt_sha256=parent_state_receipt_sha256,
        evaluation_point_id=hold30_alpha_evaluation_point_id(module),
        alpha_core_parameter_state_sha256=(
            _named_hold30_alpha_parameter_sha256(
                partition.alpha_core,
                gradients=False,
            )
        ),
        overlay_parameter_state_sha256=_named_hold30_alpha_parameter_sha256(
            partition.overlay,
            gradients=False,
        ),
        alpha_core_optimizer_state_sha256=_optimizer_state_sha256(
            alpha_core_optimizer
        ),
        overlay_optimizer_state_sha256=_optimizer_state_sha256(overlay_optimizer),
    )


@dataclass(frozen=True, slots=True)
class Hold30A06OverlayCoefficients:
    """Detached full-batch coefficients for the overlay-only objective."""

    count: int
    policy_mean: float
    policy_std: float
    benchmark_std: float
    volatility_ratio: float
    volatility_gradient_scale: float
    risk_free_excess_mean: float
    risk_free_excess_std: float
    total_excess_mean_weight: float
    total_sharpe_weight: float


def derive_hold30_a06_overlay_coefficients(
    moments: Hold30AlphaMomentSums,
    config: Hold30AlphaObjectiveConfig,
) -> Hold30A06OverlayCoefficients:
    """Pass A for the separately optimized A06 total-risk overlay."""

    if config.setting_id != "hold30a-a06-sharpe-overlay":
        raise Hold30AlphaTrainingError(
            "overlay coefficients are exclusive to registered A06"
        )
    config.require_resolved()
    n = float(moments.count)
    policy_variance = _sample_variance(
        moments.sum_policy,
        moments.sum_policy_sq,
        n,
    )
    benchmark_variance = _sample_variance(
        moments.sum_benchmark,
        moments.sum_benchmark_sq,
        n,
    )
    benchmark_std = math.sqrt(max(benchmark_variance, 0.0))
    if benchmark_std <= 0:
        raise Hold30AlphaTrainingError(
            "A06 volatility ratio requires nonzero C1 volatility"
        )
    policy_std = math.sqrt(max(policy_variance, 0.0))
    volatility_ratio = policy_std / benchmark_std
    volatility_gradient_scale = 0.0
    if policy_std > 0:
        volatility_gradient_scale = (
            -2.0
            * _finite_nonnegative(
                "lambda_volatility_ratio",
                config.lambda_volatility_ratio,
            )
            * (volatility_ratio - float(config.target_volatility_ratio))
            / (benchmark_std * (n - 1.0) * policy_std)
        )
    population_excess_variance = max(
        (
            moments.sum_risk_free_excess_sq
            - moments.sum_risk_free_excess**2 / n
        )
        / n,
        0.0,
    )
    epsilon = _finite_positive(
        "total_sharpe_epsilon",
        config.total_sharpe_epsilon,
    )
    return Hold30A06OverlayCoefficients(
        count=int(n),
        policy_mean=moments.sum_policy / n,
        policy_std=policy_std,
        benchmark_std=benchmark_std,
        volatility_ratio=volatility_ratio,
        volatility_gradient_scale=volatility_gradient_scale,
        risk_free_excess_mean=moments.sum_risk_free_excess / n,
        risk_free_excess_std=math.sqrt(population_excess_variance + epsilon),
        total_excess_mean_weight=_finite_nonnegative(
            "lambda_total_excess_mean",
            config.lambda_total_excess_mean,
        ),
        total_sharpe_weight=_finite_nonnegative(
            "lambda_total_sharpe_overlay",
            config.lambda_total_sharpe_overlay,
        ),
    )


def _hold30_a06_drawdown_weight_map(
    binding: Hold30AlphaGlobalMomentBinding,
    config: Hold30AlphaObjectiveConfig,
) -> dict[tuple[int, int], float]:
    by_path: dict[int, list[Hold30AlphaMomentRowBinding]] = {}
    for row in binding.rows:
        if row.valid:
            by_path.setdefault(row.global_path_id, []).append(row)
    result: dict[tuple[int, int], float] = {}
    for batch_row, rows in by_path.items():
        ordered = sorted(rows, key=lambda row: row.origin_row_id)
        returns = torch.tensor(
            [row.policy_net_return for row in ordered],
            dtype=torch.float64,
        )
        weights, _drawdown = drawdown_detached_log_weights(
            returns,
            drawdown_limit=_finite_nonnegative(
                "drawdown_limit",
                config.drawdown_limit,
            ),
            lambda_drawdown=_finite_nonnegative(
                "lambda_drawdown",
                config.lambda_drawdown,
            ),
        )
        for row, weight in zip(ordered, weights.tolist(), strict=True):
            result[(row.origin_row_id, batch_row)] = float(weight)
    return result


def hold30_a06_overlay_surrogate(
    batch: Hold30AlphaBatch,
    coefficients: Hold30A06OverlayCoefficients,
    drawdown_weights: torch.Tensor,
) -> torch.Tensor:
    """Pass-B contribution containing only A06 total-risk terms."""

    if tuple(drawdown_weights.shape) != tuple(batch.policy_net_return.shape):
        raise Hold30AlphaTrainingError(
            "A06 drawdown weights must match the local return rows"
        )
    mask = batch.mask
    n = float(coefficients.count)
    policy = batch.policy_net_return
    excess = batch.risk_free_excess_return
    value = float(coefficients.total_excess_mean_weight) * (
        excess.masked_select(mask).sum() / n
    )
    std = float(coefficients.risk_free_excess_std)
    if std > 0 and coefficients.total_sharpe_weight:
        mean = float(coefficients.risk_free_excess_mean)
        sharpe_gradient = math.sqrt(HOLD30_ALPHA_ANNUALIZATION) * (
            1.0 / (n * std)
            - mean * (excess.detach() - mean) / (n * std**3)
        )
        value = value + float(coefficients.total_sharpe_weight) * (
            excess * sharpe_gradient.detach()
        ).masked_select(mask).sum()
    volatility_gradient = float(coefficients.volatility_gradient_scale) * (
        policy.detach() - float(coefficients.policy_mean)
    )
    value = value + (
        policy * volatility_gradient.detach()
    ).masked_select(mask).sum()
    value = value + (
        torch.log1p(policy) * drawdown_weights.detach()
    ).masked_select(mask).sum()
    return value


def hold30_a06_overlay_two_pass_objective(
    pass_a_batches: Sequence[Hold30AlphaBatch],
    pass_b_batches: Sequence[Hold30AlphaBatch],
    config: Hold30AlphaObjectiveConfig,
    *,
    global_moments: Hold30AlphaGlobalMomentBinding,
) -> tuple[torch.Tensor, Hold30AlphaGlobalMetrics]:
    """Exact overlay-only objective; no alpha-core term enters this graph."""

    if config.setting_id != "hold30a-a06-sharpe-overlay":
        raise Hold30AlphaTrainingError(
            "overlay-only objective is exclusive to registered A06"
        )
    if len(pass_a_batches) != len(pass_b_batches) or not pass_a_batches:
        raise Hold30AlphaTrainingError(
            "A06 Pass A/B must contain the same nonzero batches"
        )
    for batch_a, batch_b in zip(pass_a_batches, pass_b_batches, strict=True):
        _verify_hold30_alpha_pass_pair(batch_a, batch_b)
    _verify_hold30_alpha_effective_batch_identity(pass_a_batches)
    _verify_hold30_alpha_effective_batch_identity(pass_b_batches)
    for batch in (*pass_a_batches, *pass_b_batches):
        config.validate_batch(batch)
        if batch.binding_kind == "receipt-bound" and config.qualification_math_test_only:
            raise Hold30AlphaTrainingError(
                "qualification-math-test configs cannot execute on receipt-bound data"
            )
        if (
            batch.binding_kind == "qualification-math-fixture"
            and not config.qualification_math_test_only
        ):
            raise Hold30AlphaTrainingError(
                "executable configs cannot consume qualification math fixtures"
            )
    _verify_hold30_alpha_global_binding(
        global_moments,
        pass_a_batches,
        require_complete=global_moments.world_size == 1,
    )
    _verify_hold30_alpha_global_binding(
        global_moments,
        pass_b_batches,
        require_complete=global_moments.world_size == 1,
    )
    coefficients = derive_hold30_a06_overlay_coefficients(
        global_moments.moments,
        config,
    )
    drawdown_by_row = _hold30_a06_drawdown_weight_map(global_moments, config)
    total: torch.Tensor | None = None
    for batch in pass_b_batches:
        keys = zip(
            batch.origin_row_ids.detach().to(device="cpu").tolist(),
            batch.global_path_ids.detach().to(device="cpu").tolist(),
            strict=True,
        )
        drawdown = batch.policy_net_return.new_tensor(
            [drawdown_by_row.get((int(origin), int(row)), 0.0) for origin, row in keys]
        )
        contribution = hold30_a06_overlay_surrogate(
            batch,
            coefficients,
            drawdown,
        )
        total = contribution if total is None else total + contribution
    assert total is not None
    return total, hold30_alpha_global_metrics(
        global_moments.moments,
        direct_sharpe_epsilon=_finite_positive(
            "total_sharpe_epsilon",
            config.total_sharpe_epsilon,
        ),
    )


def hold30_alpha_two_pass_objective(
    pass_a_batches: Sequence[Hold30AlphaBatch],
    pass_b_batches: Sequence[Hold30AlphaBatch],
    config: Hold30AlphaObjectiveConfig,
    *,
    global_moments: Hold30AlphaGlobalMomentBinding | None = None,
) -> tuple[torch.Tensor, Hold30AlphaGlobalMetrics]:
    """Compose one effective-batch Pass B objective from Pass A coefficients."""

    if len(pass_a_batches) != len(pass_b_batches) or not pass_a_batches:
        raise Hold30AlphaTrainingError("Pass A/B must contain the same nonzero batches")
    for batch_a, batch_b in zip(pass_a_batches, pass_b_batches, strict=True):
        _verify_hold30_alpha_pass_pair(batch_a, batch_b)
    _verify_hold30_alpha_effective_batch_identity(pass_a_batches)
    _verify_hold30_alpha_effective_batch_identity(pass_b_batches)
    for batch in (*pass_a_batches, *pass_b_batches):
        config.validate_batch(batch)
        if batch.binding_kind == "receipt-bound" and config.qualification_math_test_only:
            raise Hold30AlphaTrainingError(
                "qualification-math-test configs cannot execute on receipt-bound data"
            )
        if (
            batch.binding_kind == "qualification-math-fixture"
            and not config.qualification_math_test_only
        ):
            raise Hold30AlphaTrainingError(
                "executable configs cannot consume qualification math fixtures"
            )
    if global_moments is None:
        moments = aggregate_hold30_alpha_moments(pass_a_batches)
    else:
        if not isinstance(global_moments, Hold30AlphaGlobalMomentBinding):
            raise Hold30AlphaTrainingError(
                "external global moments require a typed row/evaluation receipt"
            )
        _verify_hold30_alpha_global_binding(
            global_moments,
            pass_a_batches,
            require_complete=global_moments.world_size == 1,
        )
        _verify_hold30_alpha_global_binding(
            global_moments,
            pass_b_batches,
            require_complete=global_moments.world_size == 1,
        )
        moments = global_moments.moments
    coefficients = derive_hold30_alpha_coefficients(moments, config)
    total: torch.Tensor | None = None
    for batch in pass_b_batches:
        contribution = hold30_alpha_surrogate(batch, coefficients, config)
        total = contribution if total is None else total + contribution
    assert total is not None
    epsilon = (
        _finite_nonnegative(
            "direct_sharpe_epsilon", config.direct_sharpe_epsilon
        )
        if config.direct_sharpe_enabled
        else 0.0
    )
    return total, hold30_alpha_global_metrics(
        moments,
        direct_sharpe_epsilon=epsilon,
    )


def _hold30_alpha_parameter_sha256(
    module: torch.nn.Module,
    *,
    gradients: bool,
) -> str:
    return _named_hold30_alpha_parameter_sha256(
        tuple(
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ),
        gradients=gradients,
    )


def _named_hold30_alpha_parameter_sha256(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    gradients: bool,
) -> str:
    payload: list[dict[str, str | None]] = []
    for name, parameter in named_parameters:
        value = parameter.grad if gradients else parameter
        payload.append(
            {
                "name": name,
                "sha256": None if value is None else _tensor_exact_sha256(value),
            }
        )
    return _canonical_sha256(payload)


def _sum_reduce_hold30_alpha_gradients(
    parameters: Sequence[torch.nn.Parameter],
    *,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> None:
    if world_size == 1:
        return
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


def _hold30_alpha_require_optimizer_ownership(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(
        parameter for parameter in module.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise Hold30AlphaTrainingError(
            "Hold30 alpha update requires trainable parameters"
        )
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if (
        len({id(parameter) for parameter in optimizer_parameters})
        != len(optimizer_parameters)
        or {id(parameter) for parameter in optimizer_parameters}
        != {id(parameter) for parameter in parameters}
    ):
        raise Hold30AlphaTrainingError(
            "optimizer must own every trainable alpha parameter exactly once"
        )
    return parameters


def _distributed_hold30_alpha_require_equal(
    value: object,
    *,
    name: str,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> None:
    if world_size == 1:
        return
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, value, group=group)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise Hold30AlphaTrainingError(
            f"distributed ranks disagree on {name}"
        )


def train_hold30_alpha_two_pass_update(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pass_a_batches: Sequence[Hold30AlphaBatch],
    pass_b_batches: Sequence[Hold30AlphaBatch],
    config: Hold30AlphaObjectiveConfig,
    *,
    global_moments: Hold30AlphaGlobalMomentBinding | None = None,
    group: dist.ProcessGroup | None = None,
    moment_device: torch.device | str | None = None,
    grad_clip: float = 0.0,
) -> dict[str, object]:
    """Maximize one exact global objective with one SUM-reduced update.

    The typed Pass-A binding is created collectively unless supplied by a
    receipt-complete caller. Pass-B rows are gathered and required to match
    that receipt before gradients are computed. Gradient reduction mirrors
    the qualified Hold-30 two-rank primitive: SUM, never rank averaging.
    """

    if not isinstance(module, torch.nn.Module):
        raise Hold30AlphaTrainingError("module must be a torch Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise Hold30AlphaTrainingError("optimizer must be a torch optimizer")
    if (
        isinstance(grad_clip, bool)
        or not isinstance(grad_clip, (int, float))
        or not math.isfinite(float(grad_clip))
        or float(grad_clip) < 0
    ):
        raise Hold30AlphaTrainingError(
            "grad_clip must be a finite nonnegative scalar"
        )
    parameters = _hold30_alpha_require_optimizer_ownership(module, optimizer)
    world_size = _hold30_alpha_distributed_world(group)
    initial_parameter_sha256 = _hold30_alpha_parameter_sha256(
        module,
        gradients=False,
    )
    _distributed_hold30_alpha_require_equal(
        initial_parameter_sha256,
        name="initial parameters",
        world_size=world_size,
        group=group,
    )

    if global_moments is None:
        binding = bind_hold30_alpha_global_moments(
            pass_a_batches,
            group=group,
            device=moment_device,
        )
    elif isinstance(global_moments, Hold30AlphaGlobalMomentBinding):
        binding = global_moments
    else:
        raise Hold30AlphaTrainingError(
            "external global moments require a typed row/evaluation receipt"
        )
    if binding.world_size != world_size:
        raise Hold30AlphaTrainingError(
            "global-moment receipt world size differs from optimizer update"
        )
    _verify_distributed_hold30_alpha_pass_b(
        binding,
        pass_b_batches,
        group=group,
    )

    optimizer.zero_grad(set_to_none=True)
    objective, metrics = hold30_alpha_two_pass_objective(
        pass_a_batches,
        pass_b_batches,
        config,
        global_moments=binding,
    )
    (-objective).backward()

    _sum_reduce_hold30_alpha_gradients(
        parameters,
        world_size=world_size,
        group=group,
    )
    if float(grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))

    gradient_sha256 = _hold30_alpha_parameter_sha256(module, gradients=True)
    _distributed_hold30_alpha_require_equal(
        gradient_sha256,
        name="SUM-reduced gradients",
        world_size=world_size,
        group=group,
    )
    optimizer.step()
    parameter_sha256 = _hold30_alpha_parameter_sha256(module, gradients=False)
    _distributed_hold30_alpha_require_equal(
        parameter_sha256,
        name="updated parameters",
        world_size=world_size,
        group=group,
    )

    objective_total = objective.detach().to(
        device=parameters[0].device,
        dtype=torch.float64,
    )
    if world_size == 2:
        dist.all_reduce(objective_total, op=dist.ReduceOp.SUM, group=group)
    return {
        "objective": float(objective_total.cpu()),
        "global_metrics": metrics,
        "global_moment_receipt": binding.manifest_payload(),
        "global_moment_receipt_sha256": binding.receipt_id,
        "row_identity_sha256": binding.row_identity_sha256,
        "pass_a_content_sha256": binding.pass_a_content_sha256,
        "moments_sha256": binding.moments_sha256,
        "initial_parameter_sha256": initial_parameter_sha256,
        "gradient_sha256": gradient_sha256,
        "parameter_sha256": parameter_sha256,
        "optimizer_steps": 1,
        "distributed_world_size": world_size,
        "gradient_reduction": "SUM",
    }


def _verify_hold30_a06_three_stream_contract(
    alpha_core_pass_a_batches: Sequence[Hold30AlphaBatch],
    alpha_core_pass_b_batches: Sequence[Hold30AlphaBatch],
    executed_pass_a_batches: Sequence[Hold30AlphaBatch],
    overlay_pass_b_batches: Sequence[Hold30AlphaBatch],
) -> None:
    if not (
        len(alpha_core_pass_a_batches)
        == len(alpha_core_pass_b_batches)
        == len(executed_pass_a_batches)
        == len(overlay_pass_b_batches)
        > 0
    ):
        raise Hold30AlphaTrainingError(
            "A06 requires complete core, overlay-gradient, and executed streams"
        )
    immutable_tensors = (
        "origin_row_ids",
        "global_path_ids",
        "benchmark_net_return",
        "market_return",
        "risk_free_return",
        "valid",
        "auxiliary_prediction",
        "auxiliary_target",
        "auxiliary_valid",
        "downside_30d",
    )
    for core_a, core_b, executed_a, overlay_b in zip(
        alpha_core_pass_a_batches,
        alpha_core_pass_b_batches,
        executed_pass_a_batches,
        overlay_pass_b_batches,
        strict=True,
    ):
        if core_a.stream_id != "a06-alpha-core" or core_b.stream_id != (
            "a06-alpha-core"
        ):
            raise Hold30AlphaTrainingError(
                "A06 canonical core batches require stream_id a06-alpha-core"
            )
        if executed_a.stream_id != "a06-executed-overlay" or overlay_b.stream_id != (
            "a06-executed-overlay"
        ):
            raise Hold30AlphaTrainingError(
                "A06 overlay replay/execution requires stream_id a06-executed-overlay"
            )
        _verify_hold30_alpha_pass_pair(core_a, core_b)
        _verify_hold30_alpha_pass_pair(executed_a, overlay_b)
        for name in (
            "binding_kind",
            "source_axis_id",
            "objective_inputs_id",
            "role",
            "evaluation_point_id",
        ):
            if getattr(core_a, name) != getattr(executed_a, name):
                raise Hold30AlphaTrainingError(
                    f"A06 core and executed streams differ on {name}"
                )
        for name in immutable_tensors:
            if not _tensors_match_exactly(
                getattr(core_a, name),
                getattr(executed_a, name),
            ):
                raise Hold30AlphaTrainingError(
                    f"A06 core and executed streams differ on immutable {name}"
                )
        for batch in (core_a, executed_a):
            for name in (
                "policy_net_return",
                "discretionary_turnover",
                "early_exit_mass",
                "auxiliary_prediction",
                "downside_30d",
            ):
                value = getattr(batch, name)
                if value is not None and value.requires_grad:
                    raise Hold30AlphaTrainingError(
                        "A06 Pass-A economic streams must be detached"
                    )


def train_hold30_a06_two_optimizer_update(
    module: torch.nn.Module,
    alpha_core_optimizer: torch.optim.Optimizer,
    overlay_optimizer: torch.optim.Optimizer,
    alpha_core_pass_a_batches: Sequence[Hold30AlphaBatch],
    alpha_core_pass_b_batches: Sequence[Hold30AlphaBatch],
    executed_pass_a_batches: Sequence[Hold30AlphaBatch],
    overlay_pass_b_batches: Sequence[Hold30AlphaBatch],
    config: Hold30AlphaObjectiveConfig,
    *,
    optimizer_spec_receipt: Hold30A06OptimizerSpecReceipt,
    optimizer_state_receipt: Hold30A06OptimizerStateReceipt,
    alpha_core_global_moments: Hold30AlphaGlobalMomentBinding | None = None,
    executed_global_moments: Hold30AlphaGlobalMomentBinding | None = None,
    group: dist.ProcessGroup | None = None,
    moment_device: torch.device | str | None = None,
    alpha_core_grad_clip: float = 0.0,
    overlay_grad_clip: float = 0.0,
) -> dict[str, object]:
    """Update A06 from separate core-only and final-overlay economics."""

    if not isinstance(module, torch.nn.Module):
        raise Hold30AlphaTrainingError("module must be a torch Module")
    if not isinstance(alpha_core_optimizer, torch.optim.Optimizer) or not isinstance(
        overlay_optimizer, torch.optim.Optimizer
    ):
        raise Hold30AlphaTrainingError("A06 requires two torch optimizers")
    if not isinstance(optimizer_spec_receipt, Hold30A06OptimizerSpecReceipt):
        raise Hold30AlphaTrainingError("A06 requires a typed optimizer-spec receipt")
    if not isinstance(optimizer_state_receipt, Hold30A06OptimizerStateReceipt):
        raise Hold30AlphaTrainingError("A06 requires a typed optimizer-state receipt")
    for name, value in (
        ("alpha_core_grad_clip", alpha_core_grad_clip),
        ("overlay_grad_clip", overlay_grad_clip),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise Hold30AlphaTrainingError(
                f"{name} must be a finite nonnegative scalar"
            )
    config.require_resolved()
    _verify_hold30_a06_three_stream_contract(
        alpha_core_pass_a_batches,
        alpha_core_pass_b_batches,
        executed_pass_a_batches,
        overlay_pass_b_batches,
    )
    partition = partition_hold30_a06_parameters(module, config)
    actual_spec = build_hold30_a06_optimizer_spec_receipt(
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
    )
    if actual_spec != optimizer_spec_receipt or (
        config.separate_optimizer_spec_receipt_sha256
        != optimizer_spec_receipt.receipt_id
    ):
        raise Hold30AlphaTrainingError(
            "A06 runtime does not match the immutable optimizer spec"
        )
    evaluation_point_id = _hold30_alpha_batch_identity(
        alpha_core_pass_a_batches
    )[-1]
    if evaluation_point_id != _hold30_alpha_batch_identity(executed_pass_a_batches)[-1]:
        raise Hold30AlphaTrainingError(
            "A06 core and executed streams use different evaluation points"
        )
    current_state_receipt = build_hold30_a06_optimizer_state_receipt(
        module,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        optimizer_spec_receipt,
        update_index=optimizer_state_receipt.update_index,
        parent_state_receipt_sha256=(
            optimizer_state_receipt.parent_state_receipt_sha256
        ),
    )
    if optimizer_state_receipt != current_state_receipt or (
        optimizer_state_receipt.evaluation_point_id != evaluation_point_id
    ):
        raise Hold30AlphaTrainingError(
            "A06 optimizer/model states do not match the supplied ledger receipt"
        )

    world_size = _hold30_alpha_distributed_world(group)
    for name, value in (
        ("A06 optimizer spec", optimizer_spec_receipt.receipt_id),
        ("A06 optimizer state", optimizer_state_receipt.receipt_id),
    ):
        _distributed_hold30_alpha_require_equal(
            value,
            name=name,
            world_size=world_size,
            group=group,
        )
    initial_parameter_sha256 = _hold30_alpha_parameter_sha256(
        module,
        gradients=False,
    )
    _distributed_hold30_alpha_require_equal(
        initial_parameter_sha256,
        name="initial A06 parameters",
        world_size=world_size,
        group=group,
    )

    def resolve_binding(
        batches: Sequence[Hold30AlphaBatch],
        supplied: Hold30AlphaGlobalMomentBinding | None,
    ) -> Hold30AlphaGlobalMomentBinding:
        if supplied is None:
            return bind_hold30_alpha_global_moments(
                batches,
                group=group,
                device=moment_device,
            )
        if not isinstance(supplied, Hold30AlphaGlobalMomentBinding):
            raise Hold30AlphaTrainingError(
                "external global moments require a typed row/evaluation receipt"
            )
        return supplied

    alpha_core_binding = resolve_binding(
        alpha_core_pass_a_batches,
        alpha_core_global_moments,
    )
    executed_binding = resolve_binding(
        executed_pass_a_batches,
        executed_global_moments,
    )
    if (
        alpha_core_binding.world_size != world_size
        or executed_binding.world_size != world_size
        or alpha_core_binding.receipt_id == executed_binding.receipt_id
    ):
        raise Hold30AlphaTrainingError(
            "A06 requires distinct world-complete core and executed receipts"
        )
    _verify_distributed_hold30_alpha_pass_b(
        alpha_core_binding,
        alpha_core_pass_b_batches,
        group=group,
    )
    _verify_distributed_hold30_alpha_pass_b(
        executed_binding,
        overlay_pass_b_batches,
        group=group,
    )

    alpha_core_optimizer.zero_grad(set_to_none=True)
    overlay_optimizer.zero_grad(set_to_none=True)
    alpha_core_objective, alpha_core_metrics = hold30_alpha_two_pass_objective(
        alpha_core_pass_a_batches,
        alpha_core_pass_b_batches,
        config,
        global_moments=alpha_core_binding,
    )
    (-alpha_core_objective).backward()
    if any(parameter.grad is not None for _name, parameter in partition.overlay):
        raise Hold30AlphaTrainingError(
            "A06 alpha-core objective leaked gradients into the overlay"
        )
    if not any(parameter.grad is not None for _name, parameter in partition.alpha_core):
        raise Hold30AlphaTrainingError(
            "A06 alpha-core objective produced no alpha-core gradients"
        )
    alpha_core_gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for _name, parameter in partition.alpha_core
    )
    for _name, parameter in partition.alpha_core:
        parameter.grad = None

    overlay_objective, executed_metrics = hold30_a06_overlay_two_pass_objective(
        executed_pass_a_batches,
        overlay_pass_b_batches,
        config,
        global_moments=executed_binding,
    )
    (-overlay_objective).backward()
    if any(parameter.grad is not None for _name, parameter in partition.alpha_core):
        raise Hold30AlphaTrainingError(
            "A06 overlay objective leaked gradients into the alpha core"
        )
    for (_name, parameter), gradient in zip(
        partition.alpha_core,
        alpha_core_gradients,
        strict=True,
    ):
        parameter.grad = gradient
    if not any(parameter.grad is not None for _name, parameter in partition.overlay):
        raise Hold30AlphaTrainingError(
            "A06 overlay objective produced no overlay gradients"
        )

    alpha_core_parameters = tuple(
        parameter for _name, parameter in partition.alpha_core
    )
    overlay_parameters = tuple(parameter for _name, parameter in partition.overlay)
    _sum_reduce_hold30_alpha_gradients(
        alpha_core_parameters,
        world_size=world_size,
        group=group,
    )
    _sum_reduce_hold30_alpha_gradients(
        overlay_parameters,
        world_size=world_size,
        group=group,
    )
    if float(alpha_core_grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(
            alpha_core_parameters,
            float(alpha_core_grad_clip),
        )
    if float(overlay_grad_clip) > 0:
        torch.nn.utils.clip_grad_norm_(
            overlay_parameters,
            float(overlay_grad_clip),
        )
    alpha_core_gradient_sha256 = _named_hold30_alpha_parameter_sha256(
        partition.alpha_core,
        gradients=True,
    )
    overlay_gradient_sha256 = _named_hold30_alpha_parameter_sha256(
        partition.overlay,
        gradients=True,
    )
    for name, value in (
        ("SUM-reduced A06 alpha-core gradients", alpha_core_gradient_sha256),
        ("SUM-reduced A06 overlay gradients", overlay_gradient_sha256),
    ):
        _distributed_hold30_alpha_require_equal(
            value,
            name=name,
            world_size=world_size,
            group=group,
        )

    alpha_core_optimizer.step()
    overlay_optimizer.step()
    parameter_sha256 = _hold30_alpha_parameter_sha256(module, gradients=False)
    _distributed_hold30_alpha_require_equal(
        parameter_sha256,
        name="updated A06 parameters",
        world_size=world_size,
        group=group,
    )
    post_update_state_receipt = build_hold30_a06_optimizer_state_receipt(
        module,
        partition,
        alpha_core_optimizer,
        overlay_optimizer,
        optimizer_spec_receipt,
        update_index=optimizer_state_receipt.update_index + 1,
        parent_state_receipt_sha256=optimizer_state_receipt.receipt_id,
    )
    if post_update_state_receipt.evaluation_point_id == evaluation_point_id:
        raise Hold30AlphaTrainingError(
            "A06 update did not produce a new model evaluation point"
        )
    _distributed_hold30_alpha_require_equal(
        post_update_state_receipt.receipt_id,
        name="post-update A06 optimizer state",
        world_size=world_size,
        group=group,
    )

    objective_totals = torch.stack(
        (
            alpha_core_objective.detach().to(
                device=alpha_core_parameters[0].device,
                dtype=torch.float64,
            ),
            overlay_objective.detach().to(
                device=alpha_core_parameters[0].device,
                dtype=torch.float64,
            ),
        )
    )
    if world_size == 2:
        dist.all_reduce(objective_totals, op=dist.ReduceOp.SUM, group=group)
    return {
        "alpha_core_objective": float(objective_totals[0].cpu()),
        "overlay_objective": float(objective_totals[1].cpu()),
        "alpha_core_global_metrics": alpha_core_metrics,
        "executed_global_metrics": executed_metrics,
        "alpha_core_global_moment_receipt": alpha_core_binding.manifest_payload(),
        "alpha_core_global_moment_receipt_sha256": alpha_core_binding.receipt_id,
        "executed_global_moment_receipt": executed_binding.manifest_payload(),
        "executed_global_moment_receipt_sha256": executed_binding.receipt_id,
        "optimizer_spec_receipt_sha256": optimizer_spec_receipt.receipt_id,
        "pre_update_optimizer_state_receipt_sha256": (
            optimizer_state_receipt.receipt_id
        ),
        "post_update_optimizer_state_receipt": (
            post_update_state_receipt.manifest_payload()
        ),
        "post_update_optimizer_state_receipt_sha256": (
            post_update_state_receipt.receipt_id
        ),
        "pre_update_evaluation_point_id": evaluation_point_id,
        "post_update_evaluation_point_id": (
            post_update_state_receipt.evaluation_point_id
        ),
        "initial_parameter_sha256": initial_parameter_sha256,
        "alpha_core_gradient_sha256": alpha_core_gradient_sha256,
        "overlay_gradient_sha256": overlay_gradient_sha256,
        "parameter_sha256": parameter_sha256,
        "alpha_core_optimizer_steps": 1,
        "overlay_optimizer_steps": 1,
        "distributed_world_size": world_size,
        "gradient_reduction": "SUM",
        "gradient_isolation_verified": True,
        "three_stream_contract_verified": True,
    }


@dataclass(frozen=True, slots=True)
class Hold30AlphaValidationMetrics:
    """Six-fold/five-seed aggregate metrics for one synchronized update."""

    update: int
    coverage_complete: bool
    active_return_20bp: float
    active_return_40bp: float
    tracking_error: float
    beta: float
    median_sale_age: float
    projection_distance: float
    forced_turnover_fraction: float
    median_active_return_20bp: float
    information_ratio_20bp: float
    total_sharpe_20bp: float
    max_drawdown_20bp: float
    turnover_cost_20bp: float
    trace_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
        ):
            raise Hold30AlphaTrainingError("validation update must be positive")
        for name in (
            "active_return_20bp",
            "active_return_40bp",
            "tracking_error",
            "beta",
            "median_sale_age",
            "projection_distance",
            "forced_turnover_fraction",
            "median_active_return_20bp",
            "information_ratio_20bp",
            "total_sharpe_20bp",
            "max_drawdown_20bp",
            "turnover_cost_20bp",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise Hold30AlphaTrainingError(f"{name} must be finite")
        if any(
            getattr(self, name) < 0
            for name in (
                "tracking_error",
                "median_sale_age",
                "projection_distance",
                "forced_turnover_fraction",
                "max_drawdown_20bp",
                "turnover_cost_20bp",
            )
        ):
            raise Hold30AlphaTrainingError("risk/age/cost metrics cannot be negative")
        if (
            not isinstance(self.trace_sha256, str)
            or len(self.trace_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.trace_sha256)
        ):
            raise Hold30AlphaTrainingError("trace_sha256 must be a lowercase digest")

    def eligible(
        self,
        setting_id: str,
        *,
        contract: Hold30AlphaCheckpointContract = HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
    ) -> bool:
        resolve_hold30_alpha_setting(setting_id)
        if not contract.result_moving_thresholds_complete:
            raise Hold30AlphaUnresolvedCoefficientError(
                "projection/forced checkpoint thresholds remain unresolved"
            )
        assert contract.projection_distance_max is not None
        assert contract.forced_turnover_fraction_max is not None
        return bool(
            self.coverage_complete
            and self.tracking_error >= HOLD30_ALPHA_TE_MIN_ANNUAL
            and self.tracking_error <= HOLD30_ALPHA_TE_MAX_ANNUAL
            and HOLD30_ALPHA_BETA_TARGET - HOLD30_ALPHA_BETA_TOLERANCE
            <= self.beta
            <= HOLD30_ALPHA_BETA_TARGET + HOLD30_ALPHA_BETA_TOLERANCE
            and HOLD30_ALPHA_MEDIAN_AGE_BAND[0]
            <= self.median_sale_age
            <= HOLD30_ALPHA_MEDIAN_AGE_BAND[1]
            and self.projection_distance <= contract.projection_distance_max
            and self.forced_turnover_fraction
            <= contract.forced_turnover_fraction_max
        )

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            -float(self.median_active_return_20bp),
            -float(self.information_ratio_20bp),
            -float(self.total_sharpe_20bp),
            float(self.max_drawdown_20bp),
            float(self.turnover_cost_20bp),
            float(self.update),
        )


@dataclass(frozen=True, slots=True)
class Hold30AlphaCheckpointCandidate:
    metrics: Hold30AlphaValidationMetrics
    fold_seed_updates: tuple[tuple[int, int, int], ...]
    bundle_id: str

    def __post_init__(self) -> None:
        expected = tuple(
            (fold_index, seed)
            for fold_index in range(6)
            for seed in HOLD30_SEEDS
        )
        if tuple(
            (fold_index, seed)
            for fold_index, seed, _update in self.fold_seed_updates
        ) != expected:
            raise Hold30AlphaTrainingError(
                "candidate must bind all six folds by five ordered seeds"
            )
        if any(
            update != self.metrics.update
            for _fold_index, _seed, update in self.fold_seed_updates
        ):
            raise Hold30AlphaTrainingError(
                "all thirty fold/seed checkpoints must share one update"
            )
        if not isinstance(self.bundle_id, str) or not self.bundle_id:
            raise Hold30AlphaTrainingError("bundle_id must be non-empty")


def select_hold30_alpha_checkpoint(
    setting_id: str,
    candidates: Sequence[Hold30AlphaCheckpointCandidate],
    *,
    contract: Hold30AlphaCheckpointContract = HOLD30_ALPHA_V3_CHECKPOINT_CONTRACT,
) -> Hold30AlphaCheckpointCandidate:
    """Filter qualification gates, then apply the frozen lexicographic rank."""

    rows = tuple(candidates)
    if not rows:
        raise Hold30AlphaTrainingError("checkpoint selection requires candidates")
    if len({row.metrics.update for row in rows}) != len(rows):
        raise Hold30AlphaTrainingError("checkpoint updates must be unique")
    eligible = tuple(
        row for row in rows if row.metrics.eligible(setting_id, contract=contract)
    )
    if not eligible:
        raise Hold30AlphaTrainingError("no synchronized checkpoint is eligible")
    return min(eligible, key=lambda row: (*row.metrics.rank_key, row.bundle_id))


__all__ = [
    "HOLD30_ALPHA_ANNUALIZATION",
    "HOLD30_ALPHA_BETA_TARGET",
    "HOLD30_ALPHA_BETA_TOLERANCE",
    "HOLD30_ALPHA_MEDIAN_AGE_BAND",
    "HOLD30_ALPHA_TARGET_TURNOVER",
    "HOLD30_ALPHA_TRAIN_COST_BPS",
    "HOLD30_ALPHA_VALIDATION_COSTS_BPS",
    "Hold30A06OptimizerSpecReceipt",
    "Hold30A06OptimizerStateReceipt",
    "Hold30A06OverlayCoefficients",
    "Hold30A06ParameterPartition",
    "Hold30AlphaBatch",
    "Hold30AlphaBoundObjectiveInputs",
    "Hold30AlphaCheckpointCandidate",
    "Hold30AlphaDetachedCoefficients",
    "Hold30AlphaGlobalMetrics",
    "Hold30AlphaGlobalMomentBinding",
    "Hold30AlphaMomentRowBinding",
    "Hold30AlphaMomentSums",
    "Hold30AlphaObjectiveConfig",
    "Hold30AlphaTrainingError",
    "Hold30AlphaUnresolvedCoefficientError",
    "Hold30AlphaValidationMetrics",
    "aggregate_hold30_alpha_moments",
    "bind_hold30_alpha_global_moments",
    "bind_hold30_alpha_objective_inputs",
    "build_hold30_a06_optimizer_spec_receipt",
    "build_hold30_a06_optimizer_state_receipt",
    "derive_hold30_a06_overlay_coefficients",
    "derive_hold30_alpha_coefficients",
    "direct_sharpe_detached_weights",
    "distributed_sum_hold30_alpha_moments",
    "drawdown_detached_log_weights",
    "hold30_a06_overlay_surrogate",
    "hold30_a06_overlay_two_pass_objective",
    "hold30_alpha_evaluation_point_id",
    "hold30_alpha_global_metrics",
    "hold30_alpha_surrogate",
    "hold30_alpha_two_pass_objective",
    "partition_hold30_a06_parameters",
    "select_hold30_alpha_checkpoint",
    "train_hold30_a06_two_optimizer_update",
    "train_hold30_alpha_two_pass_update",
]
