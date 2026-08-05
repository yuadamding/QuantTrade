"""Objective primitives for the M03R v5 active-alpha Hold-30 generation.

This module is intentionally disjoint from :mod:`hold30_alpha`, whose public
types and receipt semantics belong to the frozen v3 generation.  M03R removes
the tracking-error floor, controls *active* market beta around zero, and keeps
factor-risk, turnover, forced-exit, and early-exit terms individually visible.

The helpers here are production-path building blocks, not launch authority.
All result-moving multipliers must be supplied by an immutable experiment
plan. Factor identities, units, normalization, asymmetric slabs, and the exact
ordered PIT manifest hashes come only from the typed execution risk contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rl_quant.execution.hold30_m03r_projection_v5 import (
    M03RObjectiveRiskContract,
    M03RProjectionError,
    M03RQualifiedRiskManifest,
    validate_m03r_objective_risk_contract,
    validate_m03r_qualified_risk_manifest,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_DESIGN,
    M03RProtocolError,
    M03RSetting,
    resolve_m03r_v5_setting,
)

M03R_ANNUALIZATION_SESSIONS = 252
M03R_ANNUAL_TRACKING_ERROR_CEILING = (
    M03R_DESIGN.active_risk.annual_tracking_error_ceiling
)
M03R_MAX_PREFERRED_TRACKING_ERROR = (
    M03R_DESIGN.active_risk.confidence_preferred_annual_tracking_error_maximum
)
M03R_ACTIVE_BETA_TARGET = M03R_DESIGN.active_risk.active_market_beta_target
M03R_ACTIVE_BETA_ABSOLUTE_LIMIT = (
    M03R_DESIGN.active_risk.absolute_active_market_beta_maximum
)
M03R_DAILY_ONE_WAY_TURNOVER_TARGET = (
    M03R_DESIGN.active_risk.target_daily_one_way_discretionary_turnover
)
M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS = M03R_DESIGN.training_one_way_cost_basis_points
M03R_FIXED_TE_FLOOR_SETTING_ID = "A05-fixed-te-floor"
M03R_DIRECT_SHARPE_SETTING_ID = "A07-direct-sharpe"
M03R_DIRECT_SHARPE_CONSTRUCTION_ID = (
    "full-effective-batch-two-pass-return-level-gradient-surrogate-v1"
)


class M03RObjectiveError(ValueError):
    """The M03R objective or its bound tensors are incomplete or invalid."""


def _finite_nonnegative(name: str, value: float | None) -> float:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise M03RObjectiveError(
            f"{name} must be an explicitly frozen finite nonnegative scalar"
        )
    return float(value)


def _resolve_setting(setting_id: str) -> M03RSetting:
    if not isinstance(setting_id, str) or not setting_id:
        raise M03RObjectiveError("setting_id must be one exact non-empty M03R ID")
    try:
        return resolve_m03r_v5_setting(setting_id)
    except M03RProtocolError as exc:
        raise M03RObjectiveError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class M03RObjectiveConfig:
    """Manifest-owned coefficients for one exact M03R economic loss.

    ``None`` means that a term is causally inapplicable, not that a launch plan
    forgot to resolve it.  Conversely, every applicable coefficient must be
    explicit.  Only A05 may supply a tracking-error-floor coefficient and only
    A07 may supply a direct-Sharpe coefficient.
    """

    setting_id: str
    risk_contract: M03RObjectiveRiskContract
    lambda_tracking_error_ceiling: float | None = None
    lambda_tracking_error_floor: float | None = None
    lambda_active_beta: float | None = None
    lambda_factor_exposure: float | None = None
    lambda_turnover: float | None = None
    lambda_early_exit: float | None = None
    lambda_forced_turnover: float | None = None
    lambda_auxiliary: float | None = None
    lambda_direct_sharpe: float | None = None
    daily_one_way_turnover_target: float = M03R_DAILY_ONE_WAY_TURNOVER_TARGET
    training_one_way_cost_basis_points: int = M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS

    def __post_init__(self) -> None:
        _resolve_setting(self.setting_id)
        try:
            validate_m03r_objective_risk_contract(self.risk_contract)
        except M03RProjectionError as exc:
            raise M03RObjectiveError(str(exc)) from exc
        if (
            self.daily_one_way_turnover_target != M03R_DAILY_ONE_WAY_TURNOVER_TARGET
            or self.training_one_way_cost_basis_points
            != M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS
        ):
            raise M03RObjectiveError("canonical M03R turnover/cost constants drifted")

    def require_resolved(self) -> M03RSetting:
        """Resolve the exact setting and reject missing or irrelevant terms."""

        setting = _resolve_setting(self.setting_id)
        applicable = {
            "lambda_tracking_error_ceiling": (
                setting.annual_tracking_error_ceiling is not None
            ),
            "lambda_tracking_error_floor": (
                setting.setting_id == M03R_FIXED_TE_FLOOR_SETTING_ID
            ),
            "lambda_active_beta": setting.active_beta_neutrality,
            "lambda_factor_exposure": setting.factor_sector_projection,
            "lambda_turnover": True,
            "lambda_early_exit": True,
            "lambda_forced_turnover": True,
            "lambda_auxiliary": setting.residual_alpha_heads,
            "lambda_direct_sharpe": (
                setting.setting_id == M03R_DIRECT_SHARPE_SETTING_ID
            ),
        }
        for name, is_applicable in applicable.items():
            value = getattr(self, name)
            if is_applicable:
                _finite_nonnegative(name, value)
            elif value is not None:
                raise M03RObjectiveError(
                    f"{name} is irrelevant for exact setting {setting.setting_id!r} "
                    "and must be None"
                )
        if setting.setting_id == M03R_FIXED_TE_FLOOR_SETTING_ID and (
            setting.annual_tracking_error_floor != 0.02
        ):
            raise M03RObjectiveError("A05 must use its frozen 2% annual TE floor")
        return setting


@dataclass(frozen=True, slots=True)
class M03RDirectSharpeSurrogate:
    """Precomputed A07 two-pass loss term for one complete effective batch.

    The term is already signed for minimization and must retain its autograd
    graph.  It may not be computed independently per microbatch and averaged:
    Sharpe's ratio statistic requires full-effective-batch moments and a
    second gradient-enabled pass with detached return-level coefficients.
    """

    loss_term: torch.Tensor
    observation_count: int
    construction_id: str = M03R_DIRECT_SHARPE_CONSTRUCTION_ID

    def __post_init__(self) -> None:
        if (
            not isinstance(self.loss_term, torch.Tensor)
            or self.loss_term.numel() != 1
            or not self.loss_term.is_floating_point()
            or not bool(torch.isfinite(self.loss_term).all())
            or not self.loss_term.requires_grad
        ):
            raise M03RObjectiveError(
                "direct Sharpe surrogate must be one finite gradient-enabled scalar"
            )
        if (
            isinstance(self.observation_count, bool)
            or not isinstance(self.observation_count, int)
            or self.observation_count < 2
        ):
            raise M03RObjectiveError(
                "direct Sharpe surrogate observation_count must be an integer >= 2"
            )
        if self.construction_id != M03R_DIRECT_SHARPE_CONSTRUCTION_ID:
            raise M03RObjectiveError(
                "direct Sharpe surrogate must use the frozen full-batch two-pass construction"
            )


@dataclass(frozen=True, slots=True)
class M03RObjectiveInputs:
    """One complete effective batch for the additive M03R objective."""

    policy_net_return: torch.Tensor
    benchmark_net_return: torch.Tensor
    market_excess_return: torch.Tensor
    discretionary_one_way_turnover: torch.Tensor
    early_exit_notional: torch.Tensor
    forced_one_way_turnover: torch.Tensor
    asset_ids: tuple[str, ...]
    policy_weights: torch.Tensor
    benchmark_weights: torch.Tensor
    qualified_risk_manifests: tuple[M03RQualifiedRiskManifest, ...]
    risk_manifest_sha256s: tuple[str, ...]
    valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.policy_net_return.shape)
        if len(shape) != 1 or shape[0] < 2:
            raise M03RObjectiveError(
                "M03R requires at least two chronological return observations"
            )
        for name in (
            "policy_net_return",
            "benchmark_net_return",
            "market_excess_return",
            "discretionary_one_way_turnover",
            "early_exit_notional",
            "forced_one_way_turnover",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise M03RObjectiveError(f"{name} must be finite with shape {shape}")
        if (
            not isinstance(self.asset_ids, tuple)
            or len(self.asset_ids) < 2
            or len(set(self.asset_ids)) != len(self.asset_ids)
            or any(
                not isinstance(asset_id, str) or not asset_id.strip()
                for asset_id in self.asset_ids
            )
        ):
            raise M03RObjectiveError(
                "asset_ids must be distinct nonempty ordered strings"
            )
        weight_shape = (shape[0], len(self.asset_ids))
        for name in ("policy_weights", "benchmark_weights"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != weight_shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or bool((value < -1e-10).any())
                or not bool(
                    torch.allclose(
                        value.sum(dim=-1),
                        torch.ones(
                            shape[0],
                            device=value.device,
                            dtype=value.dtype,
                        ),
                        atol=1e-7,
                        rtol=1e-7,
                    )
                )
            ):
                raise M03RObjectiveError(
                    f"{name} must be finite unit-simplex [session, asset]"
                )
        if (
            self.policy_weights.device != self.benchmark_weights.device
            or self.policy_weights.dtype != self.benchmark_weights.dtype
        ):
            raise M03RObjectiveError(
                "policy and benchmark weights must share device and dtype"
            )
        if self.benchmark_weights.requires_grad:
            raise M03RObjectiveError("benchmark_weights must be detached")
        if len(self.risk_manifest_sha256s) != shape[0] or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.risk_manifest_sha256s
        ):
            raise M03RObjectiveError(
                "risk_manifest_sha256s must bind one lowercase digest per observation"
            )
        if len(self.qualified_risk_manifests) != shape[0]:
            raise M03RObjectiveError(
                "qualified_risk_manifests must bind one typed manifest per observation"
            )
        for index, (qualified, expected_digest) in enumerate(
            zip(
                self.qualified_risk_manifests,
                self.risk_manifest_sha256s,
                strict=True,
            )
        ):
            try:
                validate_m03r_qualified_risk_manifest(
                    qualified,
                    expected_manifest_sha256=expected_digest,
                )
            except M03RProjectionError as exc:
                raise M03RObjectiveError(
                    f"qualified risk manifest {index} failed validation: {exc}"
                ) from exc
            if qualified.manifest.asset_ids != self.asset_ids:
                raise M03RObjectiveError(
                    f"qualified risk manifest {index} asset axis does not match objective weights"
                )
        for name in ("policy_net_return", "benchmark_net_return"):
            if bool((getattr(self, name) <= -1.0).any()):
                raise M03RObjectiveError(f"{name} must be greater than -1")
        for name in (
            "discretionary_one_way_turnover",
            "early_exit_notional",
            "forced_one_way_turnover",
        ):
            if bool((getattr(self, name) < 0.0).any()):
                raise M03RObjectiveError(f"{name} cannot be negative")
        if self.valid is not None and (
            not isinstance(self.valid, torch.Tensor)
            or self.valid.dtype != torch.bool
            or tuple(self.valid.shape) != shape
        ):
            raise M03RObjectiveError("valid must be boolean with the return shape")
        if not bool(self.mask.any()) or int(self.mask.sum().item()) < 2:
            raise M03RObjectiveError("M03R needs at least two valid observations")
        for name in (
            "benchmark_net_return",
            "market_excess_return",
        ):
            if getattr(self, name).requires_grad:
                raise M03RObjectiveError(f"{name} must be a detached bound input")

    @property
    def mask(self) -> torch.Tensor:
        if self.valid is None:
            return torch.ones_like(self.policy_net_return, dtype=torch.bool)
        return self.valid


@dataclass(frozen=True, slots=True)
class M03RObjectiveMetrics:
    setting_id: str
    observation_count: int
    mean_policy_net_log_return: float
    mean_net_active_log_return: float
    annual_tracking_error: float
    active_market_beta: float
    mean_discretionary_one_way_turnover: float
    mean_early_exit_notional: float
    mean_forced_one_way_turnover: float
    annual_tracking_error_penalty: float
    annual_tracking_error_floor_penalty: float
    active_market_beta_penalty: float
    factor_exposure_penalty: float
    turnover_penalty: float
    direct_sharpe_surrogate_loss: float


def confidence_scaled_preferred_tracking_error(
    signal_confidence: torch.Tensor,
) -> torch.Tensor:
    """Map calibrated confidence in ``[0,1]`` to preferred TE in ``[0,4%]``."""

    if (
        not isinstance(signal_confidence, torch.Tensor)
        or not signal_confidence.is_floating_point()
        or not bool(torch.isfinite(signal_confidence).all())
        or bool(((signal_confidence < 0.0) | (signal_confidence > 1.0)).any())
    ):
        raise M03RObjectiveError(
            "signal_confidence must be a finite floating-point tensor in [0,1]"
        )
    return signal_confidence * M03R_MAX_PREFERRED_TRACKING_ERROR


def _sample_variance(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean()
    return centered.square().sum() / (value.numel() - 1)


def m03r_active_objective(
    inputs: M03RObjectiveInputs,
    config: M03RObjectiveConfig,
    *,
    auxiliary_loss: torch.Tensor | None = None,
    direct_sharpe_surrogate: M03RDirectSharpeSurrogate | None = None,
) -> tuple[torch.Tensor, M03RObjectiveMetrics]:
    """Return one setting-bound differentiable utility and detached metrics.

    The objective is computed over one complete effective batch.  Callers that
    use memory-bounded microbatches must implement an exact sufficient-statistic
    or two-pass gradient, not average separately computed ratio penalties.
    """

    setting = config.require_resolved()
    if setting.residual_alpha_heads:
        if (
            not isinstance(auxiliary_loss, torch.Tensor)
            or auxiliary_loss.numel() != 1
            or not auxiliary_loss.is_floating_point()
            or not bool(torch.isfinite(auxiliary_loss).all())
        ):
            raise M03RObjectiveError(
                "auxiliary_loss must be one finite scalar for this exact setting"
            )
    elif auxiliary_loss is not None:
        raise M03RObjectiveError(
            f"auxiliary_loss is irrelevant for exact setting {setting.setting_id!r}"
        )

    if setting.setting_id == M03R_DIRECT_SHARPE_SETTING_ID:
        if not isinstance(direct_sharpe_surrogate, M03RDirectSharpeSurrogate):
            raise M03RObjectiveError(
                "A07 requires an explicit precomputed two-pass direct Sharpe surrogate"
            )
        if direct_sharpe_surrogate.observation_count != int(inputs.mask.sum().item()):
            raise M03RObjectiveError(
                "direct Sharpe surrogate observation count must match the complete effective batch"
            )
    elif direct_sharpe_surrogate is not None:
        raise M03RObjectiveError(
            "direct Sharpe surrogate is causally applicable only to A07-direct-sharpe"
        )
    if (
        inputs.risk_manifest_sha256s
        != config.risk_contract.ordered_risk_manifest_sha256s
    ):
        raise M03RObjectiveError(
            "objective observations do not match the ordered risk-manifest contract"
        )
    factor_rows: list[torch.Tensor] = []
    for index, qualified in enumerate(inputs.qualified_risk_manifests):
        manifest = qualified.manifest
        lower = tuple(
            float(value)
            for value in manifest.exposure_lower_bounds.detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )
        upper = tuple(
            float(value)
            for value in manifest.exposure_upper_bounds.detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        )
        observed_contract = (
            manifest.exposure_names,
            manifest.exposure_families,
            manifest.exposure_units,
            manifest.exposure_normalization_ids,
            lower,
            upper,
        )
        expected_contract = (
            config.risk_contract.exposure_names,
            config.risk_contract.exposure_families,
            config.risk_contract.exposure_units,
            config.risk_contract.exposure_normalization_ids,
            config.risk_contract.exposure_lower_bounds,
            config.risk_contract.exposure_upper_bounds,
        )
        if observed_contract != expected_contract:
            raise M03RObjectiveError(
                f"qualified risk manifest {index} disagrees with objective risk contract"
            )
        loadings = manifest.exposure_loadings.detach().to(
            device=inputs.policy_weights.device,
            dtype=inputs.policy_weights.dtype,
        )
        active_weights = (
            inputs.policy_weights[index] - inputs.benchmark_weights[index]
        )
        factor_rows.append(loadings @ active_weights)
    factors_all = torch.stack(factor_rows, dim=0)

    mask = inputs.mask
    policy = inputs.policy_net_return.masked_select(mask)
    benchmark = inputs.benchmark_net_return.masked_select(mask)
    market = inputs.market_excess_return.masked_select(mask)
    discretionary = inputs.discretionary_one_way_turnover.masked_select(mask)
    early = inputs.early_exit_notional.masked_select(mask)
    forced = inputs.forced_one_way_turnover.masked_select(mask)
    factors = factors_all[mask]

    policy_log = torch.log1p(policy)
    active_log = policy_log - torch.log1p(benchmark)
    active_simple = policy - benchmark
    mean_policy_log = policy_log.mean()
    mean_active_log = active_log.mean()
    if setting.objective_mode == "absolute-net-log-return":
        mean_economic_log = mean_policy_log
    elif setting.objective_mode == "c1-active-net-log-return":
        mean_economic_log = mean_active_log
    else:  # pragma: no cover - protocol validation owns this invariant.
        raise M03RObjectiveError(
            f"unsupported objective mode {setting.objective_mode!r}"
        )
    annual_active_variance = _sample_variance(active_log).clamp_min(0.0) * float(
        M03R_ANNUALIZATION_SESSIONS
    )
    if float(annual_active_variance.detach()) > 0.0:
        annual_te = torch.sqrt(annual_active_variance)
    else:
        # sqrt has an infinite derivative at zero.  The exact TE is zero and
        # this connected zero branch gives canonical/A05 finite autograd.
        annual_te = annual_active_variance * 0.0

    centered_active = active_simple - active_simple.mean()
    centered_market = market - market.mean()
    market_ss = centered_market.square().sum()
    if float(market_ss.detach()) <= 0.0:
        raise M03RObjectiveError(
            "active market beta is undefined for a zero-variance market series"
        )
    active_beta = (centered_active * centered_market).sum() / market_ss

    # Below the ceiling, the exact one-sided loss and its derivative are zero.
    # Selecting that inactive branch from detached variance avoids the
    # undefined sqrt derivative at a legitimate zero-active-risk solution.
    te_ceiling = setting.annual_tracking_error_ceiling
    if (
        te_ceiling is not None
        and float(annual_active_variance.detach()) > float(te_ceiling) ** 2
    ):
        te_penalty = (annual_te - float(te_ceiling)).square()
    else:
        te_penalty = annual_active_variance * 0.0

    te_floor = setting.annual_tracking_error_floor
    if setting.setting_id == M03R_FIXED_TE_FLOOR_SETTING_ID and te_floor is not None:
        if float(annual_active_variance.detach()) == 0.0:
            te_floor_penalty = annual_active_variance * 0.0 + float(te_floor) ** 2
        elif float(annual_te.detach()) < float(te_floor):
            te_floor_penalty = (float(te_floor) - annual_te).square()
        else:
            te_floor_penalty = annual_active_variance * 0.0
    else:
        te_floor_penalty = annual_active_variance * 0.0

    beta_penalty = (active_beta - float(M03R_ACTIVE_BETA_TARGET)).square()
    lower_limits = factors.new_tensor(config.risk_contract.exposure_lower_bounds)
    upper_limits = factors.new_tensor(config.risk_contract.exposure_upper_bounds)
    factor_penalty = (
        torch.relu(lower_limits - factors).square()
        + torch.relu(factors - upper_limits).square()
    ).mean()
    mean_turnover = discretionary.mean()
    turnover_penalty = torch.relu(
        mean_turnover - float(config.daily_one_way_turnover_target)
    ).square()

    loss = -mean_economic_log
    loss = (
        loss
        + _finite_nonnegative("lambda_turnover", config.lambda_turnover)
        * turnover_penalty
    )
    loss = (
        loss
        + _finite_nonnegative("lambda_early_exit", config.lambda_early_exit)
        * early.mean()
    )
    loss = (
        loss
        + _finite_nonnegative("lambda_forced_turnover", config.lambda_forced_turnover)
        * forced.mean()
    )
    if setting.annual_tracking_error_ceiling is not None:
        loss = (
            loss
            + _finite_nonnegative(
                "lambda_tracking_error_ceiling",
                config.lambda_tracking_error_ceiling,
            )
            * te_penalty
        )
    if setting.setting_id == M03R_FIXED_TE_FLOOR_SETTING_ID:
        loss = (
            loss
            + _finite_nonnegative(
                "lambda_tracking_error_floor", config.lambda_tracking_error_floor
            )
            * te_floor_penalty
        )
    if setting.active_beta_neutrality:
        loss = (
            loss
            + _finite_nonnegative("lambda_active_beta", config.lambda_active_beta)
            * beta_penalty
        )
    if setting.factor_sector_projection:
        loss = (
            loss
            + _finite_nonnegative(
                "lambda_factor_exposure", config.lambda_factor_exposure
            )
            * factor_penalty
        )
    if setting.residual_alpha_heads:
        assert auxiliary_loss is not None
        loss = loss + _finite_nonnegative(
            "lambda_auxiliary", config.lambda_auxiliary
        ) * auxiliary_loss.reshape(())
    if setting.setting_id == M03R_DIRECT_SHARPE_SETTING_ID:
        assert direct_sharpe_surrogate is not None
        loss = loss + _finite_nonnegative(
            "lambda_direct_sharpe", config.lambda_direct_sharpe
        ) * direct_sharpe_surrogate.loss_term.reshape(())
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise M03RObjectiveError("M03R objective must be one finite scalar")

    def scalar(value: torch.Tensor) -> float:
        return float(value.detach().to(dtype=torch.float64, device="cpu").item())

    metrics = M03RObjectiveMetrics(
        setting_id=setting.setting_id,
        observation_count=int(mask.sum().item()),
        mean_policy_net_log_return=scalar(mean_policy_log),
        mean_net_active_log_return=scalar(mean_active_log),
        annual_tracking_error=scalar(annual_te),
        active_market_beta=scalar(active_beta),
        mean_discretionary_one_way_turnover=scalar(mean_turnover),
        mean_early_exit_notional=scalar(early.mean()),
        mean_forced_one_way_turnover=scalar(forced.mean()),
        annual_tracking_error_penalty=scalar(te_penalty),
        annual_tracking_error_floor_penalty=scalar(te_floor_penalty),
        active_market_beta_penalty=scalar(beta_penalty),
        factor_exposure_penalty=scalar(factor_penalty),
        turnover_penalty=scalar(turnover_penalty),
        direct_sharpe_surrogate_loss=(
            scalar(direct_sharpe_surrogate.loss_term)
            if direct_sharpe_surrogate is not None
            else 0.0
        ),
    )
    return loss, metrics


__all__ = [
    "M03R_ACTIVE_BETA_ABSOLUTE_LIMIT",
    "M03R_ACTIVE_BETA_TARGET",
    "M03R_ANNUALIZATION_SESSIONS",
    "M03R_ANNUAL_TRACKING_ERROR_CEILING",
    "M03R_DAILY_ONE_WAY_TURNOVER_TARGET",
    "M03R_DIRECT_SHARPE_CONSTRUCTION_ID",
    "M03R_DIRECT_SHARPE_SETTING_ID",
    "M03R_FIXED_TE_FLOOR_SETTING_ID",
    "M03R_MAX_PREFERRED_TRACKING_ERROR",
    "M03R_TRAINING_ONE_WAY_COST_BASIS_POINTS",
    "M03RDirectSharpeSurrogate",
    "M03RObjectiveConfig",
    "M03RObjectiveError",
    "M03RObjectiveInputs",
    "M03RObjectiveMetrics",
    "confidence_scaled_preferred_tracking_error",
    "m03r_active_objective",
]
