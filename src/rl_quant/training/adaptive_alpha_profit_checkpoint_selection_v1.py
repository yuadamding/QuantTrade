"""Profit-aligned checkpoint selection for adaptive-alpha engineering.

The selector consumes complete inner-validation economic traces and computes
its own selection metrics.  Rank IC and calibration remain eligibility
diagnostics; the primary ranking variable is executable annualized dollar net
profit at the declared capital.  No outer-test or lockbox observation may
enter this boundary.

This is intentionally nonauthorizing.  A future adaptive-profit protocol must
bind the source-qualified simulator, compiler, factor model, and selection
configuration before historical evidence can use this implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import math
import string
from typing import Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_PROFIT_VALIDATION_TRACE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-profit-validation-trace-v1"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_METRICS_V1_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-metrics-v1"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-selection-v1"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_RULE = (
    "eligible-then-annualized-dollar-net-profit-active-log-wealth-factor-alpha-"
    "worst-block-drawdown-turnover-capacity-complexity-earliest-epoch-v1"
)

_BUCKET_IDS = tuple(row.bucket_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
_SETTING_IDS = frozenset(row.setting_id for row in MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS)
_HEX = frozenset(string.hexdigits.lower())


class MassiveAdaptiveProfitCheckpointSelectionError(ValueError):
    """Inner-validation evidence or its deterministic selection drifted."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MassiveAdaptiveProfitCheckpointSelectionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MassiveAdaptiveProfitCheckpointSelectionError(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: object) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must be nonnegative"
        )
    return result


def _positive(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise MassiveAdaptiveProfitCheckpointSelectionError(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must be a positive integer"
        )
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must be a canonical nonempty string"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _finite_tuple(name: str, values: Sequence[object], *, length: int) -> tuple[float, ...]:
    if len(values) != length:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} must contain {length} rows"
        )
    return tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))


def _canonical_contributions(
    name: str,
    rows: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    if not rows:
        raise MassiveAdaptiveProfitCheckpointSelectionError(f"{name} is empty")
    labels = tuple(row[0] for row in rows)
    if labels != tuple(sorted(set(labels))):
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"{name} labels must be unique and sorted"
        )
    return tuple(
        (_canonical_text(f"{name} label", label), _finite(f"{name}[{label}]", value))
        for label, value in rows
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointSelectionConfigV1:
    """Engineering thresholds and deterministic economic ranking order."""

    primary_capital: float = 10_000_000.0
    minimum_validation_sessions: int = 63
    minimum_eligible_fraction: float = 0.80
    maximum_calibration_error: float = 0.05
    required_positive_rank_ic_bucket_ids: tuple[str, ...] = _BUCKET_IDS
    maximum_constraint_violation: float = 1.0e-8
    maximum_capacity_lost_notional_fraction: float = 0.05
    maximum_date_gross_profit_share: float = 0.10
    maximum_year_gross_profit_share: float = 0.50
    maximum_sector_gross_profit_share: float = 0.35
    maximum_security_gross_profit_share: float = 0.10
    annualization_sessions: int = 252
    schema: str = "rl-quant.massive-adaptive-profit-checkpoint-selection-config-v1"

    def validate(self) -> None:
        if self.schema != "rl-quant.massive-adaptive-profit-checkpoint-selection-config-v1":
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection config schema drifted"
            )
        _positive("primary_capital", self.primary_capital)
        _positive_int("minimum_validation_sessions", self.minimum_validation_sessions)
        _positive_int("annualization_sessions", self.annualization_sessions)
        for name in (
            "minimum_eligible_fraction",
            "maximum_calibration_error",
            "maximum_constraint_violation",
            "maximum_capacity_lost_notional_fraction",
            "maximum_date_gross_profit_share",
            "maximum_year_gross_profit_share",
            "maximum_sector_gross_profit_share",
            "maximum_security_gross_profit_share",
        ):
            value = _nonnegative(name, getattr(self, name))
            if name != "maximum_constraint_violation" and value > 1.0:
                raise MassiveAdaptiveProfitCheckpointSelectionError(f"{name} exceeds one")
        if not 0.0 < self.minimum_eligible_fraction <= 1.0:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "minimum eligible fraction must lie in (0, 1]"
            )
        required = self.required_positive_rank_ic_bucket_ids
        if not required or len(set(required)) != len(required):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "required rank-IC bucket inventory is empty or duplicated"
            )
        if any(bucket not in _BUCKET_IDS for bucket in required):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "required rank-IC bucket inventory is unknown"
            )
        assert_no_adaptive_hold_semantics(self)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitValidationTraceV1:
    """Complete daily inner-validation evidence for one epoch checkpoint."""

    setting_id: str
    fold_index: int
    epoch_index: int
    model_parameter_count: int
    checkpoint_state_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    prediction_receipt_sha256: str
    source_bundle_receipt_sha256: str
    validation_plan_receipt_sha256: str
    compiler_config_receipt_sha256: str
    factor_model_receipt_sha256: str
    attribution_receipt_sha256: str
    session_dates: tuple[str, ...]
    validation_block_ids: tuple[int, ...]
    compiler_decision_receipts: tuple[str, ...]
    portfolio_net_returns_20bp: tuple[float, ...]
    portfolio_net_returns_40bp: tuple[float, ...]
    benchmark_net_returns_20bp: tuple[float, ...]
    factor_residual_returns_20bp: tuple[float, ...]
    gross_signal_returns: tuple[float, ...]
    eligible_fractions: tuple[float, ...]
    constraint_violations: tuple[float, ...]
    capacity_lost_notional_fractions: tuple[float, ...]
    one_way_turnovers: tuple[float, ...]
    rank_ic_by_bucket: tuple[float, ...]
    calibration_error: float
    sector_gross_return_contributions: tuple[tuple[str, float], ...]
    security_gross_return_contributions: tuple[tuple[str, float], ...]
    semantic_receipt_sha256: str
    split_role: str = "inner_validation"
    outer_test_accessed: bool = False
    lockbox_accessed: bool = False
    profitability_reporting_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_PROFIT_VALIDATION_TRACE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_PROFIT_VALIDATION_TRACE_V1_SCHEMA:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation trace schema drifted"
            )
        if self.setting_id not in _SETTING_IDS:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation trace setting is unknown"
            )
        _nonnegative_int("fold_index", self.fold_index)
        _nonnegative_int("epoch_index", self.epoch_index)
        _positive_int("model_parameter_count", self.model_parameter_count)
        for name in (
            "checkpoint_state_receipt_sha256",
            "checkpoint_source_receipt_sha256",
            "prediction_receipt_sha256",
            "source_bundle_receipt_sha256",
            "validation_plan_receipt_sha256",
            "compiler_config_receipt_sha256",
            "factor_model_receipt_sha256",
            "attribution_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        count = len(self.session_dates)
        if count == 0:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation trace contains no sessions"
            )
        parsed_dates: list[date] = []
        for raw in self.session_dates:
            _canonical_text("session date", raw)
            try:
                parsed = date.fromisoformat(raw)
            except ValueError as exc:
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    "session date is not ISO-8601"
                ) from exc
            if parsed.isoformat() != raw:
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    "session date is not canonical"
                )
            parsed_dates.append(parsed)
        if parsed_dates != sorted(set(parsed_dates)):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation session dates must be unique and increasing"
            )
        if len(self.validation_block_ids) != count or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.validation_block_ids
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation block inventory is invalid"
            )
        if tuple(sorted(self.validation_block_ids)) != self.validation_block_ids:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation block IDs must be nondecreasing"
            )
        unique_blocks = tuple(dict.fromkeys(self.validation_block_ids))
        if unique_blocks != tuple(range(len(unique_blocks))):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation block IDs must be contiguous from zero"
            )
        if (
            len(self.compiler_decision_receipts) != count
            or len(set(self.compiler_decision_receipts)) != count
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "compiler decision receipt inventory is incomplete or duplicated"
            )
        for value in self.compiler_decision_receipts:
            _digest("compiler decision receipt", value)
        for name in (
            "portfolio_net_returns_20bp",
            "portfolio_net_returns_40bp",
            "benchmark_net_returns_20bp",
            "factor_residual_returns_20bp",
            "gross_signal_returns",
            "eligible_fractions",
            "constraint_violations",
            "capacity_lost_notional_fractions",
            "one_way_turnovers",
        ):
            values = _finite_tuple(name, getattr(self, name), length=count)
            if name in {
                "portfolio_net_returns_20bp",
                "portfolio_net_returns_40bp",
                "benchmark_net_returns_20bp",
                "factor_residual_returns_20bp",
                "gross_signal_returns",
            } and any(value <= -1.0 for value in values):
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    f"{name} contains a return at or below total loss"
                )
            if name in {
                "eligible_fractions",
                "capacity_lost_notional_fractions",
                "one_way_turnovers",
            } and any(not 0.0 <= value <= 1.0 for value in values):
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    f"{name} must lie in [0, 1]"
                )
            if name == "constraint_violations" and any(value < 0.0 for value in values):
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    "constraint violations must be nonnegative"
                )
        if any(
            stressed > primary + 1.0e-12
            for primary, stressed in zip(
                self.portfolio_net_returns_20bp,
                self.portfolio_net_returns_40bp,
                strict=True,
            )
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "40-bp returns must not exceed frozen-action 20-bp returns"
            )
        if any(
            net > gross + 1.0e-12
            for net, gross in zip(
                self.portfolio_net_returns_20bp,
                self.gross_signal_returns,
                strict=True,
            )
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "20-bp net return exceeds gross signal return"
            )
        _finite_tuple("rank_ic_by_bucket", self.rank_ic_by_bucket, length=len(_BUCKET_IDS))
        _nonnegative("calibration_error", self.calibration_error)
        sector = _canonical_contributions(
            "sector_gross_return_contributions",
            self.sector_gross_return_contributions,
        )
        security = _canonical_contributions(
            "security_gross_return_contributions",
            self.security_gross_return_contributions,
        )
        gross_total = math.fsum(self.gross_signal_returns)
        for name, rows in (
            ("sector", sector),
            ("security", security),
        ):
            if not math.isclose(
                math.fsum(value for _, value in rows),
                gross_total,
                abs_tol=1.0e-12,
                rel_tol=1.0e-12,
            ):
                raise MassiveAdaptiveProfitCheckpointSelectionError(
                    f"{name} gross contribution does not reconcile"
                )
        if (
            self.split_role != "inner_validation"
            or self.outer_test_accessed
            or self.lockbox_accessed
            or self.profitability_reporting_authorized
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection accepts untouched inner-validation evidence only"
            )
        if self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation trace does not bind the adaptive-alpha protocol"
            )
        MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.validate()
        if MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.receipt_sha256 != self.protocol_receipt_sha256:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "adaptive-alpha protocol root drifted"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "validation trace receipt differs"
            )
        assert_no_adaptive_hold_semantics(self)


def build_massive_adaptive_profit_validation_trace_v1(
    **values: object,
) -> MassiveAdaptiveProfitValidationTraceV1:
    """Build one self-verifying validation trace from explicit typed values."""

    provisional = MassiveAdaptiveProfitValidationTraceV1(
        **values, semantic_receipt_sha256="0" * 64  # type: ignore[arg-type]
    )
    trace = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    trace.validate()
    return trace


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointMetricsV1:
    """Metrics recomputed from one complete validation trace."""

    validation_trace_receipt_sha256: str
    annualized_dollar_net_profit_20bp: float
    terminal_net_return_20bp: float
    terminal_net_return_40bp: float
    net_active_log_wealth_20bp: float
    annualized_factor_residual_return_20bp: float
    worst_block_net_return_20bp: float
    maximum_drawdown_20bp: float
    mean_one_way_turnover: float
    mean_capacity_lost_notional_fraction: float
    minimum_eligible_fraction: float
    maximum_constraint_violation: float
    maximum_date_gross_profit_share: float
    maximum_year_gross_profit_share: float
    maximum_sector_gross_profit_share: float
    maximum_security_gross_profit_share: float
    rank_ic_by_bucket: tuple[float, ...]
    calibration_error: float
    model_parameter_count: int
    eligibility_failures: tuple[str, ...]
    eligible: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_METRICS_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_METRICS_V1_SCHEMA:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint metric schema drifted"
            )
        _digest("validation_trace_receipt_sha256", self.validation_trace_receipt_sha256)
        for name in (
            "annualized_dollar_net_profit_20bp",
            "terminal_net_return_20bp",
            "terminal_net_return_40bp",
            "net_active_log_wealth_20bp",
            "annualized_factor_residual_return_20bp",
            "worst_block_net_return_20bp",
        ):
            _finite(name, getattr(self, name))
        for name in (
            "maximum_drawdown_20bp",
            "mean_one_way_turnover",
            "mean_capacity_lost_notional_fraction",
            "minimum_eligible_fraction",
            "maximum_constraint_violation",
            "maximum_date_gross_profit_share",
            "maximum_year_gross_profit_share",
            "maximum_sector_gross_profit_share",
            "maximum_security_gross_profit_share",
            "calibration_error",
        ):
            _nonnegative(name, getattr(self, name))
        _finite_tuple("rank_ic_by_bucket", self.rank_ic_by_bucket, length=len(_BUCKET_IDS))
        _positive_int("model_parameter_count", self.model_parameter_count)
        if self.eligible != (not self.eligibility_failures):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint eligibility does not match its failure inventory"
            )
        if self.profitability_reporting_authorized or self.lockbox_access_authorized:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "training-only metrics cannot authorize evaluation"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint metric receipt differs"
            )
        assert_no_adaptive_hold_semantics(self)

    @property
    def selection_key(self) -> tuple[float, ...]:
        self.validate()
        return (
            self.annualized_dollar_net_profit_20bp,
            self.net_active_log_wealth_20bp,
            self.annualized_factor_residual_return_20bp,
            self.worst_block_net_return_20bp,
            -self.maximum_drawdown_20bp,
            -self.mean_one_way_turnover,
            -self.mean_capacity_lost_notional_fraction,
            -float(self.model_parameter_count),
        )


def _compound(returns: Sequence[float]) -> float:
    return math.exp(math.fsum(math.log1p(value) for value in returns)) - 1.0


def _maximum_drawdown(returns: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum = max(maximum, 1.0 - wealth / peak)
    return maximum


def _maximum_positive_share(values: Sequence[float]) -> float:
    positive = tuple(max(float(value), 0.0) for value in values)
    total = math.fsum(positive)
    if total <= 0.0:
        return 1.0
    return max(positive, default=0.0) / total


def evaluate_massive_adaptive_profit_checkpoint_v1(
    trace: MassiveAdaptiveProfitValidationTraceV1,
    *,
    config: MassiveAdaptiveProfitCheckpointSelectionConfigV1,
) -> MassiveAdaptiveProfitCheckpointMetricsV1:
    """Recompute economic metrics and the ordered eligibility ladder."""

    trace.validate()
    config.validate()
    count = len(trace.session_dates)
    policy20 = trace.portfolio_net_returns_20bp
    policy40 = trace.portfolio_net_returns_40bp
    benchmark = trace.benchmark_net_returns_20bp
    terminal20 = _compound(policy20)
    terminal40 = _compound(policy40)
    annualized_return = math.exp(
        math.log1p(terminal20) * config.annualization_sessions / count
    ) - 1.0
    annualized_dollar = config.primary_capital * annualized_return
    active_log = math.fsum(
        math.log1p(policy) - math.log1p(reference)
        for policy, reference in zip(policy20, benchmark, strict=True)
    )
    factor_alpha = config.annualization_sessions * math.fsum(
        trace.factor_residual_returns_20bp
    ) / count
    block_returns = tuple(
        _compound(
            tuple(
                value
                for value, candidate_block in zip(
                    policy20, trace.validation_block_ids, strict=True
                )
                if candidate_block == block
            )
        )
        for block in range(max(trace.validation_block_ids) + 1)
    )
    years: dict[int, list[float]] = {}
    for raw_date, gross_return in zip(
        trace.session_dates, trace.gross_signal_returns, strict=True
    ):
        years.setdefault(date.fromisoformat(raw_date).year, []).append(gross_return)
    year_contributions = tuple(math.fsum(years[year]) for year in sorted(years))
    rank_by_id = dict(zip(_BUCKET_IDS, trace.rank_ic_by_bucket, strict=True))
    max_constraint = max(trace.constraint_violations)
    max_capacity_loss = max(trace.capacity_lost_notional_fractions)
    date_share = _maximum_positive_share(trace.gross_signal_returns)
    year_share = _maximum_positive_share(year_contributions)
    sector_share = _maximum_positive_share(
        tuple(value for _, value in trace.sector_gross_return_contributions)
    )
    security_share = _maximum_positive_share(
        tuple(value for _, value in trace.security_gross_return_contributions)
    )

    failures: list[str] = []
    if count < config.minimum_validation_sessions or min(
        trace.eligible_fractions
    ) < config.minimum_eligible_fraction:
        failures.append("coverage")
    if trace.calibration_error > config.maximum_calibration_error:
        failures.append("calibration")
    for bucket_id in config.required_positive_rank_ic_bucket_ids:
        if rank_by_id[bucket_id] <= 0.0:
            failures.append(f"rank-ic:{bucket_id}")
    if terminal20 <= 0.0:
        failures.append("net-return-20bp")
    if terminal40 < 0.0:
        failures.append("net-return-40bp")
    if factor_alpha <= 0.0:
        failures.append("factor-residual-alpha")
    if max_constraint > config.maximum_constraint_violation:
        failures.append("constraint-violation")
    if max_capacity_loss > config.maximum_capacity_lost_notional_fraction:
        failures.append("capacity-loss")
    for name, value, limit in (
        ("date", date_share, config.maximum_date_gross_profit_share),
        ("year", year_share, config.maximum_year_gross_profit_share),
        ("sector", sector_share, config.maximum_sector_gross_profit_share),
        ("security", security_share, config.maximum_security_gross_profit_share),
    ):
        if value > limit:
            failures.append(f"profit-concentration:{name}")

    body = {
        "validation_trace_receipt_sha256": trace.semantic_receipt_sha256,
        "annualized_dollar_net_profit_20bp": annualized_dollar,
        "terminal_net_return_20bp": terminal20,
        "terminal_net_return_40bp": terminal40,
        "net_active_log_wealth_20bp": active_log,
        "annualized_factor_residual_return_20bp": factor_alpha,
        "worst_block_net_return_20bp": min(block_returns),
        "maximum_drawdown_20bp": _maximum_drawdown(policy20),
        "mean_one_way_turnover": math.fsum(trace.one_way_turnovers) / count,
        "mean_capacity_lost_notional_fraction": math.fsum(
            trace.capacity_lost_notional_fractions
        )
        / count,
        "minimum_eligible_fraction": min(trace.eligible_fractions),
        "maximum_constraint_violation": max_constraint,
        "maximum_date_gross_profit_share": date_share,
        "maximum_year_gross_profit_share": year_share,
        "maximum_sector_gross_profit_share": sector_share,
        "maximum_security_gross_profit_share": security_share,
        "rank_ic_by_bucket": trace.rank_ic_by_bucket,
        "calibration_error": trace.calibration_error,
        "model_parameter_count": trace.model_parameter_count,
        "eligibility_failures": tuple(failures),
        "eligible": not failures,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_METRICS_V1_SCHEMA,
    }
    result = MassiveAdaptiveProfitCheckpointMetricsV1(
        **body,
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointSelectionV1:
    """Deterministic selection result over one fold-setting candidate family."""

    setting_id: str
    fold_index: int
    selected_epoch_index: int
    selected_checkpoint_state_receipt_sha256: str
    selected_checkpoint_source_receipt_sha256: str
    selected_validation_trace_receipt_sha256: str
    selected_metrics_receipt_sha256: str
    candidate_epoch_indices: tuple[int, ...]
    candidate_checkpoint_state_receipts: tuple[str, ...]
    candidate_checkpoint_source_receipts: tuple[str, ...]
    candidate_validation_trace_receipts: tuple[str, ...]
    candidate_metrics_receipts: tuple[str, ...]
    eligible_epoch_indices: tuple[int, ...]
    config_receipt_sha256: str
    semantic_receipt_sha256: str
    selection_rule: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_RULE
    economic_training_authorized: bool = False
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_SCHEMA:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection schema drifted"
            )
        if self.setting_id not in _SETTING_IDS:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection setting is unknown"
            )
        _nonnegative_int("fold_index", self.fold_index)
        _nonnegative_int("selected_epoch_index", self.selected_epoch_index)
        for name in (
            "selected_checkpoint_state_receipt_sha256",
            "selected_checkpoint_source_receipt_sha256",
            "selected_validation_trace_receipt_sha256",
            "selected_metrics_receipt_sha256",
            "config_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            not self.candidate_validation_trace_receipts
            or len(self.candidate_epoch_indices)
            != len(self.candidate_validation_trace_receipts)
            or len(self.candidate_checkpoint_state_receipts)
            != len(self.candidate_validation_trace_receipts)
            or len(self.candidate_checkpoint_source_receipts)
            != len(self.candidate_validation_trace_receipts)
            or len(self.candidate_validation_trace_receipts)
            != len(self.candidate_metrics_receipts)
            or tuple(sorted(set(self.candidate_epoch_indices)))
            != self.candidate_epoch_indices
            or any(
                epoch not in self.candidate_epoch_indices
                for epoch in self.eligible_epoch_indices
            )
            or self.selected_epoch_index not in self.eligible_epoch_indices
            or tuple(sorted(set(self.eligible_epoch_indices)))
            != self.eligible_epoch_indices
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection inventory drifted"
            )
        selected_position = self.candidate_epoch_indices.index(self.selected_epoch_index)
        if (
            self.selected_checkpoint_state_receipt_sha256
            != self.candidate_checkpoint_state_receipts[selected_position]
            or self.selected_checkpoint_source_receipt_sha256
            != self.candidate_checkpoint_source_receipts[selected_position]
            or self.selected_validation_trace_receipt_sha256
            != self.candidate_validation_trace_receipts[selected_position]
            or self.selected_metrics_receipt_sha256
            != self.candidate_metrics_receipts[selected_position]
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "selected checkpoint does not reconcile at its epoch index"
            )
        for value in (
            *self.candidate_checkpoint_state_receipts,
            *self.candidate_checkpoint_source_receipts,
            *self.candidate_validation_trace_receipts,
            *self.candidate_metrics_receipts,
        ):
            _digest("candidate receipt", value)
        if self.selection_rule != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_RULE:
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection rule drifted"
            )
        if any(
            (
                self.economic_training_authorized,
                self.outer_evaluation_authorized,
                self.profitability_reporting_authorized,
                self.lockbox_access_authorized,
                self.reinforcement_learning_authorized,
            )
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "engineering checkpoint selection cannot authorize downstream stages"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint selection receipt differs"
            )
        assert_no_adaptive_hold_semantics(self)


def select_massive_adaptive_profit_checkpoint_v1(
    traces: Sequence[MassiveAdaptiveProfitValidationTraceV1],
    *,
    config: MassiveAdaptiveProfitCheckpointSelectionConfigV1,
) -> MassiveAdaptiveProfitCheckpointSelectionV1:
    """Select the highest-profit eligible checkpoint on common validation support."""

    config.validate()
    if not traces:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            "checkpoint selection requires candidates"
        )
    ordered = tuple(sorted(traces, key=lambda row: row.epoch_index))
    for trace in ordered:
        trace.validate()
    if len({trace.epoch_index for trace in ordered}) != len(ordered):
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            "checkpoint candidate epochs are duplicated"
        )
    if len({(trace.setting_id, trace.fold_index) for trace in ordered}) != 1:
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            "checkpoint candidates must share one setting and fold"
        )
    reference = ordered[0]
    common_fields = (
        "session_dates",
        "validation_block_ids",
        "benchmark_net_returns_20bp",
        "eligible_fractions",
        "source_bundle_receipt_sha256",
        "validation_plan_receipt_sha256",
        "compiler_config_receipt_sha256",
        "factor_model_receipt_sha256",
    )
    for trace in ordered[1:]:
        if any(getattr(trace, field) != getattr(reference, field) for field in common_fields):
            raise MassiveAdaptiveProfitCheckpointSelectionError(
                "checkpoint candidates do not share exact validation support"
            )
    metrics = tuple(
        evaluate_massive_adaptive_profit_checkpoint_v1(trace, config=config)
        for trace in ordered
    )
    eligible = tuple(
        (trace, metric)
        for trace, metric in zip(ordered, metrics, strict=True)
        if metric.eligible
    )
    if not eligible:
        failures = tuple(
            (trace.epoch_index, metric.eligibility_failures)
            for trace, metric in zip(ordered, metrics, strict=True)
        )
        raise MassiveAdaptiveProfitCheckpointSelectionError(
            f"no checkpoint passed the economic eligibility ladder: {failures}"
        )
    selected_trace, selected_metric = max(
        eligible,
        key=lambda row: (*row[1].selection_key, -float(row[0].epoch_index)),
    )
    body = {
        "setting_id": selected_trace.setting_id,
        "fold_index": selected_trace.fold_index,
        "selected_epoch_index": selected_trace.epoch_index,
        "selected_checkpoint_state_receipt_sha256": (
            selected_trace.checkpoint_state_receipt_sha256
        ),
        "selected_checkpoint_source_receipt_sha256": (
            selected_trace.checkpoint_source_receipt_sha256
        ),
        "selected_validation_trace_receipt_sha256": (
            selected_trace.semantic_receipt_sha256
        ),
        "selected_metrics_receipt_sha256": selected_metric.semantic_receipt_sha256,
        "candidate_epoch_indices": tuple(trace.epoch_index for trace in ordered),
        "candidate_checkpoint_state_receipts": tuple(
            trace.checkpoint_state_receipt_sha256 for trace in ordered
        ),
        "candidate_checkpoint_source_receipts": tuple(
            trace.checkpoint_source_receipt_sha256 for trace in ordered
        ),
        "candidate_validation_trace_receipts": tuple(
            trace.semantic_receipt_sha256 for trace in ordered
        ),
        "candidate_metrics_receipts": tuple(
            metric.semantic_receipt_sha256 for metric in metrics
        ),
        "eligible_epoch_indices": tuple(
            trace.epoch_index for trace, _ in eligible
        ),
        "config_receipt_sha256": config.receipt_sha256,
        "selection_rule": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_RULE,
        "economic_training_authorized": False,
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_SCHEMA,
    }
    selection = MassiveAdaptiveProfitCheckpointSelectionV1(
        **body,
        semantic_receipt_sha256=semantic_sha256(body),
    )
    selection.validate()
    return selection


__all__ = [
    "MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_METRICS_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_RULE",
    "MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_PROFIT_VALIDATION_TRACE_V1_SCHEMA",
    "MassiveAdaptiveProfitCheckpointMetricsV1",
    "MassiveAdaptiveProfitCheckpointSelectionConfigV1",
    "MassiveAdaptiveProfitCheckpointSelectionError",
    "MassiveAdaptiveProfitCheckpointSelectionV1",
    "MassiveAdaptiveProfitValidationTraceV1",
    "build_massive_adaptive_profit_validation_trace_v1",
    "evaluate_massive_adaptive_profit_checkpoint_v1",
    "select_massive_adaptive_profit_checkpoint_v1",
]
