"""Immutable contract for the Massive PIT adaptive-alpha generation.

This generation predicts a bucketed factor-residual return term structure.
Portfolio duration is an outcome of daily cost-aware re-optimization and is
never an input, target, loss term, reward, checkpoint criterion, or promotion
gate.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID = "massive-pit-adaptive-alpha-v1"
MASSIVE_ADAPTIVE_ALPHA_V1_DATASET_ID = "MassiveStocksDeveloperPITV1"
MASSIVE_ADAPTIVE_ALPHA_V1_SCHEMA = "rl-quant.massive-adaptive-alpha-protocol-v1"
MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256 = (
    "e1e3d12847f62b9cb842d81dfe0d2c1dbd2d40096c39fcad8d677318aca9d154"
)

FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS = frozenset(
    {
        "preferred_holding_sessions",
        "minimum_holding_sessions",
        "maximum_holding_sessions",
        "target_holding_sessions",
        "mandatory_exit_session",
        "holding_age",
        "age_bin",
        "age_bins",
        "age_distribution",
        "position_age",
        "position_age_summary",
        "sessions_since_entry",
        "persistence_coefficient",
        "persistence_bonus",
        "young_sale_penalty",
        "young_position_sale_penalty",
        "age_penalty",
        "early_exit_penalty",
        "fixed_exit_hazard",
        "age_conditioned_hazard",
        "scheduled_exit",
        "scheduled_exit_session",
        "duration_reward",
        "holding_period_reward",
        "position_age_reward",
        "duration_regularization",
        "duration_selection_metric",
        "holding_duration_promotion_gate",
        "holding_period_checkpoint_metric",
        "survival_constraint",
        "duration_conditioned_action_mask",
        "age_conditioned_release_rule",
        "cohort_age_constraint",
        "target_turnover_as_holding_proxy",
    }
)

FORBIDDEN_ADAPTIVE_IMPORT_PREFIXES = (
    "rl_quant.execution.age_aware_no_trade",
    "rl_quant.execution.hold30",
    "rl_quant.envs.hold30",
    "rl_quant.models.hold30",
    "rl_quant.training.hold30",
    "rl_quant.protocol.hold30",
)


class MassiveAdaptiveAlphaProtocolError(ValueError):
    """The immutable adaptive-alpha scientific contract drifted."""


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveAdaptiveAlphaProtocolError(
            f"{name} must be a canonical nonempty string"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveUniverseRule:
    """Protocol-layer view consumed later by the PIT universe materializer."""

    rule_id: str
    target_size: int
    ranking_metric: str
    ranking_lookback_sessions: int
    ranking_lag_sessions: int
    minimum_observed_sessions: int
    minimum_close_price: float
    minimum_average_dollar_volume: float
    eligible_security_types: tuple[str, ...]
    rebalance_frequency: str
    tie_breaker: str
    uses_future_survival: bool
    receipt_sha256: str
    schema: str = "rl-quant.pit-universe-rule-v1"

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rule_id": self.rule_id,
            "target_size": self.target_size,
            "ranking_metric": self.ranking_metric,
            "ranking_lookback_sessions": self.ranking_lookback_sessions,
            "ranking_lag_sessions": self.ranking_lag_sessions,
            "minimum_observed_sessions": self.minimum_observed_sessions,
            "minimum_close_price": self.minimum_close_price,
            "minimum_average_dollar_volume": self.minimum_average_dollar_volume,
            "eligible_security_types": list(self.eligible_security_types),
            "rebalance_frequency": self.rebalance_frequency,
            "tie_breaker": self.tie_breaker,
            "uses_future_survival": self.uses_future_survival,
        }

    def validate(self) -> None:
        _canonical_text("universe rule ID", self.rule_id)
        if self.schema != "rl-quant.pit-universe-rule-v1":
            raise MassiveAdaptiveAlphaProtocolError("universe rule schema drifted")
        _positive_int("target universe size", self.target_size)
        if self.ranking_metric != "trailing-mean-dollar-volume":
            raise MassiveAdaptiveAlphaProtocolError("universe ranking metric drifted")
        _positive_int("universe ranking lookback", self.ranking_lookback_sessions)
        _positive_int("universe ranking lag", self.ranking_lag_sessions)
        _positive_int("minimum universe observations", self.minimum_observed_sessions)
        if self.minimum_observed_sessions > self.ranking_lookback_sessions:
            raise MassiveAdaptiveAlphaProtocolError(
                "minimum observations exceed the ranking lookback"
            )
        if self.minimum_close_price < 0 or self.minimum_average_dollar_volume < 0:
            raise MassiveAdaptiveAlphaProtocolError("universe thresholds are negative")
        if self.eligible_security_types != ("common-stock",):
            raise MassiveAdaptiveAlphaProtocolError("security type inventory drifted")
        if self.rebalance_frequency != "monthly":
            raise MassiveAdaptiveAlphaProtocolError("rebalance frequency drifted")
        if self.tie_breaker != "security-id-ascending":
            raise MassiveAdaptiveAlphaProtocolError("universe tie breaker drifted")
        if self.uses_future_survival:
            raise MassiveAdaptiveAlphaProtocolError(
                "future survival cannot enter universe membership"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveAdaptiveAlphaProtocolError("universe rule receipt differs")


def build_massive_adaptive_universe_rule(
    *,
    rule_id: str,
    target_size: int,
    minimum_close_price: float,
    minimum_average_dollar_volume: float,
) -> MassiveAdaptiveUniverseRule:
    body = {
        "schema": "rl-quant.pit-universe-rule-v1",
        "rule_id": rule_id,
        "target_size": target_size,
        "ranking_metric": "trailing-mean-dollar-volume",
        "ranking_lookback_sessions": 63,
        "ranking_lag_sessions": 1,
        "minimum_observed_sessions": 50,
        "minimum_close_price": float(minimum_close_price),
        "minimum_average_dollar_volume": float(minimum_average_dollar_volume),
        "eligible_security_types": ["common-stock"],
        "rebalance_frequency": "monthly",
        "tie_breaker": "security-id-ascending",
        "uses_future_survival": False,
    }
    rule = MassiveAdaptiveUniverseRule(
        rule_id=rule_id,
        target_size=target_size,
        ranking_metric="trailing-mean-dollar-volume",
        ranking_lookback_sessions=63,
        ranking_lag_sessions=1,
        minimum_observed_sessions=50,
        minimum_close_price=float(minimum_close_price),
        minimum_average_dollar_volume=float(minimum_average_dollar_volume),
        eligible_security_types=("common-stock",),
        rebalance_frequency="monthly",
        tie_breaker="security-id-ascending",
        uses_future_survival=False,
        receipt_sha256=semantic_sha256(body),
    )
    rule.validate()
    return rule


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveAdaptiveAlphaProtocolError(f"{name} must be positive")
    return value


def assert_no_adaptive_hold_semantics(value: object, *, path: str = "root") -> None:
    """Reject duration-prior fields from adaptive configuration objects."""

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            if field.name in FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS:
                raise MassiveAdaptiveAlphaProtocolError(
                    f"forbidden adaptive duration field at {path}.{field.name}"
                )
            assert_no_adaptive_hold_semantics(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key in FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS:
                raise MassiveAdaptiveAlphaProtocolError(
                    f"forbidden adaptive duration field at {path}.{key}"
                )
            assert_no_adaptive_hold_semantics(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_no_adaptive_hold_semantics(child, path=f"{path}[{index}]")


def assert_adaptive_import_firewall(paths: Sequence[str | Path]) -> None:
    """Reject imports of historical Hold-30 and age-aware implementations."""

    for raw_path in paths:
        path = Path(raw_path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module in imported:
                if any(
                    module == prefix
                    or module.startswith(prefix + ".")
                    or module.startswith(prefix + "_")
                    for prefix in FORBIDDEN_ADAPTIVE_IMPORT_PREFIXES
                ):
                    raise MassiveAdaptiveAlphaProtocolError(
                        f"adaptive module {path} imports forbidden dependency {module}"
                    )


@dataclass(frozen=True, slots=True)
class AdaptiveAlphaReturnBucket:
    bucket_id: str
    start_offset_sessions: int
    end_offset_sessions: int

    def validate(self) -> None:
        _canonical_text("bucket ID", self.bucket_id)
        if (
            isinstance(self.start_offset_sessions, bool)
            or not isinstance(self.start_offset_sessions, int)
            or self.start_offset_sessions < 0
        ):
            raise MassiveAdaptiveAlphaProtocolError(
                "bucket start must be a nonnegative session offset"
            )
        _positive_int("bucket end", self.end_offset_sessions)
        if self.start_offset_sessions >= self.end_offset_sessions:
            raise MassiveAdaptiveAlphaProtocolError("return bucket is empty")


MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS = (
    AdaptiveAlphaReturnBucket("B01", 0, 1),
    AdaptiveAlphaReturnBucket("B02_05", 1, 5),
    AdaptiveAlphaReturnBucket("B06_10", 5, 10),
    AdaptiveAlphaReturnBucket("B11_21", 10, 21),
    AdaptiveAlphaReturnBucket("B22_42", 21, 42),
    AdaptiveAlphaReturnBucket("B43_63", 42, 63),
    AdaptiveAlphaReturnBucket("B64_126", 63, 126),
)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaSetting:
    setting_index: int
    setting_id: str
    cumulative_change: str
    promotion_eligible: bool

    def validate(self) -> None:
        if (
            isinstance(self.setting_index, bool)
            or not isinstance(self.setting_index, int)
            or self.setting_index < 0
        ):
            raise MassiveAdaptiveAlphaProtocolError("setting index is invalid")
        expected_id = f"AD{self.setting_index:02d}"
        if self.setting_id != expected_id:
            raise MassiveAdaptiveAlphaProtocolError("setting identity drifted")
        _canonical_text("setting change", self.cumulative_change)
        if not isinstance(self.promotion_eligible, bool):
            raise MassiveAdaptiveAlphaProtocolError(
                "promotion eligibility must be Boolean"
            )
        if self.promotion_eligible != (self.setting_id == "AD11"):
            raise MassiveAdaptiveAlphaProtocolError(
                "only the canonical AD11 row may be promotion eligible"
            )


_SETTING_CHANGES = (
    "daily-features-single-21-session-mean-target",
    "seven-bucket-factor-residual-term-structure",
    "cross-sectional-rank-loss",
    "quantile-and-predictive-scale-heads",
    "thirty-two-market-latents",
    "intraday-path-expert",
    "basic-quote-free-tape-flow-expert",
    "full-tape-flow-expert",
    "source-restricted-expert-fusion",
    "bucket-specific-regime-router",
    "causal-modality-pretraining",
    "five-hundred-four-session-long-context",
)

MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS = tuple(
    MassiveAdaptiveAlphaSetting(index, f"AD{index:02d}", change, index == 11)
    for index, change in enumerate(_SETTING_CHANGES)
)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveAlphaV1Protocol:
    protocol_id: str
    dataset_id: str
    scientific_objective: str
    decision_rule: str
    fill_rule: str
    portfolio_controller: str
    context_universe_rule: MassiveAdaptiveUniverseRule
    action_universe_rule: MassiveAdaptiveUniverseRule
    return_buckets: tuple[AdaptiveAlphaReturnBucket, ...]
    settings: tuple[MassiveAdaptiveAlphaSetting, ...]
    cost_ladder_basis_points: tuple[int, ...]
    canonical_cost_basis_points: int
    maximum_security_weight: float
    maximum_issuer_weight: float
    tracking_error_limit_annualized: float
    absolute_active_beta_limit: float
    maximum_daily_one_way_turnover: float
    maximum_adv_participation: float
    position_age_input_authorized: bool
    duration_objective_authorized: bool
    fixed_exit_authorized: bool
    economic_optimization_authorized: bool
    reinforcement_learning_authorized: bool
    historical_lockbox_access_authorized: bool
    prospective_access_authorized: bool
    schema: str = MASSIVE_ADAPTIVE_ALPHA_V1_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_ADAPTIVE_ALPHA_V1_SCHEMA:
            raise MassiveAdaptiveAlphaProtocolError("adaptive protocol schema drifted")
        if self.protocol_id != MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID:
            raise MassiveAdaptiveAlphaProtocolError("adaptive protocol ID drifted")
        if self.dataset_id != MASSIVE_ADAPTIVE_ALPHA_V1_DATASET_ID:
            raise MassiveAdaptiveAlphaProtocolError("adaptive dataset ID drifted")
        for name in (
            "scientific_objective",
            "decision_rule",
            "fill_rule",
            "portfolio_controller",
        ):
            _canonical_text(name, getattr(self, name))
        self.context_universe_rule.validate()
        self.action_universe_rule.validate()
        if (
            self.context_universe_rule.target_size != 1_500
            or self.action_universe_rule.target_size != 500
            or self.context_universe_rule.minimum_close_price != 1.0
            or self.action_universe_rule.minimum_close_price != 3.0
            or self.context_universe_rule.minimum_average_dollar_volume != 500_000.0
            or self.action_universe_rule.minimum_average_dollar_volume != 5_000_000.0
        ):
            raise MassiveAdaptiveAlphaProtocolError("dual PIT universe rules drifted")
        if self.return_buckets != MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS:
            raise MassiveAdaptiveAlphaProtocolError("return term structure drifted")
        previous_end = 0
        for bucket in self.return_buckets:
            bucket.validate()
            if bucket.start_offset_sessions != previous_end:
                raise MassiveAdaptiveAlphaProtocolError(
                    "return buckets must be contiguous and non-overlapping"
                )
            previous_end = bucket.end_offset_sessions
        if self.settings != MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS:
            raise MassiveAdaptiveAlphaProtocolError("setting inventory drifted")
        for setting in self.settings:
            setting.validate()
        if self.cost_ladder_basis_points != (10, 20, 40):
            raise MassiveAdaptiveAlphaProtocolError("cost ladder drifted")
        if self.canonical_cost_basis_points != 20:
            raise MassiveAdaptiveAlphaProtocolError("canonical cost rung drifted")
        expected_limits = {
            "maximum_security_weight": 0.01,
            "maximum_issuer_weight": 0.015,
            "tracking_error_limit_annualized": 0.06,
            "absolute_active_beta_limit": 0.10,
            "maximum_daily_one_way_turnover": 0.10,
            "maximum_adv_participation": 0.02,
        }
        for name, expected in expected_limits.items():
            if getattr(self, name) != expected:
                raise MassiveAdaptiveAlphaProtocolError(f"{name} drifted")
        if any(
            (
                self.position_age_input_authorized,
                self.duration_objective_authorized,
                self.fixed_exit_authorized,
                self.economic_optimization_authorized,
                self.reinforcement_learning_authorized,
                self.historical_lockbox_access_authorized,
                self.prospective_access_authorized,
            )
        ):
            raise MassiveAdaptiveAlphaProtocolError(
                "the source protocol cannot authorize duration priors or downstream stages"
            )
        assert_no_adaptive_hold_semantics(self)

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())


def build_massive_adaptive_alpha_v1_protocol() -> MassiveAdaptiveAlphaV1Protocol:
    """Build and validate the one canonical adaptive-alpha V1 protocol."""

    protocol = MassiveAdaptiveAlphaV1Protocol(
        protocol_id=MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID,
        dataset_id=MASSIVE_ADAPTIVE_ALPHA_V1_DATASET_ID,
        scientific_objective=(
            "conditional-term-structure-of-factor-residual-stock-returns"
        ),
        decision_rule="once-daily-close-plus-60-minutes",
        fill_rule="next-session-09:35-09:45-et-qualifying-trade-vwap",
        portfolio_controller="daily-cost-aware-receding-horizon-optimization",
        context_universe_rule=build_massive_adaptive_universe_rule(
            rule_id="massive-pit1500-monthly-dollar-volume-v1",
            target_size=1_500,
            minimum_close_price=1.0,
            minimum_average_dollar_volume=500_000.0,
        ),
        action_universe_rule=build_massive_adaptive_universe_rule(
            rule_id="massive-pit500-monthly-dollar-volume-v1",
            target_size=500,
            minimum_close_price=3.0,
            minimum_average_dollar_volume=5_000_000.0,
        ),
        return_buckets=MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
        settings=MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS,
        cost_ladder_basis_points=(10, 20, 40),
        canonical_cost_basis_points=20,
        maximum_security_weight=0.01,
        maximum_issuer_weight=0.015,
        tracking_error_limit_annualized=0.06,
        absolute_active_beta_limit=0.10,
        maximum_daily_one_way_turnover=0.10,
        maximum_adv_participation=0.02,
        position_age_input_authorized=False,
        duration_objective_authorized=False,
        fixed_exit_authorized=False,
        economic_optimization_authorized=False,
        reinforcement_learning_authorized=False,
        historical_lockbox_access_authorized=False,
        prospective_access_authorized=False,
    )
    protocol.validate()
    if protocol.receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256:
        raise MassiveAdaptiveAlphaProtocolError(
            "adaptive protocol receipt differs from its frozen identity"
        )
    return protocol


MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL = build_massive_adaptive_alpha_v1_protocol()


__all__ = [
    "FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS",
    "FORBIDDEN_ADAPTIVE_IMPORT_PREFIXES",
    "MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS",
    "MASSIVE_ADAPTIVE_ALPHA_V1_DATASET_ID",
    "MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL",
    "MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL_ID",
    "MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256",
    "MASSIVE_ADAPTIVE_ALPHA_V1_SETTINGS",
    "AdaptiveAlphaReturnBucket",
    "MassiveAdaptiveAlphaProtocolError",
    "MassiveAdaptiveAlphaSetting",
    "MassiveAdaptiveAlphaV1Protocol",
    "MassiveAdaptiveUniverseRule",
    "assert_adaptive_import_firewall",
    "assert_no_adaptive_hold_semantics",
    "build_massive_adaptive_universe_rule",
    "build_massive_adaptive_alpha_v1_protocol",
]
