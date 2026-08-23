"""Immutable finalized-file validation contract for Massive tape alpha.

This deliberately non-production generation asks whether finalized daily bar
and trade-tape summaries add repeatable cross-sectional alpha.  It does not
claim equivalence to the delayed close-plus-60-minute adaptive-alpha protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MassiveAdaptiveUniverseRule,
    assert_no_adaptive_hold_semantics,
    build_massive_adaptive_universe_rule,
)


MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID = "massive-finalized-alpha-validation-v0"
MASSIVE_FINALIZED_VALIDATION_V0_DATASET_ID = "MassiveFinalizedPIT500ValidationV0"
MASSIVE_FINALIZED_VALIDATION_V0_SCHEMA = (
    "rl-quant.massive-finalized-alpha-validation-protocol-v0"
)
MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256 = (
    "f1f84851304bd78c11130b04169e075f4f73cfa0cb6906c156d799497f582995"
)


class MassiveFinalizedValidationProtocolError(ValueError):
    """The finalized-validation V0 scientific contract drifted."""


def _canonical_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveFinalizedValidationProtocolError(
            f"{name} must be a canonical nonempty string"
        )
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MassiveFinalizedValidationProtocolError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class MassiveFinalizedValidationHorizon:
    horizon_id: str
    end_offset_sessions: int
    loss_weight: float

    def validate(self) -> None:
        _canonical_text("horizon ID", self.horizon_id)
        _positive_int("horizon session offset", self.end_offset_sessions)
        if self.horizon_id != f"H{self.end_offset_sessions:02d}":
            raise MassiveFinalizedValidationProtocolError(
                "horizon identity differs from its endpoint"
            )
        if self.loss_weight != 1.0:
            raise MassiveFinalizedValidationProtocolError(
                "all validation horizons must have equal scientific weight"
            )


MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS = (
    MassiveFinalizedValidationHorizon("H01", 1, 1.0),
    MassiveFinalizedValidationHorizon("H05", 5, 1.0),
    MassiveFinalizedValidationHorizon("H21", 21, 1.0),
    MassiveFinalizedValidationHorizon("H63", 63, 1.0),
)


@dataclass(frozen=True, slots=True)
class MassiveFinalizedValidationSetting:
    setting_id: str
    feature_set_id: str
    model_kind: str
    trainable: bool

    def validate(self) -> None:
        _canonical_text("setting ID", self.setting_id)
        _canonical_text("feature-set ID", self.feature_set_id)
        _canonical_text("model kind", self.model_kind)
        if not isinstance(self.trainable, bool):
            raise MassiveFinalizedValidationProtocolError(
                "setting trainability must be Boolean"
            )


MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS = (
    MassiveFinalizedValidationSetting(
        "MV00", "BARS_V0", "fixed-momentum-reversal-liquidity-composite", False
    ),
    MassiveFinalizedValidationSetting("MV01", "BARS_V0", "linear-multi-output", True),
    MassiveFinalizedValidationSetting("MV02", "BARS_V0", "two-layer-mlp", True),
    MassiveFinalizedValidationSetting("MV03", "TAPE_V0", "two-layer-mlp", True),
    MassiveFinalizedValidationSetting(
        "MV04", "BARS_PLUS_TAPE_V0", "two-layer-mlp", True
    ),
)


@dataclass(frozen=True, slots=True)
class MassiveFinalizedValidationV0Protocol:
    protocol_id: str
    dataset_id: str
    purpose: str
    production_equivalence: bool
    historical_delayed_stream_replay_required: bool
    universe_rule: MassiveAdaptiveUniverseRule
    context_universe_rule_receipt_sha256: str
    action_universe_rule_receipt_sha256: str
    source_availability_cutoff_local_time: str
    decision_local_time: str
    fill_start_local_time: str
    fill_end_local_time: str
    decision_rule: str
    input_cutoff_rule: str
    fill_rule: str
    horizons: tuple[MassiveFinalizedValidationHorizon, ...]
    horizons_equal_status: bool
    settings: tuple[MassiveFinalizedValidationSetting, ...]
    primary_contrast: tuple[str, str]
    cost_ladder_basis_points: tuple[int, ...]
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    outer_fold_count: int
    outer_fold_sessions: int
    historical_lockbox_sessions: int
    minimum_initial_training_sessions: int
    target_overlap_purge_sessions: int
    inner_validation_sessions: int
    inner_purge_sessions: int
    position_age_input_authorized: bool
    duration_objective_authorized: bool
    duration_checkpoint_selection_authorized: bool
    predictive_training_authorized: bool
    diagnostic_portfolio_evaluation_authorized: bool
    economic_optimization_authorized: bool
    historical_lockbox_access_authorized: bool
    prospective_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_FINALIZED_VALIDATION_V0_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_VALIDATION_V0_SCHEMA:
            raise MassiveFinalizedValidationProtocolError(
                "finalized-validation schema drifted"
            )
        if self.protocol_id != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID:
            raise MassiveFinalizedValidationProtocolError(
                "finalized-validation protocol ID drifted"
            )
        if self.dataset_id != MASSIVE_FINALIZED_VALIDATION_V0_DATASET_ID:
            raise MassiveFinalizedValidationProtocolError(
                "finalized-validation dataset ID drifted"
            )
        for name in (
            "purpose",
            "source_availability_cutoff_local_time",
            "decision_local_time",
            "fill_start_local_time",
            "fill_end_local_time",
            "decision_rule",
            "input_cutoff_rule",
            "fill_rule",
        ):
            _canonical_text(name, getattr(self, name))
        expected_text = {
            "purpose": "validate-incremental-finalized-trade-tape-cross-sectional-net-alpha",
            "decision_rule": (
                "first-eligible-session-after-vendor-finalized-input-availability"
            ),
            "input_cutoff_rule": "source-session-or-earlier-only",
            "fill_rule": "same-session-15:50-16:00-et-qualifying-trade-vwap",
        }
        for name, expected in expected_text.items():
            if getattr(self, name) != expected:
                raise MassiveFinalizedValidationProtocolError(f"{name} drifted")
        if self.production_equivalence is not False:
            raise MassiveFinalizedValidationProtocolError(
                "V0 cannot claim production equivalence"
            )
        if self.historical_delayed_stream_replay_required is not False:
            raise MassiveFinalizedValidationProtocolError(
                "V0 must not depend on historical delayed-stream replay"
            )
        self.universe_rule.validate()
        if (
            self.universe_rule.rule_id
            != "massive-finalized-pit500-validation-v0"
            or self.universe_rule.target_size != 500
            or self.universe_rule.ranking_lookback_sessions != 63
            or self.universe_rule.ranking_lag_sessions != 1
            or self.universe_rule.minimum_observed_sessions != 50
            or self.universe_rule.minimum_close_price != 3.0
            or self.universe_rule.minimum_average_dollar_volume != 5_000_000.0
            or self.universe_rule.rebalance_frequency != "monthly"
        ):
            raise MassiveFinalizedValidationProtocolError(
                "V0 PIT-500 universe rule drifted"
            )
        if (
            self.context_universe_rule_receipt_sha256
            != self.universe_rule.receipt_sha256
            or self.action_universe_rule_receipt_sha256
            != self.universe_rule.receipt_sha256
        ):
            raise MassiveFinalizedValidationProtocolError(
                "V0 context and action universes must be the same PIT-500 authority"
            )
        expected_times = {
            "source_availability_cutoff_local_time": "11:30:00",
            "decision_local_time": "12:30:00",
            "fill_start_local_time": "15:50:00",
            "fill_end_local_time": "16:00:00",
        }
        for name, expected in expected_times.items():
            if getattr(self, name) != expected:
                raise MassiveFinalizedValidationProtocolError(f"{name} drifted")
        if self.horizons != MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS:
            raise MassiveFinalizedValidationProtocolError(
                "V0 horizon inventory drifted"
            )
        for horizon in self.horizons:
            horizon.validate()
        if self.horizons_equal_status is not True:
            raise MassiveFinalizedValidationProtocolError(
                "V0 cannot select a primary forecast horizon"
            )
        if self.settings != MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS:
            raise MassiveFinalizedValidationProtocolError(
                "V0 setting inventory drifted"
            )
        for setting in self.settings:
            setting.validate()
        if self.primary_contrast != ("MV04", "MV02"):
            raise MassiveFinalizedValidationProtocolError(
                "V0 primary tape contrast drifted"
            )
        if self.cost_ladder_basis_points != (10, 20, 40):
            raise MassiveFinalizedValidationProtocolError("V0 cost ladder drifted")
        expected_counts: dict[str, object] = {
            "development_seeds": (0, 1),
            "confirmation_seeds": (0, 1, 2, 3, 4),
            "outer_fold_count": 4,
            "outer_fold_sessions": 126,
            "historical_lockbox_sessions": 252,
            "minimum_initial_training_sessions": 756,
            "target_overlap_purge_sessions": 63,
            "inner_validation_sessions": 126,
            "inner_purge_sessions": 63,
        }
        for name, expected_value in expected_counts.items():
            if getattr(self, name) != expected_value:
                raise MassiveFinalizedValidationProtocolError(f"{name} drifted")
        authorization_fields = (
            "position_age_input_authorized",
            "duration_objective_authorized",
            "duration_checkpoint_selection_authorized",
            "predictive_training_authorized",
            "diagnostic_portfolio_evaluation_authorized",
            "economic_optimization_authorized",
            "historical_lockbox_access_authorized",
            "prospective_access_authorized",
            "reinforcement_learning_authorized",
        )
        if any(
            not isinstance(getattr(self, name), bool)
            for name in authorization_fields
        ):
            raise MassiveFinalizedValidationProtocolError(
                "V0 authorization fields must be Boolean"
            )
        if any(getattr(self, name) for name in authorization_fields):
            raise MassiveFinalizedValidationProtocolError(
                "the source protocol cannot authorize downstream V0 stages"
            )
        assert_no_adaptive_hold_semantics(self)

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())


def build_massive_finalized_validation_v0_protocol() -> (
    MassiveFinalizedValidationV0Protocol
):
    """Build the sole finalized-file validation V0 protocol."""

    universe_rule = build_massive_adaptive_universe_rule(
        rule_id="massive-finalized-pit500-validation-v0",
        target_size=500,
        minimum_close_price=3.0,
        minimum_average_dollar_volume=5_000_000.0,
    )
    protocol = MassiveFinalizedValidationV0Protocol(
        protocol_id=MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID,
        dataset_id=MASSIVE_FINALIZED_VALIDATION_V0_DATASET_ID,
        purpose=("validate-incremental-finalized-trade-tape-cross-sectional-net-alpha"),
        production_equivalence=False,
        historical_delayed_stream_replay_required=False,
        universe_rule=universe_rule,
        context_universe_rule_receipt_sha256=universe_rule.receipt_sha256,
        action_universe_rule_receipt_sha256=universe_rule.receipt_sha256,
        source_availability_cutoff_local_time="11:30:00",
        decision_local_time="12:30:00",
        fill_start_local_time="15:50:00",
        fill_end_local_time="16:00:00",
        decision_rule=(
            "first-eligible-session-after-vendor-finalized-input-availability"
        ),
        input_cutoff_rule="source-session-or-earlier-only",
        fill_rule="same-session-15:50-16:00-et-qualifying-trade-vwap",
        horizons=MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS,
        horizons_equal_status=True,
        settings=MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS,
        primary_contrast=("MV04", "MV02"),
        cost_ladder_basis_points=(10, 20, 40),
        development_seeds=(0, 1),
        confirmation_seeds=(0, 1, 2, 3, 4),
        outer_fold_count=4,
        outer_fold_sessions=126,
        historical_lockbox_sessions=252,
        minimum_initial_training_sessions=756,
        target_overlap_purge_sessions=63,
        inner_validation_sessions=126,
        inner_purge_sessions=63,
        position_age_input_authorized=False,
        duration_objective_authorized=False,
        duration_checkpoint_selection_authorized=False,
        predictive_training_authorized=False,
        diagnostic_portfolio_evaluation_authorized=False,
        economic_optimization_authorized=False,
        historical_lockbox_access_authorized=False,
        prospective_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    protocol.validate()
    if protocol.receipt_sha256 != MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256:
        raise MassiveFinalizedValidationProtocolError(
            "finalized-validation protocol receipt differs from its frozen identity"
        )
    return protocol


MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL = (
    build_massive_finalized_validation_v0_protocol()
)


__all__ = [
    "MASSIVE_FINALIZED_VALIDATION_V0_DATASET_ID",
    "MASSIVE_FINALIZED_VALIDATION_V0_HORIZONS",
    "MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL",
    "MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL_ID",
    "MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_VALIDATION_V0_SCHEMA",
    "MASSIVE_FINALIZED_VALIDATION_V0_SETTINGS",
    "MassiveFinalizedValidationHorizon",
    "MassiveFinalizedValidationProtocolError",
    "MassiveFinalizedValidationSetting",
    "MassiveFinalizedValidationV0Protocol",
    "build_massive_finalized_validation_v0_protocol",
]
