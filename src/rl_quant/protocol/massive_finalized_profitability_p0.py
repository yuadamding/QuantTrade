"""Immutable research-only P0 profitability protocol for finalized Massive data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)


MASSIVE_FINALIZED_PROFITABILITY_P0_PROTOCOL_ID = "massive-finalized-profitability-p0"
MASSIVE_FINALIZED_PROFITABILITY_P0_DATASET_ID = "MassiveFinalizedPIT500ProfitabilityP0"
MASSIVE_FINALIZED_PROFITABILITY_P0_SCHEMA = (
    "rl-quant.massive-finalized-profitability-p0"
)


class MassiveFinalizedProfitabilityP0Error(ValueError):
    """The frozen research-only profitability protocol drifted."""


@dataclass(frozen=True, slots=True)
class MassiveFinalizedProfitabilityP0Protocol:
    protocol_id: str
    dataset_id: str
    parent_validation_protocol_receipt_sha256: str
    universe_rule_receipt_sha256: str
    source_staleness_sessions: int
    minimum_vendor_lead_time_hours: int
    decision_local_time: str
    fill_window: str
    source_cutoff_rule: str
    target_start_rule: str
    production_equivalence: bool
    historical_delayed_stream_replay_required: bool
    historical_runtime_capability_required: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    historical_lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_FINALIZED_PROFITABILITY_P0_SCHEMA

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())

    def validate(self) -> None:
        expected: dict[str, object] = {
            "schema": MASSIVE_FINALIZED_PROFITABILITY_P0_SCHEMA,
            "protocol_id": MASSIVE_FINALIZED_PROFITABILITY_P0_PROTOCOL_ID,
            "dataset_id": MASSIVE_FINALIZED_PROFITABILITY_P0_DATASET_ID,
            "parent_validation_protocol_receipt_sha256": (
                MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.receipt_sha256
            ),
            "universe_rule_receipt_sha256": (
                MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
            ),
            "source_staleness_sessions": 2,
            "minimum_vendor_lead_time_hours": 18,
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00)-America/New_York",
            "source_cutoff_rule": "source-session-and-earlier-only",
            "target_start_rule": "strictly-after-diagnostic-fill",
            "production_equivalence": False,
            "historical_delayed_stream_replay_required": False,
            "historical_runtime_capability_required": False,
            "panel_materialization_authorized": False,
            "predictive_training_authorized": False,
            "portfolio_evaluation_authorized": False,
            "historical_lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedProfitabilityP0Error(f"{name} drifted")


def build_massive_finalized_profitability_p0_protocol() -> (
    MassiveFinalizedProfitabilityP0Protocol
):
    protocol = MassiveFinalizedProfitabilityP0Protocol(
        protocol_id=MASSIVE_FINALIZED_PROFITABILITY_P0_PROTOCOL_ID,
        dataset_id=MASSIVE_FINALIZED_PROFITABILITY_P0_DATASET_ID,
        parent_validation_protocol_receipt_sha256=(
            MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.receipt_sha256
        ),
        universe_rule_receipt_sha256=(
            MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
        ),
        source_staleness_sessions=2,
        minimum_vendor_lead_time_hours=18,
        decision_local_time="12:30:00-America/New_York",
        fill_window="[15:50:00,16:00:00)-America/New_York",
        source_cutoff_rule="source-session-and-earlier-only",
        target_start_rule="strictly-after-diagnostic-fill",
        production_equivalence=False,
        historical_delayed_stream_replay_required=False,
        historical_runtime_capability_required=False,
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        portfolio_evaluation_authorized=False,
        historical_lockbox_access_authorized=False,
        reinforcement_learning_authorized=False,
    )
    protocol.validate()
    return protocol


MASSIVE_FINALIZED_PROFITABILITY_P0 = build_massive_finalized_profitability_p0_protocol()
MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256 = (
    "2b53161b3d49c6cdf69625bd3d090f3986b63ee18d48ab5eba80d1739beab66b"
)
if (
    MASSIVE_FINALIZED_PROFITABILITY_P0.receipt_sha256
    != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
):
    raise MassiveFinalizedProfitabilityP0Error(
        "profitability P0 frozen receipt drifted"
    )


__all__ = [
    "MASSIVE_FINALIZED_PROFITABILITY_P0",
    "MASSIVE_FINALIZED_PROFITABILITY_P0_DATASET_ID",
    "MASSIVE_FINALIZED_PROFITABILITY_P0_PROTOCOL_ID",
    "MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_PROFITABILITY_P0_SCHEMA",
    "MassiveFinalizedProfitabilityP0Error",
    "MassiveFinalizedProfitabilityP0Protocol",
    "build_massive_finalized_profitability_p0_protocol",
]
