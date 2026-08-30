"""Authority for carrying adaptive economic state across forecast refits.

Forecast and calibration identities may change at a causal prequential block
boundary.  That scientific refit boundary is not an economic episode boundary.
This authority proves that two adjacent block environments share the same
execution, accounting, benchmark, capital, cost, and identity semantics and
that the first decision of the next block is the close reached by the final
fill of the previous block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-economic-continuity-authority-v1"
)
MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "boundary": "forecast-refit-is-not-economic-episode-end",
        "chronology": "previous-next-fill-equals-next-decision-close",
        "carried_state": (
            "strategy-book",
            "neutral-book",
            "benchmark-book",
            "high-water-marks",
            "trailing-economic-state",
            "previous-action",
        ),
        "replaced_state": ("forecast-archive", "calibration", "inference-plan"),
        "internal_liquidation": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveEconomicContinuityAuthorityV1Error(ValueError):
    """Adjacent adaptive forecast blocks cannot share one economic episode."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveEconomicContinuityAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveEconomicContinuityAuthorityV1:
    previous_block_receipt_sha256: str
    next_block_receipt_sha256: str
    previous_forecast_archive_receipt_sha256: str
    next_forecast_archive_receipt_sha256: str
    previous_calibration_receipt_sha256: str
    next_calibration_receipt_sha256: str
    previous_environment_source_inventory_sha256: str
    next_environment_source_inventory_sha256: str
    previous_economic_compatibility_receipt_sha256: str
    next_economic_compatibility_receipt_sha256: str
    previous_terminal_decision_session_date: str
    previous_terminal_fill_session_date: str
    next_initial_decision_session_date: str
    exchange_sessions_consecutive: bool
    economic_sources_compatible: bool
    carry_books_authorized: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_continuity_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"semantic_receipt_sha256", "development_continuity_authorized"}
        }

    def validate(self) -> None:
        expected_carry = bool(
            self.exchange_sessions_consecutive and self.economic_sources_compatible
        )
        if (
            self.schema
            != MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SCHEMA
            or not self.previous_terminal_decision_session_date
            or not self.previous_terminal_fill_session_date
            or not self.next_initial_decision_session_date
            or not isinstance(self.exchange_sessions_consecutive, bool)
            or not isinstance(self.economic_sources_compatible, bool)
            or not isinstance(self.carry_books_authorized, bool)
            or not isinstance(self.source_data_qualified, bool)
            or self.exchange_sessions_consecutive
            != (
                self.previous_terminal_fill_session_date
                == self.next_initial_decision_session_date
            )
            or self.economic_sources_compatible
            != (
                self.previous_economic_compatibility_receipt_sha256
                == self.next_economic_compatibility_receipt_sha256
            )
            or self.carry_books_authorized != expected_carry
            or self.development_continuity_authorized
            != (expected_carry and self.source_data_qualified)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveEconomicContinuityAuthorityV1Error(
                "adaptive economic continuity authority differs"
            )
        for value in (
            self.previous_block_receipt_sha256,
            self.next_block_receipt_sha256,
            self.previous_forecast_archive_receipt_sha256,
            self.next_forecast_archive_receipt_sha256,
            self.previous_calibration_receipt_sha256,
            self.next_calibration_receipt_sha256,
            self.previous_environment_source_inventory_sha256,
            self.next_environment_source_inventory_sha256,
            self.previous_economic_compatibility_receipt_sha256,
            self.next_economic_compatibility_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive economic continuity authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_economic_continuity_authority_v1(
    *,
    previous_block_receipt_sha256: str,
    next_block_receipt_sha256: str,
    previous_environment: MassiveAdaptiveProfitabilityEnvV1,
    next_environment: MassiveAdaptiveProfitabilityEnvV1,
    source_data_qualified: bool,
) -> MassiveAdaptiveEconomicContinuityAuthorityV1:
    """Bind an adjacent refit boundary without using position duration state."""

    if not previous_environment.inference_plan.rows or not next_environment.inference_plan.rows:
        raise MassiveAdaptiveEconomicContinuityAuthorityV1Error(
            "adaptive continuity environment chronology is absent"
        )
    previous_row = previous_environment.inference_plan.rows[-1]
    next_row = next_environment.inference_plan.rows[0]
    previous_compatibility = previous_environment.economic_compatibility_receipt_sha256
    next_compatibility = next_environment.economic_compatibility_receipt_sha256
    body = {
        "schema": MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SCHEMA,
        "previous_block_receipt_sha256": previous_block_receipt_sha256,
        "next_block_receipt_sha256": next_block_receipt_sha256,
        "previous_forecast_archive_receipt_sha256": (
            previous_environment.forecast_archive.semantic_receipt_sha256
        ),
        "next_forecast_archive_receipt_sha256": (
            next_environment.forecast_archive.semantic_receipt_sha256
        ),
        "previous_calibration_receipt_sha256": (
            previous_environment.calibration.semantic_receipt_sha256
        ),
        "next_calibration_receipt_sha256": (
            next_environment.calibration.semantic_receipt_sha256
        ),
        "previous_environment_source_inventory_sha256": (
            previous_environment.source_inventory_sha256
        ),
        "next_environment_source_inventory_sha256": (
            next_environment.source_inventory_sha256
        ),
        "previous_economic_compatibility_receipt_sha256": previous_compatibility,
        "next_economic_compatibility_receipt_sha256": next_compatibility,
        "previous_terminal_decision_session_date": (
            previous_row.decision_session_date
        ),
        "previous_terminal_fill_session_date": previous_row.next_session_date,
        "next_initial_decision_session_date": next_row.decision_session_date,
        "exchange_sessions_consecutive": bool(
            previous_row.next_session_date == next_row.decision_session_date
        ),
        "economic_sources_compatible": bool(
            previous_compatibility == next_compatibility
        ),
        "carry_books_authorized": bool(
            previous_row.next_session_date == next_row.decision_session_date
            and previous_compatibility == next_compatibility
        ),
        "source_data_qualified": bool(source_data_qualified),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_ECONOMIC_CONTINUITY_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveEconomicContinuityAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        development_continuity_authorized=False,
    )
    carry = bool(body["carry_books_authorized"])
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        development_continuity_authorized=carry and bool(source_data_qualified),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveEconomicContinuityAuthorityV1",
    "MassiveAdaptiveEconomicContinuityAuthorityV1Error",
    "build_massive_adaptive_economic_continuity_authority_v1",
]
