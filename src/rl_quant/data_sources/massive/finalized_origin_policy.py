"""Immutable subordinate origin policy for finalized validation V0."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.data_sources.massive.finalized_daily_scan import (
    MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_listing import (
    MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
    MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
)
from rl_quant.data_sources.massive.finalized_origin import (
    MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
    MASSIVE_FINALIZED_PROCESSING_SPEC_V0,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA = (
    "rl-quant.massive-finalized-origin-policy-v0"
)


class MassiveFinalizedOriginPolicyError(ValueError):
    """The immutable finalized V0 source-origin policy drifted."""


@dataclass(frozen=True, slots=True)
class MassiveFinalizedOriginPolicyV0:
    required_daily_source_roles: tuple[str, ...]
    listing_parser_spec_sha256: str
    trade_file_scan_spec_sha256: str
    trade_partition_spec_sha256: str
    participant_time_domain: str
    processing_spec_receipt_sha256: str
    processing_capability_scope: str
    maximum_source_staleness_sessions: int
    primary_estimand_source_staleness_sessions: int
    nonprimary_staleness_context_required: bool
    source_selection_rule: str
    decision_local_time: str
    fill_window: str
    receipt_sha256: str
    schema: str = MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA:
            raise MassiveFinalizedOriginPolicyError("origin policy schema drifted")
        expected: dict[str, object] = {
            "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
            "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
            "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
            "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
            "participant_time_domain": "[regular-open,regular-close)",
            "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
            "processing_capability_scope": "source-scan-and-partition-only",
            "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
            "primary_estimand_source_staleness_sessions": 1,
            "nonprimary_staleness_context_required": True,
            "source_selection_rule": "latest-prior-source-ready-by-decision",
            "decision_local_time": "12:30:00-America/New_York",
            "fill_window": "[15:50:00,16:00:00]-America/New_York",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise MassiveFinalizedOriginPolicyError(f"{name} drifted")
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveFinalizedOriginPolicyError("origin policy receipt differs")
        if self.receipt_sha256 != MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256:
            raise MassiveFinalizedOriginPolicyError("origin policy frozen receipt drifted")


def _build_policy() -> MassiveFinalizedOriginPolicyV0:
    body: dict[str, object] = {
        "schema": MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA,
        "required_daily_source_roles": MASSIVE_FINALIZED_V0_REQUIRED_DAILY_SOURCE_ROLES,
        "listing_parser_spec_sha256": MASSIVE_FLAT_FILE_LISTING_PARSER_SPEC_SHA256,
        "trade_file_scan_spec_sha256": MASSIVE_DAILY_TRADE_FILE_SCAN_SPEC_SHA256,
        "trade_partition_spec_sha256": MASSIVE_DAILY_TRADE_PARTITION_SPEC_SHA256,
        "participant_time_domain": "[regular-open,regular-close)",
        "processing_spec_receipt_sha256": MASSIVE_FINALIZED_PROCESSING_SPEC_V0.receipt_sha256,
        "processing_capability_scope": "source-scan-and-partition-only",
        "maximum_source_staleness_sessions": MASSIVE_FINALIZED_MAXIMUM_SOURCE_STALENESS_SESSIONS,
        "primary_estimand_source_staleness_sessions": 1,
        "nonprimary_staleness_context_required": True,
        "source_selection_rule": "latest-prior-source-ready-by-decision",
        "decision_local_time": "12:30:00-America/New_York",
        "fill_window": "[15:50:00,16:00:00]-America/New_York",
    }
    return MassiveFinalizedOriginPolicyV0(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )


# This literal changes only with a new origin-policy generation.
MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256 = (
    "ce58430a6abea44032da299bc7b8580e0aeeb6dcbac24c6c3f9ba30c768b67a5"
)
MASSIVE_FINALIZED_ORIGIN_POLICY_V0 = _build_policy()
MASSIVE_FINALIZED_ORIGIN_POLICY_V0.validate()


__all__ = [
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0_RECEIPT_SHA256",
    "MASSIVE_FINALIZED_ORIGIN_POLICY_V0_SCHEMA",
    "MassiveFinalizedOriginPolicyError",
    "MassiveFinalizedOriginPolicyV0",
]
