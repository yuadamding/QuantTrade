"""Production-acquired P0 origins and exact t-1 monthly rank inputs.

This additive generation does not authorize a profitability experiment.  It
closes two narrower provenance gaps left by V1:

* the authorizing acquisition entry point constructs the Massive S3 client
  internally and exposes neither a provider client nor a clock;
* every scheduled monthly PIT rank group is independently reconstructed from
  the exact 63 exchange sessions ending at t-1, committed daily bars, and the
  vendor availability carried by authenticated listing/object acquisitions.

The generic V0/V1 capture functions remain useful deterministic test tools,
but an authority assembled from them is explicitly nonauthorizing here.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.alpha.pit_universe import (
    PITSecurityUniverseAuthority,
    SourcedSecurityMasterRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.finalized_listing import (
    canonical_massive_trade_object_key,
    coverage_session_from_massive_trade_key,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_ENDPOINT,
    MassiveCapturedFlatFileListingV0,
    capture_massive_flat_file_listing_v0,
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_object_acquisition import (
    MassiveAuthenticatedFlatFileDownloadV1,
    download_massive_flat_file_object_v1,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.data_sources.massive.source_receipts import (
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_daily_bars_v0 import (
    MASSIVE_DAILY_BARS_V0_FIELDS,
    MassiveDailyBarsArtifactV0,
    validate_massive_daily_bars_v0,
)
from rl_quant.features.massive_profitability_origin_v1 import (
    MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    MassiveProfitabilityDecisionOriginPlanV1,
    build_massive_profitability_decision_origin_plan_v1,
    materialize_massive_profitability_acquired_source_evidence_v1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

MASSIVE_PROFITABILITY_PRODUCTION_ACQUISITION_V2_SCHEMA = (
    "rl-quant.massive-profitability-production-acquisition-v2"
)
MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-monthly-rank-input-authority-v2"
)
MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA = (
    "rl-quant.massive-profitability-origin-plan-v2"
)
MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "listing_transport": "package-owned-real-clock-s3v4",
        "object_transport": "package-owned-real-clock-authenticated-get",
        "generic_and_test_capture": "nonauthorizing",
        "rank_schedule": "first-XNYS-session-each-candidate-month",
        "rank_window": "exact-63-XNYS-sessions-ending-t-minus-1",
        "rank_values": "committed-daily-bars-v0-close-and-dollar-volume",
        "rank_availability": "maximum-authenticated-listing-last-modified",
        "membership": "exact-independent-rank-group-reconciliation",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_ORIGIN_V2_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V2_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V2_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_ORIGIN_V2_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityOriginV2Error(ValueError):
    """Production acquisition or exact monthly rank evidence differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityOriginV2Error(f"{name} must be a lowercase SHA-256")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveProfitabilityOriginV2Error(f"{name} must be canonical text")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveProfitabilityOriginV2Error(f"{name} must be nonnegative")
    return value


def _session_ms(value_ns: int) -> int:
    value = _nonnegative_int("session timestamp", value_ns)
    if value % 1_000_000:
        raise MassiveProfitabilityOriginV2Error(
            "session timestamp is not millisecond aligned"
        )
    return value // 1_000_000


def _fixed_runtime_receipt(
    *,
    listings: Sequence[MassiveCapturedFlatFileListingV0],
    downloads: Sequence[MassiveAuthenticatedFlatFileDownloadV1],
) -> str:
    return semantic_sha256(
        {
            "implementation_source_sha256": (
                MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256
            ),
            "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
            "listing_acquisition_receipts": tuple(
                sorted(row.acquisition_evidence.receipt_sha256 for row in listings)
            ),
            "authenticated_download_receipts": tuple(
                sorted(row.receipt_sha256 for row in downloads)
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityProductionAcquisitionV2:
    captured_listings: tuple[MassiveCapturedFlatFileListingV0, ...]
    authenticated_downloads: tuple[MassiveAuthenticatedFlatFileDownloadV1, ...]
    entitlement_receipt_sha256: str
    listing_inventory_sha256: str
    download_inventory_sha256: str
    fixed_runtime_capture_receipt_sha256: str | None
    fixed_runtime_captured: bool
    receipt_sha256: str
    schema: str = MASSIVE_PROFITABILITY_PRODUCTION_ACQUISITION_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "captured_listing_receipts": tuple(
                row.acquisition_evidence.receipt_sha256
                for row in self.captured_listings
            ),
            "authenticated_download_receipts": tuple(
                row.receipt_sha256 for row in self.authenticated_downloads
            ),
            "entitlement_receipt_sha256": self.entitlement_receipt_sha256,
            "listing_inventory_sha256": self.listing_inventory_sha256,
            "download_inventory_sha256": self.download_inventory_sha256,
            "fixed_runtime_capture_receipt_sha256": (
                self.fixed_runtime_capture_receipt_sha256
            ),
            "fixed_runtime_captured": self.fixed_runtime_captured,
            "implementation_source_sha256": (
                MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256
            ),
            "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_PROFITABILITY_PRODUCTION_ACQUISITION_V2_SCHEMA:
            raise MassiveProfitabilityOriginV2Error(
                "production acquisition schema differs"
            )
        if not self.captured_listings or not self.authenticated_downloads:
            raise MassiveProfitabilityOriginV2Error(
                "production acquisition requires listings and downloads"
            )
        for captured in self.captured_listings:
            captured.validate()
        for download in self.authenticated_downloads:
            download.validate()
        prefixes = tuple(
            row.acquisition_evidence.prefix for row in self.captured_listings
        )
        keys = tuple(row.source_object_key for row in self.authenticated_downloads)
        if prefixes != tuple(sorted(set(prefixes))):
            raise MassiveProfitabilityOriginV2Error(
                "production listing months are not canonical"
            )
        if keys != tuple(sorted(set(keys))):
            raise MassiveProfitabilityOriginV2Error(
                "production downloads are not canonical"
            )
        entitlement = _digest(
            "production acquisition entitlement", self.entitlement_receipt_sha256
        )
        if any(
            row.loaded_acquisition.receipt.entitlement_receipt_sha256 != entitlement
            or row.loaded_listing.receipt.entitlement_receipt_sha256 != entitlement
            for row in self.captured_listings
        ) or any(
            row.loaded_source.receipt.entitlement_receipt_sha256 != entitlement
            for row in self.authenticated_downloads
        ):
            raise MassiveProfitabilityOriginV2Error(
                "production acquisition entitlement differs"
            )
        expected_listing_inventory = semantic_sha256(
            tuple(
                (
                    row.acquisition_evidence.prefix,
                    row.acquisition_evidence.receipt_sha256,
                    row.committed_listing.receipt_sha256,
                )
                for row in self.captured_listings
            )
        )
        expected_download_inventory = semantic_sha256(
            tuple(
                (row.source_object_key, row.receipt_sha256)
                for row in self.authenticated_downloads
            )
        )
        if (
            self.listing_inventory_sha256 != expected_listing_inventory
            or self.download_inventory_sha256 != expected_download_inventory
        ):
            raise MassiveProfitabilityOriginV2Error(
                "production acquisition inventories differ"
            )
        if not isinstance(self.fixed_runtime_captured, bool):
            raise MassiveProfitabilityOriginV2Error(
                "fixed-runtime acquisition state must be Boolean"
            )
        expected_runtime = _fixed_runtime_receipt(
            listings=self.captured_listings,
            downloads=self.authenticated_downloads,
        )
        if self.fixed_runtime_captured:
            if self.fixed_runtime_capture_receipt_sha256 != expected_runtime:
                raise MassiveProfitabilityOriginV2Error(
                    "fixed-runtime acquisition receipt differs"
                )
        elif self.fixed_runtime_capture_receipt_sha256 is not None:
            raise MassiveProfitabilityOriginV2Error(
                "nonauthorizing acquisition carries a runtime receipt"
            )
        for name in (
            "listing_inventory_sha256",
            "download_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginV2Error(
                "production acquisition receipt differs"
            )


def _build_acquisition_authority(
    *,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    authenticated_downloads: Sequence[MassiveAuthenticatedFlatFileDownloadV1],
    entitlement_receipt_sha256: str,
    fixed_runtime_captured: bool,
) -> MassiveProfitabilityProductionAcquisitionV2:
    listings = tuple(
        sorted(
            captured_listings,
            key=lambda row: row.acquisition_evidence.prefix,
        )
    )
    downloads = tuple(
        sorted(authenticated_downloads, key=lambda row: row.source_object_key)
    )
    listing_inventory = semantic_sha256(
        tuple(
            (
                row.acquisition_evidence.prefix,
                row.acquisition_evidence.receipt_sha256,
                row.committed_listing.receipt_sha256,
            )
            for row in listings
        )
    )
    download_inventory = semantic_sha256(
        tuple((row.source_object_key, row.receipt_sha256) for row in downloads)
    )
    body: dict[str, object] = {
        "captured_listings": listings,
        "authenticated_downloads": downloads,
        "entitlement_receipt_sha256": _digest(
            "production acquisition entitlement", entitlement_receipt_sha256
        ),
        "listing_inventory_sha256": listing_inventory,
        "download_inventory_sha256": download_inventory,
        "fixed_runtime_capture_receipt_sha256": (
            _fixed_runtime_receipt(listings=listings, downloads=downloads)
            if fixed_runtime_captured
            else None
        ),
        "fixed_runtime_captured": fixed_runtime_captured,
        "schema": MASSIVE_PROFITABILITY_PRODUCTION_ACQUISITION_V2_SCHEMA,
    }
    provisional = MassiveProfitabilityProductionAcquisitionV2(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,
    )
    result = MassiveProfitabilityProductionAcquisitionV2(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


def build_massive_profitability_acquisition_for_test_v2(
    *,
    captured_listings: Sequence[MassiveCapturedFlatFileListingV0],
    authenticated_downloads: Sequence[MassiveAuthenticatedFlatFileDownloadV1],
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityProductionAcquisitionV2:
    """Assemble deterministic fixture evidence; it can never authorize P0."""

    return _build_acquisition_authority(
        captured_listings=captured_listings,
        authenticated_downloads=authenticated_downloads,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        fixed_runtime_captured=False,
    )


def _fixed_massive_s3_client_v2(
    *, access_key_environment_variable: str, secret_key_environment_variable: str
) -> Any:
    access_key = os.environ.get(access_key_environment_variable)
    secret_key = os.environ.get(secret_key_environment_variable)
    if not access_key or not secret_key:
        raise MassiveProfitabilityOriginV2Error(
            "required Massive S3 credential environment variables are absent"
        )
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise MassiveProfitabilityOriginV2Error(
            "boto3 and botocore are required for production acquisition"
        ) from exc
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return session.client(
        "s3",
        endpoint_url=MASSIVE_FLAT_FILE_ENDPOINT,
        config=Config(signature_version="s3v4"),
    )


def capture_massive_profitability_production_acquisition_v2(
    *,
    root: str | Path,
    source_object_keys: Sequence[str],
    entitlement_receipt_sha256: str,
    access_key_environment_variable: str = "MASSIVE_S3_ACCESS_KEY_ID",
    secret_key_environment_variable: str = "MASSIVE_S3_SECRET_ACCESS_KEY",
) -> MassiveProfitabilityProductionAcquisitionV2:
    """Capture listings and GETs with a package-owned client and real clocks.

    Deliberately absent from this signature: provider clients, wall clocks, and
    monotonic clocks.  Tests must use the explicitly nonauthorizing builder.
    """

    keys = tuple(sorted(set(source_object_keys)))
    if not keys or len(keys) != len(tuple(source_object_keys)):
        raise MassiveProfitabilityOriginV2Error(
            "production source keys must be nonempty, unique, and canonical"
        )
    months: set[tuple[int, int]] = set()
    for key in keys:
        session_date = coverage_session_from_massive_trade_key(key)
        if key != canonical_massive_trade_object_key(session_date):
            raise MassiveProfitabilityOriginV2Error(
                "production source object key differs from the canonical trade key"
            )
        months.add((int(session_date[:4]), int(session_date[5:7])))
    client = _fixed_massive_s3_client_v2(
        access_key_environment_variable=access_key_environment_variable,
        secret_key_environment_variable=secret_key_environment_variable,
    )
    listings = tuple(
        capture_massive_flat_file_listing_v0(
            s3_client=client,
            root=root,
            year=year,
            month=month,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
            access_key_environment_variable=access_key_environment_variable,
            secret_key_environment_variable=secret_key_environment_variable,
        )
        for year, month in sorted(months)
    )
    listing_by_key = {
        entry.source_object_key: captured
        for captured in listings
        for entry in captured.committed_listing.entries
    }
    if any(key not in listing_by_key for key in keys):
        raise MassiveProfitabilityOriginV2Error(
            "production listing does not contain every requested object"
        )
    downloads = tuple(
        download_massive_flat_file_object_v1(
            s3_client=client,
            captured_listing=listing_by_key[key],
            source_object_key=key,
            destination_root=root,
            entitlement_receipt_sha256=entitlement_receipt_sha256,
        )
        for key in keys
    )
    result = _build_acquisition_authority(
        captured_listings=listings,
        authenticated_downloads=downloads,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        fixed_runtime_captured=True,
    )
    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=result, require_fixed_runtime=True
    )
    return result


def validate_massive_profitability_production_acquisition_v2(
    *,
    root: str | Path,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    require_fixed_runtime: bool,
) -> None:
    """Reopen every committed listing and authenticated object payload."""

    acquisition.validate()
    if require_fixed_runtime and not acquisition.fixed_runtime_captured:
        raise MassiveProfitabilityOriginV2Error(
            "P0 requires package-owned production acquisition"
        )
    by_acquisition = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    for captured in acquisition.captured_listings:
        validate_massive_captured_flat_file_listing_v0(
            root=root, captured_listing=captured
        )
    for download in acquisition.authenticated_downloads:
        download.validate()
        read_loaded_massive_source_bytes(
            root=root, loaded_source=download.loaded_source
        )
        captured = by_acquisition.get(download.listing_acquisition_receipt_sha256)
        if captured is None:
            raise MassiveProfitabilityOriginV2Error(
                "authenticated GET is absent from the captured listing inventory"
            )
        entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        if (
            download.listing_entry_receipt_sha256 != entry.receipt_sha256
            or download.etag != entry.etag
            or download.content_length != entry.content_length
        ):
            raise MassiveProfitabilityOriginV2Error(
                "authenticated GET and captured listing entry differ"
            )


def materialize_massive_profitability_acquired_source_evidence_from_acquisition_v2(
    *,
    root: str | Path,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    source_session_dates: Sequence[str],
    artifact_id: str,
    committed_at_ms: int,
) -> MassiveProfitabilityAcquiredSourceEvidenceArtifactV1:
    """Persist the V1 source view from a production-qualified acquisition."""

    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    requested = tuple(sorted(set(source_session_dates)))
    if not requested or len(requested) != len(tuple(source_session_dates)):
        raise MassiveProfitabilityOriginV2Error(
            "source evidence dates must be nonempty, unique, and canonical"
        )
    downloads = tuple(
        row
        for row in acquisition.authenticated_downloads
        if coverage_session_from_massive_trade_key(row.source_object_key) in requested
    )
    if (
        tuple(
            coverage_session_from_massive_trade_key(row.source_object_key)
            for row in downloads
        )
        != requested
    ):
        raise MassiveProfitabilityOriginV2Error(
            "production acquisition does not contain the exact source evidence dates"
        )
    return materialize_massive_profitability_acquired_source_evidence_v1(
        root=root,
        captured_listings=acquisition.captured_listings,
        loaded_trade_sources=tuple(row.loaded_source for row in downloads),
        artifact_id=artifact_id,
        committed_at_ms=committed_at_ms,
        entitlement_receipt_sha256=acquisition.entitlement_receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class MassiveMonthlyRankInputGroupV2:
    calendar_month: str
    scheduled_rebalance_session_date: str
    scheduled_effective_at_ms: int
    observation_session_dates: tuple[str, ...]
    observation_start_session_index: int
    observation_end_session_index: int
    maximum_vendor_available_at_ms: int
    daily_bar_inventory_sha256: str
    authenticated_source_inventory_sha256: str
    rank_inputs: tuple[UniverseRankInputRecord, ...]
    rank_input_group_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            len(self.calendar_month) != 7
            or self.scheduled_rebalance_session_date[:7] != self.calendar_month
            or not self.observation_session_dates
            or self.observation_session_dates
            != tuple(sorted(set(self.observation_session_dates)))
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank group calendar identity differs"
            )
        if (
            self.observation_end_session_index
            - self.observation_start_session_index
            + 1
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.ranking_lookback_sessions
            or len(self.observation_session_dates)
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.ranking_lookback_sessions
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank group lookback differs"
            )
        effective = _nonnegative_int(
            "monthly rank effective time", self.scheduled_effective_at_ms
        )
        available = _nonnegative_int(
            "monthly rank availability", self.maximum_vendor_available_at_ms
        )
        if available > effective:
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank inputs were unavailable at activation"
            )
        security_ids = tuple(row.security_id for row in self.rank_inputs)
        if not security_ids or security_ids != tuple(sorted(set(security_ids))):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank rows are not the complete canonical candidate group"
            )
        for row in self.rank_inputs:
            row.validate_for(MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule)
            if (
                row.effective_at_ms != effective
                or row.available_at_ms != available
                or row.observation_start_session_index
                != self.observation_start_session_index
                or row.observation_end_session_index
                != self.observation_end_session_index
            ):
                raise MassiveProfitabilityOriginV2Error(
                    "monthly rank row chronology differs from its group"
                )
        expected_group = semantic_sha256(
            {
                "rule_receipt_sha256": (
                    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
                ),
                "rank_inputs": [asdict(row) for row in self.rank_inputs],
            }
        )
        if self.rank_input_group_receipt_sha256 != expected_group:
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank-input group receipt differs"
            )
        for name in (
            "daily_bar_inventory_sha256",
            "authenticated_source_inventory_sha256",
            "rank_input_group_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank group receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveMonthlyRankInputAuthorityV2:
    first_candidate_decision_session_date: str
    last_candidate_decision_session_date: str
    groups: tuple[MassiveMonthlyRankInputGroupV2, ...]
    acquisition_receipt_sha256: str
    session_authority_receipt_sha256: str
    identity_authority_audit_receipt_sha256: str
    rule_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    schema: str = MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "first_candidate_decision_session_date": (
                self.first_candidate_decision_session_date
            ),
            "last_candidate_decision_session_date": (
                self.last_candidate_decision_session_date
            ),
            "groups": tuple(asdict(row) for row in self.groups),
            "acquisition_receipt_sha256": self.acquisition_receipt_sha256,
            "session_authority_receipt_sha256": self.session_authority_receipt_sha256,
            "rule_receipt_sha256": self.rule_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "source_data_qualified": self.source_data_qualified,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA
            or self.rule_receipt_sha256
            != MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule.receipt_sha256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank authority identity differs"
            )
        if not self.groups or tuple(row.calendar_month for row in self.groups) != tuple(
            sorted({row.calendar_month for row in self.groups})
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank authority groups differ"
            )
        for group in self.groups:
            group.validate()
        if not isinstance(self.source_data_qualified, bool):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank source qualification must be Boolean"
            )
        for name in (
            "acquisition_receipt_sha256",
            "session_authority_receipt_sha256",
            "identity_authority_audit_receipt_sha256",
            "rule_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank authority semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "identity_authority_audit_receipt_sha256": (
                    self.identity_authority_audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank authority audit receipt differs"
            )


def _active_security_ids(
    security_master: Sequence[SourcedSecurityMasterRecord], *, effective_at_ms: int
) -> tuple[str, ...]:
    return tuple(
        sorted(
            row.security_id
            for row in security_master
            if row.listing_at_ms <= effective_at_ms
            and (row.delisting_at_ms is None or effective_at_ms < row.delisting_at_ms)
        )
    )


def _scheduled_months(
    *,
    sessions: Sequence[MassiveExchangeSession],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> tuple[tuple[str, int, MassiveExchangeSession], ...]:
    by_month: defaultdict[str, list[tuple[int, MassiveExchangeSession]]] = defaultdict(
        list
    )
    for index, session in enumerate(sessions):
        by_month[session.session_date[:7]].append((index, session))
    months = tuple(
        sorted(
            {
                session.session_date[:7]
                for session in sessions
                if first_candidate_decision_session_date
                <= session.session_date
                <= last_candidate_decision_session_date
            }
        )
    )
    if not months:
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank candidate interval is empty"
        )
    return tuple(
        (month, *min(by_month[month], key=lambda row: row[1].session_date))
        for month in months
    )


def _source_rows_by_date(
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
) -> dict[str, tuple[int, str, str]]:
    listing_by_acquisition = {
        row.acquisition_evidence.receipt_sha256: row
        for row in acquisition.captured_listings
    }
    output: dict[str, tuple[int, str, str]] = {}
    for download in acquisition.authenticated_downloads:
        captured = listing_by_acquisition[download.listing_acquisition_receipt_sha256]
        entry = captured.committed_listing.resolve(
            source_object_key=download.source_object_key
        )
        session_date = coverage_session_from_massive_trade_key(
            download.source_object_key
        )
        if session_date in output:
            raise MassiveProfitabilityOriginV2Error(
                "authenticated acquisition duplicates one source session"
            )
        output[session_date] = (
            entry.vendor_last_modified_at_ms,
            entry.receipt_sha256,
            download.receipt_sha256,
        )
    return output


def build_massive_monthly_rank_input_authority_v2(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveMonthlyRankInputAuthorityV2:
    """Reconstruct each monthly candidate group from exact committed bars."""

    session_authority.validate()
    identity_authority.validate()
    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=False
    )
    rule = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL.universe_rule
    if identity_authority.rule.receipt_sha256 != rule.receipt_sha256:
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank authority requires the frozen PIT-500 rule"
        )
    sessions = tuple(
        row for row in session_authority.sessions if row.exchange == "XNYS"
    )
    dates = tuple(row.session_date for row in sessions)
    if (
        first_candidate_decision_session_date not in dates
        or last_candidate_decision_session_date not in dates
        or first_candidate_decision_session_date > last_candidate_decision_session_date
    ):
        raise MassiveProfitabilityOriginV2Error(
            "monthly rank candidate interval is absent or inverted"
        )
    bars_by_date: dict[str, MassiveDailyBarsArtifactV0] = {}
    for artifact in daily_bars:
        validate_massive_daily_bars_v0(root=root, artifact=artifact)
        if artifact.source_session_date in bars_by_date:
            raise MassiveProfitabilityOriginV2Error(
                "daily bars duplicate one source session"
            )
        bars_by_date[artifact.source_session_date] = artifact
    source_by_date = _source_rows_by_date(acquisition)
    actual_by_effective: defaultdict[int, list[UniverseRankInputRecord]] = defaultdict(
        list
    )
    for row in identity_authority.rank_inputs:
        actual_by_effective[row.effective_at_ms].append(row)

    close_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("close")
    dollar_index = MASSIVE_DAILY_BARS_V0_FIELDS.index("dollar_volume")
    groups: list[MassiveMonthlyRankInputGroupV2] = []
    for month, effective_index, scheduled in _scheduled_months(
        sessions=sessions,
        first_candidate_decision_session_date=first_candidate_decision_session_date,
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    ):
        expected_end = effective_index - rule.ranking_lag_sessions
        expected_start = expected_end - rule.ranking_lookback_sessions + 1
        if expected_start < 0:
            raise MassiveProfitabilityOriginV2Error(
                "scheduled monthly rank group lacks exact prehistory"
            )
        window_sessions = sessions[expected_start : expected_end + 1]
        window_dates = tuple(row.session_date for row in window_sessions)
        if set(window_dates) - set(bars_by_date):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank group lacks committed daily bars"
            )
        if set(window_dates) - set(source_by_date):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank group lacks authenticated source availability"
            )
        if any(
            source_by_date[session.session_date][0]
            < _session_ms(session.regular_close_ns)
            for session in window_sessions
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank source availability predates its session close"
            )
        effective_ms = _session_ms(scheduled.regular_open_ns)
        available_ms = max(source_by_date[value][0] for value in window_dates)
        if available_ms > effective_ms:
            raise MassiveProfitabilityOriginV2Error(
                "monthly rank source data was unavailable at scheduled activation"
            )
        bar_maps = {
            session_date: {
                row.security_id: row for row in bars_by_date[session_date].rows
            }
            for session_date in window_dates
        }
        active = _active_security_ids(
            identity_authority.security_master, effective_at_ms=effective_ms
        )
        derived_rows: list[UniverseRankInputRecord] = []
        for security_id in active:
            observations = tuple(
                bar_maps[session_date].get(security_id) for session_date in window_dates
            )
            dollar_values = tuple(
                float(row.values[dollar_index])
                for row in observations
                if row is not None and row.valid[dollar_index]
            )
            terminal = observations[-1]
            terminal_close_valid = terminal is not None and terminal.valid[close_index]
            observed_count = len(dollar_values) if terminal_close_valid else 0
            average_dollar_volume = (
                sum(dollar_values) / len(dollar_values) if observed_count else None
            )
            close_price = (
                float(terminal.values[close_index])
                if observed_count and terminal is not None
                else None
            )
            source_receipt = semantic_sha256(
                {
                    "specification_sha256": (
                        MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256
                    ),
                    "security_id": security_id,
                    "effective_at_ms": effective_ms,
                    "window": tuple(
                        {
                            "session_date": session_date,
                            "daily_bars_artifact_receipt_sha256": (
                                bars_by_date[session_date].receipt_sha256
                            ),
                            "daily_bar_row_receipt_sha256": (
                                None if row is None else row.receipt_sha256
                            ),
                            "listing_entry_receipt_sha256": source_by_date[
                                session_date
                            ][1],
                            "authenticated_download_receipt_sha256": source_by_date[
                                session_date
                            ][2],
                        }
                        for session_date, row in zip(
                            window_dates, observations, strict=True
                        )
                    ),
                }
            )
            derived_rows.append(
                UniverseRankInputRecord(
                    security_id=security_id,
                    effective_at_ms=effective_ms,
                    effective_session_index=effective_index,
                    available_at_ms=available_ms,
                    observation_start_ms=_session_ms(
                        window_sessions[0].regular_close_ns
                    ),
                    observation_end_ms=_session_ms(
                        window_sessions[-1].regular_close_ns
                    ),
                    observation_start_session_index=expected_start,
                    observation_end_session_index=expected_end,
                    observed_session_count=observed_count,
                    average_dollar_volume=average_dollar_volume,
                    close_price=close_price,
                    source_receipt_sha256=source_receipt,
                )
            )
        ordered_rows = tuple(sorted(derived_rows, key=lambda row: row.security_id))
        for row in ordered_rows:
            row.validate_for(rule)
            if (
                row.observation_end_session_index
                != row.effective_session_index - rule.ranking_lag_sessions
                or row.observation_start_session_index
                != row.observation_end_session_index
                - rule.ranking_lookback_sessions
                + 1
            ):
                raise MassiveProfitabilityOriginV2Error(
                    "monthly rank input is not the exact frozen t-1 window"
                )
        actual = tuple(
            sorted(
                actual_by_effective.get(effective_ms, ()),
                key=lambda row: row.security_id,
            )
        )
        if actual != ordered_rows:
            raise MassiveProfitabilityOriginV2Error(
                "PIT authority rank inputs differ from committed daily-bar derivation"
            )
        group_receipt = semantic_sha256(
            {
                "rule_receipt_sha256": rule.receipt_sha256,
                "rank_inputs": [asdict(row) for row in ordered_rows],
            }
        )
        membership_group = tuple(
            sorted(
                (
                    row
                    for row in identity_authority.membership_events
                    if row.effective_at_ms == effective_ms
                ),
                key=lambda row: row.security_id,
            )
        )
        if not membership_group or any(
            row.rank_input_group_receipt_sha256 != group_receipt
            for row in membership_group
        ):
            raise MassiveProfitabilityOriginV2Error(
                "monthly membership was not derived from the reconstructed rank group"
            )
        body: dict[str, object] = {
            "calendar_month": month,
            "scheduled_rebalance_session_date": scheduled.session_date,
            "scheduled_effective_at_ms": effective_ms,
            "observation_session_dates": window_dates,
            "observation_start_session_index": expected_start,
            "observation_end_session_index": expected_end,
            "maximum_vendor_available_at_ms": available_ms,
            "daily_bar_inventory_sha256": semantic_sha256(
                tuple(
                    (value, bars_by_date[value].receipt_sha256)
                    for value in window_dates
                )
            ),
            "authenticated_source_inventory_sha256": semantic_sha256(
                tuple((value, *source_by_date[value]) for value in window_dates)
            ),
            "rank_inputs": ordered_rows,
            "rank_input_group_receipt_sha256": group_receipt,
        }
        group = MassiveMonthlyRankInputGroupV2(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(
                {
                    **body,
                    "rank_inputs": tuple(asdict(row) for row in ordered_rows),
                }
            ),
        )
        group.validate()
        groups.append(group)
    ordered_groups = tuple(groups)
    semantic_body: dict[str, object] = {
        "schema": MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA,
        "first_candidate_decision_session_date": (
            first_candidate_decision_session_date
        ),
        "last_candidate_decision_session_date": last_candidate_decision_session_date,
        "groups": tuple(asdict(row) for row in ordered_groups),
        "acquisition_receipt_sha256": acquisition.receipt_sha256,
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "rule_receipt_sha256": rule.receipt_sha256,
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        "implementation_source_sha256": (MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256),
        "source_data_qualified": acquisition.fixed_runtime_captured,
    }
    semantic_receipt = semantic_sha256(semantic_body)
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "identity_authority_audit_receipt_sha256": (
                identity_authority.receipt_sha256
            ),
        }
    )
    result = MassiveMonthlyRankInputAuthorityV2(
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
        groups=ordered_groups,
        acquisition_receipt_sha256=acquisition.receipt_sha256,
        session_authority_receipt_sha256=session_authority.receipt_sha256,
        identity_authority_audit_receipt_sha256=identity_authority.receipt_sha256,
        rule_receipt_sha256=rule.receipt_sha256,
        specification_sha256=MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        implementation_source_sha256=MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256,
        source_data_qualified=acquisition.fixed_runtime_captured,
        semantic_receipt_sha256=semantic_receipt,
        audit_receipt_sha256=audit_receipt,
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDecisionOriginPlanV2:
    origin_plan_v1: MassiveProfitabilityDecisionOriginPlanV1
    production_acquisition_receipt_sha256: str
    monthly_rank_authority_semantic_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    acquisition_audit_receipt_sha256: str
    monthly_rank_audit_receipt_sha256: str
    audit_receipt_sha256: str
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "origin_plan_v1_semantic_receipt_sha256": (
                self.origin_plan_v1.semantic_receipt_sha256
            ),
            "production_acquisition_receipt_sha256": (
                self.production_acquisition_receipt_sha256
            ),
            "monthly_rank_authority_semantic_receipt_sha256": (
                self.monthly_rank_authority_semantic_receipt_sha256
            ),
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "panel_materialization_authorized": self.panel_materialization_authorized,
            "predictive_training_authorized": self.predictive_training_authorized,
            "profitability_reporting_authorized": self.profitability_reporting_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
        }

    def validate(self) -> None:
        self.origin_plan_v1.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256
            or any(
                (
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityOriginV2Error("V2 origin plan identity differs")
        for name in (
            "production_acquisition_receipt_sha256",
            "monthly_rank_authority_semantic_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "acquisition_audit_receipt_sha256",
            "monthly_rank_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityOriginV2Error(
                "V2 origin plan semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "acquisition_audit_receipt_sha256": (
                    self.acquisition_audit_receipt_sha256
                ),
                "monthly_rank_audit_receipt_sha256": (
                    self.monthly_rank_audit_receipt_sha256
                ),
                "origin_plan_v1_audit_receipt_sha256": (
                    self.origin_plan_v1.audit_receipt_sha256
                ),
            }
        ):
            raise MassiveProfitabilityOriginV2Error("V2 origin plan audit differs")


def build_massive_profitability_decision_origin_plan_v2(
    *,
    root: str | Path,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    acquisition: MassiveProfitabilityProductionAcquisitionV2,
    source_evidence_artifact: MassiveProfitabilityAcquiredSourceEvidenceArtifactV1,
    monthly_rank_authority: MassiveMonthlyRankInputAuthorityV2,
    daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    first_candidate_decision_session_date: str,
    last_candidate_decision_session_date: str,
) -> MassiveProfitabilityDecisionOriginPlanV2:
    """Bind the V1 chronology to production GETs and exact t-1 ranks."""

    validate_massive_profitability_production_acquisition_v2(
        root=root, acquisition=acquisition, require_fixed_runtime=True
    )
    monthly_rank_authority.validate()
    expected_rank_authority = build_massive_monthly_rank_input_authority_v2(
        root=root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        acquisition=acquisition,
        daily_bars=daily_bars,
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    )
    if monthly_rank_authority != expected_rank_authority:
        raise MassiveProfitabilityOriginV2Error(
            "V2 monthly rank authority was not independently rederived"
        )
    if (
        not monthly_rank_authority.source_data_qualified
        or monthly_rank_authority.acquisition_receipt_sha256
        != acquisition.receipt_sha256
        or monthly_rank_authority.first_candidate_decision_session_date
        != first_candidate_decision_session_date
        or monthly_rank_authority.last_candidate_decision_session_date
        != last_candidate_decision_session_date
    ):
        raise MassiveProfitabilityOriginV2Error(
            "V2 plan requires production-qualified rank inputs for its exact interval"
        )
    source_receipts = {
        row.loaded_source_receipt_sha256 for row in source_evidence_artifact.rows
    }
    selected_downloads = tuple(
        row
        for row in acquisition.authenticated_downloads
        if row.loaded_source.receipt_sha256 in source_receipts
    )
    if len(selected_downloads) != len(source_receipts):
        raise MassiveProfitabilityOriginV2Error(
            "source artifact is not backed by authenticated production GETs"
        )
    v1_plan = build_massive_profitability_decision_origin_plan_v1(
        root=root,
        session_authority=session_authority,
        identity_authority=identity_authority,
        source_evidence_artifact=source_evidence_artifact,
        captured_listings=acquisition.captured_listings,
        loaded_trade_sources=tuple(row.loaded_source for row in selected_downloads),
        first_candidate_decision_session_date=(first_candidate_decision_session_date),
        last_candidate_decision_session_date=last_candidate_decision_session_date,
    )
    rank_months = tuple(row.calendar_month for row in monthly_rank_authority.groups)
    origin_months = tuple(
        sorted({value[:7] for value in v1_plan.candidate_decision_session_dates})
    )
    if rank_months != origin_months:
        raise MassiveProfitabilityOriginV2Error(
            "V2 monthly rank groups do not cover the complete origin interval"
        )
    semantic_body = {
        "schema": MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA,
        "origin_plan_v1_semantic_receipt_sha256": v1_plan.semantic_receipt_sha256,
        "production_acquisition_receipt_sha256": acquisition.receipt_sha256,
        "monthly_rank_authority_semantic_receipt_sha256": (
            monthly_rank_authority.semantic_receipt_sha256
        ),
        "protocol_receipt_sha256": (MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        "specification_sha256": MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        "implementation_source_sha256": (MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256),
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic_body)
    acquisition_audit = semantic_sha256(
        {
            "acquisition_receipt_sha256": acquisition.receipt_sha256,
            "listing_inventory_sha256": acquisition.listing_inventory_sha256,
            "download_inventory_sha256": acquisition.download_inventory_sha256,
            "fixed_runtime_capture_receipt_sha256": (
                acquisition.fixed_runtime_capture_receipt_sha256
            ),
        }
    )
    audit_receipt = semantic_sha256(
        {
            "semantic_receipt_sha256": semantic_receipt,
            "acquisition_audit_receipt_sha256": acquisition_audit,
            "monthly_rank_audit_receipt_sha256": (
                monthly_rank_authority.audit_receipt_sha256
            ),
            "origin_plan_v1_audit_receipt_sha256": v1_plan.audit_receipt_sha256,
        }
    )
    result = MassiveProfitabilityDecisionOriginPlanV2(
        origin_plan_v1=v1_plan,
        production_acquisition_receipt_sha256=acquisition.receipt_sha256,
        monthly_rank_authority_semantic_receipt_sha256=(
            monthly_rank_authority.semantic_receipt_sha256
        ),
        protocol_receipt_sha256=(MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256),
        specification_sha256=MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256,
        implementation_source_sha256=(MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256),
        semantic_receipt_sha256=semantic_receipt,
        acquisition_audit_receipt_sha256=acquisition_audit,
        monthly_rank_audit_receipt_sha256=(monthly_rank_authority.audit_receipt_sha256),
        audit_receipt_sha256=audit_receipt,
        panel_materialization_authorized=False,
        predictive_training_authorized=False,
        profitability_reporting_authorized=False,
        lockbox_access_authorized=False,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_MONTHLY_RANK_INPUT_AUTHORITY_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_ORIGIN_PLAN_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_ORIGIN_V2_SPEC_SHA256",
    "MASSIVE_PROFITABILITY_PRODUCTION_ACQUISITION_V2_SCHEMA",
    "MassiveMonthlyRankInputAuthorityV2",
    "MassiveMonthlyRankInputGroupV2",
    "MassiveProfitabilityDecisionOriginPlanV2",
    "MassiveProfitabilityOriginV2Error",
    "MassiveProfitabilityProductionAcquisitionV2",
    "build_massive_monthly_rank_input_authority_v2",
    "build_massive_profitability_acquisition_for_test_v2",
    "build_massive_profitability_decision_origin_plan_v2",
    "capture_massive_profitability_production_acquisition_v2",
    "materialize_massive_profitability_acquired_source_evidence_from_acquisition_v2",
    "validate_massive_profitability_production_acquisition_v2",
]
