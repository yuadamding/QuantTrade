"""Production-clock wrapper for the finalized V0 typed source-to-order canary."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_archive_scope import (
    MassiveFinalizedArchiveScopeV2,
)
from rl_quant.data_sources.massive.finalized_execution_authority import (
    MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
    MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
    MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
    MassiveExecutionClockAuthorityV1,
    MassiveInputAvailabilityAuthorityV1,
    MassiveTypedExecutionEnvironmentAuthorityV1,
    parse_massive_execution_clock_authority_v1,
    parse_massive_input_availability_authority_v1,
    parse_massive_typed_execution_environment_v1,
)
from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MassiveCapturedFlatFileListingV0,
    validate_massive_captured_flat_file_listing_v0,
)
from rl_quant.data_sources.massive.finalized_origin_policy import (
    MASSIVE_FINALIZED_ORIGIN_POLICY_V3,
    MASSIVE_FINALIZED_ORIGIN_POLICY_V5,
)
from rl_quant.data_sources.massive.finalized_partition_manifest import (
    MassiveFinalizedFeatureDomainSpecV0,
)
from rl_quant.data_sources.massive.finalized_runtime_authority import (
    MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
    MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
    MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
    MassiveExecutionClockAuthorityV2,
    MassiveHostExecutionAuthorityV2,
    MassiveRuntimeExecutionEnvironmentAuthorityV2,
    capture_massive_execution_clock_authority_v2,
    capture_massive_host_execution_authority_v2,
    capture_massive_runtime_execution_environment_v2,
)
from rl_quant.data_sources.massive.session_calendar import (
    MassiveExchangeSession,
    MassiveSessionAuthority,
)
from rl_quant.evaluation.massive_validation_inference_v1 import (
    MassiveValidationCheckpointV1,
    parse_massive_validation_checkpoint_v1,
)
from rl_quant.features.massive_daily_bars_v0 import MassiveDailyBarsArtifactV0
from rl_quant.features.massive_daily_tape_v0 import MassiveDailyTapeArtifactV0
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
)
from rl_quant.workflows.massive_measured_typed_run_v1 import (
    MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1,
    MASSIVE_MEASURED_TYPED_STAGE_IDS_V1,
    MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1,
    MassiveMeasuredTypedRunV1,
    measure_massive_typed_finalized_run_for_test_v1,
)

MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V2 = semantic_sha256(
    (
        MASSIVE_MEASURED_TYPED_STAGE_IDS_V1,
        MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1,
        MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1,
    )
)
MASSIVE_PRODUCTION_TYPED_RUN_V2_SCHEMA = "rl-quant.massive-production-typed-run-v2"
MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "engine": "measured-typed-run-v1-development-engine",
        "wall_clock": "non-injectable-time.time_ns",
        "monotonic_clock": "non-injectable-time.perf_counter_ns",
        "clock_authority": MASSIVE_EXECUTION_CLOCK_V1_SPEC_SHA256,
        "execution_environment": MASSIVE_EXECUTION_ENVIRONMENT_V1_SPEC_SHA256,
        "input_availability": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "clock_deadline_rule": "utc-upper-bound-at-finish<=decision",
        "input_rule": "utc-earliest-run-start-after-complete-input-inventory",
        "checkpoint": "independently-reparsed-from-committed-bytes",
        "implementation_inventory": (
            MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V2
        ),
        "origin_policy": "frozen-v3-receipt-field",
        "maximum_runtime_ms": 55 * 60 * 1_000,
        "performance_authorization": False,
    }
)
MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V3 = semantic_sha256(
    (
        MASSIVE_MEASURED_TYPED_STAGE_IDS_V1,
        MASSIVE_MEASURED_TYPED_STAGE_IMPLEMENTATIONS_V1,
        MASSIVE_MEASURED_TYPED_STAGE_CONFIGURATION_V1,
        MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
        MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
    )
)
MASSIVE_PRODUCTION_TYPED_RUN_V3_SCHEMA = "rl-quant.massive-production-typed-run-v3"
MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol_receipt": MASSIVE_FINALIZED_VALIDATION_V0_RECEIPT_SHA256,
        "engine": "measured-typed-run-v1-development-engine",
        "host_authority": MASSIVE_HOST_EXECUTION_V2_SPEC_SHA256,
        "clock_authority": MASSIVE_EXECUTION_CLOCK_V2_SPEC_SHA256,
        "execution_environment": MASSIVE_RUNTIME_ENVIRONMENT_V2_SPEC_SHA256,
        "authority_capture": "fixed-runtime-entry-point-before-outer-timer",
        "wall_clock": "non-injectable-time.time_ns",
        "monotonic_clock": "non-injectable-time.perf_counter_ns",
        "input_availability": MASSIVE_INPUT_AVAILABILITY_V1_SPEC_SHA256,
        "clock_deadline_rule": "utc-upper-bound-at-finish<=decision",
        "host_rule": "clock-host==environment-host==executing-host",
        "checkpoint": "independently-reparsed-from-committed-bytes",
        "implementation_inventory": (
            MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V3
        ),
        "origin_policy": "frozen-v5-receipt-field",
        "maximum_runtime_ms": 55 * 60 * 1_000,
        "historical_capability": False,
        "performance_authorization": False,
    }
)
class MassiveProductionTypedRunV2Error(ValueError):
    """Production timing evidence differs from real-clock committed inputs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProductionTypedRunV2Error(f"{name} must be a lowercase SHA-256")
    return value


def _wall_ms() -> int:
    return time.time_ns() // 1_000_000


def _authority_available_at_ms(
    authority: PITSecurityUniverseAuthority,
) -> int:
    values = tuple(
        row.available_at_ms
        for rows in (
            authority.security_master,
            authority.ticker_history,
            authority.listing_events,
            authority.delisting_events,
            authority.rank_inputs,
            authority.membership_events,
        )
        for row in rows
        if hasattr(row, "available_at_ms")
    )
    if not values:
        raise MassiveProductionTypedRunV2Error(
            "identity authority has no availability timestamps"
        )
    return max(values)


def _required_input_rows(
    *,
    captured_listing: MassiveCapturedFlatFileListingV0,
    archive_scope: MassiveFinalizedArchiveScopeV2,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV1,
    clock_authority: MassiveExecutionClockAuthorityV1,
    execution_environment: MassiveTypedExecutionEnvironmentAuthorityV1,
) -> dict[str, tuple[str, str, int | None]]:
    expected: dict[str, tuple[str, str, int | None]] = {
        "archive-scope": (
            archive_scope.receipt_sha256,
            semantic_sha256(archive_scope.captured_listing_receipts),
            max(
                captured_listing.loaded_acquisition.verified_at_ms,
                captured_listing.loaded_listing.verified_at_ms,
            ),
        ),
        "captured-listing-acquisition": (
            captured_listing.acquisition_evidence.receipt_sha256,
            captured_listing.loaded_acquisition.receipt.receipt_sha256,
            captured_listing.loaded_acquisition.verified_at_ms,
        ),
        "captured-listing": (
            captured_listing.committed_listing.receipt_sha256,
            captured_listing.loaded_listing.receipt.receipt_sha256,
            captured_listing.loaded_listing.verified_at_ms,
        ),
        "checkpoint": (
            checkpoint.receipt_sha256,
            checkpoint.loaded_source.receipt.receipt_sha256,
            checkpoint.loaded_source.verified_at_ms,
        ),
        "clock-authority": (
            clock_authority.receipt_sha256,
            clock_authority.loaded_source.receipt.receipt_sha256,
            clock_authority.loaded_source.verified_at_ms,
        ),
        "condition-authority": (
            condition_authority.receipt_sha256,
            condition_authority.source_object_receipt_sha256,
            None,
        ),
        "correction-authority": (
            correction_authority.receipt_sha256,
            correction_authority.canary_receipt_sha256,
            None,
        ),
        "execution-environment": (
            execution_environment.receipt_sha256,
            execution_environment.loaded_source.receipt.receipt_sha256,
            execution_environment.loaded_source.verified_at_ms,
        ),
        "feature-domain": (
            feature_domain_spec.receipt_sha256,
            semantic_sha256(
                (
                    feature_domain_spec.condition_authority_receipt_sha256,
                    feature_domain_spec.correction_authority_receipt_sha256,
                )
            ),
            None,
        ),
        "identity-authority": (
            identity_authority.receipt_sha256,
            identity_authority.receipt_sha256,
            _authority_available_at_ms(identity_authority),
        ),
        "session-authority": (
            session_authority.receipt_sha256,
            session_authority.calendar_source_receipt_sha256,
            None,
        ),
    }
    for bars_artifact in prior_daily_bars:
        key = f"prior-daily-bars:{bars_artifact.source_session_date}"
        if key in expected:
            raise MassiveProductionTypedRunV2Error(
                "prior daily bars contain a duplicate source session"
            )
        expected[key] = (
            bars_artifact.receipt_sha256,
            bars_artifact.loaded_source.receipt.receipt_sha256,
            bars_artifact.loaded_source.verified_at_ms,
        )
    for tape_artifact in prior_daily_tape:
        key = f"prior-daily-tape:{tape_artifact.source_session_date}"
        if key in expected:
            raise MassiveProductionTypedRunV2Error(
                "prior daily tape contains a duplicate source session"
            )
        expected[key] = (
            tape_artifact.receipt_sha256,
            tape_artifact.loaded_source.receipt.receipt_sha256,
            tape_artifact.loaded_source.verified_at_ms,
        )
    return expected


def _validate_input_inventory(
    *,
    authority: MassiveInputAvailabilityAuthorityV1,
    expected: dict[str, tuple[str, str, int | None]],
    latest_allowed_at_ms: int,
) -> None:
    authority.validate()
    if tuple(row.input_kind for row in authority.rows) != tuple(sorted(expected)):
        raise MassiveProductionTypedRunV2Error(
            "pre-existing input availability inventory differs"
        )
    for row in authority.rows:
        artifact, evidence, derived_at = expected[row.input_kind]
        if (
            row.artifact_receipt_sha256 != artifact
            or row.evidence_receipt_sha256 != evidence
            or (derived_at is not None and row.available_at_ms != derived_at)
            or row.available_at_ms > latest_allowed_at_ms
        ):
            raise MassiveProductionTypedRunV2Error(
                f"pre-existing input availability differs for {row.input_kind}"
            )


def _required_input_rows_v3(
    *,
    captured_listing: MassiveCapturedFlatFileListingV0,
    archive_scope: MassiveFinalizedArchiveScopeV2,
    session_authority: MassiveSessionAuthority,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV1,
) -> dict[str, tuple[str, str, int | None]]:
    """Inputs that must predate V3; host, clock, and environment are captured inside it."""

    expected: dict[str, tuple[str, str, int | None]] = {
        "archive-scope": (
            archive_scope.receipt_sha256,
            semantic_sha256(archive_scope.captured_listing_receipts),
            max(
                captured_listing.loaded_acquisition.verified_at_ms,
                captured_listing.loaded_listing.verified_at_ms,
            ),
        ),
        "captured-listing-acquisition": (
            captured_listing.acquisition_evidence.receipt_sha256,
            captured_listing.loaded_acquisition.receipt.receipt_sha256,
            captured_listing.loaded_acquisition.verified_at_ms,
        ),
        "captured-listing": (
            captured_listing.committed_listing.receipt_sha256,
            captured_listing.loaded_listing.receipt.receipt_sha256,
            captured_listing.loaded_listing.verified_at_ms,
        ),
        "checkpoint": (
            checkpoint.receipt_sha256,
            checkpoint.loaded_source.receipt.receipt_sha256,
            checkpoint.loaded_source.verified_at_ms,
        ),
        "condition-authority": (
            condition_authority.receipt_sha256,
            condition_authority.source_object_receipt_sha256,
            None,
        ),
        "correction-authority": (
            correction_authority.receipt_sha256,
            correction_authority.canary_receipt_sha256,
            None,
        ),
        "feature-domain": (
            feature_domain_spec.receipt_sha256,
            semantic_sha256(
                (
                    feature_domain_spec.condition_authority_receipt_sha256,
                    feature_domain_spec.correction_authority_receipt_sha256,
                )
            ),
            None,
        ),
        "identity-authority": (
            identity_authority.receipt_sha256,
            identity_authority.receipt_sha256,
            _authority_available_at_ms(identity_authority),
        ),
        "session-authority": (
            session_authority.receipt_sha256,
            session_authority.calendar_source_receipt_sha256,
            None,
        ),
    }
    for bars_artifact in prior_daily_bars:
        key = f"prior-daily-bars:{bars_artifact.source_session_date}"
        if key in expected:
            raise MassiveProductionTypedRunV2Error(
                "prior daily bars contain a duplicate source session"
            )
        expected[key] = (
            bars_artifact.receipt_sha256,
            bars_artifact.loaded_source.receipt.receipt_sha256,
            bars_artifact.loaded_source.verified_at_ms,
        )
    for tape_artifact in prior_daily_tape:
        key = f"prior-daily-tape:{tape_artifact.source_session_date}"
        if key in expected:
            raise MassiveProductionTypedRunV2Error(
                "prior daily tape contains a duplicate source session"
            )
        expected[key] = (
            tape_artifact.receipt_sha256,
            tape_artifact.loaded_source.receipt.receipt_sha256,
            tape_artifact.loaded_source.verified_at_ms,
        )
    return expected


@dataclass(frozen=True, slots=True)
class MassiveProductionTypedRunV2:
    development_engine_run: MassiveMeasuredTypedRunV1
    clock_authority: MassiveExecutionClockAuthorityV1
    execution_environment: MassiveTypedExecutionEnvironmentAuthorityV1
    input_availability_authority: MassiveInputAvailabilityAuthorityV1
    checkpoint_receipt_sha256: str
    stage_authority_inventory_sha256: str
    outer_started_at_ms: int
    outer_finished_at_ms: int
    outer_started_monotonic_ns: int
    outer_finished_monotonic_ns: int
    runtime_ms: int
    timing_source_kind: str
    origin_policy_receipt_sha256: str
    production_run_spec_receipt_sha256: str
    production_timing_qualified: bool
    historical_availability_qualified: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_PRODUCTION_TYPED_RUN_V2_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.development_engine_run.validate()
        self.clock_authority.validate()
        self.execution_environment.validate()
        self.input_availability_authority.validate()
        engine = self.development_engine_run
        if (
            self.schema != MASSIVE_PRODUCTION_TYPED_RUN_V2_SCHEMA
            or engine.typed_timing_qualified
            or self.timing_source_kind != "production-system-clocks"
            or self.production_run_spec_receipt_sha256
            != MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256
            or self.origin_policy_receipt_sha256
            != MASSIVE_FINALIZED_ORIGIN_POLICY_V3.receipt_sha256
            or self.execution_environment.pipeline_implementation_inventory_sha256
            != MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V2
            or self.outer_started_at_ms > engine.outer_started_at_ms
            or self.outer_finished_at_ms < engine.outer_finished_at_ms
            or self.outer_started_monotonic_ns > engine.outer_started_monotonic_ns
            or self.outer_finished_monotonic_ns < engine.outer_finished_monotonic_ns
            or self.runtime_ms
            != (self.outer_finished_monotonic_ns - self.outer_started_monotonic_ns)
            // 1_000_000
            or self.runtime_ms > 55 * 60 * 1_000
            or not self.production_timing_qualified
            or self.historical_availability_qualified
            or self.panel_materialization_authorized
            or self.predictive_training_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveProductionTypedRunV2Error(
                "production typed run qualification differs"
            )
        if (
            self.clock_authority.utc_lower_bound_ms(self.outer_started_at_ms)
            < self.clock_authority.measurement_observed_at_ms
            or self.clock_authority.utc_upper_bound_ms(self.outer_finished_at_ms)
            > self.clock_authority.qualification_valid_until_ms
            or self.clock_authority.utc_upper_bound_ms(self.outer_finished_at_ms)
            > engine.decision_origin.decision_at_ms
            or self.input_availability_authority.loaded_source.verified_at_ms
            > self.clock_authority.utc_lower_bound_ms(self.outer_started_at_ms)
        ):
            raise MassiveProductionTypedRunV2Error(
                "production clock uncertainty escaped the qualified interval"
            )
        expected_stage_inventory = semantic_sha256(
            tuple(
                (
                    stage.receipt_sha256,
                    self.clock_authority.receipt_sha256,
                    self.execution_environment.receipt_sha256,
                )
                for stage in engine.stages
            )
        )
        if self.stage_authority_inventory_sha256 != expected_stage_inventory:
            raise MassiveProductionTypedRunV2Error(
                "production stage authority inventory differs"
            )
        if (
            self.checkpoint_receipt_sha256 != engine.checkpoint.receipt_sha256
            or engine.inference.setting_id != engine.checkpoint.setting_id
            or engine.inference.seed != engine.checkpoint.seed
            or engine.requested_orders.setting_id != engine.inference.setting_id
            or engine.requested_orders.seed != engine.inference.seed
            or engine.requested_orders.tensor_receipt_sha256
            != engine.decision_tensor.receipt_sha256
            or engine.requested_orders.decision_origin_receipt_sha256
            != engine.decision_origin.receipt_sha256
            or engine.requested_orders.decision_session_date
            != engine.decision_origin.decision_session_date
            or engine.requested_orders.decision_at_ms
            != engine.decision_origin.decision_at_ms
        ):
            raise MassiveProductionTypedRunV2Error(
                "production typed run output chain differs"
            )
        for name in (
            "checkpoint_receipt_sha256",
            "stage_authority_inventory_sha256",
            "production_run_spec_receipt_sha256",
            "origin_policy_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProductionTypedRunV2Error(
                "production typed run receipt differs"
            )


def measure_massive_typed_finalized_run_production_v2(
    *,
    s3_client: Any,
    captured_listing: MassiveCapturedFlatFileListingV0,
    listing_root: str | Path,
    archive_scope: MassiveFinalizedArchiveScopeV2,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV1,
    checkpoint_root: str | Path,
    clock_authority: MassiveExecutionClockAuthorityV1,
    clock_root: str | Path,
    execution_environment: MassiveTypedExecutionEnvironmentAuthorityV1,
    execution_environment_root: str | Path,
    input_availability_authority: MassiveInputAvailabilityAuthorityV1,
    input_availability_root: str | Path,
    source_root: str | Path,
    spool_root: str | Path,
    persisted_root: str | Path,
    artifact_root: str | Path,
    entitlement_receipt_sha256: str,
) -> MassiveProductionTypedRunV2:
    """Run once with real process clocks and committed timing authorities."""

    outer_started_at_ms = _wall_ms()
    outer_started_monotonic_ns = time.perf_counter_ns()
    expected_clock = parse_massive_execution_clock_authority_v1(
        root=clock_root, loaded_source=clock_authority.loaded_source
    )
    expected_environment = parse_massive_typed_execution_environment_v1(
        root=execution_environment_root,
        loaded_source=execution_environment.loaded_source,
    )
    expected_availability = parse_massive_input_availability_authority_v1(
        root=input_availability_root,
        loaded_source=input_availability_authority.loaded_source,
    )
    expected_checkpoint = parse_massive_validation_checkpoint_v1(
        root=checkpoint_root, loaded_source=checkpoint.loaded_source
    )
    if (
        expected_clock != clock_authority
        or expected_environment != execution_environment
        or expected_availability != input_availability_authority
        or expected_checkpoint != checkpoint
    ):
        raise MassiveProductionTypedRunV2Error(
            "production input was not rederived from committed bytes"
        )
    validate_massive_captured_flat_file_listing_v0(
        root=listing_root, captured_listing=captured_listing
    )
    earliest_start_at_ms = clock_authority.utc_lower_bound_ms(outer_started_at_ms)
    expected_inputs = _required_input_rows(
        captured_listing=captured_listing,
        archive_scope=archive_scope,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        feature_domain_spec=feature_domain_spec,
        prior_daily_bars=prior_daily_bars,
        prior_daily_tape=prior_daily_tape,
        checkpoint=checkpoint,
        clock_authority=clock_authority,
        execution_environment=execution_environment,
    )
    _validate_input_inventory(
        authority=input_availability_authority,
        expected=expected_inputs,
        latest_allowed_at_ms=earliest_start_at_ms,
    )
    if execution_environment.pipeline_implementation_inventory_sha256 != (
        MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V2
    ):
        raise MassiveProductionTypedRunV2Error(
            "production pipeline implementation inventory differs"
        )
    engine = measure_massive_typed_finalized_run_for_test_v1(
        s3_client=s3_client,
        captured_listing=captured_listing,
        archive_scope=archive_scope,
        session_authority=session_authority,
        source_session=source_session,
        decision_session=decision_session,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        feature_domain_spec=feature_domain_spec,
        prior_daily_bars=prior_daily_bars,
        prior_daily_tape=prior_daily_tape,
        checkpoint=checkpoint,
        source_root=source_root,
        spool_root=spool_root,
        persisted_root=persisted_root,
        artifact_root=artifact_root,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        now_ms=_wall_ms,
        monotonic_ns=time.perf_counter_ns,
    )
    outer_finished_at_ms = _wall_ms()
    outer_finished_monotonic_ns = time.perf_counter_ns()
    stage_inventory = semantic_sha256(
        tuple(
            (
                stage.receipt_sha256,
                clock_authority.receipt_sha256,
                execution_environment.receipt_sha256,
            )
            for stage in engine.stages
        )
    )
    body = {
        "schema": MASSIVE_PRODUCTION_TYPED_RUN_V2_SCHEMA,
        "development_engine_run": engine,
        "clock_authority": clock_authority,
        "execution_environment": execution_environment,
        "input_availability_authority": input_availability_authority,
        "checkpoint_receipt_sha256": checkpoint.receipt_sha256,
        "stage_authority_inventory_sha256": stage_inventory,
        "outer_started_at_ms": outer_started_at_ms,
        "outer_finished_at_ms": outer_finished_at_ms,
        "outer_started_monotonic_ns": outer_started_monotonic_ns,
        "outer_finished_monotonic_ns": outer_finished_monotonic_ns,
        "runtime_ms": (outer_finished_monotonic_ns - outer_started_monotonic_ns)
        // 1_000_000,
        "timing_source_kind": "production-system-clocks",
        "origin_policy_receipt_sha256": (
            MASSIVE_FINALIZED_ORIGIN_POLICY_V3.receipt_sha256
        ),
        "production_run_spec_receipt_sha256": MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256,
        "production_timing_qualified": True,
        "historical_availability_qualified": False,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    provisional = MassiveProductionTypedRunV2(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,  # type: ignore[arg-type]
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


class MassiveProductionTypedRunV3Error(ValueError):
    """Source-derived production host, clock, or environment evidence differs."""


def validate_massive_production_clock_interval_v3(
    *,
    clock_authority: MassiveExecutionClockAuthorityV2,
    outer_started_at_ms: int,
    outer_finished_at_ms: int,
    host_capture_finished_at_ms: int,
    environment_capture_finished_at_ms: int,
    clock_source_verified_at_ms: int,
    input_inventory_verified_at_ms: int,
    decision_at_ms: int,
) -> tuple[int, int]:
    """Validate the full run against two-sided wall-clock uncertainty bounds."""

    clock_authority.validate()
    earliest_start = clock_authority.utc_lower_bound_ms(outer_started_at_ms)
    latest_finish = clock_authority.utc_upper_bound_ms(outer_finished_at_ms)
    if (
        clock_authority.measurement_utc_upper_bound_ms > earliest_start
        or latest_finish > clock_authority.qualification_end_utc_lower_bound_ms
        or latest_finish > decision_at_ms
        or clock_authority.utc_upper_bound_ms(host_capture_finished_at_ms)
        > earliest_start
        or clock_authority.utc_upper_bound_ms(environment_capture_finished_at_ms)
        > earliest_start
        or clock_authority.utc_upper_bound_ms(clock_source_verified_at_ms)
        > earliest_start
        or input_inventory_verified_at_ms > earliest_start
    ):
        raise MassiveProductionTypedRunV3Error(
            "source-derived timing evidence escaped the qualified interval"
        )
    return earliest_start, latest_finish


@dataclass(frozen=True, slots=True)
class MassiveProductionTypedRunV3:
    development_engine_run: MassiveMeasuredTypedRunV1
    host_authority: MassiveHostExecutionAuthorityV2
    clock_authority: MassiveExecutionClockAuthorityV2
    execution_environment: MassiveRuntimeExecutionEnvironmentAuthorityV2
    input_availability_authority: MassiveInputAvailabilityAuthorityV1
    checkpoint_receipt_sha256: str
    stage_authority_inventory_sha256: str
    outer_started_at_ms: int
    outer_finished_at_ms: int
    outer_started_monotonic_ns: int
    outer_finished_monotonic_ns: int
    runtime_ms: int
    timing_source_kind: str
    origin_policy_receipt_sha256: str
    production_run_spec_receipt_sha256: str
    production_timing_qualified: bool
    historical_capability_authorized: bool
    historical_availability_qualified: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    portfolio_evaluation_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_PRODUCTION_TYPED_RUN_V3_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        self.development_engine_run.validate()
        self.host_authority.validate()
        self.clock_authority.validate()
        self.execution_environment.validate()
        self.input_availability_authority.validate()
        engine = self.development_engine_run
        if (
            self.schema != MASSIVE_PRODUCTION_TYPED_RUN_V3_SCHEMA
            or engine.typed_timing_qualified
            or self.timing_source_kind
            != "fixed-host-chrony-and-runtime-environment-capture"
            or self.production_run_spec_receipt_sha256
            != MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256
            or self.origin_policy_receipt_sha256
            != MASSIVE_FINALIZED_ORIGIN_POLICY_V5.receipt_sha256
            or self.execution_environment.pipeline_implementation_inventory_sha256
            != MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V3
            or not self.host_authority.captured_by_fixed_runtime
            or not self.clock_authority.captured_by_fixed_runtime
            or not self.execution_environment.captured_by_fixed_runtime
            or self.clock_authority.host_authority_receipt_sha256
            != self.host_authority.receipt_sha256
            or self.execution_environment.host_authority_receipt_sha256
            != self.host_authority.receipt_sha256
            or self.outer_started_at_ms > engine.outer_started_at_ms
            or self.outer_finished_at_ms < engine.outer_finished_at_ms
            or self.outer_started_monotonic_ns > engine.outer_started_monotonic_ns
            or self.outer_finished_monotonic_ns < engine.outer_finished_monotonic_ns
            or self.runtime_ms
            != (self.outer_finished_monotonic_ns - self.outer_started_monotonic_ns)
            // 1_000_000
            or self.runtime_ms > 55 * 60 * 1_000
            or not self.production_timing_qualified
            or self.historical_capability_authorized
            or self.historical_availability_qualified
            or self.panel_materialization_authorized
            or self.predictive_training_authorized
            or self.portfolio_evaluation_authorized
        ):
            raise MassiveProductionTypedRunV3Error(
                "production typed run v3 qualification differs"
            )
        validate_massive_production_clock_interval_v3(
            clock_authority=self.clock_authority,
            outer_started_at_ms=self.outer_started_at_ms,
            outer_finished_at_ms=self.outer_finished_at_ms,
            host_capture_finished_at_ms=self.host_authority.capture_finished_at_ms,
            environment_capture_finished_at_ms=(
                self.execution_environment.capture_finished_at_ms
            ),
            clock_source_verified_at_ms=(
                self.clock_authority.loaded_source.verified_at_ms
            ),
            input_inventory_verified_at_ms=(
                self.input_availability_authority.loaded_source.verified_at_ms
            ),
            decision_at_ms=engine.decision_origin.decision_at_ms,
        )
        expected_stage_inventory = semantic_sha256(
            tuple(
                (
                    stage.receipt_sha256,
                    self.host_authority.receipt_sha256,
                    self.clock_authority.receipt_sha256,
                    self.execution_environment.receipt_sha256,
                )
                for stage in engine.stages
            )
        )
        if self.stage_authority_inventory_sha256 != expected_stage_inventory:
            raise MassiveProductionTypedRunV3Error(
                "production v3 stage authority inventory differs"
            )
        if (
            self.checkpoint_receipt_sha256 != engine.checkpoint.receipt_sha256
            or engine.inference.setting_id != engine.checkpoint.setting_id
            or engine.inference.seed != engine.checkpoint.seed
            or engine.requested_orders.setting_id != engine.inference.setting_id
            or engine.requested_orders.seed != engine.inference.seed
            or engine.requested_orders.tensor_receipt_sha256
            != engine.decision_tensor.receipt_sha256
            or engine.requested_orders.decision_origin_receipt_sha256
            != engine.decision_origin.receipt_sha256
            or engine.requested_orders.decision_session_date
            != engine.decision_origin.decision_session_date
            or engine.requested_orders.decision_at_ms
            != engine.decision_origin.decision_at_ms
        ):
            raise MassiveProductionTypedRunV3Error(
                "production v3 typed output chain differs"
            )
        for name in (
            "checkpoint_receipt_sha256",
            "stage_authority_inventory_sha256",
            "origin_policy_receipt_sha256",
            "production_run_spec_receipt_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProductionTypedRunV3Error(
                "production typed run v3 receipt differs"
            )


def measure_massive_typed_finalized_run_production_v3(
    *,
    s3_client: Any,
    captured_listing: MassiveCapturedFlatFileListingV0,
    listing_root: str | Path,
    archive_scope: MassiveFinalizedArchiveScopeV2,
    session_authority: MassiveSessionAuthority,
    source_session: MassiveExchangeSession,
    decision_session: MassiveExchangeSession,
    identity_authority: PITSecurityUniverseAuthority,
    condition_authority: MassiveConditionAuthority,
    correction_authority: MassiveCorrectionAuthority,
    feature_domain_spec: MassiveFinalizedFeatureDomainSpecV0,
    prior_daily_bars: Sequence[MassiveDailyBarsArtifactV0],
    prior_daily_tape: Sequence[MassiveDailyTapeArtifactV0],
    checkpoint: MassiveValidationCheckpointV1,
    checkpoint_root: str | Path,
    input_availability_authority: MassiveInputAvailabilityAuthorityV1,
    input_availability_root: str | Path,
    host_capture_root: str | Path,
    clock_capture_root: str | Path,
    environment_capture_root: str | Path,
    storage_roots: dict[str, str | Path],
    source_root: str | Path,
    spool_root: str | Path,
    persisted_root: str | Path,
    artifact_root: str | Path,
    entitlement_receipt_sha256: str,
) -> MassiveProductionTypedRunV3:
    """Capture this host and chrony directly, then time the exact typed pipeline."""

    host_authority = capture_massive_host_execution_authority_v2(
        root=host_capture_root,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    execution_environment = capture_massive_runtime_execution_environment_v2(
        root=environment_capture_root,
        host_authority=host_authority,
        host_root=host_capture_root,
        s3_client=s3_client,
        storage_roots=storage_roots,
        pipeline_implementation_inventory_sha256=(
            MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V3
        ),
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    clock_authority = capture_massive_execution_clock_authority_v2(
        root=clock_capture_root,
        host_authority=host_authority,
        host_root=host_capture_root,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
    )
    outer_started_at_ms = _wall_ms()
    outer_started_monotonic_ns = time.perf_counter_ns()
    expected_availability = parse_massive_input_availability_authority_v1(
        root=input_availability_root,
        loaded_source=input_availability_authority.loaded_source,
    )
    expected_checkpoint = parse_massive_validation_checkpoint_v1(
        root=checkpoint_root,
        loaded_source=checkpoint.loaded_source,
    )
    if (
        expected_availability != input_availability_authority
        or expected_checkpoint != checkpoint
    ):
        raise MassiveProductionTypedRunV3Error(
            "production v3 input was not rederived from committed bytes"
        )
    validate_massive_captured_flat_file_listing_v0(
        root=listing_root,
        captured_listing=captured_listing,
    )
    earliest_start_at_ms = clock_authority.utc_lower_bound_ms(outer_started_at_ms)
    expected_inputs = _required_input_rows_v3(
        captured_listing=captured_listing,
        archive_scope=archive_scope,
        session_authority=session_authority,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        feature_domain_spec=feature_domain_spec,
        prior_daily_bars=prior_daily_bars,
        prior_daily_tape=prior_daily_tape,
        checkpoint=checkpoint,
    )
    _validate_input_inventory(
        authority=input_availability_authority,
        expected=expected_inputs,
        latest_allowed_at_ms=earliest_start_at_ms,
    )
    engine = measure_massive_typed_finalized_run_for_test_v1(
        s3_client=s3_client,
        captured_listing=captured_listing,
        archive_scope=archive_scope,
        session_authority=session_authority,
        source_session=source_session,
        decision_session=decision_session,
        identity_authority=identity_authority,
        condition_authority=condition_authority,
        correction_authority=correction_authority,
        feature_domain_spec=feature_domain_spec,
        prior_daily_bars=prior_daily_bars,
        prior_daily_tape=prior_daily_tape,
        checkpoint=checkpoint,
        source_root=source_root,
        spool_root=spool_root,
        persisted_root=persisted_root,
        artifact_root=artifact_root,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        now_ms=_wall_ms,
        monotonic_ns=time.perf_counter_ns,
    )
    outer_finished_at_ms = _wall_ms()
    outer_finished_monotonic_ns = time.perf_counter_ns()
    stage_inventory = semantic_sha256(
        tuple(
            (
                stage.receipt_sha256,
                host_authority.receipt_sha256,
                clock_authority.receipt_sha256,
                execution_environment.receipt_sha256,
            )
            for stage in engine.stages
        )
    )
    body = {
        "schema": MASSIVE_PRODUCTION_TYPED_RUN_V3_SCHEMA,
        "development_engine_run": engine,
        "host_authority": host_authority,
        "clock_authority": clock_authority,
        "execution_environment": execution_environment,
        "input_availability_authority": input_availability_authority,
        "checkpoint_receipt_sha256": checkpoint.receipt_sha256,
        "stage_authority_inventory_sha256": stage_inventory,
        "outer_started_at_ms": outer_started_at_ms,
        "outer_finished_at_ms": outer_finished_at_ms,
        "outer_started_monotonic_ns": outer_started_monotonic_ns,
        "outer_finished_monotonic_ns": outer_finished_monotonic_ns,
        "runtime_ms": (
            outer_finished_monotonic_ns - outer_started_monotonic_ns
        )
        // 1_000_000,
        "timing_source_kind": "fixed-host-chrony-and-runtime-environment-capture",
        "origin_policy_receipt_sha256": (
            MASSIVE_FINALIZED_ORIGIN_POLICY_V5.receipt_sha256
        ),
        "production_run_spec_receipt_sha256": (
            MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256
        ),
        "production_timing_qualified": True,
        "historical_capability_authorized": False,
        "historical_availability_qualified": False,
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "portfolio_evaluation_authorized": False,
    }
    provisional = MassiveProductionTypedRunV3(
        **body,  # type: ignore[arg-type]
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V2",
    "MASSIVE_PRODUCTION_TYPED_PIPELINE_IMPLEMENTATION_INVENTORY_V3",
    "MASSIVE_PRODUCTION_TYPED_RUN_V2_SPEC_SHA256",
    "MASSIVE_PRODUCTION_TYPED_RUN_V3_SPEC_SHA256",
    "MassiveProductionTypedRunV2",
    "MassiveProductionTypedRunV2Error",
    "MassiveProductionTypedRunV3",
    "MassiveProductionTypedRunV3Error",
    "measure_massive_typed_finalized_run_production_v2",
    "measure_massive_typed_finalized_run_production_v3",
    "validate_massive_production_clock_interval_v3",
]
