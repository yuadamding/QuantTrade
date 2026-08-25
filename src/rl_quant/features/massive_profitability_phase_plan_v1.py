"""Candidate-date phases and purged folds for the Massive P0 experiment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_origin_v2 import (
    MassiveProfitabilityDecisionOriginPlanV2,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)

MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA = (
    "rl-quant.massive-profitability-phase-plan-v1"
)
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_DATASET = "massive-profitability-phase-plan-v1"
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))

_PROTOCOL = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
_OUTER_COUNT = _PROTOCOL.outer_fold_count
_OUTER_SESSIONS = _PROTOCOL.outer_fold_sessions
_OUTER_PURGE = _PROTOCOL.target_overlap_purge_sessions
_INNER_VALIDATION = _PROTOCOL.inner_validation_sessions
_INNER_PURGE = _PROTOCOL.inner_purge_sessions
_MINIMUM_FIT = _PROTOCOL.minimum_initial_training_sessions
_LOCKBOX = _PROTOCOL.historical_lockbox_sessions

MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "phase_basis": "archive-frozen-candidate-session-dates",
        "outer_folds": (_OUTER_COUNT, _OUTER_SESSIONS),
        "outer_purge": _OUTER_PURGE,
        "inner_validation": _INNER_VALIDATION,
        "inner_purge": _INNER_PURGE,
        "minimum_fit": _MINIMUM_FIT,
        "lockbox": _LOCKBOX,
        "phase_shift_after_freeze": "prohibited",
        "confirmation_and_lockbox_skips": "gate-failure-not-date-shift",
        "performance_authorization": False,
    }
)

MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PANEL_MATERIALIZATION_AUTHORIZED = False
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PREDICTIVE_TRAINING_AUTHORIZED = False
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PROFITABILITY_REPORTING_AUTHORIZED = False
MASSIVE_PROFITABILITY_PHASE_PLAN_V1_LOCKBOX_ACCESS_AUTHORIZED = False


class MassiveProfitabilityPhasePlanV1Error(ValueError):
    """Frozen phase geometry or origin coverage differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPhasePlanV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveProfitabilityPhasePlanV1Error(f"{name} must be canonical text")
    return value


def _artifact_id(value: object) -> str:
    result = _text("phase plan artifact ID", value)
    if any(not (character.isalnum() or character in "-_") for character in result):
        raise MassiveProfitabilityPhasePlanV1Error(
            "phase plan artifact ID is not path safe"
        )
    return result


def _canonical_inventory(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or result != tuple(sorted(set(result))):
        raise MassiveProfitabilityPhasePlanV1Error(
            f"{name} must be sorted, unique, and nonempty"
        )
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityOuterFoldPlanV1:
    fold_index: int
    fit_session_dates: tuple[str, ...]
    inner_purge_session_dates: tuple[str, ...]
    inner_validation_session_dates: tuple[str, ...]
    outer_purge_session_dates: tuple[str, ...]
    outer_test_session_dates: tuple[str, ...]
    fit_inventory_sha256: str
    inner_validation_inventory_sha256: str
    outer_test_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or not 0 <= self.fold_index < _OUTER_COUNT
        ):
            raise MassiveProfitabilityPhasePlanV1Error("outer fold index differs")
        fit = _canonical_inventory("fold fit sessions", self.fit_session_dates)
        inner_purge = _canonical_inventory(
            "fold inner purge sessions", self.inner_purge_session_dates
        )
        inner_validation = _canonical_inventory(
            "fold inner validation sessions", self.inner_validation_session_dates
        )
        outer_purge = _canonical_inventory(
            "fold outer purge sessions", self.outer_purge_session_dates
        )
        outer_test = _canonical_inventory(
            "fold outer test sessions", self.outer_test_session_dates
        )
        if (
            len(fit) < _MINIMUM_FIT
            or len(inner_purge) != _INNER_PURGE
            or len(inner_validation) != _INNER_VALIDATION
            or len(outer_purge) != _OUTER_PURGE
            or len(outer_test) != _OUTER_SESSIONS
            or not (
                fit[-1]
                < inner_purge[0]
                < inner_validation[0]
                < outer_purge[0]
                < outer_test[0]
            )
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "outer fold chronology or counts differ"
            )
        union = fit + inner_purge + inner_validation + outer_purge + outer_test
        if len(union) != len(set(union)):
            raise MassiveProfitabilityPhasePlanV1Error("outer fold partitions overlap")
        if (
            self.fit_inventory_sha256 != semantic_sha256(fit)
            or self.inner_validation_inventory_sha256
            != semantic_sha256(inner_validation)
            or self.outer_test_inventory_sha256 != semantic_sha256(outer_test)
        ):
            raise MassiveProfitabilityPhasePlanV1Error("outer fold inventories differ")
        for name in (
            "fit_inventory_sha256",
            "inner_validation_inventory_sha256",
            "outer_test_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityPhasePlanV1Error("outer fold receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPhasePlanV1:
    candidate_session_dates: tuple[str, ...]
    initial_development_session_dates: tuple[str, ...]
    outer_folds: tuple[MassiveProfitabilityOuterFoldPlanV1, ...]
    lockbox_session_dates: tuple[str, ...]
    candidate_inventory_sha256: str
    archive_freeze_semantic_receipt_sha256: str
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    archive_freeze_audit_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    archive_source_transport_qualified: bool
    archive_rank_bar_data_qualified: bool
    panel_materialization_authorized: bool
    predictive_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "candidate_session_dates": self.candidate_session_dates,
            "initial_development_session_dates": (
                self.initial_development_session_dates
            ),
            "outer_folds": tuple(asdict(row) for row in self.outer_folds),
            "lockbox_session_dates": self.lockbox_session_dates,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "archive_freeze_semantic_receipt_sha256": (
                self.archive_freeze_semantic_receipt_sha256
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
        if (
            self.schema != MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SHA256
        ):
            raise MassiveProfitabilityPhasePlanV1Error("phase plan identity differs")
        candidates = _canonical_inventory(
            "phase candidate sessions", self.candidate_session_dates
        )
        development = _canonical_inventory(
            "initial development sessions", self.initial_development_session_dates
        )
        lockbox = _canonical_inventory(
            "phase lockbox sessions", self.lockbox_session_dates
        )
        expected_development_count = (
            len(candidates) - _LOCKBOX - _OUTER_COUNT * _OUTER_SESSIONS
        )
        if (
            len(self.outer_folds) != _OUTER_COUNT
            or tuple(row.fold_index for row in self.outer_folds)
            != tuple(range(_OUTER_COUNT))
            or len(lockbox) != _LOCKBOX
            or development != candidates[:expected_development_count]
            or lockbox != candidates[-_LOCKBOX:]
        ):
            raise MassiveProfitabilityPhasePlanV1Error("phase plan partitions differ")
        for fold in self.outer_folds:
            fold.validate()
        if self.outer_folds != _folds(candidates):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase plan folds do not match the frozen candidate geometry"
            )
        expected_outer = tuple(
            value
            for fold in self.outer_folds
            for value in fold.outer_test_session_dates
        )
        if (
            expected_outer + lockbox
            != candidates[-(len(expected_outer) + len(lockbox)) :]
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase plan outer tests and lockbox are not the candidate tail"
            )
        if (
            not isinstance(self.archive_source_transport_qualified, bool)
            or not isinstance(self.archive_rank_bar_data_qualified, bool)
            or self.archive_rank_bar_data_qualified
            and not self.archive_source_transport_qualified
            or any(
                (
                    self.panel_materialization_authorized,
                    self.predictive_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase qualification or authorization differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SCHEMA_SHA256
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase plan source transaction differs"
            )
        for name in (
            "candidate_inventory_sha256",
            "archive_freeze_semantic_receipt_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
            "archive_freeze_audit_receipt_sha256",
            "audit_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if self.candidate_inventory_sha256 != semantic_sha256(candidates):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase candidate inventory differs"
            )
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase plan semantic receipt differs"
            )
        if self.audit_receipt_sha256 != semantic_sha256(
            {
                "semantic_receipt_sha256": self.semantic_receipt_sha256,
                "archive_freeze_audit_receipt_sha256": (
                    self.archive_freeze_audit_receipt_sha256
                ),
                "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
            }
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "phase plan audit receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPhaseOriginGateV1:
    phase_plan_semantic_receipt_sha256: str
    origin_plan_semantic_receipt_sha256: str
    candidate_status_inventory_sha256: str
    development_skip_inventory: tuple[tuple[str, str], ...]
    outer_test_skip_inventory: tuple[tuple[str, str], ...]
    lockbox_skip_inventory: tuple[tuple[str, str], ...]
    outer_test_complete: bool
    lockbox_complete: bool
    data_gate_passed: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        for name in (
            "phase_plan_semantic_receipt_sha256",
            "origin_plan_semantic_receipt_sha256",
            "candidate_status_inventory_sha256",
            "receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.outer_test_complete is not (not self.outer_test_skip_inventory)
            or self.lockbox_complete is not (not self.lockbox_skip_inventory)
            or self.data_gate_passed is not False
        ):
            raise MassiveProfitabilityPhasePlanV1Error(
                "origin phase gate state differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityPhasePlanV1Error(
                "origin phase gate receipt differs"
            )


def _folds(
    candidates: tuple[str, ...],
) -> tuple[MassiveProfitabilityOuterFoldPlanV1, ...]:
    outer_flat_start = len(candidates) - _LOCKBOX - _OUTER_COUNT * _OUTER_SESSIONS
    if (
        outer_flat_start
        < _MINIMUM_FIT + _INNER_PURGE + _INNER_VALIDATION + _OUTER_PURGE
    ):
        raise MassiveProfitabilityPhasePlanV1Error(
            "candidate history cannot support the first frozen outer fold"
        )
    output = []
    for fold_index in range(_OUTER_COUNT):
        outer_start = outer_flat_start + fold_index * _OUTER_SESSIONS
        outer_test = candidates[outer_start : outer_start + _OUTER_SESSIONS]
        outer_purge = candidates[outer_start - _OUTER_PURGE : outer_start]
        inner_validation_end = outer_start - _OUTER_PURGE
        inner_validation = candidates[
            inner_validation_end - _INNER_VALIDATION : inner_validation_end
        ]
        inner_purge = candidates[
            inner_validation_end
            - _INNER_VALIDATION
            - _INNER_PURGE : inner_validation_end - _INNER_VALIDATION
        ]
        fit = candidates[: inner_validation_end - _INNER_VALIDATION - _INNER_PURGE]
        body: dict[str, object] = {
            "fold_index": fold_index,
            "fit_session_dates": fit,
            "inner_purge_session_dates": inner_purge,
            "inner_validation_session_dates": inner_validation,
            "outer_purge_session_dates": outer_purge,
            "outer_test_session_dates": outer_test,
            "fit_inventory_sha256": semantic_sha256(fit),
            "inner_validation_inventory_sha256": semantic_sha256(inner_validation),
            "outer_test_inventory_sha256": semantic_sha256(outer_test),
        }
        row = MassiveProfitabilityOuterFoldPlanV1(
            fold_index=fold_index,
            fit_session_dates=fit,
            inner_purge_session_dates=inner_purge,
            inner_validation_session_dates=inner_validation,
            outer_purge_session_dates=outer_purge,
            outer_test_session_dates=outer_test,
            fit_inventory_sha256=semantic_sha256(fit),
            inner_validation_inventory_sha256=semantic_sha256(inner_validation),
            outer_test_inventory_sha256=semantic_sha256(outer_test),
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        output.append(row)
    return tuple(output)


def _semantic_payload(
    *, archive_freeze: MassiveProfitabilityArchiveFreezeV1
) -> dict[str, object]:
    candidates = archive_freeze.fixed_candidate_session_dates
    folds = _folds(candidates)
    initial_development_end = (
        len(candidates) - _LOCKBOX - _OUTER_COUNT * _OUTER_SESSIONS
    )
    return {
        "schema": MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA,
        "candidate_session_dates": candidates,
        "initial_development_session_dates": candidates[:initial_development_end],
        "outer_folds": tuple(asdict(row) for row in folds),
        "lockbox_session_dates": archive_freeze.fixed_lockbox_session_dates,
        "candidate_inventory_sha256": archive_freeze.candidate_inventory_sha256,
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze.semantic_receipt_sha256
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SHA256
        ),
        "panel_materialization_authorized": False,
        "predictive_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }


def materialize_massive_profitability_phase_plan_v1(
    *,
    root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityPhasePlanV1:
    """Persist fixed candidate-date folds without consulting origin outcomes."""

    archive_freeze.validate()
    semantic = _semantic_payload(archive_freeze=archive_freeze)
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "semantic_receipt_sha256": semantic_receipt,
        "archive_freeze_audit_receipt_sha256": archive_freeze.audit_receipt_sha256,
    }
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/phase-plan-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PHASE_PLAN_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "phase plan entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-PHASE-PLAN-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_phase_plan_v1(root=root, loaded_source=loaded)
    result = replace(
        parsed,
        archive_source_transport_qualified=(archive_freeze.source_transport_qualified),
        archive_rank_bar_data_qualified=archive_freeze.rank_bar_data_qualified,
    )
    result.validate()
    return result


def parse_massive_profitability_phase_plan_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityPhasePlanV1:
    """Reload exact phase bytes; generic reload never qualifies the archive."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityPhasePlanV1Error(
            "phase plan source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityPhasePlanV1Error(
            "phase plan source is not canonical JSON"
        )
    expected_fields = {
        "schema",
        "candidate_session_dates",
        "initial_development_session_dates",
        "outer_folds",
        "lockbox_session_dates",
        "candidate_inventory_sha256",
        "archive_freeze_semantic_receipt_sha256",
        "protocol_receipt_sha256",
        "specification_sha256",
        "implementation_source_sha256",
        "panel_materialization_authorized",
        "predictive_training_authorized",
        "profitability_reporting_authorized",
        "lockbox_access_authorized",
        "semantic_receipt_sha256",
        "archive_freeze_audit_receipt_sha256",
    }
    if set(payload) != expected_fields:
        raise MassiveProfitabilityPhasePlanV1Error("phase plan field inventory differs")
    try:
        folds = tuple(
            MassiveProfitabilityOuterFoldPlanV1(
                **{
                    **row,
                    "fit_session_dates": tuple(row["fit_session_dates"]),
                    "inner_purge_session_dates": tuple(
                        row["inner_purge_session_dates"]
                    ),
                    "inner_validation_session_dates": tuple(
                        row["inner_validation_session_dates"]
                    ),
                    "outer_purge_session_dates": tuple(
                        row["outer_purge_session_dates"]
                    ),
                    "outer_test_session_dates": tuple(row["outer_test_session_dates"]),
                }
            )
            for row in payload["outer_folds"]
        )
        result = MassiveProfitabilityPhasePlanV1(
            schema=payload["schema"],
            candidate_session_dates=tuple(payload["candidate_session_dates"]),
            initial_development_session_dates=tuple(
                payload["initial_development_session_dates"]
            ),
            outer_folds=folds,
            lockbox_session_dates=tuple(payload["lockbox_session_dates"]),
            candidate_inventory_sha256=payload["candidate_inventory_sha256"],
            archive_freeze_semantic_receipt_sha256=payload[
                "archive_freeze_semantic_receipt_sha256"
            ],
            protocol_receipt_sha256=payload["protocol_receipt_sha256"],
            specification_sha256=payload["specification_sha256"],
            implementation_source_sha256=payload["implementation_source_sha256"],
            semantic_receipt_sha256=payload["semantic_receipt_sha256"],
            archive_freeze_audit_receipt_sha256=payload[
                "archive_freeze_audit_receipt_sha256"
            ],
            audit_receipt_sha256=semantic_sha256(
                {
                    "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                    "archive_freeze_audit_receipt_sha256": payload[
                        "archive_freeze_audit_receipt_sha256"
                    ],
                    "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
                }
            ),
            loaded_source=loaded_source,
            archive_source_transport_qualified=False,
            archive_rank_bar_data_qualified=False,
            panel_materialization_authorized=payload[
                "panel_materialization_authorized"
            ],
            predictive_training_authorized=payload["predictive_training_authorized"],
            profitability_reporting_authorized=payload[
                "profitability_reporting_authorized"
            ],
            lockbox_access_authorized=payload["lockbox_access_authorized"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityPhasePlanV1Error(
            "phase plan values are malformed"
        ) from exc
    result.validate()
    return result


def build_massive_profitability_phase_origin_gate_v1(
    *,
    phase_plan: MassiveProfitabilityPhasePlanV1,
    origin_plan: MassiveProfitabilityDecisionOriginPlanV2,
) -> MassiveProfitabilityPhaseOriginGateV1:
    """Report skips without ever moving frozen confirmation/lockbox dates."""

    phase_plan.validate()
    origin_plan.validate()
    v1 = origin_plan.origin_plan_v1
    if v1.candidate_decision_session_dates != phase_plan.candidate_session_dates:
        raise MassiveProfitabilityPhasePlanV1Error(
            "origin candidates differ from the archive-frozen phase plan"
        )
    skip_by_date = {
        row.decision_session_date: row.reason for row in v1.skipped_decisions
    }
    origin_by_date = {
        row.decision_session_date: row.receipt_sha256 for row in v1.origins
    }
    statuses = tuple(
        (
            session_date,
            "skip" if session_date in skip_by_date else "origin",
            skip_by_date.get(session_date, origin_by_date.get(session_date, "")),
        )
        for session_date in phase_plan.candidate_session_dates
    )
    if any(not value[2] for value in statuses):
        raise MassiveProfitabilityPhasePlanV1Error(
            "origin plan does not partition every frozen candidate"
        )
    outer_dates = {
        value
        for fold in phase_plan.outer_folds
        for value in fold.outer_test_session_dates
    }
    lockbox_dates = set(phase_plan.lockbox_session_dates)
    development_skips = tuple(
        sorted(
            (session_date, reason)
            for session_date, reason in skip_by_date.items()
            if session_date not in outer_dates and session_date not in lockbox_dates
        )
    )
    outer_skips = tuple(
        sorted(
            (session_date, reason)
            for session_date, reason in skip_by_date.items()
            if session_date in outer_dates
        )
    )
    lockbox_skips = tuple(
        sorted(
            (session_date, reason)
            for session_date, reason in skip_by_date.items()
            if session_date in lockbox_dates
        )
    )
    body: dict[str, object] = {
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "origin_plan_semantic_receipt_sha256": origin_plan.semantic_receipt_sha256,
        "candidate_status_inventory_sha256": semantic_sha256(statuses),
        "development_skip_inventory": development_skips,
        "outer_test_skip_inventory": outer_skips,
        "lockbox_skip_inventory": lockbox_skips,
        "outer_test_complete": not outer_skips,
        "lockbox_complete": not lockbox_skips,
        "data_gate_passed": False,
    }
    result = MassiveProfitabilityPhaseOriginGateV1(
        phase_plan_semantic_receipt_sha256=phase_plan.semantic_receipt_sha256,
        origin_plan_semantic_receipt_sha256=origin_plan.semantic_receipt_sha256,
        candidate_status_inventory_sha256=semantic_sha256(statuses),
        development_skip_inventory=development_skips,
        outer_test_skip_inventory=outer_skips,
        lockbox_skip_inventory=lockbox_skips,
        outer_test_complete=not outer_skips,
        lockbox_complete=not lockbox_skips,
        data_gate_passed=False,
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_DATASET",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_LOCKBOX_ACCESS_AUTHORIZED",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PANEL_MATERIALIZATION_AUTHORIZED",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PREDICTIVE_TRAINING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_PROFITABILITY_REPORTING_AUTHORIZED",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V1_SPEC_SHA256",
    "MassiveProfitabilityOuterFoldPlanV1",
    "MassiveProfitabilityPhaseOriginGateV1",
    "MassiveProfitabilityPhasePlanV1",
    "MassiveProfitabilityPhasePlanV1Error",
    "build_massive_profitability_phase_origin_gate_v1",
    "materialize_massive_profitability_phase_plan_v1",
    "parse_massive_profitability_phase_plan_v1",
]
