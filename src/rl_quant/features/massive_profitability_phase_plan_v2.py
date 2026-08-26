"""Embargoed candidate-date phases for the Massive P0 profitability test."""

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
from rl_quant.features.massive_profitability_phase_plan_v1 import (
    MassiveProfitabilityOuterFoldPlanV1,
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

MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA = (
    "rl-quant.massive-profitability-phase-plan-v2"
)
MASSIVE_PROFITABILITY_PHASE_PLAN_V2_DATASET = "massive-profitability-phase-plan-v2"
MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SHA256 = file_sha256(Path(__file__))

_PROTOCOL = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
_OUTER_COUNT = _PROTOCOL.outer_fold_count
_OUTER_SESSIONS = _PROTOCOL.outer_fold_sessions
_OUTER_PURGE = _PROTOCOL.target_overlap_purge_sessions
_INNER_VALIDATION = _PROTOCOL.inner_validation_sessions
_INNER_PURGE = _PROTOCOL.inner_purge_sessions
_MINIMUM_FIT = _PROTOCOL.minimum_initial_training_sessions
_LOCKBOX = _PROTOCOL.historical_lockbox_sessions
MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2 = 63
MASSIVE_PROFITABILITY_MINIMUM_CANDIDATE_SESSIONS_V2 = (
    _MINIMUM_FIT
    + _INNER_PURGE
    + _INNER_VALIDATION
    + _OUTER_PURGE
    + _OUTER_COUNT * _OUTER_SESSIONS
    + MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2
    + _LOCKBOX
)

MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "phase_basis": "archive-frozen-candidate-session-dates",
        "outer_folds": (_OUTER_COUNT, _OUTER_SESSIONS),
        "outer_purge": _OUTER_PURGE,
        "inner_validation": _INNER_VALIDATION,
        "inner_purge": _INNER_PURGE,
        "minimum_fit": _MINIMUM_FIT,
        "outer_to_lockbox_embargo": (
            MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2
        ),
        "embargo_use": "outer-position-maturation-only-no-entry-or-retuning",
        "lockbox": _LOCKBOX,
        "phase_shift_after_freeze": "prohibited",
        "performance_authorization": False,
    }
)


class MassiveProfitabilityPhasePlanV2Error(ValueError):
    """The embargoed phase geometry differs from the frozen contract."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityPhasePlanV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityPhasePlanV2Error(
            "phase plan V2 artifact ID is not path safe"
        )
    return value


def _canonical_dates(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or result != tuple(sorted(set(result))):
        raise MassiveProfitabilityPhasePlanV2Error(
            f"{name} must be sorted, unique, and nonempty"
        )
    return result


def _folds_v2(
    candidates: tuple[str, ...],
) -> tuple[MassiveProfitabilityOuterFoldPlanV1, ...]:
    outer_start = (
        len(candidates)
        - _LOCKBOX
        - MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2
        - _OUTER_COUNT * _OUTER_SESSIONS
    )
    if outer_start < _MINIMUM_FIT + _INNER_PURGE + _INNER_VALIDATION + _OUTER_PURGE:
        raise MassiveProfitabilityPhasePlanV2Error(
            "candidate history cannot support the embargoed first outer fold"
        )
    rows: list[MassiveProfitabilityOuterFoldPlanV1] = []
    for fold_index in range(_OUTER_COUNT):
        test_start = outer_start + fold_index * _OUTER_SESSIONS
        outer_test = candidates[test_start : test_start + _OUTER_SESSIONS]
        outer_purge = candidates[test_start - _OUTER_PURGE : test_start]
        validation_end = test_start - _OUTER_PURGE
        inner_validation = candidates[
            validation_end - _INNER_VALIDATION : validation_end
        ]
        inner_purge = candidates[
            validation_end - _INNER_VALIDATION - _INNER_PURGE : validation_end
            - _INNER_VALIDATION
        ]
        fit = candidates[: validation_end - _INNER_VALIDATION - _INNER_PURGE]
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
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityPhasePlanV2:
    candidate_session_dates: tuple[str, ...]
    initial_development_session_dates: tuple[str, ...]
    outer_folds: tuple[MassiveProfitabilityOuterFoldPlanV1, ...]
    outer_to_lockbox_embargo_session_dates: tuple[str, ...]
    lockbox_session_dates: tuple[str, ...]
    candidate_inventory_sha256: str
    embargo_inventory_sha256: str
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
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    evaluator_retuning_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "archive_freeze_audit_receipt_sha256",
                "audit_receipt_sha256",
                "loaded_source",
                "archive_source_transport_qualified",
                "archive_rank_bar_data_qualified",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SHA256
        ):
            raise MassiveProfitabilityPhasePlanV2Error("phase plan V2 identity differs")
        candidates = _canonical_dates(
            "phase V2 candidates", self.candidate_session_dates
        )
        development = _canonical_dates(
            "phase V2 development", self.initial_development_session_dates
        )
        embargo = _canonical_dates(
            "phase V2 embargo", self.outer_to_lockbox_embargo_session_dates
        )
        lockbox = _canonical_dates("phase V2 lockbox", self.lockbox_session_dates)
        if len(candidates) < MASSIVE_PROFITABILITY_MINIMUM_CANDIDATE_SESSIONS_V2:
            raise MassiveProfitabilityPhasePlanV2Error(
                "phase plan V2 candidate inventory is too short"
            )
        folds = _folds_v2(candidates)
        outer_dates = tuple(
            value for fold in folds for value in fold.outer_test_session_dates
        )
        development_end = (
            len(candidates) - len(outer_dates) - len(embargo) - len(lockbox)
        )
        if (
            self.outer_folds != folds
            or len(embargo) != MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2
            or len(lockbox) != _LOCKBOX
            or development != candidates[:development_end]
            or outer_dates + embargo + lockbox
            != candidates[-(len(outer_dates) + len(embargo) + len(lockbox)) :]
        ):
            raise MassiveProfitabilityPhasePlanV2Error(
                "phase plan V2 partitions or embargo differ"
            )
        if (
            self.candidate_inventory_sha256 != semantic_sha256(candidates)
            or self.embargo_inventory_sha256 != semantic_sha256(embargo)
            or not isinstance(self.archive_source_transport_qualified, bool)
            or not isinstance(self.archive_rank_bar_data_qualified, bool)
            or self.archive_rank_bar_data_qualified
            and not self.archive_source_transport_qualified
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.evaluator_retuning_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityPhasePlanV2Error(
                "phase plan V2 qualification or authorization differs"
            )
        for value in (
            self.candidate_inventory_sha256,
            self.embargo_inventory_sha256,
            self.archive_freeze_semantic_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.archive_freeze_audit_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("phase plan V2", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityPhasePlanV2Error(
                "phase plan V2 semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SCHEMA_SHA256
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "archive_freeze_audit_receipt_sha256": (
                        self.archive_freeze_audit_receipt_sha256
                    ),
                    "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
                }
            )
        ):
            raise MassiveProfitabilityPhasePlanV2Error(
                "phase plan V2 committed source differs"
            )


def _semantic_payload(
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
) -> dict[str, object]:
    candidates = archive_freeze.fixed_candidate_session_dates
    folds = _folds_v2(candidates)
    outer_dates = tuple(
        value for fold in folds for value in fold.outer_test_session_dates
    )
    embargo_start = (
        len(candidates)
        - _LOCKBOX
        - (MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2)
    )
    embargo = candidates[embargo_start : len(candidates) - _LOCKBOX]
    development_end = len(candidates) - len(outer_dates) - len(embargo) - _LOCKBOX
    return {
        "schema": MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA,
        "candidate_session_dates": candidates,
        "initial_development_session_dates": candidates[:development_end],
        "outer_folds": tuple(asdict(row) for row in folds),
        "outer_to_lockbox_embargo_session_dates": embargo,
        "lockbox_session_dates": candidates[-_LOCKBOX:],
        "candidate_inventory_sha256": semantic_sha256(candidates),
        "embargo_inventory_sha256": semantic_sha256(embargo),
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze.semantic_receipt_sha256
        ),
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SHA256,
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "evaluator_retuning_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def materialize_massive_profitability_phase_plan_v2(
    *,
    root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    artifact_id: str,
    committed_at_ms: int,
    entitlement_receipt_sha256: str,
) -> MassiveProfitabilityPhasePlanV2:
    """Publish the outer folds, embargo, and lockbox as immutable dates."""

    archive_freeze.validate()
    semantic = _semantic_payload(archive_freeze)
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "semantic_receipt_sha256": semantic_receipt,
        "archive_freeze_audit_receipt_sha256": archive_freeze.audit_receipt_sha256,
    }
    identifier = _artifact_id(artifact_id)
    relative = f"massive-profitability/phase-plan-v2/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_PHASE_PLAN_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "phase plan V2 entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"P0-PHASE-PLAN-V2-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_phase_plan_v2(root=root, loaded_source=loaded)
    result = replace(
        parsed,
        archive_source_transport_qualified=archive_freeze.source_transport_qualified,
        archive_rank_bar_data_qualified=archive_freeze.rank_bar_data_qualified,
    )
    result.validate()
    return result


def parse_massive_profitability_phase_plan_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityPhasePlanV2:
    """Reload the immutable V2 phase plan without promoting archive qualification."""

    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityPhasePlanV2Error(
            "phase plan V2 source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityPhasePlanV2Error(
            "phase plan V2 source is not canonical JSON"
        )
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
            for row in payload.pop("outer_folds")
        )
        for name in (
            "candidate_session_dates",
            "initial_development_session_dates",
            "outer_to_lockbox_embargo_session_dates",
            "lockbox_session_dates",
        ):
            payload[name] = tuple(payload[name])
        result = MassiveProfitabilityPhasePlanV2(
            **payload,
            outer_folds=folds,
            loaded_source=loaded_source,
            audit_receipt_sha256=semantic_sha256(
                {
                    "semantic_receipt_sha256": payload["semantic_receipt_sha256"],
                    "archive_freeze_audit_receipt_sha256": payload[
                        "archive_freeze_audit_receipt_sha256"
                    ],
                    "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
                }
            ),
            archive_source_transport_qualified=False,
            archive_rank_bar_data_qualified=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MassiveProfitabilityPhasePlanV2Error(
            "phase plan V2 values are malformed"
        ) from exc
    result.validate()
    expected = result.semantic_unsigned() | {
        "outer_folds": tuple(asdict(row) for row in result.outer_folds),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
        "archive_freeze_audit_receipt_sha256": (
            result.archive_freeze_audit_receipt_sha256
        ),
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityPhasePlanV2Error(
            "phase plan V2 canonical bytes differ"
        )
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_MINIMUM_CANDIDATE_SESSIONS_V2",
    "MASSIVE_PROFITABILITY_OUTER_LOCKBOX_EMBARGO_SESSIONS_V2",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V2_DATASET",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SCHEMA",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SOURCE_SHA256",
    "MASSIVE_PROFITABILITY_PHASE_PLAN_V2_SPEC_SHA256",
    "MassiveProfitabilityPhasePlanV2",
    "MassiveProfitabilityPhasePlanV2Error",
    "materialize_massive_profitability_phase_plan_v2",
    "parse_massive_profitability_phase_plan_v2",
]
