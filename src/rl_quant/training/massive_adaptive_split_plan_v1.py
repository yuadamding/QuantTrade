"""Frozen 126-session split geometry for adaptive alpha development.

The longest supervised target ends 126 exchange sessions after its origin.
This plan therefore moves every inner/outer purge and the final lockbox
embargo to 126 sessions.  It is a deterministic function of one sorted
candidate-session inventory; callers cannot provide role indices or a free
plan digest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SCHEMA = "rl-quant.massive-adaptive-split-plan-v1"
MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1 = 504
MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_INITIAL_FIT_SESSIONS_V1 = 756
MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1 = 4
MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1 = 126
MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1 = 252
MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1 = (
    MASSIVE_ADAPTIVE_INITIAL_FIT_SESSIONS_V1
    + MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1
    + MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
    + MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1
    + MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
    * MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
    + MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1
    + MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1
)
MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "maximum_target_sessions": MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1,
        "maximum_context_sessions": MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
        "initial_fit_sessions": MASSIVE_ADAPTIVE_INITIAL_FIT_SESSIONS_V1,
        "inner_purge_sessions": MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1,
        "inner_validation_sessions": MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1,
        "outer_purge_sessions": MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1,
        "outer_folds": (
            MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
            MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1,
        ),
        "outer_to_lockbox_embargo_sessions": (
            MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1
        ),
        "lockbox_sessions": MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1,
        "duration_prior": False,
        "training_authorization": False,
        "outer_authorization": False,
    }
)


class MassiveAdaptiveSplitPlanV1Error(ValueError):
    """Adaptive split geometry does not protect the 126-session target."""


def _canonical_dates(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if (
        len(result) < MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1
        or result != tuple(sorted(set(result)))
    ):
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive candidate sessions are not a sufficient sorted inventory"
        )
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveSplitFoldV1:
    fold_index: int
    fit_session_dates: tuple[str, ...]
    inner_purge_session_dates: tuple[str, ...]
    inner_validation_session_dates: tuple[str, ...]
    outer_purge_session_dates: tuple[str, ...]
    outer_test_session_dates: tuple[str, ...]
    fit_target_stop_exclusive_index: int
    validation_target_stop_exclusive_index: int
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    def validate(self, candidate_session_dates: tuple[str, ...]) -> None:
        if (
            isinstance(self.fold_index, bool)
            or not 0 <= self.fold_index < MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or len(self.fit_session_dates)
            < MASSIVE_ADAPTIVE_INITIAL_FIT_SESSIONS_V1
            or len(self.inner_purge_session_dates)
            != MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1
            or len(self.inner_validation_session_dates)
            != MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
            or len(self.outer_purge_session_dates)
            != MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1
            or len(self.outer_test_session_dates)
            != MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveSplitPlanV1Error(
                "adaptive fold geometry or receipt differs"
            )
        contiguous = (
            self.fit_session_dates
            + self.inner_purge_session_dates
            + self.inner_validation_session_dates
            + self.outer_purge_session_dates
            + self.outer_test_session_dates
        )
        if contiguous != candidate_session_dates[: len(contiguous)]:
            raise MassiveAdaptiveSplitPlanV1Error(
                "adaptive fold is not a contiguous candidate prefix"
            )
        validation_start = len(self.fit_session_dates) + len(
            self.inner_purge_session_dates
        )
        outer_start = validation_start + len(self.inner_validation_session_dates) + len(
            self.outer_purge_session_dates
        )
        if (
            self.fit_target_stop_exclusive_index != validation_start
            or self.validation_target_stop_exclusive_index != outer_start
        ):
            raise MassiveAdaptiveSplitPlanV1Error(
                "adaptive fold target maturity boundary differs"
            )


def _build_folds(
    candidates: tuple[str, ...],
) -> tuple[MassiveAdaptiveSplitFoldV1, ...]:
    outer_start = (
        len(candidates)
        - MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1
        - MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1
        - MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
        * MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
    )
    minimum = (
        MASSIVE_ADAPTIVE_INITIAL_FIT_SESSIONS_V1
        + MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1
        + MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
        + MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1
    )
    if outer_start < minimum:
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive candidates cannot support the first outer fold"
        )
    rows: list[MassiveAdaptiveSplitFoldV1] = []
    for fold_index in range(MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1):
        test_start = (
            outer_start
            + fold_index * MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
        )
        validation_stop = test_start - MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1
        validation_start = (
            validation_stop - MASSIVE_ADAPTIVE_INNER_VALIDATION_SESSIONS_V1
        )
        fit_stop = validation_start - MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1
        body = {
            "fold_index": fold_index,
            "fit_session_dates": candidates[:fit_stop],
            "inner_purge_session_dates": candidates[fit_stop:validation_start],
            "inner_validation_session_dates": candidates[
                validation_start:validation_stop
            ],
            "outer_purge_session_dates": candidates[validation_stop:test_start],
            "outer_test_session_dates": candidates[
                test_start : test_start
                + MASSIVE_ADAPTIVE_OUTER_TEST_SESSIONS_V1
            ],
            "fit_target_stop_exclusive_index": validation_start,
            "validation_target_stop_exclusive_index": test_start,
        }
        row = MassiveAdaptiveSplitFoldV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate(candidates)
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveSplitPlanV1:
    candidate_session_dates: tuple[str, ...]
    outer_folds: tuple[MassiveAdaptiveSplitFoldV1, ...]
    outer_to_lockbox_embargo_session_dates: tuple[str, ...]
    lockbox_session_dates: tuple[str, ...]
    candidate_inventory_sha256: str
    fold_inventory_sha256: str
    session_authority_receipt_sha256: str
    candidate_authority_receipt_sha256: str
    candidate_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    source_geometry_replayed: bool
    development_training_authorized: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        candidates = _canonical_dates(self.candidate_session_dates)
        expected_folds = _build_folds(candidates)
        embargo_start = len(candidates) - MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1 - (
            MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1
        )
        expected_embargo = candidates[
            embargo_start : len(candidates) - MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1
        ]
        expected_lockbox = candidates[-MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1:]
        if (
            self.schema != MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SCHEMA
            or self.outer_folds != expected_folds
            or self.outer_to_lockbox_embargo_session_dates != expected_embargo
            or self.lockbox_session_dates != expected_lockbox
            or self.candidate_inventory_sha256 != semantic_sha256(candidates)
            or self.fold_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in expected_folds))
            or not isinstance(self.candidate_source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SOURCE_SHA256
            or not self.source_geometry_replayed
            or self.development_training_authorized
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveSplitPlanV1Error(
                "adaptive split-plan identity or authorization differs"
            )
        for value in (
            self.candidate_inventory_sha256,
            self.fold_inventory_sha256,
            self.session_authority_receipt_sha256,
            self.candidate_authority_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise MassiveAdaptiveSplitPlanV1Error(
                    "adaptive split-plan digest differs"
                )
        assert_no_adaptive_hold_semantics(asdict(self))


def build_massive_adaptive_split_plan_v1(
    *,
    candidate_session_dates: Sequence[str],
    session_authority: MassiveSessionAuthority,
) -> MassiveAdaptiveSplitPlanV1:
    """Build a nonauthorizing engineering plan from a calendar inventory."""

    session_authority.validate()
    candidates = _canonical_dates(candidate_session_dates)
    exchange_dates = tuple(
        row.session_date for row in session_authority.sessions if row.exchange == "XNYS"
    )
    if not exchange_dates:
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive split authority has no XNYS sessions"
        )
    try:
        start = exchange_dates.index(candidates[0])
    except ValueError as exc:
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive candidate start is absent from the session authority"
        ) from exc
    if exchange_dates[start : start + len(candidates)] != candidates:
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive candidates are not consecutive XNYS authority sessions"
        )
    folds = _build_folds(candidates)
    embargo_start = len(candidates) - MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1 - (
        MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SCHEMA,
        "candidate_session_dates": candidates,
        "outer_folds": folds,
        "outer_to_lockbox_embargo_session_dates": candidates[
            embargo_start : len(candidates) - MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1
        ],
        "lockbox_session_dates": candidates[-MASSIVE_ADAPTIVE_LOCKBOX_SESSIONS_V1:],
        "candidate_inventory_sha256": semantic_sha256(candidates),
        "fold_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in folds)
        ),
        "session_authority_receipt_sha256": session_authority.receipt_sha256,
        "candidate_authority_receipt_sha256": semantic_sha256(
            {
                "kind": "engineering-candidate-inventory",
                "session_authority": session_authority.receipt_sha256,
                "candidate_session_dates": candidates,
            }
        ),
        "candidate_source_data_qualified": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SOURCE_SHA256,
        "source_geometry_replayed": True,
        "development_training_authorized": False,
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_body = {
        **body,
        "outer_folds": tuple(asdict(row) for row in folds),
    }
    result = MassiveAdaptiveSplitPlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(semantic_body),
    )
    result.validate()
    return result


def build_massive_adaptive_split_plan_from_archive_v1(
    *,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    session_authority: MassiveSessionAuthority,
) -> MassiveAdaptiveSplitPlanV1:
    """Promote only the exact create-only archive candidate inventory."""

    archive_freeze.validate()
    engineering = build_massive_adaptive_split_plan_v1(
        candidate_session_dates=archive_freeze.fixed_candidate_session_dates,
        session_authority=session_authority,
    )
    if (
        archive_freeze.session_authority_receipt_sha256
        != session_authority.receipt_sha256
        or archive_freeze.candidate_inventory_sha256
        != engineering.candidate_inventory_sha256
    ):
        raise MassiveAdaptiveSplitPlanV1Error(
            "adaptive archive, calendar, or candidate inventory differs"
        )
    qualified = (
        archive_freeze.source_transport_qualified
        and archive_freeze.rank_bar_data_qualified
        and archive_freeze.calendar_geometry_complete
    )
    provisional = replace(
        engineering,
        candidate_authority_receipt_sha256=(
            archive_freeze.semantic_receipt_sha256
        ),
        candidate_source_data_qualified=qualified,
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_INNER_PURGE_SESSIONS_V1",
    "MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1",
    "MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1",
    "MASSIVE_ADAPTIVE_MINIMUM_CANDIDATE_SESSIONS_V1",
    "MASSIVE_ADAPTIVE_OUTER_LOCKBOX_EMBARGO_SESSIONS_V1",
    "MASSIVE_ADAPTIVE_OUTER_PURGE_SESSIONS_V1",
    "MassiveAdaptiveSplitFoldV1",
    "MassiveAdaptiveSplitPlanV1",
    "MassiveAdaptiveSplitPlanV1Error",
    "build_massive_adaptive_split_plan_v1",
    "build_massive_adaptive_split_plan_from_archive_v1",
]
