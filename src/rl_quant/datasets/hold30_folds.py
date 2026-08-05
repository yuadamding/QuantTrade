"""Role-exact Hold-30 fold materialization with outer-access gating.

Training and inner validation are materialized together from a qualified
point-in-time parent sequence.  The outer sequence has a separate API and is
unavailable until an exact five-seed checkpoint-selection receipt and an
append-only access-marker digest are supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

import torch

from rl_quant.datasets.hold30 import (
    Hold30AsOfEvidence,
    Hold30DatasetError,
    Hold30DatasetSequence,
)
from rl_quant.datasets.hold30_qualification import (
    verify_hold30_dataset_against_qualification,
)
from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import Hold30Fold, render_hold30_folds, sha256_payload
from rl_quant.training.hold30_coordinator import (
    Hold30CoordinationError,
    verify_hold30_cohort_receipt,
)


@dataclass(frozen=True, slots=True)
class Hold30DevelopmentFold:
    """Training and inner-validation sequences; deliberately no outer tensor."""

    fold: Hold30Fold
    parent_axis_id: str
    data_qualification_sha256: str
    fold_sha256: str
    training: Hold30DatasetSequence
    inner_validation: Hold30DatasetSequence
    training_absolute_range: tuple[int, int]
    validation_absolute_range: tuple[int, int]
    outer_absolute_range: tuple[int, int]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class Hold30OuterFold:
    fold_index: int
    sequence: Hold30DatasetSequence
    absolute_range: tuple[int, int]
    development_receipt_sha256: str
    cohort_receipt_sha256: str
    access_marker_sha256: str
    receipt_sha256: str


def _require_digest(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hold30DatasetError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _axis_dates(sequence: Hold30DatasetSequence) -> tuple[str, ...]:
    timestamps = sequence.decision_timestamps_ms.detach().to(device="cpu", dtype=torch.int64)
    return tuple(
        datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date().isoformat()
        for value in timestamps.tolist()
    )


def _slice_asof(evidence: Hold30AsOfEvidence, start: int, stop: int) -> Hold30AsOfEvidence:
    return replace(
        evidence,
        **{
            name: getattr(evidence, name)[start:stop]
            for name in evidence.__dataclass_fields__
        },
    )


def _slice_sequence(
    parent: Hold30DatasetSequence,
    absolute_range: tuple[int, int],
) -> Hold30DatasetSequence:
    start, stop = absolute_range
    if not 0 <= start < stop <= parent.n_positions:
        raise Hold30DatasetError("fold slice lies outside the qualified parent axis")
    if stop - start < 95:
        raise Hold30DatasetError("a Hold-30 fold slice needs at least 95 positions")
    return replace(
        parent,
        decision_timestamps_ms=parent.decision_timestamps_ms[start:stop],
        fill_timestamps_ms=parent.fill_timestamps_ms[start:stop],
        decision_state=parent.decision_state[start:stop],
        decision_membership=parent.decision_membership[start:stop],
        decision_tradability=parent.decision_tradability[start:stop],
        fill_membership=parent.fill_membership[start:stop],
        fill_tradability=parent.fill_tradability[start:stop],
        asset_returns=parent.asset_returns[start : stop - 1],
        ordinary_return_valid=parent.ordinary_return_valid[start : stop - 1],
        mandatory_return_mask=parent.mandatory_return_mask[start : stop - 1],
        c1_benchmark_weights=parent.c1_benchmark_weights[start:stop],
        c1_benchmark_net_returns=parent.c1_benchmark_net_returns[start : stop - 1],
        risk_asset_caps=parent.risk_asset_caps[start:stop],
        risk_gross_max=parent.risk_gross_max[start:stop],
        cost_rate=parent.cost_rate[start : stop - 1],
        asof_evidence=_slice_asof(parent.asof_evidence, start, stop),
    )


def _absolute_role_indices(
    mask: torch.Tensor,
    *,
    start: int,
) -> tuple[int, ...]:
    return tuple(int(value) + start for value in torch.where(mask)[0].tolist())


def _range_values(value: tuple[int, int]) -> tuple[int, ...]:
    return tuple(range(*value))


def _validate_training_roles(
    sequence: Hold30DatasetSequence,
    fold: Hold30Fold,
    *,
    start: int,
) -> None:
    checks = (
        (sequence.roles.warmup, fold.training_warmup, "training warm-up"),
        (sequence.roles.score, fold.training_anchors, "training anchors"),
        (sequence.roles.support, fold.training_support, "training support"),
    )
    for mask, expected, name in checks:
        if _absolute_role_indices(mask, start=start) != _range_values(expected):
            raise Hold30DatasetError(f"{name} does not match the frozen fold")
    terminal = _absolute_role_indices(sequence.roles.terminal, start=start)
    if terminal != (fold.training_terminal_observation,):
        raise Hold30DatasetError("training terminal observation does not match the fold")


def _validate_evaluation_roles(
    sequence: Hold30DatasetSequence,
    *,
    start: int,
    score_range: tuple[int, int],
    support_range: tuple[int, int],
    name: str,
) -> None:
    if _absolute_role_indices(sequence.roles.score, start=start) != _range_values(score_range):
        raise Hold30DatasetError(f"{name} score rows do not match the frozen fold")
    support_and_terminal = sequence.roles.support | sequence.roles.terminal
    if _absolute_role_indices(support_and_terminal, start=start) != _range_values(
        support_range
    ):
        raise Hold30DatasetError(f"{name} support/terminal rows do not match the frozen fold")
    if _absolute_role_indices(sequence.roles.warmup, start=start) != tuple(
        range(start, score_range[0])
    ):
        raise Hold30DatasetError(f"{name} causal warm-up is not exactly 63 positions")


def _qualified_parent(
    parent: Hold30DatasetSequence,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    receipt: Mapping[str, Any],
) -> str:
    verify_hold30_dataset_against_qualification(
        parent,
        monthly_rebalance,
        external_artifacts,
        receipt,
    )
    return _require_digest("data qualification receipt", receipt["receipt_sha256"])


def materialize_hold30_development_fold(
    parent: Hold30DatasetSequence,
    fold: Hold30Fold,
    *,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    data_qualification_receipt: Mapping[str, Any],
) -> Hold30DevelopmentFold:
    """Materialize training/validation only; never expose outer tensors."""

    if not isinstance(parent, Hold30DatasetSequence) or not isinstance(fold, Hold30Fold):
        raise TypeError("parent and fold must be validated Hold-30 objects")
    qualification_sha = _qualified_parent(
        parent,
        monthly_rebalance,
        external_artifacts,
        data_qualification_receipt,
    )
    try:
        rendered = render_hold30_folds(_axis_dates(parent))
    except ValueError as exc:
        raise Hold30DatasetError(f"cannot reconstruct frozen folds: {exc}") from exc
    if fold.fold_index not in range(len(rendered)) or fold != rendered[fold.fold_index]:
        raise Hold30DatasetError("requested fold differs from the qualified rendered fold")

    training_range = fold.expanding_train
    validation_range = (fold.inner_validation[0] - 63, fold.validation_support[1])
    outer_range = (fold.outer_score[0] - 63, fold.outer_support[1])
    training = _slice_sequence(parent, training_range)
    validation = _slice_sequence(parent, validation_range)
    _validate_training_roles(training, fold, start=training_range[0])
    _validate_evaluation_roles(
        validation,
        start=validation_range[0],
        score_range=fold.inner_validation,
        support_range=fold.validation_support,
        name="inner validation",
    )
    fold_sha = sha256_payload(asdict(fold))
    payload = {
        "schema": "rl-quant.hold30.development-fold",
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "parent_axis_id": parent.axis_id,
        "data_qualification_sha256": qualification_sha,
        "fold": asdict(fold),
        "fold_sha256": fold_sha,
        "training_absolute_range": list(training_range),
        "validation_absolute_range": list(validation_range),
        "outer_absolute_range": list(outer_range),
        "training_axis_id": training.axis_id,
        "inner_validation_axis_id": validation.axis_id,
        "outer_materialized": False,
        "outer_access": False,
    }
    receipt_sha = sha256_payload(payload)
    return Hold30DevelopmentFold(
        fold=fold,
        parent_axis_id=parent.axis_id,
        data_qualification_sha256=qualification_sha,
        fold_sha256=fold_sha,
        training=training,
        inner_validation=validation,
        training_absolute_range=training_range,
        validation_absolute_range=validation_range,
        outer_absolute_range=outer_range,
        receipt_sha256=receipt_sha,
    )


def materialize_hold30_outer_fold(
    parent: Hold30DatasetSequence,
    development: Hold30DevelopmentFold,
    *,
    monthly_rebalance: torch.Tensor,
    external_artifacts: Mapping[str, str],
    data_qualification_receipt: Mapping[str, Any],
    cohort_selection_receipt: Mapping[str, Any],
    access_marker_sha256: str,
) -> Hold30OuterFold:
    """Open one outer fold only after shared checkpoint selection is terminal."""

    marker = _require_digest("access_marker_sha256", access_marker_sha256)
    qualification_sha = _qualified_parent(
        parent,
        monthly_rebalance,
        external_artifacts,
        data_qualification_receipt,
    )
    if not isinstance(development, Hold30DevelopmentFold):
        raise TypeError("development must be Hold30DevelopmentFold")
    if (
        development.parent_axis_id != parent.axis_id
        or development.data_qualification_sha256 != qualification_sha
    ):
        raise Hold30DatasetError("development fold belongs to another qualified parent")
    expected_development = materialize_hold30_development_fold(
        parent,
        development.fold,
        monthly_rebalance=monthly_rebalance,
        external_artifacts=external_artifacts,
        data_qualification_receipt=data_qualification_receipt,
    )
    observed_metadata = (
        development.receipt_sha256,
        development.fold_sha256,
        development.training_absolute_range,
        development.validation_absolute_range,
        development.outer_absolute_range,
        development.training.axis_id,
        development.inner_validation.axis_id,
    )
    expected_metadata = (
        expected_development.receipt_sha256,
        expected_development.fold_sha256,
        expected_development.training_absolute_range,
        expected_development.validation_absolute_range,
        expected_development.outer_absolute_range,
        expected_development.training.axis_id,
        expected_development.inner_validation.axis_id,
    )
    if observed_metadata != expected_metadata:
        raise Hold30DatasetError("development fold receipt or slice tensors were tampered")
    try:
        outcome = verify_hold30_cohort_receipt(cohort_selection_receipt)
    except Hold30CoordinationError as exc:
        raise Hold30DatasetError(f"invalid checkpoint selection receipt: {exc}") from exc
    if (
        outcome.identity.fold_index != development.fold.fold_index
        or outcome.identity.fold_sha256 != development.fold_sha256
        or outcome.identity.inner_validation_sequence_sha256
        != development.inner_validation.axis_id
    ):
        raise Hold30DatasetError("checkpoint selection does not bind this development fold")
    outer = _slice_sequence(parent, development.outer_absolute_range)
    _validate_evaluation_roles(
        outer,
        start=development.outer_absolute_range[0],
        score_range=development.fold.outer_score,
        support_range=development.fold.outer_support,
        name="outer",
    )
    cohort_sha = _require_digest(
        "cohort receipt", cohort_selection_receipt["receipt_sha256"]
    )
    payload = {
        "schema": "rl-quant.hold30.outer-fold-access",
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "fold_index": development.fold.fold_index,
        "parent_axis_id": parent.axis_id,
        "development_receipt_sha256": development.receipt_sha256,
        "data_qualification_sha256": qualification_sha,
        "cohort_receipt_sha256": cohort_sha,
        "access_marker_sha256": marker,
        "absolute_range": list(development.outer_absolute_range),
        "outer_axis_id": outer.axis_id,
        "selected_update": outcome.selected_validation.update,
        "outer_access": True,
    }
    return Hold30OuterFold(
        fold_index=development.fold.fold_index,
        sequence=outer,
        absolute_range=development.outer_absolute_range,
        development_receipt_sha256=development.receipt_sha256,
        cohort_receipt_sha256=cohort_sha,
        access_marker_sha256=marker,
        receipt_sha256=sha256_payload(payload),
    )


__all__ = [
    "Hold30DevelopmentFold",
    "Hold30OuterFold",
    "materialize_hold30_development_fold",
    "materialize_hold30_outer_fold",
]
