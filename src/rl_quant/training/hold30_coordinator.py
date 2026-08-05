"""Synchronous five-seed checkpoint coordination for Hold-30 v2.

The trial driver owns optimizer steps and checkpoint durability.  This module
owns the higher-level rule that all five seeds advance and validate at the same
update, that early stopping applies to the deployed five-member ensemble, and
that one shared update is selected without searching combinations of
per-seed checkpoints.

Only an explicitly bound inner-validation trace is accepted here.  The outer
score is deliberately absent from every callback and data structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION, resolve_hold30_setting
from rl_quant.protocol.hold30_freeze import (
    HOLD30_FOLDS,
    HOLD30_SEEDS,
    sha256_payload,
)


HOLD30_VALIDATION_CADENCE = 8
HOLD30_MIN_UPDATES = 32
HOLD30_MAX_UPDATES = 128
HOLD30_VALIDATION_PATIENCE = 4
HOLD30_SELECTION_TOLERANCE = 1e-4
HOLD30_VALIDATION_COST_BPS = 20.0
HOLD30_ENSEMBLE_MEMBERS = 5
HOLD30_COHORT_FINALIZATION_SCHEMA = "rl-quant.hold30.cohort-finalization"
HOLD30_TRIAL_FINALIZATION_SCHEMA = "rl-quant.hold30.trial-cohort-finalization"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class Hold30CoordinationError(RuntimeError):
    """A seed cohort or its validation evidence violates the frozen rule."""


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_finite(name: str, value: float, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite scalar")
    if not math.isfinite(float(value)) or (nonnegative and float(value) < 0):
        raise ValueError(f"{name} must be a finite{' non-negative' if nonnegative else ''} scalar")


@dataclass(frozen=True, slots=True)
class Hold30CohortIdentity:
    setting_id: str
    fold_index: int
    executable_manifest_sha256: str
    fold_sha256: str
    inner_validation_sequence_sha256: str
    protocol_generation: str = HOLD30_PROTOCOL_GENERATION

    def __post_init__(self) -> None:
        resolve_hold30_setting(self.setting_id)
        if self.protocol_generation != HOLD30_PROTOCOL_GENERATION:
            raise ValueError("cohort protocol generation mismatch")
        if (
            isinstance(self.fold_index, bool)
            or not isinstance(self.fold_index, int)
            or self.fold_index not in range(HOLD30_FOLDS)
        ):
            raise ValueError(f"fold_index must be in [0, {HOLD30_FOLDS - 1}]")
        for name in (
            "executable_manifest_sha256",
            "fold_sha256",
            "inner_validation_sequence_sha256",
        ):
            _require_digest(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class Hold30CheckpointReference:
    seed: int
    update: int
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_receipt_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed not in HOLD30_SEEDS:
            raise ValueError(f"seed must be one of {HOLD30_SEEDS}")
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or not 0 <= self.update <= HOLD30_MAX_UPDATES
        ):
            raise ValueError(f"update must be in [0, {HOLD30_MAX_UPDATES}]")
        if self.update and self.update % HOLD30_VALIDATION_CADENCE:
            raise ValueError("non-initial checkpoints must lie on the validation cadence")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty")
        _require_digest("checkpoint_sha256", self.checkpoint_sha256)
        _require_digest("checkpoint_receipt_sha256", self.checkpoint_receipt_sha256)


def _ordered_seed_refs(
    references: Sequence[Hold30CheckpointReference],
    *,
    expected_update: int,
) -> tuple[Hold30CheckpointReference, ...]:
    refs = tuple(references)
    if len(refs) != HOLD30_ENSEMBLE_MEMBERS:
        raise Hold30CoordinationError("a cohort checkpoint requires exactly five seeds")
    if tuple(reference.seed for reference in refs) != HOLD30_SEEDS:
        raise Hold30CoordinationError(
            f"checkpoint references must use ordered seeds {HOLD30_SEEDS}"
        )
    if any(reference.update != expected_update for reference in refs):
        raise Hold30CoordinationError("all five seed checkpoints must share one update")
    if len({reference.checkpoint_id for reference in refs}) != len(refs):
        raise Hold30CoordinationError("seed checkpoint IDs must be unique")
    return refs


@dataclass(frozen=True, slots=True)
class Hold30ValidationScore:
    update: int
    active_log_wealth: float
    discretionary_turnover: float
    trace_sha256: str
    inner_validation_sequence_sha256: str
    role: str = "inner_validation"
    cost_bps: float = HOLD30_VALIDATION_COST_BPS
    continuing_wealth: bool = True
    outer_access: bool = False
    ensemble_member_count: int = HOLD30_ENSEMBLE_MEMBERS

    def __post_init__(self) -> None:
        if (
            isinstance(self.update, bool)
            or not isinstance(self.update, int)
            or self.update <= 0
            or self.update > HOLD30_MAX_UPDATES
            or self.update % HOLD30_VALIDATION_CADENCE
        ):
            raise ValueError("validation update is outside the frozen cadence")
        _require_finite("active_log_wealth", self.active_log_wealth)
        _require_finite(
            "discretionary_turnover", self.discretionary_turnover, nonnegative=True
        )
        for name in ("trace_sha256", "inner_validation_sequence_sha256"):
            _require_digest(name, getattr(self, name))
        if self.role != "inner_validation":
            raise ValueError("checkpoint selection accepts only inner_validation evidence")
        if float(self.cost_bps) != HOLD30_VALIDATION_COST_BPS:
            raise ValueError("checkpoint selection requires continuing 20-bp validation")
        if self.continuing_wealth is not True:
            raise ValueError("checkpoint selection requires continuing wealth")
        if self.outer_access is not False:
            raise ValueError("outer-score access is forbidden during checkpoint selection")
        if self.ensemble_member_count != HOLD30_ENSEMBLE_MEMBERS:
            raise ValueError("validation must use the exact five-member ensemble")


@dataclass(frozen=True, slots=True)
class Hold30ValidationRecord:
    score: Hold30ValidationScore
    checkpoints: tuple[Hold30CheckpointReference, ...]

    def __post_init__(self) -> None:
        _ordered_seed_refs(self.checkpoints, expected_update=self.score.update)

    @property
    def update(self) -> int:
        return self.score.update

    @property
    def bundle_id(self) -> str:
        return "|".join(reference.checkpoint_id for reference in self.checkpoints)


@dataclass(frozen=True, slots=True)
class Hold30CohortOutcome:
    identity: Hold30CohortIdentity
    initial_checkpoints: tuple[Hold30CheckpointReference, ...]
    validations: tuple[Hold30ValidationRecord, ...]
    selected_validation: Hold30ValidationRecord
    final_checkpoints: tuple[Hold30CheckpointReference, ...]
    stopped_update: int
    stop_reason: str

    def __post_init__(self) -> None:
        _ordered_seed_refs(self.initial_checkpoints, expected_update=0)
        _validate_validation_prefix(self.validations, self.identity)
        if self.selected_validation not in self.validations:
            raise Hold30CoordinationError("selected validation is outside the cohort history")
        if self.stopped_update != self.validations[-1].update:
            raise Hold30CoordinationError("stopped_update must equal the final validation update")
        _ordered_seed_refs(self.final_checkpoints, expected_update=self.stopped_update)
        if self.final_checkpoints != self.validations[-1].checkpoints:
            raise Hold30CoordinationError("final checkpoints must be the final validated cohort")
        if self.stop_reason not in {"validation_patience_exhausted", "maximum_updates"}:
            raise Hold30CoordinationError("unknown cohort stop reason")
        expected_stop = _validated_stop_reason(self.validations)
        if self.stop_reason != expected_stop:
            raise Hold30CoordinationError(
                "cohort stop reason/history does not satisfy the frozen patience rule"
            )
        expected_reason = _stop_reason_for_prefix(self.validations)
        if expected_reason != self.stop_reason:
            raise Hold30CoordinationError(
                "cohort stop reason/update does not satisfy the frozen patience rule"
            )
        selected = select_hold30_shared_checkpoint(self.validations, self.identity)
        if selected != self.selected_validation:
            raise Hold30CoordinationError("selected checkpoint does not satisfy the frozen rule")

    def receipt(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "rl-quant.hold30.seed-cohort-selection",
            "schema_version": 1,
            "protocol_generation": HOLD30_PROTOCOL_GENERATION,
            "identity": asdict(self.identity),
            "identity_sha256": sha256_payload(asdict(self.identity)),
            "selection_rule": {
                "validation_cadence": HOLD30_VALIDATION_CADENCE,
                "minimum_updates": HOLD30_MIN_UPDATES,
                "maximum_updates": HOLD30_MAX_UPDATES,
                "patience_validations": HOLD30_VALIDATION_PATIENCE,
                "selection_tolerance_active_log_wealth": HOLD30_SELECTION_TOLERANCE,
                "priority": ["earliest_update", "lower_turnover", "lexical_bundle_id"],
                "strict_new_max_resets_patience": True,
            },
            "initial_checkpoints": [asdict(value) for value in self.initial_checkpoints],
            "validations": [
                {
                    "score": asdict(record.score),
                    "checkpoints": [asdict(value) for value in record.checkpoints],
                    "bundle_id": record.bundle_id,
                }
                for record in self.validations
            ],
            "selected_update": self.selected_validation.update,
            "selected_checkpoints": [
                asdict(value) for value in self.selected_validation.checkpoints
            ],
            "final_checkpoints": [asdict(value) for value in self.final_checkpoints],
            "stopped_update": self.stopped_update,
            "stop_reason": self.stop_reason,
            "outer_access": False,
            "checkpoint_selection_complete": True,
            "scientific_qualification": False,
            "promotion_authorized": False,
        }
        payload["receipt_sha256"] = sha256_payload(payload)
        return payload


def _stop_reason_for_prefix(
    validations: Sequence[Hold30ValidationRecord],
) -> str | None:
    best = -math.inf
    without_new_max = 0
    for index, record in enumerate(validations):
        value = float(record.score.active_log_wealth)
        if value > best:
            best = value
            without_new_max = 0
        else:
            without_new_max += 1
        if (
            record.update >= HOLD30_MIN_UPDATES
            and without_new_max >= HOLD30_VALIDATION_PATIENCE
        ):
            return (
                "validation_patience_exhausted"
                if index == len(validations) - 1
                else None
            )
    if validations and validations[-1].update == HOLD30_MAX_UPDATES:
        return "maximum_updates"
    return None


def _validate_validation_prefix(
    validations: Sequence[Hold30ValidationRecord],
    identity: Hold30CohortIdentity,
) -> tuple[Hold30ValidationRecord, ...]:
    rows = tuple(validations)
    if not rows:
        raise Hold30CoordinationError("a seed cohort has no validation evidence")
    expected_updates = tuple(
        range(HOLD30_VALIDATION_CADENCE, rows[-1].update + 1, HOLD30_VALIDATION_CADENCE)
    )
    if tuple(row.update for row in rows) != expected_updates:
        raise Hold30CoordinationError("validation updates must form an exact cadence prefix")
    traces: set[str] = set()
    for row in rows:
        if row.score.inner_validation_sequence_sha256 != identity.inner_validation_sequence_sha256:
            raise Hold30CoordinationError("validation used an unbound sequence")
        _ordered_seed_refs(row.checkpoints, expected_update=row.update)
        if row.score.trace_sha256 in traces:
            raise Hold30CoordinationError("validation trace digests must be update-specific")
        traces.add(row.score.trace_sha256)
    return rows


def _validated_stop_reason(
    validations: Sequence[Hold30ValidationRecord],
) -> str:
    """Return the only legal terminal reason for an exact validation prefix."""

    best = -math.inf
    stale = 0
    rows = tuple(validations)
    for index, row in enumerate(rows):
        value = float(row.score.active_log_wealth)
        if value > best:
            best = value
            stale = 0
        else:
            stale += 1
        patience_hit = (
            row.update >= HOLD30_MIN_UPDATES and stale >= HOLD30_VALIDATION_PATIENCE
        )
        if patience_hit:
            if index != len(rows) - 1:
                raise Hold30CoordinationError(
                    "validation continued after frozen patience was exhausted"
                )
            return "validation_patience_exhausted"
    if rows[-1].update != HOLD30_MAX_UPDATES:
        raise Hold30CoordinationError(
            "cohort validation prefix is an interruption, not a terminal outcome"
        )
    return "maximum_updates"


def select_hold30_shared_checkpoint(
    validations: Sequence[Hold30ValidationRecord],
    identity: Hold30CohortIdentity,
) -> Hold30ValidationRecord:
    """Apply the common earliest-within-one-basis-point selection rule."""

    rows = _validate_validation_prefix(validations, identity)
    selectable = tuple(row for row in rows if row.update >= HOLD30_MIN_UPDATES)
    if not selectable:
        raise Hold30CoordinationError(
            "checkpoint selection requires the frozen 32-update minimum"
        )
    maximum = max(float(row.score.active_log_wealth) for row in selectable)
    eligible = tuple(
        row
        for row in selectable
        if float(row.score.active_log_wealth) >= maximum - HOLD30_SELECTION_TOLERANCE
    )
    return min(
        eligible,
        key=lambda row: (
            row.update,
            float(row.score.discretionary_turnover),
            row.bundle_id,
        ),
    )


AdvanceCohort = Callable[[int], Sequence[Hold30CheckpointReference]]
ValidateCohort = Callable[
    [int, tuple[Hold30CheckpointReference, ...]], Hold30ValidationScore
]


def coordinate_hold30_seed_cohort(
    identity: Hold30CohortIdentity,
    initial_checkpoints: Sequence[Hold30CheckpointReference],
    *,
    advance_cohort: AdvanceCohort,
    validate_ensemble: ValidateCohort,
) -> Hold30CohortOutcome:
    """Advance five seeds synchronously and select one shared update.

    ``advance_cohort`` is the only training boundary and must durably return
    five same-update checkpoint receipts before validation begins.
    ``validate_ensemble`` receives only those receipts and is required to bind
    the predeclared inner-validation sequence in its result.
    """

    initial = _ordered_seed_refs(initial_checkpoints, expected_update=0)
    validations: list[Hold30ValidationRecord] = []
    best = -math.inf
    validations_without_new_max = 0
    stop_reason = "maximum_updates"
    for update in range(
        HOLD30_VALIDATION_CADENCE,
        HOLD30_MAX_UPDATES + 1,
        HOLD30_VALIDATION_CADENCE,
    ):
        checkpoints = _ordered_seed_refs(
            advance_cohort(update), expected_update=update
        )
        score = validate_ensemble(update, checkpoints)
        if not isinstance(score, Hold30ValidationScore):
            raise Hold30CoordinationError(
                "validation callback must return Hold30ValidationScore"
            )
        if score.update != update:
            raise Hold30CoordinationError("validation callback changed the update")
        if score.inner_validation_sequence_sha256 != identity.inner_validation_sequence_sha256:
            raise Hold30CoordinationError("validation callback used an unbound sequence")
        record = Hold30ValidationRecord(score, checkpoints)
        validations.append(record)
        value = float(score.active_log_wealth)
        if value > best:
            best = value
            validations_without_new_max = 0
        else:
            validations_without_new_max += 1
        if (
            update >= HOLD30_MIN_UPDATES
            and validations_without_new_max >= HOLD30_VALIDATION_PATIENCE
        ):
            stop_reason = "validation_patience_exhausted"
            break
    frozen = tuple(validations)
    selected = select_hold30_shared_checkpoint(frozen, identity)
    return Hold30CohortOutcome(
        identity=identity,
        initial_checkpoints=initial,
        validations=frozen,
        selected_validation=selected,
        final_checkpoints=frozen[-1].checkpoints,
        stopped_update=frozen[-1].update,
        stop_reason=stop_reason,
    )


def verify_hold30_cohort_receipt(
    receipt: Mapping[str, Any],
) -> Hold30CohortOutcome:
    """Reconstruct and verify every edge of a cohort-selection receipt."""

    expected_fields = {
        "schema",
        "schema_version",
        "protocol_generation",
        "identity",
        "identity_sha256",
        "selection_rule",
        "initial_checkpoints",
        "validations",
        "selected_update",
        "selected_checkpoints",
        "final_checkpoints",
        "stopped_update",
        "stop_reason",
        "outer_access",
        "checkpoint_selection_complete",
        "scientific_qualification",
        "promotion_authorized",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise Hold30CoordinationError("cohort receipt is partial or has unknown fields")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    if claimed != sha256_payload(unsigned):
        raise Hold30CoordinationError("cohort receipt self-hash mismatch")
    if (
        receipt["schema"] != "rl-quant.hold30.seed-cohort-selection"
        or receipt["schema_version"] != 1
        or receipt["protocol_generation"] != HOLD30_PROTOCOL_GENERATION
    ):
        raise Hold30CoordinationError("unsupported cohort receipt schema")
    if (
        receipt["outer_access"] is not False
        or receipt["checkpoint_selection_complete"] is not True
        or receipt["scientific_qualification"] is not False
        or receipt["promotion_authorized"] is not False
    ):
        raise Hold30CoordinationError("cohort receipt status flags are unsafe")
    try:
        identity_payload = receipt["identity"]
        if not isinstance(identity_payload, Mapping):
            raise TypeError("identity")
        identity = Hold30CohortIdentity(**identity_payload)
        if receipt["identity_sha256"] != sha256_payload(asdict(identity)):
            raise Hold30CoordinationError("cohort identity digest mismatch")
        expected_rule = {
            "validation_cadence": HOLD30_VALIDATION_CADENCE,
            "minimum_updates": HOLD30_MIN_UPDATES,
            "maximum_updates": HOLD30_MAX_UPDATES,
            "patience_validations": HOLD30_VALIDATION_PATIENCE,
            "selection_tolerance_active_log_wealth": HOLD30_SELECTION_TOLERANCE,
            "priority": ["earliest_update", "lower_turnover", "lexical_bundle_id"],
            "strict_new_max_resets_patience": True,
        }
        if receipt["selection_rule"] != expected_rule:
            raise Hold30CoordinationError("cohort selection rule drifted")

        def parse_refs(value: Any) -> tuple[Hold30CheckpointReference, ...]:
            if not isinstance(value, list):
                raise TypeError("checkpoint references")
            return tuple(Hold30CheckpointReference(**item) for item in value)

        initial = parse_refs(receipt["initial_checkpoints"])
        validation_rows = receipt["validations"]
        if not isinstance(validation_rows, list):
            raise TypeError("validations")
        validations: list[Hold30ValidationRecord] = []
        for row in validation_rows:
            if not isinstance(row, Mapping) or set(row) != {
                "score",
                "checkpoints",
                "bundle_id",
            }:
                raise Hold30CoordinationError(
                    "cohort validation row is partial or has unknown fields"
                )
            score = Hold30ValidationScore(**row["score"])
            record = Hold30ValidationRecord(score, parse_refs(row["checkpoints"]))
            if row["bundle_id"] != record.bundle_id:
                raise Hold30CoordinationError("validation checkpoint bundle ID mismatch")
            validations.append(record)
        selected_refs = parse_refs(receipt["selected_checkpoints"])
        selected_update = receipt["selected_update"]
        selected = next(
            (
                row
                for row in validations
                if row.update == selected_update and row.checkpoints == selected_refs
            ),
            None,
        )
        if selected is None:
            raise Hold30CoordinationError(
                "selected checkpoints do not identify a validation row"
            )
        outcome = Hold30CohortOutcome(
            identity=identity,
            initial_checkpoints=initial,
            validations=tuple(validations),
            selected_validation=selected,
            final_checkpoints=parse_refs(receipt["final_checkpoints"]),
            stopped_update=receipt["stopped_update"],
            stop_reason=receipt["stop_reason"],
        )
    except Hold30CoordinationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise Hold30CoordinationError("cohort receipt payload is malformed") from exc
    if outcome.receipt() != dict(receipt):
        raise Hold30CoordinationError("cohort receipt is not canonical")
    return outcome


def checkpoint_reference_from_trial(
    trial_root: str | Path,
    update: int,
) -> Hold30CheckpointReference:
    """Construct a coordinator reference only from a verified retained artifact."""

    from rl_quant.training.hold30_driver import inspect_hold30_trial_checkpoint

    evidence = inspect_hold30_trial_checkpoint(trial_root, update)
    return Hold30CheckpointReference(
        seed=evidence["seed"],
        update=update,
        checkpoint_id=evidence["checkpoint_id"],
        checkpoint_sha256=evidence["checkpoint_sha256"],
        checkpoint_receipt_sha256=evidence["checkpoint_receipt_sha256"],
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Hold30CoordinationError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Hold30CoordinationError(f"required cohort artifact is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise Hold30CoordinationError(f"cohort artifact must be a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Hold30CoordinationError(f"invalid cohort JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise Hold30CoordinationError("cohort JSON artifact must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Hold30CoordinationError(
                f"refusing to overwrite existing cohort artifact: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _reference_payload(reference: Hold30CheckpointReference) -> dict[str, Any]:
    return asdict(reference)


def _build_cohort_finalization(
    outcome: Hold30CohortOutcome,
    trial_roots: Mapping[int, Path],
    receipt_parent: Path,
) -> dict[str, Any]:
    from rl_quant.training.hold30_driver import inspect_hold30_trial_checkpoints

    verified_outcome = verify_hold30_cohort_receipt(outcome.receipt())
    if verified_outcome != outcome:
        raise Hold30CoordinationError("cohort outcome is not canonical")
    if outcome.stopped_update >= HOLD30_MAX_UPDATES:
        raise Hold30CoordinationError(
            "maximum-update cohorts use ordinary complete-trial finalization"
        )
    if set(trial_roots) != set(HOLD30_SEEDS):
        raise Hold30CoordinationError(f"trial roots must bind exact seeds {HOLD30_SEEDS}")
    resolved = {seed: Path(trial_roots[seed]).resolve() for seed in HOLD30_SEEDS}
    if len(set(resolved.values())) != HOLD30_ENSEMBLE_MEMBERS:
        raise Hold30CoordinationError("every seed must have a distinct trial root")

    expected_by_update: dict[int, tuple[Hold30CheckpointReference, ...]] = {0: outcome.initial_checkpoints}
    expected_by_update.update({row.update: row.checkpoints for row in outcome.validations})
    actual_lists: dict[int, list[Hold30CheckpointReference]] = {
        update: [] for update in expected_by_update
    }
    trial_evidence: dict[int, dict[str, Any]] = {}
    updates = tuple(expected_by_update)
    for seed_index, seed in enumerate(HOLD30_SEEDS):
        evidence_by_update = inspect_hold30_trial_checkpoints(resolved[seed], updates)
        for update, expected_refs in expected_by_update.items():
            expected = expected_refs[seed_index]
            evidence = evidence_by_update[update]
            actual = Hold30CheckpointReference(
                seed=evidence["seed"],
                update=update,
                checkpoint_id=evidence["checkpoint_id"],
                checkpoint_sha256=evidence["checkpoint_sha256"],
                checkpoint_receipt_sha256=evidence["checkpoint_receipt_sha256"],
            )
            if actual != expected:
                raise Hold30CoordinationError(
                    f"seed {seed} update {update} checkpoint does not match the cohort receipt"
                )
            trial = evidence["trial"]
            expected_identity = outcome.identity
            identity_checks = {
                "protocol_generation": expected_identity.protocol_generation,
                "setting_id": expected_identity.setting_id,
                "fold_index": expected_identity.fold_index,
                "seed": seed,
                "executable_manifest_sha256": expected_identity.executable_manifest_sha256,
                "fold_sha256": expected_identity.fold_sha256,
            }
            if any(trial.get(name) != value for name, value in identity_checks.items()):
                raise Hold30CoordinationError("trial identity differs from cohort identity")
            if evidence["retained_update_count"] != outcome.stopped_update:
                raise Hold30CoordinationError(
                    "every seed must stop at exactly the common final update"
                )
            trial_evidence[seed] = evidence
            actual_lists[update].append(actual)
    actual_by_update = {
        update: tuple(values) for update, values in actual_lists.items()
    }

    selected_update = outcome.selected_validation.update
    rows = []
    for seed in HOLD30_SEEDS:
        root = resolved[seed]
        evidence = trial_evidence[seed]
        rows.append(
            {
                "seed": seed,
                "trial_root": os.path.relpath(root, receipt_parent),
                "identity_receipt_sha256": evidence["identity_receipt_sha256"],
                "retained_update_count": evidence["retained_update_count"],
                "initial_checkpoint": _reference_payload(actual_by_update[0][HOLD30_SEEDS.index(seed)]),
                "selected_checkpoint": _reference_payload(
                    actual_by_update[selected_update][HOLD30_SEEDS.index(seed)]
                ),
                "final_checkpoint": _reference_payload(
                    actual_by_update[outcome.stopped_update][HOLD30_SEEDS.index(seed)]
                ),
            }
        )
    cohort_receipt = outcome.receipt()
    payload: dict[str, Any] = {
        "schema": HOLD30_COHORT_FINALIZATION_SCHEMA,
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "cohort_selection": cohort_receipt,
        "cohort_selection_receipt_sha256": cohort_receipt["receipt_sha256"],
        "selected_update": selected_update,
        "stopped_update": outcome.stopped_update,
        "stop_reason": outcome.stop_reason,
        "trial_artifacts": rows,
        "initial_selected_final_retained": True,
        "per_seed_update_mix": False,
        "production_early_stop_finalized": outcome.stopped_update < HOLD30_MAX_UPDATES,
        "scientific_qualification": False,
        "promotion_authorized": False,
    }
    payload["artifact_graph_sha256"] = sha256_payload(rows)
    payload["receipt_sha256"] = sha256_payload(payload)
    return payload


def verify_hold30_cohort_finalization(
    receipt: Mapping[str, Any],
    *,
    receipt_path: str | Path,
) -> Hold30CohortOutcome:
    """Verify selection semantics and every retained seed artifact."""

    expected_fields = {
        "schema",
        "schema_version",
        "protocol_generation",
        "cohort_selection",
        "cohort_selection_receipt_sha256",
        "selected_update",
        "stopped_update",
        "stop_reason",
        "trial_artifacts",
        "initial_selected_final_retained",
        "per_seed_update_mix",
        "production_early_stop_finalized",
        "scientific_qualification",
        "promotion_authorized",
        "artifact_graph_sha256",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise Hold30CoordinationError("cohort finalization is partial or has unknown fields")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    if claimed != sha256_payload(unsigned):
        raise Hold30CoordinationError("cohort finalization self-hash mismatch")
    if (
        receipt["schema"] != HOLD30_COHORT_FINALIZATION_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["protocol_generation"] != HOLD30_PROTOCOL_GENERATION
        or receipt["initial_selected_final_retained"] is not True
        or receipt["per_seed_update_mix"] is not False
        or receipt["scientific_qualification"] is not False
        or receipt["promotion_authorized"] is not False
    ):
        raise Hold30CoordinationError("cohort finalization status/schema is unsafe")
    cohort_receipt = receipt["cohort_selection"]
    outcome = verify_hold30_cohort_receipt(cohort_receipt)
    if receipt["cohort_selection_receipt_sha256"] != cohort_receipt["receipt_sha256"]:
        raise Hold30CoordinationError("cohort selection receipt digest mismatch")
    if (
        receipt["selected_update"] != outcome.selected_validation.update
        or receipt["stopped_update"] != outcome.stopped_update
        or receipt["stop_reason"] != outcome.stop_reason
        or receipt["production_early_stop_finalized"]
        is not (outcome.stopped_update < HOLD30_MAX_UPDATES)
    ):
        raise Hold30CoordinationError("cohort finalization stop/selection binding mismatch")
    rows = receipt["trial_artifacts"]
    if not isinstance(rows, list) or len(rows) != HOLD30_ENSEMBLE_MEMBERS:
        raise Hold30CoordinationError("cohort finalization must bind five trial artifacts")
    if receipt["artifact_graph_sha256"] != sha256_payload(rows):
        raise Hold30CoordinationError("cohort finalization artifact graph mismatch")
    roots: dict[int, Path] = {}
    parent = Path(receipt_path).resolve().parent
    for row in rows:
        if not isinstance(row, Mapping) or row.get("seed") not in HOLD30_SEEDS:
            raise Hold30CoordinationError("cohort trial-artifact row is malformed")
        seed = int(row["seed"])
        if seed in roots:
            raise Hold30CoordinationError("cohort finalization duplicates a seed")
        relative = row.get("trial_root")
        if not isinstance(relative, str) or not relative:
            raise Hold30CoordinationError("cohort trial root is missing")
        roots[seed] = (parent / relative).resolve()
    expected = _build_cohort_finalization(outcome, roots, parent)
    if expected != dict(receipt):
        raise Hold30CoordinationError("cohort finalization is not canonical")
    return outcome


def publish_hold30_cohort_finalization(
    outcome: Hold30CohortOutcome,
    trial_roots: Mapping[int, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Exclusively publish one verified five-seed early-stop authorization."""

    path = Path(output_path).resolve()
    resolved = {seed: Path(root).resolve() for seed, root in trial_roots.items()}
    for root in resolved.values():
        try:
            path.relative_to(root)
        except ValueError:
            pass
        else:
            raise Hold30CoordinationError(
                "cohort finalization receipt must live outside every trial root"
            )
    if path.exists():
        existing = _read_json_object(path)
        verified = verify_hold30_cohort_finalization(existing, receipt_path=path)
        if verified != outcome:
            raise Hold30CoordinationError("existing cohort finalization has a different identity")
        bound_roots = {
            int(row["seed"]): (path.parent / row["trial_root"]).resolve()
            for row in existing["trial_artifacts"]
        }
        if bound_roots != resolved:
            raise Hold30CoordinationError("existing cohort finalization binds different trial roots")
        expected = existing
    else:
        expected = _build_cohort_finalization(outcome, resolved, path.parent)
        _write_new_json(path, expected)
    file_sha = _file_sha256(path)
    for row in expected["trial_artifacts"]:
        seed = int(row["seed"])
        root = resolved[seed]
        marker_path = root / "cohort-finalization.json"
        marker: dict[str, Any] = {
            "schema": HOLD30_TRIAL_FINALIZATION_SCHEMA,
            "schema_version": 1,
            "protocol_generation": HOLD30_PROTOCOL_GENERATION,
            "seed": seed,
            "cohort_finalization_path": os.path.relpath(path, root),
            "cohort_finalization_file_sha256": file_sha,
            "cohort_finalization_receipt_sha256": expected["receipt_sha256"],
            "selected_update": expected["selected_update"],
            "stopped_update": expected["stopped_update"],
            "stop_reason": expected["stop_reason"],
        }
        marker["receipt_sha256"] = sha256_payload(marker)
        if marker_path.exists():
            existing_marker = _read_json_object(marker_path)
            if existing_marker != marker:
                raise Hold30CoordinationError(
                    f"seed {seed} has an unsafe existing cohort marker"
                )
        else:
            _write_new_json(marker_path, marker)
    return expected


__all__ = [
    "HOLD30_COHORT_FINALIZATION_SCHEMA",
    "HOLD30_TRIAL_FINALIZATION_SCHEMA",
    "HOLD30_ENSEMBLE_MEMBERS",
    "HOLD30_MAX_UPDATES",
    "HOLD30_MIN_UPDATES",
    "HOLD30_SELECTION_TOLERANCE",
    "HOLD30_VALIDATION_CADENCE",
    "HOLD30_VALIDATION_COST_BPS",
    "HOLD30_VALIDATION_PATIENCE",
    "Hold30CheckpointReference",
    "Hold30CohortIdentity",
    "Hold30CohortOutcome",
    "Hold30CoordinationError",
    "Hold30ValidationRecord",
    "Hold30ValidationScore",
    "checkpoint_reference_from_trial",
    "coordinate_hold30_seed_cohort",
    "publish_hold30_cohort_finalization",
    "select_hold30_shared_checkpoint",
    "verify_hold30_cohort_receipt",
    "verify_hold30_cohort_finalization",
]
