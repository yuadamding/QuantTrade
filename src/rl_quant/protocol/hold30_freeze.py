"""Fail-closed freeze geometry for ``prelockbox-hold30-mech8-v2``.

The renderer is deliberately pure: it validates an already materialized,
point-in-time decision axis and returns content-addressable Python objects.  It
does not discover files, infer missing provenance, touch a lockbox, or launch
compute.  A workflow may serialize the returned values only after supplying
all required bindings explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
from typing import Any, Sequence

from rl_quant.protocol.hold30 import (
    HOLD30_BASE_DESIGN,
    HOLD30_MECH8_SETTINGS,
    HOLD30_PROTOCOL_GENERATION,
)


HOLD30_FOLDS = 6
HOLD30_SEEDS = (17, 29, 43, 71, 101)
HOLD30_GPUS_PER_SETTING = 2
HOLD30_GPU_PRODUCT = "NVIDIA-H100-80GB-HBM3"
HOLD30_MIN_AXIS_POSITIONS = 1811
HOLD30_FOLD_ADVANCE = 224
HOLD30_TRAILING_GEOMETRY = 1339
HOLD30_MIN_INITIAL_TRAIN = 472
HOLD30_SCORE_DAYS = 63
HOLD30_SUPPORT_POSITIONS = 31
HOLD30_EMBARGO_POSITIONS = 5
HOLD30_PRELOCKBOX_CUTOFF = date(2026, 1, 1)


class Hold30FreezeError(ValueError):
    """A launch-affecting protocol invariant is absent or inconsistent."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _parse_session(value: str) -> date:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise Hold30FreezeError(f"invalid decision-axis timestamp {value!r}") from exc
    return parsed.date()


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Hold30FreezeError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise Hold30FreezeError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Hold30Fold:
    fold_index: int
    expanding_train: tuple[int, int]
    train_validation_separation: tuple[int, int]
    inner_validation: tuple[int, int]
    validation_support: tuple[int, int]
    outer_score: tuple[int, int]
    outer_support: tuple[int, int]
    embargo: tuple[int, int]
    training_warmup: tuple[int, int]
    training_anchors: tuple[int, int]
    training_support: tuple[int, int]
    training_terminal_observation: int

    def intervals(self) -> tuple[tuple[int, int], ...]:
        return (
            self.expanding_train,
            self.train_validation_separation,
            self.inner_validation,
            self.validation_support,
            self.outer_score,
            self.outer_support,
            self.embargo,
        )


def render_hold30_folds(decision_axis: Sequence[str]) -> tuple[Hold30Fold, ...]:
    """Materialize the six frozen folds from an explicit pre-2026 axis."""

    axis = tuple(decision_axis)
    if len(axis) < HOLD30_MIN_AXIS_POSITIONS:
        raise Hold30FreezeError(
            f"Hold-30 requires N >= {HOLD30_MIN_AXIS_POSITIONS}; got {len(axis)}"
        )
    parsed = tuple(_parse_session(value) for value in axis)
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise Hold30FreezeError("decision axis must be strictly increasing and unique")
    if parsed[-1] >= HOLD30_PRELOCKBOX_CUTOFF:
        raise Hold30FreezeError("every materialized position must be before 2026-01-01")

    initial_train = len(axis) - HOLD30_TRAILING_GEOMETRY
    if initial_train < HOLD30_MIN_INITIAL_TRAIN:
        raise Hold30FreezeError(
            f"initial expanding train must be >= {HOLD30_MIN_INITIAL_TRAIN}; got {initial_train}"
        )
    folds: list[Hold30Fold] = []
    for fold_index in range(HOLD30_FOLDS):
        anchor = initial_train + HOLD30_FOLD_ADVANCE * fold_index
        embargo_end = anchor + 224 if fold_index < HOLD30_FOLDS - 1 else anchor + 219
        fold = Hold30Fold(
            fold_index=fold_index,
            expanding_train=(0, anchor),
            train_validation_separation=(anchor, anchor + 31),
            inner_validation=(anchor + 31, anchor + 94),
            validation_support=(anchor + 94, anchor + 125),
            outer_score=(anchor + 125, anchor + 188),
            outer_support=(anchor + 188, anchor + 219),
            embargo=(anchor + 219, embargo_end),
            training_warmup=(0, 63),
            training_anchors=(63, anchor - 31),
            training_support=(anchor - 31, anchor - 1),
            training_terminal_observation=anchor - 1,
        )
        if fold.training_anchors[1] - fold.training_anchors[0] < 378:
            raise Hold30FreezeError("each fold needs at least 378 training anchors")
        if fold.inner_validation[1] - fold.inner_validation[0] != HOLD30_SCORE_DAYS:
            raise AssertionError("inner-validation geometry drifted")
        if fold.outer_score[1] - fold.outer_score[0] != HOLD30_SCORE_DAYS:
            raise AssertionError("outer-score geometry drifted")
        if fold.validation_support[1] - fold.validation_support[0] != HOLD30_SUPPORT_POSITIONS:
            raise AssertionError("validation support geometry drifted")
        if fold.outer_support[1] - fold.outer_support[0] != HOLD30_SUPPORT_POSITIONS:
            raise AssertionError("outer support geometry drifted")
        if fold.embargo[1] > len(axis):
            raise Hold30FreezeError("fold geometry exceeds the bound decision axis")
        nontraining = fold.intervals()[1:]
        for left, right in zip(nontraining, nontraining[1:]):
            if left[1] != right[0]:
                raise AssertionError("fold intervals are not exact adjacent partitions")
        folds.append(fold)
    if folds[-1].outer_support[1] != len(axis):
        raise Hold30FreezeError("final outer holding support must end at the final axis position")
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class Hold30FreezeBindings:
    repository_url: str
    git_commit: str
    git_tree: str
    clean_worktree: bool
    dirty_patch_sha256: str | None
    source_archive_sha256: str
    dependency_lock_sha256: str
    container_image_digest: str
    rfc_sha256: str
    base_experiment_sha256: str
    v2_specification_sha256: str
    data_snapshot_sha256: str
    decision_axis_sha256: str
    universe_events_sha256: str
    corporate_actions_sha256: str
    benchmark_trace_sha256: str
    split_arrays_sha256: str
    component_qualification_sha256: str
    software_qualification_sha256: str
    data_qualification_sha256: str
    capacity_qualification_sha256: str
    training_plan_sha256: str
    stage1_plan_sha256: str
    control_plan_sha256: str
    inference_plan_sha256: str
    artifact_inventory_sha256: str
    recovery_policy_sha256: str
    worker_template_sha256: str
    admitted_job_template_sha256: str
    namespace: str
    service_account: str
    executable_approval_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.repository_url:
            raise Hold30FreezeError("repository_url is required")
        if not isinstance(self.clean_worktree, bool):
            raise Hold30FreezeError("clean_worktree must be bool")
        for name in ("git_commit", "git_tree"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise Hold30FreezeError(
                    f"{name} must be a lowercase full 40-character Git SHA"
                )
        if self.clean_worktree and self.dirty_patch_sha256 is not None:
            raise Hold30FreezeError("a clean worktree cannot bind a dirty patch")
        if not self.clean_worktree and self.dirty_patch_sha256 is None:
            raise Hold30FreezeError("a dirty worktree must bind dirty_patch_sha256")
        digest_fields = (
            "source_archive_sha256",
            "dependency_lock_sha256",
            "rfc_sha256",
            "base_experiment_sha256",
            "v2_specification_sha256",
            "data_snapshot_sha256",
            "decision_axis_sha256",
            "universe_events_sha256",
            "corporate_actions_sha256",
            "benchmark_trace_sha256",
            "split_arrays_sha256",
            "component_qualification_sha256",
            "software_qualification_sha256",
            "data_qualification_sha256",
            "capacity_qualification_sha256",
            "training_plan_sha256",
            "stage1_plan_sha256",
            "control_plan_sha256",
            "inference_plan_sha256",
            "artifact_inventory_sha256",
            "recovery_policy_sha256",
            "worker_template_sha256",
            "admitted_job_template_sha256",
        )
        for name in digest_fields:
            _require_digest(name, getattr(self, name))
        if self.dirty_patch_sha256 is not None:
            _require_digest("dirty_patch_sha256", self.dirty_patch_sha256)
        if self.executable_approval_sha256 is not None:
            _require_digest("executable_approval_sha256", self.executable_approval_sha256)
        if not self.container_image_digest.startswith("sha256:"):
            raise Hold30FreezeError("container_image_digest must be digest-pinned")
        _require_digest("container_image_digest", self.container_image_digest[7:])
        if self.namespace != "yn-gpu-workload":
            raise Hold30FreezeError("Hold-30 namespace must be yn-gpu-workload")
        if not isinstance(self.service_account, str) or not self.service_account:
            raise Hold30FreezeError("service_account is required")


def hold30_trial_inventory() -> tuple[dict[str, Any], ...]:
    """Return the exact 8 x 6 x 5 training inventory (240 trials)."""

    rows: list[dict[str, Any]] = []
    for setting in HOLD30_MECH8_SETTINGS:
        for fold_index in range(HOLD30_FOLDS):
            for seed in HOLD30_SEEDS:
                rows.append(
                    {
                        "setting_index": setting.setting_index,
                        "setting_id": setting.setting_id,
                        "mechanism": setting.mechanism,
                        "fold_index": fold_index,
                        "seed": seed,
                        "promotion_eligible": setting.promotion_eligible,
                    }
                )
    return tuple(rows)


def render_hold30_manifest(
    decision_axis: Sequence[str],
    bindings: Hold30FreezeBindings,
    *,
    approval_state: str = "dry_run",
) -> dict[str, Any]:
    """Render a deterministic, non-discovering manifest.

    ``approval_state='executable'`` is accepted only as an explicit caller
    assertion after external approval.  Rendering never grants that approval.
    """

    if approval_state not in {"dry_run", "software_qualified", "executable"}:
        raise Hold30FreezeError("approval_state must be dry_run, software_qualified, or executable")
    if approval_state == "executable" and bindings.executable_approval_sha256 is None:
        raise Hold30FreezeError(
            "executable manifests require an explicit executable_approval_sha256"
        )
    if approval_state != "executable" and bindings.executable_approval_sha256 is not None:
        raise Hold30FreezeError(
            "executable approval cannot be attached to a non-executable manifest"
        )
    folds = render_hold30_folds(decision_axis)
    axis_hash = sha256_payload(tuple(decision_axis))
    if axis_hash != bindings.decision_axis_sha256:
        raise Hold30FreezeError("bound decision-axis digest does not match materialized axis")
    split_payload = [asdict(fold) for fold in folds]
    if sha256_payload(split_payload) != bindings.split_arrays_sha256:
        raise Hold30FreezeError("bound split-arrays digest does not match rendered folds")
    inventory = hold30_trial_inventory()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "design": HOLD30_BASE_DESIGN,
        "approval_state": approval_state,
        "render_grants_launch_authority": False,
        "lockbox_consumed": False,
        "decision_axis": {
            "count": len(decision_axis),
            "first": decision_axis[0],
            "last": decision_axis[-1],
            "sha256": axis_hash,
        },
        "compute": {
            "gpu_product": HOLD30_GPU_PRODUCT,
            "gpus_per_setting": HOLD30_GPUS_PER_SETTING,
            "world_size_per_trial": 2,
            "distributed_strategy": "explicit-sum-origin-shard-two-rank-v1",
            "concurrent_setting_workers": 8,
            "maximum_h100": 16,
            "namespace": bindings.namespace,
            "service_account": bindings.service_account,
            "worker_template_sha256": bindings.worker_template_sha256,
            "admitted_job_template_sha256": bindings.admitted_job_template_sha256,
            "scientific_fields_inferred_from_gpu_count": False,
        },
        "plans": {
            "training_plan_sha256": bindings.training_plan_sha256,
            "stage1_plan_sha256": bindings.stage1_plan_sha256,
            "control_plan_sha256": bindings.control_plan_sha256,
            "inference_plan_sha256": bindings.inference_plan_sha256,
            "artifact_inventory_sha256": bindings.artifact_inventory_sha256,
            "recovery_policy_sha256": bindings.recovery_policy_sha256,
        },
        "seeds": list(HOLD30_SEEDS),
        "folds": split_payload,
        "settings": [asdict(setting) for setting in HOLD30_MECH8_SETTINGS],
        "trial_inventory_count": len(inventory),
        "trial_inventory_sha256": sha256_payload(inventory),
        "bindings": asdict(bindings),
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload


__all__ = [
    "HOLD30_FOLDS",
    "HOLD30_GPUS_PER_SETTING",
    "HOLD30_MIN_AXIS_POSITIONS",
    "HOLD30_SEEDS",
    "Hold30Fold",
    "Hold30FreezeBindings",
    "Hold30FreezeError",
    "hold30_trial_inventory",
    "render_hold30_folds",
    "render_hold30_manifest",
    "sha256_payload",
]
