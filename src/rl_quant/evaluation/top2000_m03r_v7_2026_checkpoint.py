"""Fail-closed seed-17 checkpoint loading for the 2026 retrospective.

The frozen 2026 plan binds checkpoint paths and hashes, but inference still
needs to reconstruct the exact seed-17 training plan and model.  This module
closes that final trust boundary without importing the workflow module (and
therefore without creating an evaluation/workflow import cycle).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import torch

from rl_quant.evaluation.top2000_m03r_v7_dev import model_state_sha256
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DevelopmentFold,
    Top2000M03RV7DevelopmentPolicy,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.workflows import top2000_m03r_v7_dev as base_worker
from rl_quant.workflows.top2000_m03r_v7_seed17_dev import (
    Top2000M03RV7Seed17TrainingPlan,
)

TOP2000_M03R_V7_2026_CHECKPOINT_LOAD_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-checkpoint-load-v1"
)
TOP2000_M03R_V7_2026_FROZEN_CHECKPOINT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-2026-ytd-checkpoint-binding-v1"
)


class Top2000M03RV72026CheckpointError(RuntimeError):
    """A frozen checkpoint, training plan, or model payload drifted."""


class Top2000M03RV72026FrozenCheckpointBinding(Protocol):
    """Structural surface supplied by the pre-access frozen workflow plan."""

    @property
    def completion_index(self) -> int: ...

    @property
    def setting_index(self) -> int: ...

    @property
    def setting_id(self) -> str: ...

    @property
    def runtime_setting_id(self) -> str: ...

    @property
    def training_fold_index(self) -> int: ...

    @property
    def seed(self) -> int: ...

    @property
    def writer_rank(self) -> int: ...

    @property
    def optimizer_steps(self) -> int: ...

    @property
    def checkpoint_role(self) -> str: ...

    @property
    def training_root_relative_path(self) -> str: ...

    @property
    def model_relative_path(self) -> str: ...

    @property
    def model_file_sha256(self) -> str: ...

    @property
    def model_state_sha256(self) -> str: ...

    @property
    def training_plan_file_sha256(self) -> str: ...

    @property
    def training_plan_receipt_sha256(self) -> str: ...

    @property
    def source_protocol_sha256(self) -> str: ...

    @property
    def schema(self) -> str: ...

    @property
    def development_only(self) -> bool: ...

    @property
    def future_selected_universe(self) -> bool: ...

    @property
    def scientific_reporting_eligible(self) -> bool: ...

    @property
    def promotion_eligible(self) -> bool: ...

    @property
    def receipt_sha256(self) -> str: ...


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV72026CheckpointError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _safe_relative_file(root: Path, relative: str, *, label: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise Top2000M03RV72026CheckpointError(
            f"{label} is not a safe relative path"
        )
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise Top2000M03RV72026CheckpointError(
            f"{label} must be a regular non-symlink file"
        )
    return path


def _binding_value(binding: object, name: str) -> Any:
    try:
        return getattr(binding, name)
    except AttributeError as exc:
        raise Top2000M03RV72026CheckpointError(
            f"frozen checkpoint binding omitted {name}"
        ) from exc


def _validate_binding(
    binding: Top2000M03RV72026FrozenCheckpointBinding,
) -> None:
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    setting_index = _binding_value(binding, "setting_index")
    fold_index = _binding_value(binding, "training_fold_index")
    completion_index = _binding_value(binding, "completion_index")
    if (
        isinstance(setting_index, bool)
        or not isinstance(setting_index, int)
        or not 0 <= setting_index < len(contract.settings)
        or isinstance(fold_index, bool)
        or not isinstance(fold_index, int)
        or fold_index not in range(6)
        or isinstance(completion_index, bool)
        or not isinstance(completion_index, int)
        or completion_index not in range(12)
    ):
        raise Top2000M03RV72026CheckpointError(
            "frozen checkpoint setting, fold, or completion index is invalid"
        )
    setting = contract.settings[setting_index]
    expected_role = (
        "headline"
        if fold_index == contract.checkpoint_rule.headline_training_fold_index
        else "cutoff-sensitivity"
    )
    expected_training_root = (
        f"completion-{completion_index:02d}-setting-{setting_index:02d}/training"
    )
    expected_model = (
        f"{expected_training_root}/cells/fold-{fold_index:02d}-seed-17/"
        "model.rank-00.pt"
    )
    if (
        _binding_value(binding, "schema")
        != TOP2000_M03R_V7_2026_FROZEN_CHECKPOINT_SCHEMA
        or _binding_value(binding, "setting_id") != setting.seed17_setting_id
        or _binding_value(binding, "runtime_setting_id")
        != setting.runtime_setting_id
        or _binding_value(binding, "seed") != contract.checkpoint_rule.seed
        or _binding_value(binding, "writer_rank")
        != contract.checkpoint_rule.checkpoint_writer_rank
        or _binding_value(binding, "optimizer_steps") != 64
        or _binding_value(binding, "checkpoint_role") != expected_role
        or _binding_value(binding, "training_root_relative_path")
        != expected_training_root
        or _binding_value(binding, "model_relative_path") != expected_model
        or _binding_value(binding, "source_protocol_sha256")
        != contract.source_training_protocol_sha256
        or _binding_value(binding, "development_only") is not True
        or _binding_value(binding, "future_selected_universe") is not True
        or _binding_value(binding, "scientific_reporting_eligible") is not False
        or _binding_value(binding, "promotion_eligible") is not False
    ):
        raise Top2000M03RV72026CheckpointError(
            "frozen checkpoint binding drifted from the seed-17 evaluation rule"
        )
    for name in (
        "model_file_sha256",
        "model_state_sha256",
        "training_plan_file_sha256",
        "training_plan_receipt_sha256",
        "source_protocol_sha256",
        "receipt_sha256",
    ):
        _require_digest(name, _binding_value(binding, name))


def _load_exact_training_plan(
    training_output_root: Path,
    binding: Top2000M03RV72026FrozenCheckpointBinding,
) -> tuple[Top2000M03RV7Seed17TrainingPlan, Path]:
    training_relative = str(_binding_value(binding, "training_root_relative_path"))
    setting_root_relative = str(PurePosixPath(training_relative).parent)
    plan_path = _safe_relative_file(
        training_output_root,
        f"{setting_root_relative}/training-plan.json",
        label="seed-17 training plan",
    )
    expected_file_sha256 = _require_digest(
        "training_plan_file_sha256",
        _binding_value(binding, "training_plan_file_sha256"),
    )
    if _file_sha256(plan_path) != expected_file_sha256:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 training-plan file SHA-256 drifted"
        )
    try:
        payload = json.loads(plan_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 training plan is not valid JSON"
        ) from exc
    expected_fields = {field.name for field in fields(Top2000M03RV7Seed17TrainingPlan)}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 training plan fields drifted"
        )
    typed_payload = dict(payload)
    if isinstance(typed_payload.get("fold_indices"), list):
        typed_payload["fold_indices"] = tuple(typed_payload["fold_indices"])
    if isinstance(typed_payload.get("paired_seeds"), list):
        typed_payload["paired_seeds"] = tuple(typed_payload["paired_seeds"])
    try:
        plan = Top2000M03RV7Seed17TrainingPlan(**typed_payload)
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 training plan failed typed reconstruction"
        ) from exc
    expected_setting_root = training_output_root.joinpath(
        *PurePosixPath(setting_root_relative).parts
    )
    if (
        plan.setting_index != _binding_value(binding, "setting_index")
        or plan.setting_id != _binding_value(binding, "setting_id")
        or plan.runtime_setting_id != _binding_value(binding, "runtime_setting_id")
        or plan.output_root != str(expected_setting_root)
        or plan.expected_world_size != 2
        or plan.total_optimizer_steps_per_fold_seed != 64
        or plan.receipt_sha256
        != _binding_value(binding, "training_plan_receipt_sha256")
    ):
        raise Top2000M03RV72026CheckpointError(
            "seed-17 training plan does not match the frozen checkpoint binding"
        )
    return plan, plan_path


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026CheckpointLoadReceipt:
    """Content receipt for one inference-only reconstructed checkpoint."""

    frozen_checkpoint_binding_sha256: str
    training_plan_file_sha256: str
    training_plan_receipt_sha256: str
    fold_receipt_sha256: str
    model_file_sha256: str
    model_state_sha256: str
    setting_index: int
    setting_id: str
    runtime_setting_id: str
    training_fold_index: int
    checkpoint_role: str
    seed: int = 17
    writer_rank: int = 0
    world_size: int = 2
    completed_optimizer_steps: int = 64
    policy_training_enabled: bool = False
    development_only: bool = True
    future_selected_universe: bool = True
    scientific_reporting_eligible: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_2026_CHECKPOINT_LOAD_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "frozen_checkpoint_binding_sha256",
            "training_plan_file_sha256",
            "training_plan_receipt_sha256",
            "fold_receipt_sha256",
            "model_file_sha256",
            "model_state_sha256",
        ):
            _require_digest(name, getattr(self, name))
        contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
        if not 0 <= self.setting_index < len(contract.settings):
            raise Top2000M03RV72026CheckpointError(
                "checkpoint load receipt setting index is invalid"
            )
        setting = contract.settings[self.setting_index]
        expected_role = "headline" if self.training_fold_index == 5 else "cutoff-sensitivity"
        if (
            self.schema != TOP2000_M03R_V7_2026_CHECKPOINT_LOAD_SCHEMA
            or self.setting_id != setting.seed17_setting_id
            or self.runtime_setting_id != setting.runtime_setting_id
            or self.training_fold_index not in range(6)
            or self.checkpoint_role != expected_role
            or self.seed != 17
            or self.writer_rank != 0
            or self.world_size != 2
            or self.completed_optimizer_steps != 64
            or self.policy_training_enabled
            or not self.development_only
            or not self.future_selected_universe
            or self.scientific_reporting_eligible
            or self.promotion_eligible
        ):
            raise Top2000M03RV72026CheckpointError(
                "checkpoint load receipt overclaims or changed inference identity"
            )

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV72026LoadedCheckpoint:
    policy: Top2000M03RV7DevelopmentPolicy
    training_plan: Top2000M03RV7Seed17TrainingPlan
    training_fold: Top2000M03RV7DevelopmentFold
    receipt: Top2000M03RV72026CheckpointLoadReceipt


def load_top2000_m03r_v7_seed17_2026_checkpoint(
    binding: Top2000M03RV72026FrozenCheckpointBinding,
    *,
    training_output_root: str | Path,
    device: str | torch.device,
) -> Top2000M03RV72026LoadedCheckpoint:
    """Load one exact frozen final-update checkpoint for inference only.

    The existing numerical-route loader validates model and optimizer state.
    This wrapper additionally validates the seed-17 training-plan artifact and
    the two-rank, 64-update terminal fields that the shared loader predates.
    """

    _validate_binding(binding)
    root = Path(training_output_root)
    if not root.is_dir() or root.is_symlink():
        raise Top2000M03RV72026CheckpointError(
            "training output root must be a regular non-symlink directory"
        )
    plan, plan_path = _load_exact_training_plan(root, binding)
    folds = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )
    fold = folds[int(_binding_value(binding, "training_fold_index"))]
    model_path = _safe_relative_file(
        root,
        str(_binding_value(binding, "model_relative_path")),
        label="seed-17 rank-00 model",
    )
    expected_model_file_sha256 = _require_digest(
        "model_file_sha256", _binding_value(binding, "model_file_sha256")
    )
    if _file_sha256(model_path) != expected_model_file_sha256:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 checkpoint file SHA-256 drifted"
        )
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Top2000M03RV72026CheckpointError(
            "seed-17 checkpoint cannot be inspected"
        ) from exc
    if not isinstance(payload, Mapping) or (
        payload.get("plan_file_sha256")
        != _binding_value(binding, "training_plan_file_sha256")
        or payload.get("world_size") != 2
        or payload.get("completed_optimizer_steps") != 64
    ):
        raise Top2000M03RV72026CheckpointError(
            "checkpoint does not prove the exact two-rank 64-update training plan"
        )
    try:
        policy, observed_model_state_sha256 = base_worker._load_saved_seed_policy(
            model_path,
            expected_file_sha256=expected_model_file_sha256,
            plan=cast(Any, plan),
            fold=fold,
            seed=17,
            device=torch.device(device),
        )
    except (ValueError, RuntimeError) as exc:
        raise Top2000M03RV72026CheckpointError(
            "shared checkpoint loader rejected the frozen seed-17 model"
        ) from exc
    expected_model_state_sha256 = _require_digest(
        "model_state_sha256", _binding_value(binding, "model_state_sha256")
    )
    if (
        observed_model_state_sha256 != expected_model_state_sha256
        or model_state_sha256(policy) != expected_model_state_sha256
    ):
        raise Top2000M03RV72026CheckpointError(
            "loaded policy state does not match the frozen model-state hash"
        )
    policy.requires_grad_(False)
    policy.eval()
    plan_path_sha256 = _file_sha256(plan_path)
    receipt = Top2000M03RV72026CheckpointLoadReceipt(
        frozen_checkpoint_binding_sha256=str(
            _binding_value(binding, "receipt_sha256")
        ),
        training_plan_file_sha256=plan_path_sha256,
        training_plan_receipt_sha256=plan.receipt_sha256,
        fold_receipt_sha256=fold.receipt_sha256,
        model_file_sha256=expected_model_file_sha256,
        model_state_sha256=expected_model_state_sha256,
        setting_index=plan.setting_index,
        setting_id=plan.setting_id,
        runtime_setting_id=plan.runtime_setting_id,
        training_fold_index=fold.fold_index,
        checkpoint_role=str(_binding_value(binding, "checkpoint_role")),
    )
    if plan_path_sha256 != _binding_value(binding, "training_plan_file_sha256"):
        raise Top2000M03RV72026CheckpointError(
            "training plan changed during checkpoint reconstruction"
        )
    return Top2000M03RV72026LoadedCheckpoint(
        policy=policy,
        training_plan=plan,
        training_fold=fold,
        receipt=receipt,
    )


__all__ = [
    "TOP2000_M03R_V7_2026_CHECKPOINT_LOAD_SCHEMA",
    "TOP2000_M03R_V7_2026_FROZEN_CHECKPOINT_SCHEMA",
    "Top2000M03RV72026CheckpointError",
    "Top2000M03RV72026CheckpointLoadReceipt",
    "Top2000M03RV72026FrozenCheckpointBinding",
    "Top2000M03RV72026LoadedCheckpoint",
    "load_top2000_m03r_v7_seed17_2026_checkpoint",
]
