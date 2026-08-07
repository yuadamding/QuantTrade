"""Two-rank worker for the immutable TOP2000 M03R-v7 seed-17 diagnostic.

The numerical training implementation is shared with the five-seed worker,
but the package, training-plan, progress, validation, fold-execution, and
terminal identities are disjoint.  A validation sentinel is the only bounded
qualification mode: it must cross the seed-validation and one-member fold
execution boundary before it can pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    M03R_SEED17_TOP2000_DATA_ROLE,
    M03R_SEED17_TOP2000_DESIGN_ID,
    M03R_SEED17_TOP2000_FOLDS,
    M03R_SEED17_TOP2000_PROTOCOL_GENERATION,
    M03R_SEED17_TOP2000_PROTOCOL_SHA256,
    M03R_SEED17_TOP2000_SEEDS,
    M03R_SEED17_TOP2000_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_dev import (
    runtime_setting_id as resolve_runtime_setting_id,
)
from rl_quant.training.hold30_alpha_m03r_v7_seed17_package import (
    M03RV7Seed17PackageError,
    M03RV7Seed17PackagePlan,
    load_m03r_v7_seed17_top2000_package_plan,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
    TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS,
    TOP2000_M03R_V7_DEV_PURGE_DECISIONS,
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS,
    TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
)
from rl_quant.workflows import top2000_m03r_v7_dev as base_worker

SEED17_WORKER_SCHEMA = "rl-quant.top2000-dev.m03r-v7-seed17-worker-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Top2000M03RV7Seed17WorkerError(RuntimeError):
    """The seed-17 package, plan, or worker invocation is inconsistent."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Top2000M03RV7Seed17WorkerError(
            "seed-17 worker payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Top2000M03RV7Seed17TrainingPlan:
    """One six-cell setting plan with an unchanged numerical route."""

    setting_index: int
    setting_id: str
    runtime_setting_id: str
    cache_path: str
    cache_sha256: str
    output_root: str
    total_optimizer_steps_per_fold_seed: int = 64
    max_origin_batch: int = 32
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    token_dim: int = 512
    raw_stock_chunk: int = 512
    expected_world_size: int = 2
    activation_checkpointing: bool = False
    mixed_precision: str = "bfloat16"
    fold_indices: tuple[int, ...] = M03R_SEED17_TOP2000_FOLDS
    paired_seeds: tuple[int, ...] = M03R_SEED17_TOP2000_SEEDS
    protocol_generation: str = M03R_SEED17_TOP2000_PROTOCOL_GENERATION
    design_id: str = M03R_SEED17_TOP2000_DESIGN_ID
    protocol_sha256: str = M03R_SEED17_TOP2000_PROTOCOL_SHA256
    data_role: str = M03R_SEED17_TOP2000_DATA_ROLE
    schema: str = (
        "rl-quant.top2000-dev.m03r-v7-seed17-training-plan-v1"
    )
    development_only: bool = True
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if (
            not 0 <= self.setting_index < 12
            or self.setting_id
            != M03R_SEED17_TOP2000_SETTING_IDS[self.setting_index]
            or self.runtime_setting_id
            != resolve_runtime_setting_id(self.setting_id)
            or not self.cache_path
            or _SHA256_RE.fullmatch(self.cache_sha256) is None
            or not self.output_root
            or self.total_optimizer_steps_per_fold_seed != 64
            or self.max_origin_batch != 32
            or self.learning_rate != 1.0e-4
            or self.weight_decay != 1.0e-4
            or self.grad_clip != 1.0
            or self.token_dim != 512
            or self.raw_stock_chunk != 512
            or self.expected_world_size != 2
            or self.activation_checkpointing
            or self.mixed_precision != "bfloat16"
            or self.fold_indices != tuple(range(6))
            or self.paired_seeds != (17,)
            or self.protocol_generation
            != M03R_SEED17_TOP2000_PROTOCOL_GENERATION
            or self.design_id != M03R_SEED17_TOP2000_DESIGN_ID
            or self.protocol_sha256 != M03R_SEED17_TOP2000_PROTOCOL_SHA256
            or self.data_role != M03R_SEED17_TOP2000_DATA_ROLE
            or not self.development_only
            or self.promotion_eligible
        ):
            raise Top2000M03RV7Seed17WorkerError(
                "seed-17 training plan identity or runtime profile drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))

    @property
    def episode_schedule_sha256(self) -> str:
        return _sha256(
            {
                "schema": (
                    "rl-quant.top2000-dev.m03r-v7-seed17-episode-schedule-v1"
                ),
                "protocol_sha256": self.protocol_sha256,
                "cache_sha256": self.cache_sha256,
                "required_state_rows": TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
                "episode_state_rows": TOP2000_M03R_V7_DEV_EPISODE_STATE_ROWS,
                "warmup_decisions": TOP2000_M03R_V7_DEV_WARMUP_DECISIONS,
                "label_support_decisions": (
                    TOP2000_M03R_V7_DEV_LABEL_SUPPORT_DECISIONS
                ),
                "validation_decisions": (
                    TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
                ),
                "purge_decisions": TOP2000_M03R_V7_DEV_PURGE_DECISIONS,
                "fold_indices": list(self.fold_indices),
                "paired_seeds": list(self.paired_seeds),
            }
        )


def load_package_plan(
    path: str | Path,
    *,
    expected_package_plan_sha256: str,
) -> M03RV7Seed17PackagePlan:
    """Load and fully reconstruct one pinned seed-17 package plan."""

    try:
        return load_m03r_v7_seed17_top2000_package_plan(
            path,
            expected_package_plan_sha256=expected_package_plan_sha256,
            require_file_location_matches_plan=True,
        )
    except M03RV7Seed17PackageError as exc:
        raise Top2000M03RV7Seed17WorkerError(
            "seed-17 package failed typed validation"
        ) from exc


def resolve_completion_index(explicit: int | None) -> int:
    environment = os.environ.get("JOB_COMPLETION_INDEX")
    from_environment: int | None = None
    if environment is not None:
        try:
            from_environment = int(environment)
        except ValueError as exc:
            raise Top2000M03RV7Seed17WorkerError(
                "JOB_COMPLETION_INDEX must be an integer"
            ) from exc
    if explicit is not None and from_environment is not None and (
        explicit != from_environment
    ):
        raise Top2000M03RV7Seed17WorkerError(
            "completion argument disagrees with JOB_COMPLETION_INDEX"
        )
    value = explicit if explicit is not None else from_environment
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 12:
        raise Top2000M03RV7Seed17WorkerError(
            "seed-17 completion index must lie in [0, 11]"
        )
    return value


def plan_from_package_completion(
    package: M03RV7Seed17PackagePlan,
    *,
    package_plan_path: str | Path,
    completion_index: int,
    output_root: str | Path,
) -> tuple[Top2000M03RV7Seed17TrainingPlan, str]:
    row = package.indices[completion_index]
    package_path = Path(package_plan_path)
    setting_root = (
        Path(output_root)
        / f"completion-{completion_index:02d}-setting-{row.setting_index:02d}"
    )
    profile = package.runtime_profile
    plan = Top2000M03RV7Seed17TrainingPlan(
        setting_index=row.setting_index,
        setting_id=row.setting_id,
        runtime_setting_id=row.runtime_setting_id,
        cache_path=str(package_path.parent / "cache.pt"),
        cache_sha256=package.artifacts.cache_artifact_sha256,
        output_root=str(setting_root),
        total_optimizer_steps_per_fold_seed=profile.optimizer_steps_per_fold_seed,
        max_origin_batch=profile.max_origin_batch,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        grad_clip=profile.grad_clip,
        token_dim=profile.token_dim,
        raw_stock_chunk=profile.raw_stock_chunk,
        expected_world_size=profile.expected_world_size,
        activation_checkpointing=profile.activation_checkpointing,
        mixed_precision=profile.mixed_precision,
    )
    plan_path = setting_root / "training-plan.json"
    plan_file_sha256 = base_worker._write_immutable_json(
        plan_path,
        asdict(plan),
    )
    binding = {
        "schema": "rl-quant.top2000-dev.m03r-v7-seed17-worker-binding-v1",
        "worker_schema": SEED17_WORKER_SCHEMA,
        "package_plan_sha256": package.package_plan_sha256,
        "package_plan_path": str(package_path),
        "completion": asdict(row),
        "training_plan": asdict(plan),
        "training_plan_path": str(plan_path),
        "training_plan_file_sha256": plan_file_sha256,
        "training_plan_receipt_sha256": plan.receipt_sha256,
        "episode_schedule_sha256": plan.episode_schedule_sha256,
        "output_root": str(setting_root),
        "prior_training_evidence_imported": False,
        "one_member_fold_execution": True,
        "development_only": True,
        "promotion_eligible": False,
    }
    binding_path = setting_root / "execution-plan-binding.json"
    base_worker._write_immutable_json(binding_path, binding)
    return plan, plan_file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--completion-index", type=int)
    parser.add_argument(
        "--validation-sentinel",
        action="store_true",
        help=(
            "run four updates for fold 0/seed 17 and require validation plus "
            "one-member fold-execution receipts"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    package = load_package_plan(
        args.package_plan,
        expected_package_plan_sha256=args.package_plan_sha256,
    )
    completion_index = resolve_completion_index(args.completion_index)
    plan, plan_file_sha256 = plan_from_package_completion(
        package,
        package_plan_path=args.package_plan,
        completion_index=completion_index,
        output_root=args.output_root,
    )
    terminal = base_worker.run_worker(
        plan,
        plan_file_sha256=plan_file_sha256,
        qualification_only=args.validation_sentinel,
        qualification_steps=4 if args.validation_sentinel else 1,
        seed17_diagnostic=True,
        seed17_validation_sentinel=args.validation_sentinel,
    )
    if terminal is not None:
        print(_canonical_json(terminal).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SEED17_WORKER_SCHEMA",
    "Top2000M03RV7Seed17TrainingPlan",
    "Top2000M03RV7Seed17WorkerError",
    "load_package_plan",
    "main",
    "plan_from_package_completion",
    "resolve_completion_index",
]
