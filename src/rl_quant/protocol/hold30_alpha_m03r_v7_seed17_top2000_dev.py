"""Immutable one-seed TOP2000 diagnostic derived from M03R v7.

This generation deliberately does not weaken or alias the five-seed TOP2000
package.  It is a smaller, development-only diagnostic in which every setting
owns the same six folds and only seed 17.  Its evidence cannot satisfy the
five-seed M03R-v7 completion or ensemble contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v7_schedule import (
    M03R_V7_ADMISSION_ORDER,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_SETTING_IDS,
)

M03R_SEED17_TOP2000_PROTOCOL_GENERATION = (
    "top2000-dev-hold30-active-alpha-m03r-v7-seed17-v1"
)
M03R_SEED17_TOP2000_DESIGN_ID = (
    "daily_ohlcv_aggregated_top2000_dev_hold30_m03r_v7_seed17_v1"
)
M03R_SEED17_TOP2000_DATA_ROLE: Literal[
    "development-only-nonreportable"
] = "development-only-nonreportable"
M03R_SEED17_TOP2000_FOLDS = (0, 1, 2, 3, 4, 5)
M03R_SEED17_TOP2000_SEEDS = (17,)
M03R_SEED17_TOP2000_ADMISSION_ORDER = M03R_V7_ADMISSION_ORDER

M03R_SEED17_TOP2000_SETTING_IDS = tuple(
    setting_id.replace("-top2000-dev-v1", "-top2000-seed17-dev-v1")
    for setting_id in M03R_TOP2000_DEV_SETTING_IDS
)
M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID = dict(
    zip(
        M03R_SEED17_TOP2000_SETTING_IDS,
        M03R_TOP2000_DEV_SETTING_IDS,
        strict=True,
    )
)

M03R_SEED17_TOP2000_PROTOCOL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-protocol-v1"
)
M03R_SEED17_TOP2000_PACKAGE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-package-v1"
)
M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-package-plan-file-v1"
)
M03R_SEED17_TOP2000_TRAINING_PLAN_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-training-plan-v1"
)
M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-validation-v1"
)
M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-fold-execution-v1"
)
M03R_SEED17_TOP2000_COMPLETION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed17-completion-v1"
)


class M03RV7Seed17ProtocolError(ValueError):
    """The immutable seed-17 diagnostic inventory drifted."""


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
        raise M03RV7Seed17ProtocolError(
            "seed-17 protocol payload is not canonical-JSON safe"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7Seed17PanelContract:
    """Twelve paired settings, six folds, and exactly one seed."""

    setting_ids: tuple[str, ...] = M03R_SEED17_TOP2000_SETTING_IDS
    runtime_setting_ids: tuple[str, ...] = M03R_TOP2000_DEV_SETTING_IDS
    admission_order: tuple[int, ...] = M03R_SEED17_TOP2000_ADMISSION_ORDER
    fold_indices: tuple[int, ...] = M03R_SEED17_TOP2000_FOLDS
    paired_seeds: tuple[int, ...] = M03R_SEED17_TOP2000_SEEDS
    gpu_count_per_worker: int = 2
    maximum_concurrent_workers: int = 8
    worker_count: int = 12
    one_member_fold_execution: bool = True
    five_seed_ensemble_eligible: bool = False
    development_only: bool = True
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if (
            self.setting_ids != M03R_SEED17_TOP2000_SETTING_IDS
            or self.runtime_setting_ids != M03R_TOP2000_DEV_SETTING_IDS
            or self.admission_order != M03R_V7_ADMISSION_ORDER
            or self.fold_indices != tuple(range(6))
            or self.paired_seeds != (17,)
            or self.gpu_count_per_worker != 2
            or self.maximum_concurrent_workers != 8
            or self.worker_count != 12
            or not self.one_member_fold_execution
            or self.five_seed_ensemble_eligible
            or not self.development_only
            or self.promotion_eligible
        ):
            raise M03RV7Seed17ProtocolError(
                "seed-17 panel must remain 12 settings x 6 folds x seed 17"
            )
        if len(set(self.setting_ids)) != 12:
            raise M03RV7Seed17ProtocolError(
                "seed-17 setting identities must be unique"
            )

    @property
    def cells_per_setting(self) -> int:
        return len(self.fold_indices) * len(self.paired_seeds)

    @property
    def total_cells(self) -> int:
        return self.worker_count * self.cells_per_setting

    @property
    def receipt_sha256(self) -> str:
        return _sha256(
            {
                "schema": M03R_SEED17_TOP2000_PROTOCOL_SCHEMA,
                **asdict(self),
            }
        )


M03R_SEED17_TOP2000_PANEL = M03RV7Seed17PanelContract()
M03R_SEED17_TOP2000_PROTOCOL_SHA256 = M03R_SEED17_TOP2000_PANEL.receipt_sha256


def runtime_setting_id(seed17_setting_id: str) -> str:
    """Return the unchanged numerical-route identity for a diagnostic row."""

    try:
        return M03R_SEED17_TOP2000_RUNTIME_SETTING_BY_ID[seed17_setting_id]
    except KeyError as exc:
        raise M03RV7Seed17ProtocolError(
            f"unknown seed-17 setting {seed17_setting_id!r}"
        ) from exc


__all__ = [
    "M03R_SEED17_TOP2000_ADMISSION_ORDER",
    "M03R_SEED17_TOP2000_COMPLETION_SCHEMA",
    "M03R_SEED17_TOP2000_DATA_ROLE",
    "M03R_SEED17_TOP2000_DESIGN_ID",
    "M03R_SEED17_TOP2000_FOLDS",
    "M03R_SEED17_TOP2000_FOLD_EXECUTION_SCHEMA",
    "M03R_SEED17_TOP2000_PACKAGE_FILE_SCHEMA",
    "M03R_SEED17_TOP2000_PACKAGE_SCHEMA",
    "M03R_SEED17_TOP2000_PANEL",
    "M03R_SEED17_TOP2000_PROTOCOL_GENERATION",
    "M03R_SEED17_TOP2000_PROTOCOL_SHA256",
    "M03R_SEED17_TOP2000_SEEDS",
    "M03R_SEED17_TOP2000_SEED_VALIDATION_SCHEMA",
    "M03R_SEED17_TOP2000_SETTING_IDS",
    "M03R_SEED17_TOP2000_TRAINING_PLAN_SCHEMA",
    "M03RV7Seed17PanelContract",
    "M03RV7Seed17ProtocolError",
    "runtime_setting_id",
]
