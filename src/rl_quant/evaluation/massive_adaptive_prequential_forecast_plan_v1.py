"""Leakage-safe schedule over replayed adaptive forecast blocks.

The supervised checkpoint for each block is trained only on an earlier causal
window.  Its already replayed ForecastArchive V2 supplies the complete,
target-free inner-validation block.  This plan aggregates an exact prefix of
the registered folds without opening targets or granting RL access.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)


MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_BLOCK_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prequential-forecast-block-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SCHEMA = (
    "rl-quant.massive-adaptive-prequential-forecast-plan-v1"
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "blocks": "exact-fold-prefix-of-replayed-inner-validation-forecasts",
        "checkpoint_cutoff": "strictly-before-each-forecast-block",
        "target_archive": "inaccessible",
        "model_selection_authority": False,
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptivePrequentialForecastPlanV1Error(ValueError):
    """Forecast blocks are not one chronological, leakage-safe prefix."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptivePrequentialForecastPlanV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePrequentialForecastBlockV1:
    block_index: int
    fold_index: int
    training_cutoff_session_date: str
    forecast_session_dates: tuple[str, ...]
    checkpoint_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    source_forecast_archive_receipt_sha256: str
    source_forecast_row_inventory_sha256: str
    source_data_qualified: bool
    receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_BLOCK_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_BLOCK_V1_SCHEMA
            or self.block_index < 0
            or self.fold_index < 0
            or not self.training_cutoff_session_date
            or not self.forecast_session_dates
            or self.forecast_session_dates
            != tuple(sorted(set(self.forecast_session_dates)))
            or self.training_cutoff_session_date >= self.forecast_session_dates[0]
            or not isinstance(self.source_data_qualified, bool)
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptivePrequentialForecastPlanV1Error(
                "prequential forecast block geometry or receipt differs"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.training_window_plan_receipt_sha256,
            self.source_forecast_archive_receipt_sha256,
            self.source_forecast_row_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("prequential forecast block", value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptivePrequentialForecastPlanV1:
    blocks: tuple[MassiveAdaptivePrequentialForecastBlockV1, ...]
    fold_indices: tuple[int, ...]
    origin_session_dates: tuple[str, ...]
    source_forecast_archive_receipts: tuple[str, ...]
    source_forecast_row_receipts: tuple[str, ...]
    block_inventory_sha256: str
    source_forecast_archive_inventory_sha256: str
    source_forecast_row_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    source_schedule_replayed: bool
    development_prequential_forecast_authorized: bool = False
    checkpoint_selection_authorized: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for block in self.blocks:
            block.validate()
        expected_dates = tuple(
            date for block in self.blocks for date in block.forecast_session_dates
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SCHEMA
            or not self.blocks
            or len(self.blocks) > MASSIVE_ADAPTIVE_OUTER_FOLD_COUNT_V1
            or self.fold_indices != tuple(range(len(self.blocks)))
            or tuple(block.block_index for block in self.blocks)
            != tuple(range(len(self.blocks)))
            or tuple(block.fold_index for block in self.blocks) != self.fold_indices
            or expected_dates != tuple(sorted(set(expected_dates)))
            or self.origin_session_dates != expected_dates
            or self.source_forecast_archive_receipts
            != tuple(
                block.source_forecast_archive_receipt_sha256
                for block in self.blocks
            )
            or self.block_inventory_sha256
            != semantic_sha256(tuple(block.receipt_sha256 for block in self.blocks))
            or self.source_forecast_archive_inventory_sha256
            != semantic_sha256(self.source_forecast_archive_receipts)
            or self.source_forecast_row_inventory_sha256
            != semantic_sha256(self.source_forecast_row_receipts)
            or not isinstance(self.source_data_qualified, bool)
            or self.source_data_qualified
            != all(block.source_data_qualified for block in self.blocks)
            or not self.source_schedule_replayed
            or self.development_prequential_forecast_authorized
            or self.checkpoint_selection_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptivePrequentialForecastPlanV1Error(
                "prequential forecast plan identity or chronology differs"
            )
        for value in (
            *self.source_forecast_archive_receipts,
            *self.source_forecast_row_receipts,
            self.block_inventory_sha256,
            self.source_forecast_archive_inventory_sha256,
            self.source_forecast_row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("prequential forecast plan", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_prequential_forecast_plan_v1(
    *,
    forecast_archives: Sequence[MassiveAdaptiveForecastArchiveV2],
    training_window_plans: Sequence[MassiveAdaptiveWindowPlanV1],
) -> MassiveAdaptivePrequentialForecastPlanV1:
    """Bind a chronological prefix of independently replayed forecast blocks."""

    archives = tuple(forecast_archives)
    window_plans = tuple(training_window_plans)
    if not archives or len(archives) != len(window_plans):
        raise MassiveAdaptivePrequentialForecastPlanV1Error(
            "prequential forecasts and training windows are not parallel"
        )
    blocks: list[MassiveAdaptivePrequentialForecastBlockV1] = []
    row_receipts: list[str] = []
    for block_index, (archive, window_plan) in enumerate(
        zip(archives, window_plans, strict=True)
    ):
        archive.validate()
        window_plan.validate()
        if (
            archive.runtime_rows is None
            or not archive.runtime_forecasts_replayed
            or archive.inference_role != "inner_validation"
            or archive.fold_index != block_index
            or window_plan.fold_index != block_index
            or window_plan.split_role != "training"
            or archive.training_window_plan_receipt_sha256
            != window_plan.semantic_receipt_sha256
            or archive.checkpoint_receipt_sha256 == ""
        ):
            raise MassiveAdaptivePrequentialForecastPlanV1Error(
                "prequential block is not one replayed fold forecast"
            )
        training_dates = tuple(row.origin_session_date for row in window_plan.rows)
        if (
            not training_dates
            or training_dates != tuple(sorted(set(training_dates)))
            or training_dates[-1] >= archive.origin_session_dates[0]
        ):
            raise MassiveAdaptivePrequentialForecastPlanV1Error(
                "prequential checkpoint cutoff is not earlier than its forecast block"
            )
        body = {
            "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_BLOCK_V1_SCHEMA,
            "block_index": block_index,
            "fold_index": archive.fold_index,
            "training_cutoff_session_date": training_dates[-1],
            "forecast_session_dates": archive.origin_session_dates,
            "checkpoint_receipt_sha256": archive.checkpoint_receipt_sha256,
            "training_window_plan_receipt_sha256": (
                window_plan.semantic_receipt_sha256
            ),
            "source_forecast_archive_receipt_sha256": (
                archive.semantic_receipt_sha256
            ),
            "source_forecast_row_inventory_sha256": archive.row_inventory_sha256,
            "source_data_qualified": archive.development_forecast_authorized,
        }
        block = MassiveAdaptivePrequentialForecastBlockV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        block.validate()
        blocks.append(block)
        row_receipts.extend(archive.row_receipts)
    if any(
        left.forecast_session_dates[-1] >= right.forecast_session_dates[0]
        or left.training_cutoff_session_date >= right.training_cutoff_session_date
        for left, right in zip(blocks, blocks[1:], strict=False)
    ):
        raise MassiveAdaptivePrequentialForecastPlanV1Error(
            "prequential forecast blocks or training cutoffs are not chronological"
        )
    ordered = tuple(blocks)
    archive_receipts = tuple(
        block.source_forecast_archive_receipt_sha256 for block in ordered
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SCHEMA,
        "blocks": ordered,
        "fold_indices": tuple(block.fold_index for block in ordered),
        "origin_session_dates": tuple(
            date for block in ordered for date in block.forecast_session_dates
        ),
        "source_forecast_archive_receipts": archive_receipts,
        "source_forecast_row_receipts": tuple(row_receipts),
        "block_inventory_sha256": semantic_sha256(
            tuple(block.receipt_sha256 for block in ordered)
        ),
        "source_forecast_archive_inventory_sha256": semantic_sha256(
            archive_receipts
        ),
        "source_forecast_row_inventory_sha256": semantic_sha256(
            tuple(row_receipts)
        ),
        "source_data_qualified": all(
            block.source_data_qualified for block in ordered
        ),
        "source_schedule_replayed": True,
        "development_prequential_forecast_authorized": False,
        "checkpoint_selection_authorized": False,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SOURCE_SHA256
        ),
    }
    semantic_body = {**body, "blocks": tuple(asdict(block) for block in ordered)}
    result = MassiveAdaptivePrequentialForecastPlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(semantic_body),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_PREQUENTIAL_FORECAST_PLAN_V1_SCHEMA",
    "MassiveAdaptivePrequentialForecastBlockV1",
    "MassiveAdaptivePrequentialForecastPlanV1",
    "MassiveAdaptivePrequentialForecastPlanV1Error",
    "build_massive_adaptive_prequential_forecast_plan_v1",
]
