"""Causal adaptive RL-training authority over genuine fit-only forecasts.

V1 aggregated inner-validation archives.  V2 consumes only replayed
``MassiveAdaptiveRLFitForecastArchiveV1`` blocks whose union is the exact
expanding fit prefix for the requested outer fold.  Every block is bound to a
training-loss-only checkpoint choice and checkpoint-specific calibration with
all cutoffs strictly before the block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    MassiveAdaptiveRLFitForecastArchiveV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveCausalCheckpointChoiceV1,
    MassiveAdaptiveRLTrainingForecastBlockV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)


MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-training-forecast-authority-v2"
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "forecast_source": "replayed-rl-fit-forecast-archive-v1",
        "fit_prefix": "exact-fold-fit-tail-126-times-fold-index-plus-one",
        "blocks": "complete-package-derived-21-or-63-session-inventory",
        "checkpoint_choice": "minimum-final-training-loss-only",
        "cutoffs": "training-target-choice-and-calibration-before-each-block",
        "caller_dates": False,
        "validation_access": False,
        "outer_access": False,
        "lockbox_access": False,
        "profitability_reporting": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLTrainingForecastAuthorityV2Error(ValueError):
    """The fit-only forecast inventory or one causal block differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrainingForecastAuthorityV2:
    outer_fold_index: int
    block_sessions: int
    blocks: tuple[MassiveAdaptiveRLTrainingForecastBlockV1, ...]
    origin_session_dates: tuple[str, ...]
    rl_fit_prefix_inventory_sha256: str
    block_inventory_sha256: str
    forecast_row_inventory_sha256: str
    source_forecast_archive_receipts: tuple[str, ...]
    source_forecast_archive_inventory_sha256: str
    source_inference_plan_inventory_sha256: str
    checkpoint_choice_inventory_sha256: str
    calibration_inventory_sha256: str
    prequential_plan_receipt_sha256: str
    prequential_archive_receipt_sha256: str
    split_plan_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_forecasts_replayed: bool
    development_rl_training_forecast_authorized: bool
    reinforcement_learning_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for block in self.blocks:
            block.validate()
        expected_sessions = 126 * (self.outer_fold_index + 1)
        expected_blocks = expected_sessions // self.block_sessions
        dates = tuple(
            date for block in self.blocks for date in block.forecast_session_dates
        )
        row_receipts = tuple(
            receipt for block in self.blocks for receipt in block.forecast_row_receipts
        )
        expected_authorized = (
            self.runtime_forecasts_replayed and self.source_data_qualified
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SCHEMA
            or isinstance(self.outer_fold_index, bool)
            or self.outer_fold_index not in range(4)
            or self.block_sessions not in {21, 63}
            or expected_sessions % self.block_sessions
            or len(self.blocks) != expected_blocks
            or tuple(block.block_index for block in self.blocks)
            != tuple(range(expected_blocks))
            or dates != tuple(sorted(set(dates)))
            or len(dates) != expected_sessions
            or self.origin_session_dates != dates
            or self.rl_fit_prefix_inventory_sha256 != semantic_sha256(dates)
            or self.block_inventory_sha256
            != semantic_sha256(
                tuple(block.semantic_receipt_sha256 for block in self.blocks)
            )
            or self.forecast_row_inventory_sha256 != semantic_sha256(row_receipts)
            or self.source_forecast_archive_receipts
            != tuple(
                block.source_forecast_archive_receipt_sha256 for block in self.blocks
            )
            or self.source_forecast_archive_inventory_sha256
            != semantic_sha256(self.source_forecast_archive_receipts)
            or not isinstance(self.source_data_qualified, bool)
            or not self.runtime_forecasts_replayed
            or self.development_rl_training_forecast_authorized != expected_authorized
            or self.reinforcement_learning_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
                "adaptive RL training forecast authority V2 differs"
            )
        for value in (
            self.rl_fit_prefix_inventory_sha256,
            self.block_inventory_sha256,
            self.forecast_row_inventory_sha256,
            *self.source_forecast_archive_receipts,
            self.source_forecast_archive_inventory_sha256,
            self.source_inference_plan_inventory_sha256,
            self.checkpoint_choice_inventory_sha256,
            self.calibration_inventory_sha256,
            self.prequential_plan_receipt_sha256,
            self.prequential_archive_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL training forecast authority V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_training_forecast_authority_v2(
    *,
    outer_fold_index: int,
    block_sessions: int,
    split_plan: MassiveAdaptiveSplitPlanV1,
    forecast_archives: Sequence[MassiveAdaptiveRLFitForecastArchiveV1],
    training_window_plans: Sequence[MassiveAdaptiveWindowPlanV1],
    checkpoint_choices: Sequence[MassiveAdaptiveCausalCheckpointChoiceV1],
    calibrations: Sequence[MassiveAdaptiveForecastCalibrationV2],
) -> MassiveAdaptiveRLTrainingForecastAuthorityV2:
    """Bind the exact causal RL-fit prefix and authorize it for PPO fitting."""

    split_plan.validate()
    if (
        isinstance(outer_fold_index, bool)
        or outer_fold_index not in range(len(split_plan.outer_folds))
        or block_sessions not in {21, 63}
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
            "adaptive RL training forecast V2 fold or block size is unsupported"
        )
    expected_dates = split_plan.outer_folds[outer_fold_index].fit_session_dates[
        -(126 * (outer_fold_index + 1)) :
    ]
    expected_block_count = len(expected_dates) // block_sessions
    archives = tuple(sorted(forecast_archives, key=lambda value: value.block_index))
    windows = tuple(training_window_plans)
    choices = tuple(checkpoint_choices)
    calibration_rows = tuple(calibrations)
    if (
        len(archives) != expected_block_count
        or len(windows) != outer_fold_index + 1
        or len(choices) != outer_fold_index + 1
        or len(calibration_rows) != outer_fold_index + 1
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
            "adaptive RL training forecast V2 inventory is incomplete"
        )
    for window, choice, calibration in zip(
        windows, choices, calibration_rows, strict=True
    ):
        window.validate()
        choice.validate()
        calibration.validate()
    blocks: list[MassiveAdaptiveRLTrainingForecastBlockV1] = []
    inference_plan_receipts: list[str] = []
    for expected_block_index, archive in enumerate(archives):
        archive.validate()
        if (
            archive.runtime_rows is None
            or not archive.runtime_forecasts_replayed
            or archive.inference_role != "rl_fit"
            or archive.outer_fold_index != outer_fold_index
            or archive.block_index != expected_block_index
            or archive.block_sessions != block_sessions
            or archive.origin_session_dates
            != expected_dates[
                expected_block_index * block_sessions : (expected_block_index + 1)
                * block_sessions
            ]
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
                "adaptive RL-fit archive is not the next exact prefix block"
            )
        source_fold_index = archive.source_fold_index
        try:
            window = windows[source_fold_index]
            choice = choices[source_fold_index]
            calibration = calibration_rows[source_fold_index]
        except IndexError as error:
            raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
                "adaptive RL-fit block has no causal training lineage"
            ) from error
        cutoffs = (
            archive.supervised_training_cutoff_session_date,
            archive.target_maturity_cutoff_session_date,
            choice.selection_cutoff_session_date,
            calibration.calibration_fit_stop_session_date,
        )
        if (
            window.fold_index != source_fold_index
            or choice.fold_index != source_fold_index
            or calibration.fold_index != source_fold_index
            or choice.selected_checkpoint_receipt_sha256
            != archive.checkpoint_receipt_sha256
            or choice.selected_checkpoint_source_receipt_sha256
            != archive.checkpoint_source_receipt_sha256
            or choice.selected_model_state_receipt_sha256
            != archive.model_state_receipt_sha256
            or choice.training_window_plan_receipt_sha256
            != archive.training_window_plan_receipt_sha256
            or calibration.checkpoint_receipt_sha256
            != archive.checkpoint_receipt_sha256
            or calibration.checkpoint_source_receipt_sha256
            != archive.checkpoint_source_receipt_sha256
            or calibration.model_state_receipt_sha256
            != archive.model_state_receipt_sha256
            or calibration.training_window_plan_receipt_sha256
            != archive.training_window_plan_receipt_sha256
            or window.semantic_receipt_sha256
            != archive.training_window_plan_receipt_sha256
            or max(cutoffs) >= archive.origin_session_dates[0]
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
                "adaptive RL-fit block checkpoint choice or calibration differs"
            )
        block_body = {
            "schema": "rl-quant.massive-adaptive-rl-training-forecast-block-v1",
            "block_index": expected_block_index,
            "source_fold_index": source_fold_index,
            "forecast_session_dates": archive.origin_session_dates,
            "supervised_training_cutoff_session_date": cutoffs[0],
            "target_maturity_cutoff_session_date": cutoffs[1],
            "checkpoint_selection_cutoff_session_date": cutoffs[2],
            "calibration_fit_cutoff_session_date": cutoffs[3],
            "checkpoint_choice_receipt_sha256": choice.semantic_receipt_sha256,
            "checkpoint_receipt_sha256": archive.checkpoint_receipt_sha256,
            "checkpoint_source_receipt_sha256": (
                archive.checkpoint_source_receipt_sha256
            ),
            "model_state_receipt_sha256": archive.model_state_receipt_sha256,
            "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
            "training_window_plan_receipt_sha256": (
                archive.training_window_plan_receipt_sha256
            ),
            "source_forecast_archive_receipt_sha256": (archive.semantic_receipt_sha256),
            "forecast_row_receipts": archive.row_receipts,
            "forecast_row_inventory_sha256": archive.row_inventory_sha256,
            "source_data_qualified": bool(
                archive.development_forecast_authorized
                and choice.source_data_qualified
                and calibration.development_calibration_authorized
            ),
        }
        block = MassiveAdaptiveRLTrainingForecastBlockV1(
            **block_body,  # type: ignore[arg-type]
            semantic_receipt_sha256=semantic_sha256(block_body),
        )
        block.validate()
        blocks.append(block)
        inference_plan_receipts.append(archive.inference_plan_receipt_sha256)

    ordered = tuple(blocks)
    origin_dates = tuple(
        date for block in ordered for date in block.forecast_session_dates
    )
    if origin_dates != expected_dates:
        raise MassiveAdaptiveRLTrainingForecastAuthorityV2Error(
            "adaptive RL-fit archive union differs from the derived fold prefix"
        )
    archive_receipts = tuple(
        block.source_forecast_archive_receipt_sha256 for block in ordered
    )
    choice_receipts = tuple(choice.semantic_receipt_sha256 for choice in choices)
    calibration_receipts = tuple(
        calibration.semantic_receipt_sha256 for calibration in calibration_rows
    )
    source_qualified = bool(
        split_plan.candidate_source_data_qualified
        and all(block.source_data_qualified for block in ordered)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SCHEMA,
        "outer_fold_index": outer_fold_index,
        "block_sessions": block_sessions,
        "blocks": ordered,
        "origin_session_dates": origin_dates,
        "rl_fit_prefix_inventory_sha256": semantic_sha256(origin_dates),
        "block_inventory_sha256": semantic_sha256(
            tuple(block.semantic_receipt_sha256 for block in ordered)
        ),
        "forecast_row_inventory_sha256": semantic_sha256(
            tuple(
                receipt for block in ordered for receipt in block.forecast_row_receipts
            )
        ),
        "source_forecast_archive_receipts": archive_receipts,
        "source_forecast_archive_inventory_sha256": semantic_sha256(archive_receipts),
        "source_inference_plan_inventory_sha256": semantic_sha256(
            tuple(inference_plan_receipts)
        ),
        "checkpoint_choice_inventory_sha256": semantic_sha256(choice_receipts),
        "calibration_inventory_sha256": semantic_sha256(calibration_receipts),
        "prequential_plan_receipt_sha256": semantic_sha256(
            tuple(inference_plan_receipts)
        ),
        "prequential_archive_receipt_sha256": semantic_sha256(archive_receipts),
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "source_data_qualified": source_qualified,
        "runtime_forecasts_replayed": True,
        "development_rl_training_forecast_authorized": source_qualified,
        "reinforcement_learning_authorized": source_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLTrainingForecastAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(
            {**body, "blocks": tuple(asdict(block) for block in ordered)}
        ),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveRLTrainingForecastAuthorityV2",
    "MassiveAdaptiveRLTrainingForecastAuthorityV2Error",
    "build_massive_adaptive_rl_training_forecast_authority_v2",
]
