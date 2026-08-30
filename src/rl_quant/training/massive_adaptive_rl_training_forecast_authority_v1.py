"""Causal forecast authority for adaptive RL development training.

The prequential V1 archive proves that each forecast was produced by an
earlier supervised checkpoint.  This composite authority closes the remaining
selection boundary: the checkpoint is selected from training-loss state only,
its target maturation and calibration fit end before the forecast block, and
only an exact chronological prefix preceding one registered outer fold may be
used for RL optimization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import math

from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastArchiveV2,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_prequential_forecast_archive_v1 import (
    MassiveAdaptivePrequentialForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_prequential_forecast_plan_v1 import (
    MassiveAdaptivePrequentialForecastPlanV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)


MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-causal-checkpoint-choice-v1"
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_BLOCK_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-training-forecast-block-v1"
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-training-forecast-authority-v1"
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "forecast_source": "create-only-prequential-forecast-archive-v1",
        "checkpoint_choice": "minimum-final-training-loss-only",
        "cutoffs": "training-target-choice-and-calibration-before-block",
        "block_sessions": (21, 63),
        "outer_access": False,
        "lockbox_access": False,
        "profitability_reporting": False,
    }
)


class MassiveAdaptiveRLTrainingForecastAuthorityV1Error(ValueError):
    """The RL forecast chronology, selection, or calibration is not causal."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _checkpoint_source_receipt(checkpoint: MassiveAdaptiveCheckpointV1) -> str:
    return _digest(
        "checkpoint source receipt",
        getattr(getattr(checkpoint, "loaded_source", None), "receipt_sha256", None),
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveCausalCheckpointChoiceV1:
    fold_index: int
    selection_method: str
    selection_cutoff_session_date: str
    candidate_checkpoint_receipts: tuple[str, ...]
    candidate_model_state_receipts: tuple[str, ...]
    candidate_final_training_losses: tuple[float, ...]
    selected_checkpoint_receipt_sha256: str
    selected_checkpoint_source_receipt_sha256: str
    selected_model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    training_config_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        count = len(self.candidate_checkpoint_receipts)
        if (
            self.schema != MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index < 0
            or self.selection_method != "minimum-final-training-loss-only"
            or not self.selection_cutoff_session_date
            or count == 0
            or len(self.candidate_model_state_receipts) != count
            or len(self.candidate_final_training_losses) != count
            or len(set(self.candidate_checkpoint_receipts)) != count
            or any(not math.isfinite(value) for value in self.candidate_final_training_losses)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "causal supervised checkpoint choice differs"
            )
        selected_index = min(
            range(count),
            key=lambda index: (
                self.candidate_final_training_losses[index],
                self.candidate_checkpoint_receipts[index],
            ),
        )
        if (
            self.selected_checkpoint_receipt_sha256
            != self.candidate_checkpoint_receipts[selected_index]
            or self.selected_model_state_receipt_sha256
            != self.candidate_model_state_receipts[selected_index]
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "causal supervised checkpoint winner differs"
            )
        for value in (
            *self.candidate_checkpoint_receipts,
            *self.candidate_model_state_receipts,
            self.selected_checkpoint_source_receipt_sha256,
            self.training_window_plan_receipt_sha256,
            self.training_config_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("causal checkpoint choice", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_causal_checkpoint_choice_v1(
    *,
    checkpoints: Sequence[MassiveAdaptiveCheckpointV1],
    training_window_plan: MassiveAdaptiveWindowPlanV1,
) -> MassiveAdaptiveCausalCheckpointChoiceV1:
    """Select a checkpoint using its final training loss and no later outcomes."""

    candidates = tuple(checkpoints)
    training_window_plan.validate()
    if not candidates or training_window_plan.split_role != "training":
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            "causal checkpoint candidates or training window are absent"
        )
    for checkpoint in candidates:
        checkpoint.validate()
    reference = candidates[0]
    if (
        any(checkpoint.runtime_state is None for checkpoint in candidates)
        or any(not checkpoint.runtime_checkpoint_replayed for checkpoint in candidates)
        or any(
            checkpoint.window_plan_receipt_sha256
            != training_window_plan.semantic_receipt_sha256
            or checkpoint.training_authority_receipt_sha256
            != reference.training_authority_receipt_sha256
            or checkpoint.training_config_receipt_sha256
            != reference.training_config_receipt_sha256
            or checkpoint.model_spec_receipt_sha256
            != reference.model_spec_receipt_sha256
            for checkpoint in candidates
        )
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            "causal checkpoint candidates do not share one training experiment"
        )
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    if any(
        checkpoint.runtime_state is None or not checkpoint.runtime_state.loss_trace
        for checkpoint in ordered
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            "causal checkpoint candidate has no training-loss state"
        )
    losses = tuple(
        float(checkpoint.runtime_state.loss_trace[-1])  # type: ignore[union-attr]
        for checkpoint in ordered
    )
    winner_index = min(
        range(len(ordered)),
        key=lambda index: (losses[index], ordered[index].semantic_receipt_sha256),
    )
    winner = ordered[winner_index]
    cutoff = max(row.origin_session_date for row in training_window_plan.rows)
    body = {
        "schema": MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA,
        "fold_index": training_window_plan.fold_index,
        "selection_method": "minimum-final-training-loss-only",
        "selection_cutoff_session_date": cutoff,
        "candidate_checkpoint_receipts": tuple(
            checkpoint.semantic_receipt_sha256 for checkpoint in ordered
        ),
        "candidate_model_state_receipts": tuple(
            checkpoint.model_state_receipt_sha256 for checkpoint in ordered
        ),
        "candidate_final_training_losses": losses,
        "selected_checkpoint_receipt_sha256": winner.semantic_receipt_sha256,
        "selected_checkpoint_source_receipt_sha256": _checkpoint_source_receipt(winner),
        "selected_model_state_receipt_sha256": winner.model_state_receipt_sha256,
        "training_window_plan_receipt_sha256": training_window_plan.semantic_receipt_sha256,
        "training_config_receipt_sha256": winner.training_config_receipt_sha256,
        "source_data_qualified": all(
            checkpoint.development_training_authorized for checkpoint in ordered
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveCausalCheckpointChoiceV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrainingForecastBlockV1:
    block_index: int
    source_fold_index: int
    forecast_session_dates: tuple[str, ...]
    supervised_training_cutoff_session_date: str
    target_maturity_cutoff_session_date: str
    checkpoint_selection_cutoff_session_date: str
    calibration_fit_cutoff_session_date: str
    checkpoint_choice_receipt_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    calibration_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    source_forecast_archive_receipt_sha256: str
    forecast_row_receipts: tuple[str, ...]
    forecast_row_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_BLOCK_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        cutoffs = (
            self.supervised_training_cutoff_session_date,
            self.target_maturity_cutoff_session_date,
            self.checkpoint_selection_cutoff_session_date,
            self.calibration_fit_cutoff_session_date,
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_BLOCK_V1_SCHEMA
            or isinstance(self.block_index, bool)
            or self.block_index < 0
            or isinstance(self.source_fold_index, bool)
            or self.source_fold_index < 0
            or not self.forecast_session_dates
            or self.forecast_session_dates
            != tuple(sorted(set(self.forecast_session_dates)))
            or any(not cutoff or cutoff >= self.forecast_session_dates[0] for cutoff in cutoffs)
            or not self.forecast_row_receipts
            or len(self.forecast_row_receipts) != len(self.forecast_session_dates)
            or self.forecast_row_inventory_sha256
            != semantic_sha256(self.forecast_row_receipts)
            or not isinstance(self.source_data_qualified, bool)
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "RL training forecast block differs"
            )
        for value in (
            self.checkpoint_choice_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.calibration_receipt_sha256,
            self.training_window_plan_receipt_sha256,
            self.source_forecast_archive_receipt_sha256,
            *self.forecast_row_receipts,
            self.forecast_row_inventory_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("RL training forecast block", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrainingForecastAuthorityV1:
    outer_fold_index: int
    block_sessions: int
    blocks: tuple[MassiveAdaptiveRLTrainingForecastBlockV1, ...]
    origin_session_dates: tuple[str, ...]
    block_inventory_sha256: str
    forecast_row_inventory_sha256: str
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
        MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        for block in self.blocks:
            block.validate()
        dates = tuple(date for block in self.blocks for date in block.forecast_session_dates)
        expected_authorized = self.runtime_forecasts_replayed and self.source_data_qualified
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SCHEMA
            or isinstance(self.outer_fold_index, bool)
            or self.outer_fold_index < 0
            or self.block_sessions not in {21, 63}
            or not self.blocks
            or tuple(block.block_index for block in self.blocks)
            != tuple(range(len(self.blocks)))
            or dates != tuple(sorted(set(dates)))
            or self.origin_session_dates != dates
            or self.block_inventory_sha256
            != semantic_sha256(tuple(block.semantic_receipt_sha256 for block in self.blocks))
            or self.forecast_row_inventory_sha256
            != semantic_sha256(
                tuple(
                    receipt
                    for block in self.blocks
                    for receipt in block.forecast_row_receipts
                )
            )
            or not isinstance(self.source_data_qualified, bool)
            or not self.runtime_forecasts_replayed
            or self.development_rl_training_forecast_authorized != expected_authorized
            or self.reinforcement_learning_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "RL training forecast authority differs"
            )
        for value in (
            self.block_inventory_sha256,
            self.forecast_row_inventory_sha256,
            self.prequential_plan_receipt_sha256,
            self.prequential_archive_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("RL training forecast authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_training_forecast_authority_v1(
    *,
    outer_fold_index: int,
    block_sessions: int,
    prequential_plan: MassiveAdaptivePrequentialForecastPlanV1,
    prequential_archive: MassiveAdaptivePrequentialForecastArchiveV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    forecast_archives: Sequence[MassiveAdaptiveForecastArchiveV2],
    training_window_plans: Sequence[MassiveAdaptiveWindowPlanV1],
    checkpoint_choices: Sequence[MassiveAdaptiveCausalCheckpointChoiceV1],
    calibrations: Sequence[MassiveAdaptiveForecastCalibrationV2],
) -> MassiveAdaptiveRLTrainingForecastAuthorityV1:
    """Bind every forecast, selection, target, and calibration cutoff."""

    prequential_plan.validate()
    prequential_archive.validate()
    split_plan.validate()
    archives = tuple(forecast_archives)
    windows = tuple(training_window_plans)
    choices = tuple(checkpoint_choices)
    calibration_rows = tuple(calibrations)
    if (
        block_sessions not in {21, 63}
        or outer_fold_index < 0
        or outer_fold_index >= len(split_plan.outer_folds)
        or len(
            {
                len(archives),
                len(windows),
                len(choices),
                len(calibration_rows),
                len(prequential_plan.blocks),
            }
        )
        != 1
        or prequential_archive.runtime_rows is None
        or not prequential_archive.runtime_prequential_forecasts_replayed
        or prequential_archive.prequential_plan_receipt_sha256
        != prequential_plan.semantic_receipt_sha256
        or prequential_plan.fold_indices != tuple(range(outer_fold_index + 1))
        or prequential_archive.source_forecast_archive_receipts
        != tuple(archive.semantic_receipt_sha256 for archive in archives)
    ):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            "RL training forecast inputs are not one replayed outer-fold prefix"
        )
    outer_start = split_plan.outer_folds[outer_fold_index].outer_test_session_dates[0]
    blocks: list[MassiveAdaptiveRLTrainingForecastBlockV1] = []
    row_cursor = 0
    for source_block, archive, window, choice, calibration in zip(
        prequential_plan.blocks,
        archives,
        windows,
        choices,
        calibration_rows,
        strict=True,
    ):
        archive.validate()
        window.validate()
        choice.validate()
        calibration.validate()
        if (
            archive.runtime_rows is None
            or not archive.runtime_forecasts_replayed
            or archive.semantic_receipt_sha256
            != source_block.source_forecast_archive_receipt_sha256
            or archive.fold_index != source_block.fold_index
            or window.fold_index != source_block.fold_index
            or choice.fold_index != source_block.fold_index
            or calibration.fold_index != source_block.fold_index
            or choice.selected_checkpoint_receipt_sha256
            != archive.checkpoint_receipt_sha256
            or choice.selected_model_state_receipt_sha256
            != archive.model_state_receipt_sha256
            or choice.training_window_plan_receipt_sha256
            != window.semantic_receipt_sha256
            or calibration.checkpoint_receipt_sha256
            != archive.checkpoint_receipt_sha256
            or calibration.model_state_receipt_sha256
            != archive.model_state_receipt_sha256
            or calibration.training_window_plan_receipt_sha256
            != window.semantic_receipt_sha256
            or archive.origin_session_dates != source_block.forecast_session_dates
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "RL training forecast block provenance differs"
            )
        maturity_indices = tuple(
            row.candidate_origin_index
            + MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1
            for row in window.rows
        )
        maturity_cutoff = split_plan.candidate_session_dates[max(maturity_indices)]
        training_cutoff = source_block.training_cutoff_session_date
        first_forecast = archive.origin_session_dates[0]
        if (
            max(
                training_cutoff,
                maturity_cutoff,
                choice.selection_cutoff_session_date,
                calibration.calibration_fit_stop_session_date,
            )
            >= first_forecast
            or archive.origin_session_dates[-1] >= outer_start
        ):
            raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
                "RL training forecast cutoff reaches its forecast or outer block"
            )
        for start in range(0, len(archive.origin_session_dates), block_sessions):
            stop = min(start + block_sessions, len(archive.origin_session_dates))
            dates = archive.origin_session_dates[start:stop]
            receipts = archive.row_receipts[start:stop]
            body = {
                "schema": MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_BLOCK_V1_SCHEMA,
                "block_index": len(blocks),
                "source_fold_index": source_block.fold_index,
                "forecast_session_dates": dates,
                "supervised_training_cutoff_session_date": training_cutoff,
                "target_maturity_cutoff_session_date": maturity_cutoff,
                "checkpoint_selection_cutoff_session_date": (
                    choice.selection_cutoff_session_date
                ),
                "calibration_fit_cutoff_session_date": (
                    calibration.calibration_fit_stop_session_date
                ),
                "checkpoint_choice_receipt_sha256": choice.semantic_receipt_sha256,
                "checkpoint_receipt_sha256": archive.checkpoint_receipt_sha256,
                "checkpoint_source_receipt_sha256": (
                    choice.selected_checkpoint_source_receipt_sha256
                ),
                "model_state_receipt_sha256": archive.model_state_receipt_sha256,
                "calibration_receipt_sha256": calibration.semantic_receipt_sha256,
                "training_window_plan_receipt_sha256": window.semantic_receipt_sha256,
                "source_forecast_archive_receipt_sha256": archive.semantic_receipt_sha256,
                "forecast_row_receipts": receipts,
                "forecast_row_inventory_sha256": semantic_sha256(receipts),
                "source_data_qualified": bool(
                    archive.development_forecast_authorized
                    and choice.source_data_qualified
                    and calibration.development_calibration_authorized
                ),
            }
            block = MassiveAdaptiveRLTrainingForecastBlockV1(
                **body,  # type: ignore[arg-type]
                semantic_receipt_sha256=semantic_sha256(body),
            )
            block.validate()
            blocks.append(block)
            row_cursor += len(receipts)
    ordered = tuple(blocks)
    if row_cursor != len(prequential_archive.row_receipts):
        raise MassiveAdaptiveRLTrainingForecastAuthorityV1Error(
            "RL training forecast row inventory is incomplete"
        )
    source_qualified = bool(
        prequential_archive.development_prequential_forecast_authorized
        and split_plan.candidate_source_data_qualified
        and all(block.source_data_qualified for block in ordered)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SCHEMA,
        "outer_fold_index": outer_fold_index,
        "block_sessions": block_sessions,
        "blocks": ordered,
        "origin_session_dates": tuple(
            date for block in ordered for date in block.forecast_session_dates
        ),
        "block_inventory_sha256": semantic_sha256(
            tuple(block.semantic_receipt_sha256 for block in ordered)
        ),
        "forecast_row_inventory_sha256": semantic_sha256(
            tuple(
                receipt
                for block in ordered
                for receipt in block.forecast_row_receipts
            )
        ),
        "prequential_plan_receipt_sha256": prequential_plan.semantic_receipt_sha256,
        "prequential_archive_receipt_sha256": (
            prequential_archive.semantic_receipt_sha256
        ),
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
            MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_TRAINING_FORECAST_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLTrainingForecastAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(
            {**body, "blocks": tuple(asdict(block) for block in ordered)}
        ),
    )
    result.validate()
    return result


__all__ = [
    "MassiveAdaptiveCausalCheckpointChoiceV1",
    "MassiveAdaptiveRLTrainingForecastAuthorityV1",
    "MassiveAdaptiveRLTrainingForecastAuthorityV1Error",
    "MassiveAdaptiveRLTrainingForecastBlockV1",
    "build_massive_adaptive_causal_checkpoint_choice_v1",
    "build_massive_adaptive_rl_training_forecast_authority_v1",
]
