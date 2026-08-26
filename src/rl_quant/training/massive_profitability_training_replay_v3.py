"""Deterministic CPU training replay for the Massive P0 tournament."""

from __future__ import annotations

import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_HORIZONS_V1,
    MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1,
    MassiveProfitabilityTabularModelV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.training.alpha_supervised import (
    AlphaSupervisedBatch,
    alpha_supervised_loss,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
    MassiveProfitabilityTrainedRunV1,
    MassiveProfitabilityTrainingConfigV1,
    _collate_dates,
    _ordered_state,
    _validation_rank_ic,
    fit_massive_profitability_normalization_v1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentDatasetV2,
    MassiveProfitabilityTournamentPlanV2,
    MassiveProfitabilityTournamentV2Error,
    MassiveProfitabilityTrainingFoldV2,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_THREADS = 1


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingRuntimeV3:
    python_version: str
    torch_version: str
    numpy_version: str
    platform: str
    machine: str
    torch_config_sha256: str
    device: str
    dtype: str
    intraop_threads: int
    interop_threads: int
    deterministic_algorithms: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            self.device != "cpu"
            or self.dtype != "float32"
            or self.intraop_threads != MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_THREADS
            or self.interop_threads <= 0
            or not self.deterministic_algorithms
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "training replay V3 runtime differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingEpochV3:
    epoch: int
    validation_rank_ic: tuple[float, ...]
    validation_mean_rank_ic: float
    model_state_sha256: str
    selected_as_best: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            self.epoch < 0
            or len(self.validation_rank_ic) != len(MASSIVE_PROFITABILITY_HORIZONS_V1)
            or not math.isfinite(self.validation_mean_rank_ic)
            or any(not math.isfinite(value) for value in self.validation_rank_ic)
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityTournamentV2Error(
                "training replay V3 epoch trace differs"
            )


def massive_profitability_training_runtime_v3() -> MassiveProfitabilityTrainingRuntimeV3:
    body = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_config_sha256": semantic_sha256(torch.__config__.show()),
        "device": "cpu",
        "dtype": "float32",
        "intraop_threads": MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_THREADS,
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": True,
    }
    result = MassiveProfitabilityTrainingRuntimeV3(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _epoch_row(
    *,
    epoch: int,
    rank_ic: tuple[float, ...],
    model: MassiveProfitabilityTabularModelV1,
    selected_as_best: bool,
) -> MassiveProfitabilityTrainingEpochV3:
    body = {
        "epoch": epoch,
        "validation_rank_ic": rank_ic,
        "validation_mean_rank_ic": sum(rank_ic) / len(rank_ic),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "selected_as_best": selected_as_best,
    }
    result = MassiveProfitabilityTrainingEpochV3(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def train_massive_profitability_fold_replay_v3(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    seed: int,
    config: MassiveProfitabilityTrainingConfigV1 | None = None,
) -> tuple[
    MassiveProfitabilityTrainedRunV1,
    tuple[MassiveProfitabilityTrainingEpochV3, ...],
    MassiveProfitabilityTrainingRuntimeV3,
]:
    """Replay the complete frozen fit on deterministic one-thread CPU."""

    dataset.validate()
    tournament_plan.validate()
    fold.validate()
    training_config = MassiveProfitabilityTrainingConfigV1() if config is None else config
    training_config.validate()
    if (
        not training_config.is_frozen_authorizing_contract
        or setting_id not in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
        or seed not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
        or dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
        or set(fold.outer_test_session_dates) & set(dataset.maturation_only_session_dates)
    ):
        raise MassiveProfitabilityTournamentV2Error(
            "training replay V3 is detached from the frozen tournament"
        )
    runtime = massive_profitability_training_runtime_v3()
    mapping = dataset.by_date()
    fit_rows = tuple(mapping[value] for value in fold.fit_session_dates)
    validation_rows = tuple(
        mapping[value] for value in fold.inner_validation_session_dates
    )
    normalization = fit_massive_profitability_normalization_v1(
        dataset=dataset,  # type: ignore[arg-type]
        fit_session_dates=fold.fit_session_dates,
        target_scale_floor=training_config.target_scale_floor,
    )
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_THREADS)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        device = torch.device("cpu")
        model = MassiveProfitabilityTabularModelV1(setting_id=setting_id).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        best_metric = -math.inf
        best_epoch = -1
        best_state: tuple[tuple[str, torch.Tensor], ...] | None = None
        best_rank_ic: tuple[float, ...] | None = None
        patience = 0
        trace: list[MassiveProfitabilityTrainingEpochV3] = []
        for epoch in range(training_config.maximum_epochs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed * 1_000_003 + epoch)
            order = torch.randperm(len(fit_rows), generator=generator).tolist()
            model.train()
            for start in range(0, len(order), training_config.complete_dates_per_batch):
                selected = tuple(
                    fit_rows[index]
                    for index in order[
                        start : start + training_config.complete_dates_per_batch
                    ]
                )
                bars, bars_valid, tape, tape_valid, targets, valid = _collate_dates(
                    rows=selected,
                    normalization=normalization,
                    setting_id=setting_id,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                distribution = model(
                    bars_values=bars,
                    bars_valid=bars_valid,
                    tape_values=tape,
                    tape_valid=tape_valid,
                    source_staleness=torch.zeros(
                        bars.shape[:-1] + (1,), dtype=bars.dtype, device=device
                    ),
                )
                loss = alpha_supervised_loss(
                    AlphaSupervisedBatch(
                        distribution=distribution,
                        target=targets,
                        valid=valid,
                        executable_score=distribution.mean,
                    ),
                    training_config.objective,
                )
                loss.total.backward()
                optimizer.step()
            rank_ic = _validation_rank_ic(
                model=model,
                rows=validation_rows,
                normalization=normalization,
                setting_id=setting_id,
                complete_dates_per_batch=training_config.complete_dates_per_batch,
                device=device,
            )
            metric = sum(rank_ic) / len(rank_ic)
            selected_as_best = metric > best_metric
            trace.append(
                _epoch_row(
                    epoch=epoch,
                    rank_ic=rank_ic,
                    model=model,
                    selected_as_best=selected_as_best,
                )
            )
            if selected_as_best:
                best_metric = metric
                best_epoch = epoch
                best_state = _ordered_state(model)
                best_rank_ic = rank_ic
                patience = 0
            else:
                patience += 1
                if patience >= training_config.early_stopping_patience:
                    break
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        torch.set_num_threads(prior_threads)
    if best_state is None or best_rank_ic is None or best_epoch < 0:
        raise MassiveProfitabilityTournamentV2Error(
            "training replay V3 selected no checkpoint"
        )
    model_state = {name: value for name, value in best_state}
    training_source = semantic_sha256(
        {
            "fit": tuple(row.source_array_sha256 for row in fit_rows),
            "inner_validation": tuple(
                row.source_array_sha256 for row in validation_rows
            ),
        }
    )
    receipt_body = {
        "setting_id": setting_id,
        "fold_index": fold.fold_index,
        "seed": seed,
        "best_epoch": best_epoch,
        "completed_epochs": len(trace),
        "validation_rank_ic": best_rank_ic,
        "normalization_receipt_sha256": normalization.receipt_sha256,
        "model_state_sha256": state_dict_sha256(model_state),
        "fit_inventory_sha256": fold.fit_inventory_sha256,
        "validation_inventory_sha256": fold.inner_validation_inventory_sha256,
        "training_source_receipt_sha256": training_source,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "training_config_receipt_sha256": training_config.receipt_sha256,
        "outer_prediction_authorized": tournament_plan.outer_prediction_authorized,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    run = MassiveProfitabilityTrainedRunV1(
        setting_id=setting_id,
        fold_index=fold.fold_index,
        seed=seed,
        best_epoch=best_epoch,
        completed_epochs=len(trace),
        validation_rank_ic=best_rank_ic,
        normalization=normalization,
        model_state_sha256=state_dict_sha256(model_state),
        model_state=best_state,
        fit_inventory_sha256=fold.fit_inventory_sha256,
        validation_inventory_sha256=fold.inner_validation_inventory_sha256,
        training_source_receipt_sha256=training_source,
        tournament_plan_receipt_sha256=tournament_plan.receipt_sha256,
        training_config_receipt_sha256=training_config.receipt_sha256,
        run_receipt_sha256=semantic_sha256(receipt_body),
        outer_prediction_authorized=tournament_plan.outer_prediction_authorized,
    )
    run.validate()
    return run, tuple(trace), runtime


__all__ = [
    "MASSIVE_PROFITABILITY_TRAINING_REPLAY_V3_SOURCE_SHA256",
    "MassiveProfitabilityTrainingEpochV3",
    "MassiveProfitabilityTrainingRuntimeV3",
    "massive_profitability_training_runtime_v3",
    "train_massive_profitability_fold_replay_v3",
]
