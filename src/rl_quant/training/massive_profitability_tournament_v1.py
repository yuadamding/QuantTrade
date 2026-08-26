"""Purged, date-balanced training for the minimal Massive P0 model tournament."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
    MASSIVE_PROFITABILITY_HORIZONS_V1,
    MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
    MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
    MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1,
    MassiveProfitabilityTabularModelV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.protocol.massive_finalized_validation_v0 import (
    MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL,
)
from rl_quant.training.alpha_supervised import (
    AlphaObjectiveConfig,
    AlphaSupervisedBatch,
    alpha_supervised_loss,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA = (
    "rl-quant.massive-profitability-tournament-v1"
)
MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA = (
    "rl-quant.massive-profitability-model-checkpoint-v1"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_DATASET = (
    "massive-profitability-model-checkpoint-v1"
)
MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA,
        "encoding": "canonical-json-float32-state",
        "publication": "create-only-source-transaction",
        "pickle": False,
    }
)
MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1 = (0, 1)
MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1 = (0, 1, 2, 3, 4)
MASSIVE_PROFITABILITY_SHUFFLE_SEED_V1 = 17012026

_PROTOCOL = MASSIVE_FINALIZED_VALIDATION_V0_PROTOCOL
MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "settings": MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1,
        "primary_contrast": ("MV04", "MV02"),
        "placebo_contrast": ("MV04", "MV04-SHUFFLE"),
        "development_seeds": MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1,
        "confirmation_seeds": MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
        "horizons": MASSIVE_PROFITABILITY_HORIZONS_V1,
        "folds": {
            "count": _PROTOCOL.outer_fold_count,
            "minimum_fit": _PROTOCOL.minimum_initial_training_sessions,
            "inner_purge": _PROTOCOL.inner_purge_sessions,
            "inner_validation": _PROTOCOL.inner_validation_sessions,
            "outer_purge": _PROTOCOL.target_overlap_purge_sessions,
            "outer_test": _PROTOCOL.outer_fold_sessions,
        },
        "normalization": "fit-only-valid-value-mean-population-std",
        "target_scaling": "fit-only-median-1.4826-MAD-with-1e-6-floor",
        "shuffle": "within-date-security-permutation-tape-only",
        "batch": "complete-date-cross-sections",
        "optimizer": ("AdamW", 3e-4, 1e-4),
        "objective": (1.0, 0.2, 0.25, 0.1, 0.0),
        "maximum_epochs": 100,
        "early_stopping_patience": 10,
        "lockbox": "targets-excluded",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityTournamentV1Error(ValueError):
    """The minimal P0 tournament differs from its frozen information set."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTournamentV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingConfigV1:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    maximum_epochs: int = 100
    early_stopping_patience: int = 10
    complete_dates_per_batch: int = 8
    target_scale_floor: float = 1e-6
    objective: AlphaObjectiveConfig = field(
        default_factory=lambda: AlphaObjectiveConfig(
            huber_weight=1.0,
            rank_weight=0.2,
            quantile_weight=0.25,
            calibration_weight=0.1,
            residual_ssl_weight=0.0,
        )
    )

    def validate(self) -> None:
        self.objective.validate()
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or isinstance(self.maximum_epochs, bool)
            or not isinstance(self.maximum_epochs, int)
            or self.maximum_epochs <= 0
            or isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
            or self.early_stopping_patience >= self.maximum_epochs
            or isinstance(self.complete_dates_per_batch, bool)
            or not isinstance(self.complete_dates_per_batch, int)
            or self.complete_dates_per_batch <= 0
            or not math.isfinite(self.target_scale_floor)
            or self.target_scale_floor <= 0.0
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "training configuration is invalid"
            )

    @property
    def is_frozen_authorizing_contract(self) -> bool:
        return self == MassiveProfitabilityTrainingConfigV1()

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityNormalizationV1:
    bars_mean: tuple[float, ...]
    bars_scale: tuple[float, ...]
    bars_observed: tuple[bool, ...]
    tape_mean: tuple[float, ...]
    tape_scale: tuple[float, ...]
    tape_observed: tuple[bool, ...]
    target_median: tuple[float, ...]
    target_scale: tuple[float, ...]
    fit_session_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        groups = (
            (
                self.bars_mean,
                self.bars_scale,
                self.bars_observed,
                MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
            ),
            (
                self.tape_mean,
                self.tape_scale,
                self.tape_observed,
                MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
            ),
            (
                self.target_median,
                self.target_scale,
                (True,) * len(MASSIVE_PROFITABILITY_HORIZONS_V1),
                len(MASSIVE_PROFITABILITY_HORIZONS_V1),
            ),
        )
        for means, scales, observed, length in groups:
            if (
                len(means) != length
                or len(scales) != length
                or len(observed) != length
                or any(not isinstance(value, bool) for value in observed)
                or any(not math.isfinite(value) for value in means)
                or any(not math.isfinite(value) or value <= 0.0 for value in scales)
            ):
                raise MassiveProfitabilityTournamentV1Error(
                    "fit-only normalization differs"
                )
        _digest("normalization fit inventory", self.fit_session_inventory_sha256)
        _digest("normalization receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTournamentV1Error(
                "normalization receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityDateTensorV1:
    decision_session_date: str
    security_ids: tuple[str, ...]
    bars_values: torch.Tensor
    bars_valid: torch.Tensor
    tape_values: torch.Tensor
    tape_valid: torch.Tensor
    target_values: torch.Tensor
    target_valid: torch.Tensor
    feature_semantic_receipt_sha256: str
    target_semantic_receipt_sha256: str
    source_array_sha256: str

    def validate(self) -> None:
        asset_count = len(self.security_ids)
        tensors = (
            self.bars_values,
            self.tape_values,
            self.target_values,
        )
        if (
            not self.decision_session_date
            or asset_count < 2
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or self.bars_values.shape
            != (asset_count, MASSIVE_PROFITABILITY_BARS_DIMENSION_V1)
            or self.bars_valid.shape != self.bars_values.shape
            or self.tape_values.shape
            != (asset_count, MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1)
            or self.tape_valid.shape != self.tape_values.shape
            or self.target_values.shape
            != (asset_count, len(MASSIVE_PROFITABILITY_HORIZONS_V1))
            or self.target_valid.shape != self.target_values.shape
            or self.bars_valid.dtype != torch.bool
            or self.tape_valid.dtype != torch.bool
            or self.target_valid.dtype != torch.bool
            or any(value.dtype != torch.float32 or value.device.type != "cpu" for value in tensors)
            or any(not bool(torch.isfinite(value).all()) for value in tensors)
            or bool((~self.bars_valid & (self.bars_values != 0.0)).any())
            or bool((~self.tape_valid & (self.tape_values != 0.0)).any())
            or bool((~self.target_valid & (self.target_values != 0.0)).any())
            or bool((self.target_valid.sum(dim=0) < 2).any())
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "one tournament date tensor is malformed"
            )
        for value in (
            self.feature_semantic_receipt_sha256,
            self.target_semantic_receipt_sha256,
            self.source_array_sha256,
        ):
            _digest("date tensor", value)
        expected = semantic_sha256(
            {
                "decision_session_date": self.decision_session_date,
                "security_ids": self.security_ids,
                "bars_values": _tensor_sha256(self.bars_values),
                "bars_valid": _tensor_sha256(self.bars_valid),
                "tape_values": _tensor_sha256(self.tape_values),
                "tape_valid": _tensor_sha256(self.tape_valid),
                "target_values": _tensor_sha256(self.target_values),
                "target_valid": _tensor_sha256(self.target_valid),
                "feature_receipt": self.feature_semantic_receipt_sha256,
                "target_receipt": self.target_semantic_receipt_sha256,
            }
        )
        if self.source_array_sha256 != expected:
            raise MassiveProfitabilityTournamentV1Error(
                "date tensor source identity differs"
            )


def build_massive_profitability_date_tensor_v1(
    *,
    decision_session_date: str,
    security_ids: Sequence[str],
    bars_values: torch.Tensor,
    bars_valid: torch.Tensor,
    tape_values: torch.Tensor,
    tape_valid: torch.Tensor,
    target_values: torch.Tensor,
    target_valid: torch.Tensor,
    feature_semantic_receipt_sha256: str,
    target_semantic_receipt_sha256: str,
) -> MassiveProfitabilityDateTensorV1:
    """Bind one primitive complete-date cross-section to exact source arrays."""

    keys = tuple(security_ids)
    tensors = tuple(
        value.detach().to(device="cpu").contiguous()
        for value in (
            bars_values,
            bars_valid,
            tape_values,
            tape_valid,
            target_values,
            target_valid,
        )
    )
    identity = {
        "decision_session_date": decision_session_date,
        "security_ids": keys,
        "bars_values": _tensor_sha256(tensors[0]),
        "bars_valid": _tensor_sha256(tensors[1]),
        "tape_values": _tensor_sha256(tensors[2]),
        "tape_valid": _tensor_sha256(tensors[3]),
        "target_values": _tensor_sha256(tensors[4]),
        "target_valid": _tensor_sha256(tensors[5]),
        "feature_receipt": feature_semantic_receipt_sha256,
        "target_receipt": target_semantic_receipt_sha256,
    }
    result = MassiveProfitabilityDateTensorV1(
        decision_session_date=decision_session_date,
        security_ids=keys,
        bars_values=tensors[0],
        bars_valid=tensors[1],
        tape_values=tensors[2],
        tape_valid=tensors[3],
        target_values=tensors[4],
        target_valid=tensors[5],
        feature_semantic_receipt_sha256=feature_semantic_receipt_sha256,
        target_semantic_receipt_sha256=target_semantic_receipt_sha256,
        source_array_sha256=semantic_sha256(identity),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTournamentDatasetV1:
    dates: tuple[MassiveProfitabilityDateTensorV1, ...]
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    dataset_receipt_sha256: str

    def validate(self) -> None:
        session_dates = tuple(row.decision_session_date for row in self.dates)
        if not self.dates or session_dates != tuple(sorted(set(session_dates))):
            raise MassiveProfitabilityTournamentV1Error(
                "tournament dataset dates differ"
            )
        for row in self.dates:
            row.validate()
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.dataset_receipt_sha256,
        ):
            _digest("tournament dataset", value)
        if self.dataset_receipt_sha256 != semantic_sha256(
            {
                "dates": tuple(row.source_array_sha256 for row in self.dates),
                "data_gate": self.data_gate_semantic_receipt_sha256,
                "phase_plan": self.phase_plan_semantic_receipt_sha256,
            }
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "tournament dataset receipt differs"
            )

    def by_date(self) -> dict[str, MassiveProfitabilityDateTensorV1]:
        self.validate()
        return {row.decision_session_date: row for row in self.dates}


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingFoldV1:
    """Feature-agnostic copy of one already-frozen purged fold geometry."""

    fold_index: int
    fit_session_dates: tuple[str, ...]
    inner_purge_session_dates: tuple[str, ...]
    inner_validation_session_dates: tuple[str, ...]
    outer_purge_session_dates: tuple[str, ...]
    outer_test_session_dates: tuple[str, ...]
    fit_inventory_sha256: str
    inner_validation_inventory_sha256: str
    outer_test_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        inventories = (
            self.fit_session_dates,
            self.inner_purge_session_dates,
            self.inner_validation_session_dates,
            self.outer_purge_session_dates,
            self.outer_test_session_dates,
        )
        if (
            isinstance(self.fold_index, bool)
            or not 0 <= self.fold_index < _PROTOCOL.outer_fold_count
            or any(not values or values != tuple(sorted(set(values))) for values in inventories)
            or len(self.fit_session_dates) < _PROTOCOL.minimum_initial_training_sessions
            or len(self.inner_purge_session_dates) != _PROTOCOL.inner_purge_sessions
            or len(self.inner_validation_session_dates)
            != _PROTOCOL.inner_validation_sessions
            or len(self.outer_purge_session_dates)
            != _PROTOCOL.target_overlap_purge_sessions
            or len(self.outer_test_session_dates) != _PROTOCOL.outer_fold_sessions
            or not (
                self.fit_session_dates[-1]
                < self.inner_purge_session_dates[0]
                < self.inner_validation_session_dates[0]
                < self.outer_purge_session_dates[0]
                < self.outer_test_session_dates[0]
            )
            or len(tuple(value for values in inventories for value in values))
            != len({value for values in inventories for value in values})
            or self.fit_inventory_sha256
            != semantic_sha256(self.fit_session_dates)
            or self.inner_validation_inventory_sha256
            != semantic_sha256(self.inner_validation_session_dates)
            or self.outer_test_inventory_sha256
            != semantic_sha256(self.outer_test_session_dates)
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "training fold geometry differs"
            )
        for value in (
            self.fit_inventory_sha256,
            self.inner_validation_inventory_sha256,
            self.outer_test_inventory_sha256,
            self.receipt_sha256,
        ):
            _digest("training fold", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTournamentV1Error(
                "training fold receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTournamentPlanV1:
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    fold_receipts: tuple[str, ...]
    settings: tuple[str, ...]
    development_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    specification_sha256: str
    implementation_source_sha256: str
    input_adapter_source_sha256: str
    receipt_sha256: str
    development_training_authorized: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA
            or len(self.fold_receipts) != _PROTOCOL.outer_fold_count
            or self.settings != MASSIVE_PROFITABILITY_TOURNAMENT_SETTINGS_V1
            or self.development_seeds != MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1
            or self.confirmation_seeds != MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_V1_SOURCE_SHA256
            or not self.development_training_authorized
            or not self.outer_prediction_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "tournament plan identity or authorization differs"
            )
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            *self.fold_receipts,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.input_adapter_source_sha256,
            self.receipt_sha256,
        ):
            _digest("tournament plan", value)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveProfitabilityTournamentV1Error(
                "tournament plan receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainedRunV1:
    setting_id: str
    fold_index: int
    seed: int
    best_epoch: int
    completed_epochs: int
    validation_rank_ic: tuple[float, ...]
    normalization: MassiveProfitabilityNormalizationV1
    model_state_sha256: str
    model_state: tuple[tuple[str, torch.Tensor], ...]
    fit_inventory_sha256: str
    validation_inventory_sha256: str
    training_source_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    training_config_receipt_sha256: str
    run_receipt_sha256: str
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.model_state}

    def validate(self) -> None:
        if (
            self.setting_id not in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
            or isinstance(self.fold_index, bool)
            or not 0 <= self.fold_index < _PROTOCOL.outer_fold_count
            or self.seed not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or isinstance(self.best_epoch, bool)
            or not 0 <= self.best_epoch < self.completed_epochs
            or len(self.validation_rank_ic)
            != len(MASSIVE_PROFITABILITY_HORIZONS_V1)
            or any(not math.isfinite(value) for value in self.validation_rank_ic)
            or not self.model_state
            or tuple(name for name, _ in self.model_state)
            != tuple(sorted(name for name, _ in self.model_state))
            or not isinstance(self.outer_prediction_authorized, bool)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "trained run identity or authorization differs"
            )
        self.normalization.validate()
        for value in (
            self.model_state_sha256,
            self.fit_inventory_sha256,
            self.validation_inventory_sha256,
            self.training_source_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.training_config_receipt_sha256,
            self.run_receipt_sha256,
        ):
            _digest("trained run", value)
        state = self.state_dict()
        if self.model_state_sha256 != state_dict_sha256(state):
            raise MassiveProfitabilityTournamentV1Error(
                "trained model state differs"
            )
        expected = semantic_sha256(
            {
                "setting_id": self.setting_id,
                "fold_index": self.fold_index,
                "seed": self.seed,
                "best_epoch": self.best_epoch,
                "completed_epochs": self.completed_epochs,
                "validation_rank_ic": self.validation_rank_ic,
                "normalization_receipt_sha256": self.normalization.receipt_sha256,
                "model_state_sha256": self.model_state_sha256,
                "fit_inventory_sha256": self.fit_inventory_sha256,
                "validation_inventory_sha256": self.validation_inventory_sha256,
                "training_source_receipt_sha256": self.training_source_receipt_sha256,
                "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
                "training_config_receipt_sha256": self.training_config_receipt_sha256,
                "outer_prediction_authorized": self.outer_prediction_authorized,
                "profitability_reporting_authorized": False,
                "lockbox_access_authorized": False,
                "reinforcement_learning_authorized": False,
            }
        )
        if self.run_receipt_sha256 != expected:
            raise MassiveProfitabilityTournamentV1Error(
                "trained run receipt differs"
            )


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityModelCheckpointV1:
    """One safely serialized, create-only selected model state."""

    run: MassiveProfitabilityTrainedRunV1
    loaded_source: LoadedMassiveSourceObject
    schema: str = MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA

    def validate(self) -> None:
        self.run.validate()
        self.loaded_source.validate()
        if (
            self.schema != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.run.tournament_plan_receipt_sha256
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "model checkpoint source transaction differs"
            )


def _state_payload(
    state: tuple[tuple[str, torch.Tensor], ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for name, value in state:
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        rows.append(
            {
                "name": name,
                "dtype": "float32",
                "shape": tuple(tensor.shape),
                "values": tuple(float(item) for item in tensor.reshape(-1)),
            }
        )
    return tuple(rows)


def _run_payload(run: MassiveProfitabilityTrainedRunV1) -> dict[str, object]:
    run.validate()
    return {
        "setting_id": run.setting_id,
        "fold_index": run.fold_index,
        "seed": run.seed,
        "best_epoch": run.best_epoch,
        "completed_epochs": run.completed_epochs,
        "validation_rank_ic": run.validation_rank_ic,
        "normalization": asdict(run.normalization),
        "model_state_sha256": run.model_state_sha256,
        "model_state": _state_payload(run.model_state),
        "fit_inventory_sha256": run.fit_inventory_sha256,
        "validation_inventory_sha256": run.validation_inventory_sha256,
        "training_source_receipt_sha256": run.training_source_receipt_sha256,
        "tournament_plan_receipt_sha256": run.tournament_plan_receipt_sha256,
        "training_config_receipt_sha256": run.training_config_receipt_sha256,
        "run_receipt_sha256": run.run_receipt_sha256,
        "outer_prediction_authorized": run.outer_prediction_authorized,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _run_from_payload(payload: dict[str, object]) -> MassiveProfitabilityTrainedRunV1:
    normalization_raw = payload.pop("normalization")
    if not isinstance(normalization_raw, dict):
        raise MassiveProfitabilityTournamentV1Error(
            "checkpoint normalization payload differs"
        )
    normalization_payload = dict(cast(dict[str, object], normalization_raw))
    for name in (
        "bars_mean",
        "bars_scale",
        "bars_observed",
        "tape_mean",
        "tape_scale",
        "tape_observed",
        "target_median",
        "target_scale",
    ):
        normalization_payload[name] = tuple(normalization_payload[name])  # type: ignore[arg-type]
    normalization = MassiveProfitabilityNormalizationV1(**normalization_payload)  # type: ignore[arg-type]
    state_raw = payload.pop("model_state")
    if not isinstance(state_raw, (list, tuple)):
        raise MassiveProfitabilityTournamentV1Error(
            "checkpoint state payload differs"
        )
    state_rows: tuple[object, ...] = tuple(state_raw)
    state: list[tuple[str, torch.Tensor]] = []
    for item in state_rows:
        if not isinstance(item, dict):
            raise MassiveProfitabilityTournamentV1Error(
                "checkpoint tensor row differs"
            )
        row = dict(cast(dict[str, object], item))
        if row.get("dtype") != "float32":
            raise MassiveProfitabilityTournamentV1Error(
                "checkpoint tensor dtype differs"
            )
        shape_raw = row.get("shape")
        values_raw = row.get("values")
        if (
            not isinstance(shape_raw, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape_raw)
            or not isinstance(values_raw, (list, tuple))
        ):
            raise MassiveProfitabilityTournamentV1Error(
                "checkpoint tensor dimensions differ"
            )
        shape: tuple[int, ...] = tuple(shape_raw)
        tensor = torch.tensor(values_raw, dtype=torch.float32).reshape(shape)
        state.append((str(row["name"]), tensor))
    payload["validation_rank_ic"] = tuple(payload["validation_rank_ic"])  # type: ignore[arg-type]
    result = MassiveProfitabilityTrainedRunV1(
        **payload,  # type: ignore[arg-type]
        normalization=normalization,
        model_state=tuple(state),
    )
    result.validate()
    return result


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "model checkpoint artifact ID is not path safe"
        )
    return value


def publish_massive_profitability_model_checkpoint_v1(
    *,
    root: str | Path,
    artifact_id: str,
    run: MassiveProfitabilityTrainedRunV1,
    committed_at_ms: int,
) -> MassiveProfitabilityModelCheckpointV1:
    """Publish a selected state without pickle or executable deserialization."""

    run.validate()
    artifact = _artifact_id(artifact_id)
    payload = {
        "schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA,
        "run": _run_payload(run),
    }
    relative = f"massive-profitability-model-checkpoint-v1/{artifact}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=run.tournament_plan_receipt_sha256,
        committed_at_ms=committed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    result = MassiveProfitabilityModelCheckpointV1(run=run, loaded_source=loaded)
    result.validate()
    return result


def parse_massive_profitability_model_checkpoint_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityModelCheckpointV1:
    """Reload exact checkpoint bytes through the safe tensor-list encoding."""

    exact = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(exact)
    if payload.get("schema") != MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA:
        raise MassiveProfitabilityTournamentV1Error(
            "model checkpoint payload schema differs"
        )
    run = _run_from_payload(dict(payload["run"]))
    result = MassiveProfitabilityModelCheckpointV1(run=run, loaded_source=loaded_source)
    result.validate()
    if canonical_json_file_bytes(
        {"schema": MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA, "run": _run_payload(run)}
    ) != exact:
        raise MassiveProfitabilityTournamentV1Error(
            "model checkpoint canonical bytes differ"
        )
    return result


def _feature_statistics(
    rows: Sequence[MassiveProfitabilityDateTensorV1],
    *,
    value_name: str,
    valid_name: str,
    dimension: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[bool, ...]]:
    means: list[float] = []
    scales: list[float] = []
    observed: list[bool] = []
    for index in range(dimension):
        selected = torch.cat(
            [
                getattr(row, value_name)[:, index][getattr(row, valid_name)[:, index]]
                for row in rows
            ]
        )
        if selected.numel() == 0:
            means.append(0.0)
            scales.append(1.0)
            observed.append(False)
            continue
        mean = float(selected.mean(dtype=torch.float64))
        scale = float(selected.to(torch.float64).std(unbiased=False))
        means.append(mean)
        scales.append(scale if scale > 0.0 else 1.0)
        observed.append(True)
    return tuple(means), tuple(scales), tuple(observed)


def fit_massive_profitability_normalization_v1(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    fit_session_dates: Sequence[str],
    target_scale_floor: float = 1e-6,
) -> MassiveProfitabilityNormalizationV1:
    """Fit all feature and target scaling strictly on one fold's fit dates."""

    dataset.validate()
    fit_dates = tuple(fit_session_dates)
    if (
        not fit_dates
        or fit_dates != tuple(sorted(set(fit_dates)))
        or not math.isfinite(target_scale_floor)
        or target_scale_floor <= 0.0
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "fit normalization date inventory differs"
        )
    mapping = dataset.by_date()
    if any(value not in mapping for value in fit_dates):
        raise MassiveProfitabilityTournamentV1Error(
            "fit normalization references a missing date"
        )
    rows = tuple(mapping[value] for value in fit_dates)
    bars_mean, bars_scale, bars_observed = _feature_statistics(
        rows,
        value_name="bars_values",
        valid_name="bars_valid",
        dimension=MASSIVE_PROFITABILITY_BARS_DIMENSION_V1,
    )
    tape_mean, tape_scale, tape_observed = _feature_statistics(
        rows,
        value_name="tape_values",
        valid_name="tape_valid",
        dimension=MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1,
    )
    target_median: list[float] = []
    target_scale: list[float] = []
    for index in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1)):
        selected = torch.cat(
            [row.target_values[:, index][row.target_valid[:, index]] for row in rows]
        ).to(torch.float64)
        if selected.numel() < 2:
            raise MassiveProfitabilityTournamentV1Error(
                "fit target horizon has fewer than two observations"
            )
        median = float(selected.median())
        mad = float((selected - median).abs().median())
        target_median.append(median)
        target_scale.append(max(1.4826 * mad, target_scale_floor))
    body = {
        "bars_mean": bars_mean,
        "bars_scale": bars_scale,
        "bars_observed": bars_observed,
        "tape_mean": tape_mean,
        "tape_scale": tape_scale,
        "tape_observed": tape_observed,
        "target_median": tuple(target_median),
        "target_scale": tuple(target_scale),
        "fit_session_inventory_sha256": semantic_sha256(fit_dates),
    }
    result = MassiveProfitabilityNormalizationV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def massive_profitability_tape_permutation_v1(
    *, decision_session_date: str, security_ids: Sequence[str]
) -> torch.Tensor:
    """Return the frozen target-independent within-date tape permutation."""

    keys = tuple(security_ids)
    if not decision_session_date or len(keys) < 2 or keys != tuple(sorted(set(keys))):
        raise MassiveProfitabilityTournamentV1Error(
            "shuffle security inventory differs"
        )
    ordered = sorted(
        range(len(keys)),
        key=lambda index: semantic_sha256(
            (MASSIVE_PROFITABILITY_SHUFFLE_SEED_V1, decision_session_date, keys[index])
        ),
    )
    if ordered == list(range(len(keys))):
        ordered = ordered[1:] + ordered[:1]
    return torch.tensor(ordered, dtype=torch.long)


def _normalized(
    values: torch.Tensor,
    valid: torch.Tensor,
    mean: tuple[float, ...],
    scale: tuple[float, ...],
) -> torch.Tensor:
    location = torch.tensor(mean, dtype=values.dtype, device=values.device)
    spread = torch.tensor(scale, dtype=values.dtype, device=values.device)
    return torch.where(valid, (values - location) / spread, torch.zeros_like(values))


def _collate_dates(
    *,
    rows: Sequence[MassiveProfitabilityDateTensorV1],
    normalization: MassiveProfitabilityNormalizationV1,
    setting_id: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise MassiveProfitabilityTournamentV1Error("empty complete-date batch")
    maximum_assets = max(len(row.security_ids) for row in rows)
    date_count = len(rows)
    bars_values = torch.zeros(
        (date_count, maximum_assets, MASSIVE_PROFITABILITY_BARS_DIMENSION_V1),
        dtype=torch.float32,
        device=device,
    )
    bars_valid = torch.zeros_like(bars_values, dtype=torch.bool)
    tape_values = torch.zeros(
        (date_count, maximum_assets, MASSIVE_PROFITABILITY_TAPE_DIMENSION_V1),
        dtype=torch.float32,
        device=device,
    )
    tape_valid = torch.zeros_like(tape_values, dtype=torch.bool)
    targets = torch.zeros(
        (date_count, maximum_assets, len(MASSIVE_PROFITABILITY_HORIZONS_V1)),
        dtype=torch.float32,
        device=device,
    )
    target_valid = torch.zeros_like(targets, dtype=torch.bool)
    target_median = torch.tensor(
        normalization.target_median, dtype=torch.float32, device=device
    )
    target_scale = torch.tensor(
        normalization.target_scale, dtype=torch.float32, device=device
    )
    for date_index, row in enumerate(rows):
        count = len(row.security_ids)
        raw_bars = row.bars_values.to(device)
        raw_bars_valid = row.bars_valid.to(device)
        raw_tape = row.tape_values.to(device)
        raw_tape_valid = row.tape_valid.to(device)
        if setting_id == "MV04-SHUFFLE":
            permutation = massive_profitability_tape_permutation_v1(
                decision_session_date=row.decision_session_date,
                security_ids=row.security_ids,
            ).to(device)
            raw_tape = raw_tape.index_select(0, permutation)
            raw_tape_valid = raw_tape_valid.index_select(0, permutation)
        bars_values[date_index, :count] = _normalized(
            raw_bars, raw_bars_valid, normalization.bars_mean, normalization.bars_scale
        )
        bars_valid[date_index, :count] = raw_bars_valid
        tape_values[date_index, :count] = _normalized(
            raw_tape, raw_tape_valid, normalization.tape_mean, normalization.tape_scale
        )
        tape_valid[date_index, :count] = raw_tape_valid
        raw_target = row.target_values.to(device)
        raw_target_valid = row.target_valid.to(device)
        targets[date_index, :count] = torch.where(
            raw_target_valid,
            (raw_target - target_median) / target_scale,
            torch.zeros_like(raw_target),
        )
        target_valid[date_index, :count] = raw_target_valid
    return bars_values, bars_valid, tape_values, tape_valid, targets, target_valid


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    sorted_values = values.index_select(0, order)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(prediction: torch.Tensor, target: torch.Tensor) -> float:
    first = _average_ranks(prediction.detach().to(device="cpu", dtype=torch.float64))
    second = _average_ranks(target.detach().to(device="cpu", dtype=torch.float64))
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return 0.0 if float(denominator) == 0.0 else float((first * second).sum() / denominator)


def _validation_rank_ic(
    *,
    model: MassiveProfitabilityTabularModelV1,
    rows: Sequence[MassiveProfitabilityDateTensorV1],
    normalization: MassiveProfitabilityNormalizationV1,
    setting_id: str,
    complete_dates_per_batch: int,
    device: torch.device,
) -> tuple[float, ...]:
    values: list[list[float]] = [
        [] for _ in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1))
    ]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), complete_dates_per_batch):
            selected = rows[start : start + complete_dates_per_batch]
            bars, bars_valid, tape, tape_valid, targets, valid = _collate_dates(
                rows=selected,
                normalization=normalization,
                setting_id=setting_id,
                device=device,
            )
            distribution = model(
                bars_values=bars,
                bars_valid=bars_valid,
                tape_values=tape,
                tape_valid=tape_valid,
                source_staleness=torch.zeros(
                    bars.shape[:-1] + (1,), dtype=bars.dtype, device=device
                ),
            )
            for date_index in range(len(selected)):
                for horizon_index in range(len(MASSIVE_PROFITABILITY_HORIZONS_V1)):
                    mask = valid[date_index, :, horizon_index]
                    values[horizon_index].append(
                        _spearman(
                            distribution.mean[date_index, mask, horizon_index],
                            targets[date_index, mask, horizon_index],
                        )
                    )
    if any(not row for row in values):
        raise MassiveProfitabilityTournamentV1Error(
            "validation has no supported horizon"
        )
    return tuple(sum(row) / len(row) for row in values)


def _ordered_state(model: MassiveProfitabilityTabularModelV1) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, value.detach().to(device="cpu").clone())
        for name, value in sorted(model.state_dict().items())
    )


def train_massive_profitability_fold_v1(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV1,
    tournament_plan: MassiveProfitabilityTournamentPlanV1,
    fold: MassiveProfitabilityTrainingFoldV1,
    setting_id: str,
    seed: int,
    config: MassiveProfitabilityTrainingConfigV1 | None = None,
    device: str | torch.device = "cpu",
) -> MassiveProfitabilityTrainedRunV1:
    """Fit one trainable setting without reading purge, outer, or lockbox targets."""

    dataset.validate()
    tournament_plan.validate()
    fold.validate()
    config = MassiveProfitabilityTrainingConfigV1() if config is None else config
    config.validate()
    if (
        setting_id not in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
        or seed not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
        or dataset.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityTournamentV1Error(
            "training run is detached from the tournament plan"
        )
    mapping = dataset.by_date()
    fit_rows = tuple(mapping[value] for value in fold.fit_session_dates)
    validation_rows = tuple(
        mapping[value] for value in fold.inner_validation_session_dates
    )
    normalization = fit_massive_profitability_normalization_v1(
        dataset=dataset,
        fit_session_dates=fold.fit_session_dates,
        target_scale_floor=config.target_scale_floor,
    )
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise MassiveProfitabilityTournamentV1Error("requested CUDA is unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = MassiveProfitabilityTabularModelV1(setting_id=setting_id).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_metric = -math.inf
    best_epoch = -1
    best_state: tuple[tuple[str, torch.Tensor], ...] | None = None
    best_rank_ic: tuple[float, ...] | None = None
    completed_epochs = 0
    patience = 0
    objective = config.objective
    for epoch in range(config.maximum_epochs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 1_000_003 + epoch)
        order = torch.randperm(len(fit_rows), generator=generator).tolist()
        model.train()
        for start in range(0, len(order), config.complete_dates_per_batch):
            selected = tuple(
                fit_rows[index]
                for index in order[start : start + config.complete_dates_per_batch]
            )
            bars, bars_valid, tape, tape_valid, targets, valid = _collate_dates(
                rows=selected,
                normalization=normalization,
                setting_id=setting_id,
                device=target_device,
            )
            optimizer.zero_grad(set_to_none=True)
            distribution = model(
                bars_values=bars,
                bars_valid=bars_valid,
                tape_values=tape,
                tape_valid=tape_valid,
                source_staleness=torch.zeros(
                    bars.shape[:-1] + (1,), dtype=bars.dtype, device=target_device
                ),
            )
            loss = alpha_supervised_loss(
                AlphaSupervisedBatch(
                    distribution=distribution,
                    target=targets,
                    valid=valid,
                    executable_score=distribution.mean,
                ),
                objective,
            )
            loss.total.backward()
            optimizer.step()
        rank_ic = _validation_rank_ic(
            model=model,
            rows=validation_rows,
            normalization=normalization,
            setting_id=setting_id,
            complete_dates_per_batch=config.complete_dates_per_batch,
            device=target_device,
        )
        metric = sum(rank_ic) / len(rank_ic)
        completed_epochs = epoch + 1
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = _ordered_state(model)
            best_rank_ic = rank_ic
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None or best_rank_ic is None or best_epoch < 0:
        raise MassiveProfitabilityTournamentV1Error("training selected no checkpoint")
    model_state = {name: value for name, value in best_state}
    authorized = (
        tournament_plan.outer_prediction_authorized
        and config.is_frozen_authorizing_contract
    )
    receipt_body = {
        "setting_id": setting_id,
        "fold_index": fold.fold_index,
        "seed": seed,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "validation_rank_ic": best_rank_ic,
        "normalization_receipt_sha256": normalization.receipt_sha256,
        "model_state_sha256": state_dict_sha256(model_state),
        "fit_inventory_sha256": fold.fit_inventory_sha256,
        "validation_inventory_sha256": fold.inner_validation_inventory_sha256,
        "training_source_receipt_sha256": semantic_sha256(
            {
                "fit": tuple(row.source_array_sha256 for row in fit_rows),
                "inner_validation": tuple(
                    row.source_array_sha256 for row in validation_rows
                ),
            }
        ),
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "training_config_receipt_sha256": config.receipt_sha256,
        "outer_prediction_authorized": authorized,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveProfitabilityTrainedRunV1(
        setting_id=setting_id,
        fold_index=fold.fold_index,
        seed=seed,
        best_epoch=best_epoch,
        completed_epochs=completed_epochs,
        validation_rank_ic=best_rank_ic,
        normalization=normalization,
        model_state_sha256=state_dict_sha256(model_state),
        model_state=best_state,
        fit_inventory_sha256=fold.fit_inventory_sha256,
        validation_inventory_sha256=fold.inner_validation_inventory_sha256,
        training_source_receipt_sha256=semantic_sha256(
            {
                "fit": tuple(row.source_array_sha256 for row in fit_rows),
                "inner_validation": tuple(
                    row.source_array_sha256 for row in validation_rows
                ),
            }
        ),
        tournament_plan_receipt_sha256=tournament_plan.receipt_sha256,
        training_config_receipt_sha256=config.receipt_sha256,
        run_receipt_sha256=semantic_sha256(receipt_body),
        outer_prediction_authorized=authorized,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1",
    "MASSIVE_PROFITABILITY_DEVELOPMENT_SEEDS_V1",
    "MASSIVE_PROFITABILITY_MODEL_CHECKPOINT_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TOURNAMENT_V1_SCHEMA",
    "MassiveProfitabilityDateTensorV1",
    "MassiveProfitabilityModelCheckpointV1",
    "MassiveProfitabilityNormalizationV1",
    "MassiveProfitabilityTournamentDatasetV1",
    "MassiveProfitabilityTournamentPlanV1",
    "MassiveProfitabilityTournamentV1Error",
    "MassiveProfitabilityTrainedRunV1",
    "MassiveProfitabilityTrainingConfigV1",
    "MassiveProfitabilityTrainingFoldV1",
    "build_massive_profitability_date_tensor_v1",
    "fit_massive_profitability_normalization_v1",
    "massive_profitability_tape_permutation_v1",
    "parse_massive_profitability_model_checkpoint_v1",
    "publish_massive_profitability_model_checkpoint_v1",
    "train_massive_profitability_fold_v1",
]
