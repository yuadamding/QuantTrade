"""Create-only target-free forecast archives for adaptive RL fitting.

Each archive covers exactly one package-derived RL-fit inference block.  Its
supervised checkpoint and mature training targets must both precede the first
forecast date.  Generic reopening exposes metadata only; promotion rebuilds
the compatibility proof and reruns every model output exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256,
    MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES,
    MassiveAdaptiveForecastRowV2,
    replay_massive_adaptive_forecast_rows_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    MassiveAdaptiveRLFitInferencePlanV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
    MassiveAdaptiveAlphaModelSpecV1,
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


MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fit-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_DATASET = (
    "massive-adaptive-rl-fit-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA,
        "encoding": "torch-tensor-archive-loaded-weights-only",
        "publication": "create-only-source-transaction",
        "runtime": "withheld-until-causality-and-inference-replay",
    }
)
MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "role": "rl_fit",
        "schedule": "package-derived-expanding-fit-prefix-block",
        "checkpoint": "replayed-training-checkpoint-strictly-before-block",
        "target_maturity": "strictly-before-block",
        "target_archive_during_inference": False,
        "inference": "cpu-float32-eval-no-grad",
        "generic_reload": "nonauthorizing",
        "rl_authority": "downstream-composite-only",
        "profitability_reporting": False,
        "outer": False,
        "lockbox": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLFitForecastArchiveV1Error(ValueError):
    """An RL-fit forecast block differs from its causal replay roots."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            "adaptive RL-fit forecast artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitForecastEligibilityV1:
    outer_fold_index: int
    source_fold_index: int
    block_index: int
    block_sessions: int
    inference_role: str
    forecast_session_dates: tuple[str, ...]
    supervised_training_cutoff_session_date: str
    target_maturity_cutoff_session_date: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_tensor_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    inference_tensor_receipt_sha256: str
    inference_plan_receipt_sha256: str
    split_plan_receipt_sha256: str
    model_spec_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_eligibility_replayed: bool
    development_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.inference_role != "rl_fit"
            or self.source_fold_index < 0
            or self.source_fold_index > self.outer_fold_index
            or self.block_sessions not in {21, 63}
            or not self.forecast_session_dates
            or len(self.forecast_session_dates) != self.block_sessions
            or self.forecast_session_dates
            != tuple(sorted(set(self.forecast_session_dates)))
            or max(
                self.supervised_training_cutoff_session_date,
                self.target_maturity_cutoff_session_date,
            )
            >= self.forecast_session_dates[0]
            or not self.runtime_eligibility_replayed
            or not isinstance(self.source_data_qualified, bool)
            or self.development_forecast_authorized != self.source_data_qualified
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFitForecastArchiveV1Error(
                "adaptive RL-fit forecast eligibility differs"
            )
        for value in (
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_tensor_receipt_sha256,
            self.training_window_plan_receipt_sha256,
            self.inference_tensor_receipt_sha256,
            self.inference_plan_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL-fit forecast eligibility", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFitForecastArchiveV1:
    origin_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    row_inventory_sha256: str
    eligibility_receipt_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_tensor_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    supervised_training_cutoff_session_date: str
    target_maturity_cutoff_session_date: str
    inference_tensor_receipt_sha256: str
    inference_full_decision_root_inventory_sha256: str
    inference_origin_decision_root_inventory_sha256: str
    inference_plan_receipt_sha256: str
    inference_role: str
    outer_fold_index: int
    source_fold_index: int
    block_index: int
    block_sessions: int
    split_plan_receipt_sha256: str
    model_spec_receipt_sha256: str
    model_source_sha256: str
    normalization_receipt_sha256: str
    calibration_receipt_sha256: str
    committed_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rows: tuple[MassiveAdaptiveForecastRowV2, ...] | None
    runtime_forecasts_replayed: bool
    development_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA

    @property
    def fold_index(self) -> int:
        return self.outer_fold_index

    def semantic_unsigned(self) -> dict[str, object]:
        excluded = {
            "loaded_source",
            "runtime_rows",
            "runtime_forecasts_replayed",
            "development_forecast_authorized",
            "semantic_receipt_sha256",
        }
        return {
            key: value for key, value in asdict(self).items() if key not in excluded
        }

    def validate(self) -> None:
        runtime_present = self.runtime_rows is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA
            or self.inference_role != "rl_fit"
            or not self.origin_session_dates
            or len(self.origin_session_dates) != self.block_sessions
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.row_receipts) != len(self.origin_session_dates)
            or self.row_inventory_sha256 != semantic_sha256(self.row_receipts)
            or max(
                self.supervised_training_cutoff_session_date,
                self.target_maturity_cutoff_session_date,
            )
            >= self.origin_session_dates[0]
            or self.source_fold_index < 0
            or self.source_fold_index > self.outer_fold_index
            or self.block_sessions not in {21, 63}
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SHA256
            or self.normalization_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
            or self.calibration_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
            or self.model_source_sha256 != MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.runtime_forecasts_replayed != runtime_present
            or self.development_forecast_authorized
            != (runtime_present and self.committed_source_data_qualified)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveRLFitForecastArchiveV1Error(
                "adaptive RL-fit forecast archive identity differs"
            )
        for value in (
            *self.row_receipts,
            self.row_inventory_sha256,
            self.eligibility_receipt_sha256,
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.training_tensor_receipt_sha256,
            self.training_window_plan_receipt_sha256,
            self.inference_tensor_receipt_sha256,
            self.inference_full_decision_root_inventory_sha256,
            self.inference_origin_decision_root_inventory_sha256,
            self.inference_plan_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.model_source_sha256,
            self.normalization_receipt_sha256,
            self.calibration_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL-fit forecast archive", value)
        if self.runtime_rows is not None:
            for row in self.runtime_rows:
                row.validate()
            if (
                tuple(row.decision_session_date for row in self.runtime_rows)
                != self.origin_session_dates
                or tuple(row.security_ids for row in self.runtime_rows)
                != (self.security_ids,) * len(self.runtime_rows)
                or tuple(row.receipt_sha256 for row in self.runtime_rows)
                != self.row_receipts
            ):
                raise MassiveAdaptiveRLFitForecastArchiveV1Error(
                    "adaptive RL-fit runtime forecast inventory differs"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.eligibility_receipt_sha256
        ):
            raise MassiveAdaptiveRLFitForecastArchiveV1Error(
                "adaptive RL-fit forecast source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _source_fold_index(plan: MassiveAdaptiveRLFitInferencePlanV1) -> int:
    blocks_per_macro = (
        MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1 // plan.block_sessions
    )
    return plan.block_index // blocks_per_macro


def build_massive_adaptive_rl_fit_forecast_eligibility_v1(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveRLFitInferencePlanV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveRLFitForecastEligibilityV1:
    checkpoint.validate()
    training_window_plan.validate()
    inference_tensor.validate()
    inference_plan.validate()
    split_plan.validate()
    model_spec.validate()
    if (
        checkpoint.runtime_state is None
        or not checkpoint.runtime_checkpoint_replayed
        or inference_tensor.runtime_tensor is None
        or not inference_tensor.runtime_source_replayed
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            "adaptive RL-fit checkpoint or inference tensor has not been replayed"
        )
    ordered_roots = tuple(
        sorted(inference_decision_roots, key=lambda row: row.decision_session_date)
    )
    for root in ordered_roots:
        root.validate()
    source_fold_index = _source_fold_index(inference_plan)
    training_dates = tuple(row.origin_session_date for row in training_window_plan.rows)
    maturity_indices = tuple(
        row.candidate_origin_index + MASSIVE_ADAPTIVE_MAXIMUM_TARGET_SESSIONS_V1
        for row in training_window_plan.rows
    )
    maturity_cutoff = split_plan.candidate_session_dates[max(maturity_indices)]
    training_origins = set(training_dates)
    forecast_dates = inference_plan.origin_session_dates
    full_receipts = tuple(root.semantic_receipt_sha256 for root in ordered_roots)
    origin_receipts = tuple(
        row.decision_root_receipt_sha256 for row in inference_plan.rows
    )
    if (
        training_window_plan.split_role != "training"
        or inference_plan.inference_role != "rl_fit"
        or training_window_plan.fold_index != source_fold_index
        or source_fold_index > inference_plan.outer_fold_index
        or checkpoint.window_plan_receipt_sha256
        != training_window_plan.semantic_receipt_sha256
        or checkpoint.decision_tensor_receipt_sha256
        != training_window_plan.decision_tensor_receipt_sha256
        or checkpoint.split_plan_receipt_sha256 != split_plan.semantic_receipt_sha256
        or training_window_plan.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
        or inference_plan.split_plan_receipt_sha256
        != split_plan.semantic_receipt_sha256
        or checkpoint.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or inference_plan.model_spec_receipt_sha256 != model_spec.receipt_sha256
        or inference_plan.decision_tensor_receipt_sha256
        != inference_tensor.semantic_receipt_sha256
        or checkpoint.decision_tensor_receipt_sha256
        == inference_tensor.semantic_receipt_sha256
        or tuple(root.decision_session_date for root in ordered_roots)
        != inference_tensor.decision_session_dates
        or tuple(root.feature_semantic_receipt_sha256 for root in ordered_roots)
        != inference_tensor.feature_semantic_receipts
        or tuple(root.action_origin_receipt_sha256 for root in ordered_roots)
        != inference_tensor.action_origin_receipts
        or inference_plan.full_decision_root_inventory_sha256
        != semantic_sha256(full_receipts)
        or inference_plan.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_receipts)
        or training_origins.intersection(forecast_dates)
        or max(training_dates[-1], maturity_cutoff) >= forecast_dates[0]
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            "adaptive RL-fit training and inference provenance is incompatible"
        )
    qualified = bool(
        checkpoint.committed_development_training_authorized
        and inference_plan.source_data_qualified
        and inference_tensor.committed_source_data_qualified
        and split_plan.candidate_source_data_qualified
        and all(root.source_data_qualified for root in ordered_roots)
    )
    body = {
        "outer_fold_index": inference_plan.outer_fold_index,
        "source_fold_index": source_fold_index,
        "block_index": inference_plan.block_index,
        "block_sessions": inference_plan.block_sessions,
        "inference_role": "rl_fit",
        "forecast_session_dates": forecast_dates,
        "supervised_training_cutoff_session_date": training_dates[-1],
        "target_maturity_cutoff_session_date": maturity_cutoff,
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": checkpoint.loaded_source.receipt_sha256,
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "training_tensor_receipt_sha256": checkpoint.decision_tensor_receipt_sha256,
        "training_window_plan_receipt_sha256": (
            training_window_plan.semantic_receipt_sha256
        ),
        "inference_tensor_receipt_sha256": inference_tensor.semantic_receipt_sha256,
        "inference_plan_receipt_sha256": inference_plan.semantic_receipt_sha256,
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "source_data_qualified": qualified,
        "runtime_eligibility_replayed": True,
        "development_forecast_authorized": qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    result = MassiveAdaptiveRLFitForecastEligibilityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _metadata(
    *,
    rows: tuple[MassiveAdaptiveForecastRowV2, ...],
    eligibility: MassiveAdaptiveRLFitForecastEligibilityV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_plan: MassiveAdaptiveRLFitInferencePlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> dict[str, object]:
    row_receipts = tuple(row.receipt_sha256 for row in rows)
    return {
        "schema": MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA,
        "origin_session_dates": eligibility.forecast_session_dates,
        "security_ids": inference_tensor.security_ids,
        "row_receipts": row_receipts,
        "row_inventory_sha256": semantic_sha256(row_receipts),
        "eligibility_receipt_sha256": eligibility.semantic_receipt_sha256,
        "checkpoint_receipt_sha256": eligibility.checkpoint_receipt_sha256,
        "checkpoint_source_receipt_sha256": (
            eligibility.checkpoint_source_receipt_sha256
        ),
        "model_state_receipt_sha256": eligibility.model_state_receipt_sha256,
        "training_tensor_receipt_sha256": eligibility.training_tensor_receipt_sha256,
        "training_window_plan_receipt_sha256": (
            eligibility.training_window_plan_receipt_sha256
        ),
        "supervised_training_cutoff_session_date": (
            eligibility.supervised_training_cutoff_session_date
        ),
        "target_maturity_cutoff_session_date": (
            eligibility.target_maturity_cutoff_session_date
        ),
        "inference_tensor_receipt_sha256": inference_tensor.semantic_receipt_sha256,
        "inference_full_decision_root_inventory_sha256": (
            inference_plan.full_decision_root_inventory_sha256
        ),
        "inference_origin_decision_root_inventory_sha256": (
            inference_plan.origin_decision_root_inventory_sha256
        ),
        "inference_plan_receipt_sha256": inference_plan.semantic_receipt_sha256,
        "inference_role": "rl_fit",
        "outer_fold_index": eligibility.outer_fold_index,
        "source_fold_index": eligibility.source_fold_index,
        "block_index": eligibility.block_index,
        "block_sessions": eligibility.block_sessions,
        "split_plan_receipt_sha256": eligibility.split_plan_receipt_sha256,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "model_source_sha256": MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
        "normalization_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
        ),
        "calibration_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
        ),
        "committed_source_data_qualified": eligibility.source_data_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _eligibility_and_rows(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveRLFitInferencePlanV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> tuple[
    MassiveAdaptiveRLFitForecastEligibilityV1,
    tuple[MassiveAdaptiveForecastRowV2, ...],
]:
    eligibility = build_massive_adaptive_rl_fit_forecast_eligibility_v1(
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        split_plan=split_plan,
        model_spec=model_spec,
    )
    rows = replay_massive_adaptive_forecast_rows_v2(
        checkpoint=checkpoint,
        decision_tensor=inference_tensor,
        plan_rows=inference_plan.rows,
        model_spec=model_spec,
    )
    return eligibility, rows


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[dict[str, object], tuple[MassiveAdaptiveForecastRowV2, ...]]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
        or not isinstance(payload.get("rows"), Sequence)
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            "adaptive RL-fit forecast payload is malformed"
        )
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    rows: list[MassiveAdaptiveForecastRowV2] = []
    for value in cast(Sequence[object], payload["rows"]):
        if not isinstance(value, Mapping):
            raise MassiveAdaptiveRLFitForecastArchiveV1Error(
                "adaptive RL-fit forecast row payload is malformed"
            )
        row_payload = dict(cast(Mapping[str, object], value))
        row_payload["security_ids"] = tuple(
            cast(Sequence[str], row_payload["security_ids"])
        )
        row_payload["array_receipts"] = tuple(
            cast(Sequence[str], row_payload["array_receipts"])
        )
        row = MassiveAdaptiveForecastRowV2(**row_payload)  # type: ignore[arg-type]
        row.validate()
        rows.append(row)
    for name in ("origin_session_dates", "security_ids", "row_receipts"):
        metadata[name] = tuple(cast(Sequence[object], metadata[name]))
    return metadata, tuple(rows)


def parse_massive_adaptive_rl_fit_forecast_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFitForecastArchiveV1:
    metadata, _rows = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLFitForecastArchiveV1(
        **metadata,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_rows=None,
        runtime_forecasts_replayed=False,
        development_forecast_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fit_forecast_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveRLFitInferencePlanV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveRLFitForecastArchiveV1:
    """Publish and replay one target-free RL-fit forecast block."""

    identifier = _artifact_id(artifact_id)
    eligibility, rows = _eligibility_and_rows(
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        split_plan=split_plan,
        model_spec=model_spec,
    )
    metadata = _metadata(
        rows=rows,
        eligibility=eligibility,
        inference_tensor=inference_tensor,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    receipt = semantic_sha256(metadata)
    stream = BytesIO()
    archive_payload: dict[str, object] = {
        "metadata": {**metadata, "semantic_receipt_sha256": receipt},
        "rows": tuple(row.payload() for row in rows),
    }
    torch.save(archive_payload, stream)
    stream.seek(0)
    relative = f"massive-adaptive/rl-fit-forecast-archive-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=eligibility.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FIT-FORECAST-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_rl_fit_forecast_archive_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_rl_fit_forecast_archive_v1(
        root=root,
        archive=generic,
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        split_plan=split_plan,
        model_spec=model_spec,
    )


def authorize_massive_adaptive_rl_fit_forecast_archive_v1(
    *,
    root: str | Path,
    archive: MassiveAdaptiveRLFitForecastArchiveV1,
    checkpoint: MassiveAdaptiveCheckpointV1,
    training_window_plan: MassiveAdaptiveWindowPlanV1,
    inference_tensor: MassiveAdaptiveDecisionTensorV1,
    inference_decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    inference_plan: MassiveAdaptiveRLFitInferencePlanV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveRLFitForecastArchiveV1:
    """Rebuild causality and every committed RL-fit forecast array."""

    parsed = parse_massive_adaptive_rl_fit_forecast_archive_v1(
        root=root, loaded_source=archive.loaded_source
    )
    committed_metadata, committed_rows = _load_payload(
        root=root, loaded_source=archive.loaded_source
    )
    eligibility, rebuilt_rows = _eligibility_and_rows(
        checkpoint=checkpoint,
        training_window_plan=training_window_plan,
        inference_tensor=inference_tensor,
        inference_decision_roots=inference_decision_roots,
        inference_plan=inference_plan,
        split_plan=split_plan,
        model_spec=model_spec,
    )
    expected_metadata = _metadata(
        rows=rebuilt_rows,
        eligibility=eligibility,
        inference_tensor=inference_tensor,
        inference_plan=inference_plan,
        model_spec=model_spec,
    )
    if (
        parsed.semantic_receipt_sha256 != archive.semantic_receipt_sha256
        or committed_metadata
        != {
            **expected_metadata,
            "semantic_receipt_sha256": semantic_sha256(expected_metadata),
        }
        or len(committed_rows) != len(rebuilt_rows)
        or any(
            committed.payload().keys() != rebuilt.payload().keys()
            or any(
                not torch.equal(
                    cast(torch.Tensor, getattr(committed, name)),
                    cast(torch.Tensor, getattr(rebuilt, name)),
                )
                for name in (*MASSIVE_ADAPTIVE_FORECAST_V2_FLOAT_ARRAY_NAMES, "valid")
            )
            or committed.unsigned() != rebuilt.unsigned()
            or committed.receipt_sha256 != rebuilt.receipt_sha256
            for committed, rebuilt in zip(committed_rows, rebuilt_rows, strict=True)
        )
    ):
        raise MassiveAdaptiveRLFitForecastArchiveV1Error(
            "adaptive RL-fit forecast archive does not replay from causal roots"
        )
    result = replace(
        parsed,
        runtime_rows=rebuilt_rows,
        runtime_forecasts_replayed=True,
        development_forecast_authorized=(parsed.committed_source_data_qualified),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA",
    "MassiveAdaptiveRLFitForecastArchiveV1",
    "MassiveAdaptiveRLFitForecastArchiveV1Error",
    "MassiveAdaptiveRLFitForecastEligibilityV1",
    "authorize_massive_adaptive_rl_fit_forecast_archive_v1",
    "build_massive_adaptive_rl_fit_forecast_eligibility_v1",
    "materialize_massive_adaptive_rl_fit_forecast_archive_v1",
    "parse_massive_adaptive_rl_fit_forecast_archive_v1",
]
