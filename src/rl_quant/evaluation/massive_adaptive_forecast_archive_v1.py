"""Create-only, checkpoint-replayed adaptive forecast archive.

The archive commits model outputs for the exact origin rows selected by a
package-owned window plan.  A generic reload exposes metadata only.  Runtime
forecast tensors are restored only after the package reloads the committed
checkpoint state, reruns inference over the promoted decision tensor, and
requires exact per-array equality with the immutable payload.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from math import sqrt
from pathlib import Path
from typing import cast

import torch

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1,
    MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
    MassiveAdaptiveAlphaModelSpecV1,
    MassiveAdaptiveAlphaTermStructureModelV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS,
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
    MassiveAdaptiveWindowRowV1,
)


MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_DATASET = (
    "massive-adaptive-forecast-archive-v1"
)
MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA,
        "encoding": "torch-tensor-archive-loaded-weights-only",
        "publication": "create-only-source-transaction",
        "runtime": "withheld-until-checkpoint-and-tensor-replay",
    }
)
MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "checkpoint": "exact-runtime-replayed-adaptive-checkpoint-v1",
        "tensor": "exact-runtime-replayed-adaptive-decision-tensor-v1",
        "roots": "full-chronology-and-origin-subset-bound",
        "windows": "exact-package-owned-window-plan-v1",
        "inference": "cpu-float32-eval-no-grad",
        "normalization": "feature-v3-values-no-additional-transform-v1",
        "calibration": "uncalibrated-model-distribution-output-v1",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)
MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256 = semantic_sha256(
    {
        "schema": "rl-quant.massive-adaptive-forecast-normalization-v1",
        "mode": "feature-v3-values-no-additional-transform",
        "fit_population": "none",
    }
)
MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256 = semantic_sha256(
    {
        "schema": "rl-quant.massive-adaptive-forecast-calibration-v1",
        "mode": "uncalibrated-model-distribution-output",
        "profitability_authorizing": False,
    }
)


class MassiveAdaptiveForecastArchiveV1Error(ValueError):
    """Adaptive forecast content or root replay differs."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive forecast artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _tensor_receipt(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


_FLOAT_ARRAY_NAMES = (
    "residual_mean",
    "residual_downside_quantile",
    "residual_median",
    "residual_upside_quantile",
    "residual_scale",
    "residual_positive_probability",
    "raw_mean",
    "raw_downside_quantile",
    "raw_median",
    "raw_upside_quantile",
    "raw_scale",
    "raw_positive_probability",
    "factor_return_mean",
    "executable_score",
    "bucket_router_weights",
    "router_weights",
    "stock_context",
    "market_context",
)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastRowV1:
    decision_session_date: str
    security_ids: tuple[str, ...]
    residual_mean: torch.Tensor
    residual_downside_quantile: torch.Tensor
    residual_median: torch.Tensor
    residual_upside_quantile: torch.Tensor
    residual_scale: torch.Tensor
    residual_positive_probability: torch.Tensor
    raw_mean: torch.Tensor
    raw_downside_quantile: torch.Tensor
    raw_median: torch.Tensor
    raw_upside_quantile: torch.Tensor
    raw_scale: torch.Tensor
    raw_positive_probability: torch.Tensor
    factor_return_mean: torch.Tensor
    executable_score: torch.Tensor
    bucket_router_weights: torch.Tensor
    router_weights: torch.Tensor
    stock_context: torch.Tensor
    market_context: torch.Tensor
    valid: torch.Tensor
    decision_root_receipt_sha256: str
    window_row_receipt_sha256: str
    array_receipts: tuple[str, ...]
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            "decision_session_date": self.decision_session_date,
            "security_ids": self.security_ids,
            "decision_root_receipt_sha256": self.decision_root_receipt_sha256,
            "window_row_receipt_sha256": self.window_row_receipt_sha256,
            "array_receipts": self.array_receipts,
        }

    def validate(self) -> None:
        assets = len(self.security_ids)
        buckets = len(MASSIVE_ADAPTIVE_ALPHA_V1_BUCKETS)
        experts = len(MASSIVE_ADAPTIVE_ALPHA_EXPERT_IDS_V1)
        if (
            not self.decision_session_date
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or self.valid.shape != (assets,)
            or self.valid.dtype != torch.bool
            or self.factor_return_mean.shape != (buckets,)
            or self.executable_score.shape != (assets,)
            or self.bucket_router_weights.shape != (assets, buckets)
            or self.router_weights.shape != (assets, buckets, experts)
            or self.stock_context.ndim != 2
            or self.stock_context.shape[0] != assets
            or self.market_context.ndim != 2
            or len(self.array_receipts) != len(_FLOAT_ARRAY_NAMES) + 1
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast row shape or identity differs"
            )
        for name in (
            "residual_mean",
            "residual_downside_quantile",
            "residual_median",
            "residual_upside_quantile",
            "residual_scale",
            "residual_positive_probability",
            "raw_mean",
            "raw_downside_quantile",
            "raw_median",
            "raw_upside_quantile",
            "raw_scale",
            "raw_positive_probability",
        ):
            if getattr(self, name).shape != (assets, buckets):
                raise MassiveAdaptiveForecastArchiveV1Error(
                    "adaptive forecast distribution shape differs"
                )
        tensors = tuple(getattr(self, name) for name in _FLOAT_ARRAY_NAMES)
        if any(
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            for value in tensors
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast array is not finite CPU float32"
            )
        if (
            bool((self.residual_scale <= 0.0).any())
            or bool((self.raw_scale <= 0.0).any())
            or bool((self.residual_positive_probability < 0.0).any())
            or bool((self.residual_positive_probability > 1.0).any())
            or bool((self.raw_positive_probability < 0.0).any())
            or bool((self.raw_positive_probability > 1.0).any())
            or bool((~self.valid & (self.executable_score != 0.0)).any())
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast numerical domain differs"
            )
        expected_arrays = tuple(_tensor_receipt(value) for value in tensors) + (
            _tensor_receipt(self.valid),
        )
        if (
            self.array_receipts != expected_arrays
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast row receipt differs"
            )
        for value in (
            self.decision_root_receipt_sha256,
            self.window_row_receipt_sha256,
            *self.array_receipts,
            self.receipt_sha256,
        ):
            _digest("adaptive forecast row", value)

    def payload(self) -> dict[str, object]:
        return {
            **self.unsigned(),
            **{name: getattr(self, name) for name in _FLOAT_ARRAY_NAMES},
            "valid": self.valid,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveForecastArchiveV1:
    origin_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    row_inventory_sha256: str
    checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    window_plan_receipt_sha256: str
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
    runtime_rows: tuple[MassiveAdaptiveForecastRowV1, ...] | None
    runtime_forecasts_replayed: bool
    development_forecast_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "origin_session_dates": self.origin_session_dates,
            "security_ids": self.security_ids,
            "row_receipts": self.row_receipts,
            "row_inventory_sha256": self.row_inventory_sha256,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "checkpoint_source_receipt_sha256": (
                self.checkpoint_source_receipt_sha256
            ),
            "model_state_receipt_sha256": self.model_state_receipt_sha256,
            "decision_tensor_receipt_sha256": (
                self.decision_tensor_receipt_sha256
            ),
            "full_decision_root_inventory_sha256": (
                self.full_decision_root_inventory_sha256
            ),
            "origin_decision_root_inventory_sha256": (
                self.origin_decision_root_inventory_sha256
            ),
            "window_plan_receipt_sha256": self.window_plan_receipt_sha256,
            "model_spec_receipt_sha256": self.model_spec_receipt_sha256,
            "model_source_sha256": self.model_source_sha256,
            "normalization_receipt_sha256": (
                self.normalization_receipt_sha256
            ),
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "committed_source_data_qualified": (
                self.committed_source_data_qualified
            ),
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": (
                self.reinforcement_learning_authorized
            ),
        }

    def validate(self) -> None:
        runtime_present = self.runtime_rows is not None
        if (
            self.schema != MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA
            or not self.origin_session_dates
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.row_receipts) != len(self.origin_session_dates)
            or self.row_inventory_sha256 != semantic_sha256(self.row_receipts)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SHA256
            or self.normalization_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
            or self.calibration_receipt_sha256
            != MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.runtime_forecasts_replayed != runtime_present
            or self.development_forecast_authorized
            != (runtime_present and self.committed_source_data_qualified)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast archive identity or authorization differs"
            )
        for value in (
            *self.row_receipts,
            self.row_inventory_sha256,
            self.checkpoint_receipt_sha256,
            self.checkpoint_source_receipt_sha256,
            self.model_state_receipt_sha256,
            self.decision_tensor_receipt_sha256,
            self.full_decision_root_inventory_sha256,
            self.origin_decision_root_inventory_sha256,
            self.window_plan_receipt_sha256,
            self.model_spec_receipt_sha256,
            self.model_source_sha256,
            self.normalization_receipt_sha256,
            self.calibration_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive forecast archive", value)
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
                raise MassiveAdaptiveForecastArchiveV1Error(
                    "adaptive runtime forecast inventory differs"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.checkpoint_receipt_sha256
        ):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _positive_probability(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(mean / (scale * sqrt(2.0))))


def _forecast_row(
    *,
    model: MassiveAdaptiveAlphaTermStructureModelV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    window_row: MassiveAdaptiveWindowRowV1,
) -> MassiveAdaptiveForecastRowV1:
    window_row.validate()
    runtime = decision_tensor.runtime_tensor
    if runtime is None:
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive forecast tensor runtime is absent"
        )
    indices = torch.tensor(window_row.context_tensor_indices, dtype=torch.long)
    with torch.no_grad():
        output = model.forward_sequence(
            bars_values=runtime.bars_values.index_select(0, indices).unsqueeze(0),
            bars_valid=runtime.bars_valid.index_select(0, indices).unsqueeze(0),
            tape_values=runtime.tape_values.index_select(0, indices).unsqueeze(0),
            tape_valid=runtime.tape_valid.index_select(0, indices).unsqueeze(0),
            source_staleness=runtime.source_staleness.index_select(
                0, indices
            ).unsqueeze(0),
            context_membership=runtime.context_membership.index_select(
                0, indices
            ).unsqueeze(0),
            action_mask=runtime.action_mask.index_select(0, indices).unsqueeze(0),
        )
    position = window_row.origin_output_position
    residual = output.residual_distribution
    raw = output.raw_distribution
    arrays = {
        "residual_mean": residual.mean[0, position],
        "residual_downside_quantile": residual.downside_quantile[0, position],
        "residual_median": residual.median[0, position],
        "residual_upside_quantile": residual.upside_quantile[0, position],
        "residual_scale": residual.scale[0, position],
        "residual_positive_probability": _positive_probability(
            residual.mean[0, position], residual.scale[0, position]
        ),
        "raw_mean": raw.mean[0, position],
        "raw_downside_quantile": raw.downside_quantile[0, position],
        "raw_median": raw.median[0, position],
        "raw_upside_quantile": raw.upside_quantile[0, position],
        "raw_scale": raw.scale[0, position],
        "raw_positive_probability": _positive_probability(
            raw.mean[0, position], raw.scale[0, position]
        ),
        "factor_return_mean": output.factor_return_mean[0, position],
        "executable_score": output.executable_score[0, position],
        "bucket_router_weights": output.bucket_router_weights[0, position],
        "router_weights": output.router_weights[0, position],
        "stock_context": output.stock_context[0, position],
        "market_context": output.market_context[0, position],
    }
    arrays = {
        name: value.detach().cpu().to(torch.float32).contiguous()
        for name, value in arrays.items()
    }
    valid = output.valid[0, position].detach().cpu().to(torch.bool).contiguous()
    array_receipts = tuple(
        _tensor_receipt(arrays[name]) for name in _FLOAT_ARRAY_NAMES
    ) + (_tensor_receipt(valid),)
    body = {
        "decision_session_date": window_row.origin_session_date,
        "security_ids": decision_tensor.security_ids,
        "decision_root_receipt_sha256": (
            window_row.decision_root_receipt_sha256
        ),
        "window_row_receipt_sha256": window_row.receipt_sha256,
        "array_receipts": array_receipts,
    }
    result = MassiveAdaptiveForecastRowV1(
        **body,  # type: ignore[arg-type]
        **arrays,  # type: ignore[arg-type]
        valid=valid,
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _replay_rows(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> tuple[MassiveAdaptiveForecastRowV1, ...]:
    checkpoint.validate()
    decision_tensor.validate()
    window_plan.validate()
    model_spec.validate()
    if (
        checkpoint.runtime_state is None
        or not checkpoint.runtime_checkpoint_replayed
        or decision_tensor.runtime_tensor is None
        or not decision_tensor.runtime_source_replayed
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive checkpoint or tensor has not been runtime replayed"
        )
    ordered_roots = tuple(
        sorted(decision_roots, key=lambda row: row.decision_session_date)
    )
    for row in ordered_roots:
        row.validate()
    full_receipts = tuple(row.semantic_receipt_sha256 for row in ordered_roots)
    origin_receipts = tuple(
        row.decision_root_receipt_sha256 for row in window_plan.rows
    )
    if (
        tuple(row.decision_session_date for row in ordered_roots)
        != decision_tensor.decision_session_dates
        or tuple(row.feature_semantic_receipt_sha256 for row in ordered_roots)
        != decision_tensor.feature_semantic_receipts
        or tuple(row.action_origin_receipt_sha256 for row in ordered_roots)
        != decision_tensor.action_origin_receipts
        or checkpoint.decision_tensor_receipt_sha256
        != decision_tensor.semantic_receipt_sha256
        or checkpoint.full_decision_root_inventory_sha256
        != semantic_sha256(full_receipts)
        or checkpoint.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_receipts)
        or checkpoint.window_plan_receipt_sha256
        != window_plan.semantic_receipt_sha256
        or window_plan.decision_tensor_receipt_sha256
        != decision_tensor.semantic_receipt_sha256
        or window_plan.full_decision_root_inventory_sha256
        != semantic_sha256(full_receipts)
        or window_plan.origin_decision_root_inventory_sha256
        != semantic_sha256(origin_receipts)
        or checkpoint.model_spec_receipt_sha256 != model_spec.receipt_sha256
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive forecast checkpoint, tensor, root, window, or model differs"
        )
    model = MassiveAdaptiveAlphaTermStructureModelV1(model_spec)
    model.load_state_dict(checkpoint.runtime_state.model_state, strict=True)
    model.eval()
    return tuple(
        _forecast_row(
            model=model,
            decision_tensor=decision_tensor,
            window_row=row,
        )
        for row in window_plan.rows
    )


def _metadata(
    *,
    rows: tuple[MassiveAdaptiveForecastRowV1, ...],
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> dict[str, object]:
    full_roots = tuple(
        sorted(decision_roots, key=lambda row: row.decision_session_date)
    )
    row_receipts = tuple(row.receipt_sha256 for row in rows)
    return {
        "schema": MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA,
        "origin_session_dates": tuple(
            row.decision_session_date for row in rows
        ),
        "security_ids": decision_tensor.security_ids,
        "row_receipts": row_receipts,
        "row_inventory_sha256": semantic_sha256(row_receipts),
        "checkpoint_receipt_sha256": checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": (
            checkpoint.loaded_source.receipt.receipt_sha256
        ),
        "model_state_receipt_sha256": checkpoint.model_state_receipt_sha256,
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in full_roots)
        ),
        "origin_decision_root_inventory_sha256": semantic_sha256(
            tuple(row.decision_root_receipt_sha256 for row in window_plan.rows)
        ),
        "window_plan_receipt_sha256": window_plan.semantic_receipt_sha256,
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "model_source_sha256": MASSIVE_ADAPTIVE_ALPHA_MODEL_V1_SOURCE_SHA256,
        "normalization_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256
        ),
        "calibration_receipt_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256
        ),
        "committed_source_data_qualified": (
            checkpoint.committed_development_training_authorized
            and decision_tensor.committed_source_data_qualified
            and all(row.source_data_qualified for row in full_roots)
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SHA256
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[dict[str, object], tuple[MassiveAdaptiveForecastRowV1, ...]]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
        or not isinstance(payload.get("rows"), Sequence)
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive forecast payload is malformed"
        )
    metadata = dict(cast(Mapping[str, object], payload["metadata"]))
    rows = []
    for value in cast(Sequence[object], payload["rows"]):
        if not isinstance(value, Mapping):
            raise MassiveAdaptiveForecastArchiveV1Error(
                "adaptive forecast row payload is malformed"
            )
        row_payload = dict(cast(Mapping[str, object], value))
        row_payload["security_ids"] = tuple(
            cast(Sequence[str], row_payload["security_ids"])
        )
        row_payload["array_receipts"] = tuple(
            cast(Sequence[str], row_payload["array_receipts"])
        )
        row = MassiveAdaptiveForecastRowV1(**row_payload)  # type: ignore[arg-type]
        row.validate()
        rows.append(row)
    for name in ("origin_session_dates", "security_ids", "row_receipts"):
        metadata[name] = tuple(cast(Sequence[object], metadata[name]))
    return metadata, tuple(rows)


def parse_massive_adaptive_forecast_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveForecastArchiveV1:
    metadata, _rows = _load_payload(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveForecastArchiveV1(
        **metadata,  # type: ignore[arg-type]
        loaded_source=loaded_source,
        runtime_rows=None,
        runtime_forecasts_replayed=False,
        development_forecast_authorized=False,
    )
    result.validate()
    return result


def materialize_massive_adaptive_forecast_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    committed_at_ms: int,
) -> MassiveAdaptiveForecastArchiveV1:
    """Publish exact inference tensors and immediately replay them."""

    identifier = _artifact_id(artifact_id)
    rows = _replay_rows(
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    metadata = _metadata(
        rows=rows,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
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
    relative = f"massive-adaptive/forecast-archive-v1/{identifier}.pt"
    publish_massive_source_object(
        stream=stream,
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=checkpoint.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-FORECAST-ARCHIVE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_forecast_archive_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_forecast_archive_v1(
        root=root,
        archive=generic,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )


def authorize_massive_adaptive_forecast_archive_v1(
    *,
    root: str | Path,
    archive: MassiveAdaptiveForecastArchiveV1,
    checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    window_plan: MassiveAdaptiveWindowPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveForecastArchiveV1:
    """Reopen the archive and reproduce every committed forecast array."""

    parsed = parse_massive_adaptive_forecast_archive_v1(
        root=root, loaded_source=archive.loaded_source
    )
    committed_metadata, committed_rows = _load_payload(
        root=root, loaded_source=archive.loaded_source
    )
    rebuilt_rows = _replay_rows(
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    expected_metadata = _metadata(
        rows=rebuilt_rows,
        checkpoint=checkpoint,
        decision_tensor=decision_tensor,
        decision_roots=decision_roots,
        window_plan=window_plan,
        model_spec=model_spec,
    )
    if (
        parsed.semantic_receipt_sha256 != archive.semantic_receipt_sha256
        or committed_metadata
        != {**expected_metadata, "semantic_receipt_sha256": semantic_sha256(expected_metadata)}
        or len(committed_rows) != len(rebuilt_rows)
        or any(
            committed.payload().keys() != rebuilt.payload().keys()
            or any(
                not torch.equal(
                    cast(torch.Tensor, getattr(committed, name)),
                    cast(torch.Tensor, getattr(rebuilt, name)),
                )
                for name in (*_FLOAT_ARRAY_NAMES, "valid")
            )
            or committed.unsigned() != rebuilt.unsigned()
            or committed.receipt_sha256 != rebuilt.receipt_sha256
            for committed, rebuilt in zip(
                committed_rows, rebuilt_rows, strict=True
            )
        )
    ):
        raise MassiveAdaptiveForecastArchiveV1Error(
            "adaptive forecast archive does not replay from its checkpoint roots"
        )
    result = replace(
        parsed,
        runtime_rows=rebuilt_rows,
        runtime_forecasts_replayed=True,
        development_forecast_authorized=(
            parsed.committed_source_data_qualified
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_FORECAST_ARCHIVE_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V1_RECEIPT_SHA256",
    "MASSIVE_ADAPTIVE_FORECAST_NORMALIZATION_V1_RECEIPT_SHA256",
    "MassiveAdaptiveForecastArchiveV1",
    "MassiveAdaptiveForecastArchiveV1Error",
    "MassiveAdaptiveForecastRowV1",
    "authorize_massive_adaptive_forecast_archive_v1",
    "materialize_massive_adaptive_forecast_archive_v1",
    "parse_massive_adaptive_forecast_archive_v1",
]
