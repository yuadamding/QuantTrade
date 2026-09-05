"""Package-owned Manifest-V5 outer inputs opened after access commitment.

The public operation accepts only an already persisted, state-gated outer
access commitment.  It derives the supervised forecast lineage and the exact
outer chronology from the retained runtime-source graph, commits the decision
tensor after access, deterministically rebuilds the target-free forecast, and
returns the access authority with its private economic environments attached.

All four folds use the exact global development-origin inventories committed
by the runtime-source graph.  Their overlap with the fold-two/fold-three
validation roots is identity-checked when that graph is authorized.  This
operation opens only the requested state-gated outer role; no delayed
validation-release artifact is created.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_forecast_archive_v2 import (
    MassiveAdaptiveForecastRowV2,
    replay_massive_adaptive_forecast_rows_v2,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v2 import (
    MassiveAdaptiveOuterAccessCommitmentV2,
    _authorize_massive_adaptive_outer_access_environment_v2,
)
from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
    authorize_massive_adaptive_decision_tensor_v1,
    materialize_massive_adaptive_decision_tensor_v1,
    parse_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaModelSpecV1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-input-authority-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-outer-input-authority-v1"
)
MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "prerequisite": "persisted-state-gated-outer-access-v2",
        "predictor": "exact-causal-supervised-lineage-for-outer-fold",
        "chronology": "complete-126-session-outer-role-with-causal-context",
        "predictor_roots": (
            "global-development-inventories-bound-before-state-gated-access"
        ),
        "forecast": "cpu-float32-eval-no-grad-deterministic-replay",
        "persistence": "decision-tensor-plus-forecast-row-receipt-authority",
        "caller_environment": False,
        "caller_forecast": False,
        "caller_dates": False,
        "profitability_reporting": False,
        "lockbox": False,
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA,
        "encoding": "canonical-json-forecast-row-receipt-authority",
        "generic_reload": "nonauthorizing",
    }
)


class MassiveAdaptiveRLOuterInputsV1Error(ValueError):
    """Outer inputs are unavailable, premature, or detached from access."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(f"{name} must be a lowercase SHA-256")
    return value


def _time(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLOuterInputsV1Error(f"{name} is absent or invalid")
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterInferenceRowV1:
    decision_session_date: str
    fold_index: int
    candidate_origin_index: int
    tensor_origin_index: int
    context_session_dates: tuple[str, ...]
    context_tensor_indices: tuple[int, ...]
    origin_output_position: int
    decision_root_receipt_sha256: str
    next_session_date: str
    next_session_schedule_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "receipt_sha256"
        }

    def validate(self, *, maximum_context_sessions: int) -> None:
        if (
            not self.decision_session_date
            or self.fold_index not in range(4)
            or self.candidate_origin_index < 0
            or self.tensor_origin_index < 0
            or len(self.context_session_dates) != maximum_context_sessions
            or len(self.context_tensor_indices) != maximum_context_sessions
            or self.context_tensor_indices
            != tuple(
                range(
                    self.context_tensor_indices[0],
                    self.context_tensor_indices[-1] + 1,
                )
            )
            or self.context_tensor_indices[-1] != self.tensor_origin_index
            or self.origin_output_position != maximum_context_sessions - 1
            or self.next_session_date <= self.decision_session_date
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer inference row geometry differs"
            )
        for value in (
            self.decision_root_receipt_sha256,
            self.next_session_schedule_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("outer inference row", value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterInferencePlanV1:
    fold_index: int
    rows: tuple[MassiveAdaptiveRLOuterInferenceRowV1, ...]
    outer_access_commitment_receipt_sha256: str
    checkpoint_choice_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    full_decision_root_inventory_sha256: str
    origin_decision_root_inventory_sha256: str
    split_plan_receipt_sha256: str
    session_authority_receipt_sha256: str
    model_spec_receipt_sha256: str
    maximum_context_sessions: int
    row_inventory_sha256: str
    next_session_schedule_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    outer_inference_authorized: bool

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            name: value
            for name, value in asdict(self).items()
            if name not in {"semantic_receipt_sha256", "outer_inference_authorized"}
        }

    def validate(self) -> None:
        for row in self.rows:
            row.validate(maximum_context_sessions=self.maximum_context_sessions)
        if (
            self.fold_index not in range(4)
            or len(self.rows) != 126
            or tuple(row.decision_session_date for row in self.rows)
            != tuple(sorted(set(row.decision_session_date for row in self.rows)))
            or any(row.fold_index != self.fold_index for row in self.rows)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.next_session_schedule_inventory_sha256
            != semantic_sha256(
                tuple(row.next_session_schedule_receipt_sha256 for row in self.rows)
            )
            or not self.source_data_qualified
            or not self.outer_inference_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterInputsV1Error("outer inference plan differs")
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)


def _build_outer_inference_plan_v1(
    *,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    checkpoint_choice_receipt_sha256: str,
    selected_checkpoint: MassiveAdaptiveCheckpointV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    split_plan: MassiveAdaptiveSplitPlanV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
) -> MassiveAdaptiveRLOuterInferencePlanV1:
    selected_checkpoint.validate()
    decision_tensor.validate()
    split_plan.validate()
    model_spec.validate()
    initial_inputs = outer_access.frozen_policy.policy_selection_authority.fold_validation_authority.release_authority.initial_validation_inputs
    runtime_sources = initial_inputs.runtime_sources_v2.base_runtime_sources_v1
    if (
        not outer_access.outer_input_access_authorized
        or decision_tensor.runtime_tensor is None
        or not decision_tensor.runtime_source_replayed
        or selected_checkpoint.semantic_receipt_sha256
        != runtime_sources.fold(outer_access.fold_index)
        .supervised_lineage(outer_access.fold_index)
        .selected_checkpoint.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer plan requires the committed causal supervised lineage"
        )
    fold_index = outer_access.fold_index
    candidates = split_plan.candidate_session_dates
    role_dates = outer_access.outer_decision_session_dates
    tensor_dates = decision_tensor.decision_session_dates
    tensor_index = {date: index for index, date in enumerate(tensor_dates)}
    candidate_index = {date: index for index, date in enumerate(candidates)}
    roots = tuple(sorted(decision_roots, key=lambda row: row.decision_session_date))
    root_by_date = {row.decision_session_date: row for row in roots}
    maximum_context = min(
        int(getattr(model_spec, "maximum_context_sessions")),
        MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    )
    if (
        any(date not in tensor_index or date not in root_by_date for date in role_dates)
        or tuple(root.decision_session_date for root in roots) != tensor_dates
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer tensor does not cover the committed chronology"
        )
    rows: list[MassiveAdaptiveRLOuterInferenceRowV1] = []
    for date in role_dates:
        origin = tensor_index[date]
        candidate = candidate_index[date]
        start = origin - maximum_context + 1
        if start < 0 or candidate + 1 >= len(candidates):
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer origin lacks context or a following session"
            )
        indices = tuple(range(start, origin + 1))
        next_date = candidates[candidate + 1]
        schedule = semantic_sha256(
            {
                "session_authority": split_plan.session_authority_receipt_sha256,
                "decision_session_date": date,
                "next_session_date": next_date,
                "outer_access_commitment": outer_access.semantic_receipt_sha256,
            }
        )
        body = {
            "decision_session_date": date,
            "fold_index": fold_index,
            "candidate_origin_index": candidate,
            "tensor_origin_index": origin,
            "context_session_dates": tuple(tensor_dates[index] for index in indices),
            "context_tensor_indices": indices,
            "origin_output_position": maximum_context - 1,
            "decision_root_receipt_sha256": root_by_date[date].semantic_receipt_sha256,
            "next_session_date": next_date,
            "next_session_schedule_receipt_sha256": schedule,
        }
        row = MassiveAdaptiveRLOuterInferenceRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate(maximum_context_sessions=maximum_context)
        rows.append(row)
    body = {
        "fold_index": fold_index,
        "rows": tuple(rows),
        "outer_access_commitment_receipt_sha256": outer_access.semantic_receipt_sha256,
        "checkpoint_choice_receipt_sha256": checkpoint_choice_receipt_sha256,
        "selected_checkpoint_receipt_sha256": selected_checkpoint.semantic_receipt_sha256,
        "decision_tensor_receipt_sha256": decision_tensor.semantic_receipt_sha256,
        "full_decision_root_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in roots)
        ),
        "origin_decision_root_inventory_sha256": semantic_sha256(
            tuple(row.decision_root_receipt_sha256 for row in rows)
        ),
        "split_plan_receipt_sha256": split_plan.semantic_receipt_sha256,
        "session_authority_receipt_sha256": (
            split_plan.session_authority_receipt_sha256
        ),
        "model_spec_receipt_sha256": model_spec.receipt_sha256,
        "maximum_context_sessions": maximum_context,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "next_session_schedule_inventory_sha256": semantic_sha256(
            tuple(row.next_session_schedule_receipt_sha256 for row in rows)
        ),
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLOuterInferencePlanV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        outer_inference_authorized=True,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterInputAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    outer_access_commitment_receipt_sha256: str
    outer_access_source_receipt_sha256: str
    outer_access_commit_receipt_sha256: str
    outer_access_committed_at_ms: int
    fold_index: int
    origin_session_dates: tuple[str, ...]
    security_ids: tuple[str, ...]
    row_receipts: tuple[str, ...]
    row_inventory_sha256: str
    checkpoint_choice_receipt_sha256: str
    selected_checkpoint_receipt_sha256: str
    checkpoint_source_receipt_sha256: str
    model_state_receipt_sha256: str
    training_window_plan_receipt_sha256: str
    decision_tensor_receipt_sha256: str
    decision_tensor_source_receipt_sha256: str
    decision_tensor_commit_receipt_sha256: str
    decision_tensor_committed_at_ms: int
    outer_inference_plan_receipt_sha256: str
    split_plan_receipt_sha256: str
    decision_root_inventory_sha256: str
    context_origin_inventory_sha256: str
    runtime_sources_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_rows: tuple[MassiveAdaptiveForecastRowV2, ...] | None
    runtime_forecasts_replayed: bool
    outer_forecast_authorized: bool
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA
    _runtime_tensor: MassiveAdaptiveDecisionTensorV1 | None = field(
        default=None, compare=False, repr=False
    )
    _runtime_plan: MassiveAdaptiveRLOuterInferencePlanV1 | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name
            not in {
                "semantic_receipt_sha256",
                "loaded_source",
                "runtime_rows",
                "runtime_forecasts_replayed",
                "outer_forecast_authorized",
                "_runtime_tensor",
                "_runtime_plan",
            }
        }

    @property
    def source_receipt_sha256(self) -> str:
        return self.loaded_source.receipt.receipt_sha256

    @property
    def source_transaction_receipt_sha256(self) -> str:
        return self.loaded_source.commit.receipt_sha256

    @property
    def source_transaction_committed_at_ms(self) -> int:
        return self.loaded_source.commit.committed_at_ms

    def validate(self) -> None:
        runtime = (
            self.runtime_rows is not None
            and self._runtime_tensor is not None
            and self._runtime_plan is not None
        )
        any_runtime = any(
            value is not None
            for value in (self.runtime_rows, self._runtime_tensor, self._runtime_plan)
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA
            or not self.experiment_id
            or self.fold_index not in range(4)
            or len(self.origin_session_dates) != 126
            or self.origin_session_dates
            != tuple(sorted(set(self.origin_session_dates)))
            or not self.security_ids
            or self.security_ids != tuple(sorted(set(self.security_ids)))
            or len(self.row_receipts) != 126
            or self.row_inventory_sha256 != semantic_sha256(self.row_receipts)
            or any_runtime != runtime
            or self.runtime_forecasts_replayed != runtime
            or self.outer_forecast_authorized
            != bool(runtime and self.source_data_qualified)
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SHA256
        ):
            raise MassiveAdaptiveRLOuterInputsV1Error("outer input authority differs")
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self.loaded_source.commit.committed_at_ms
            <= max(
                self.outer_access_committed_at_ms, self.decision_tensor_committed_at_ms
            )
        ):
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer input source transaction differs"
            )
        if runtime:
            assert self.runtime_rows is not None
            assert self._runtime_tensor is not None
            assert self._runtime_plan is not None
            self._runtime_tensor.validate()
            self._runtime_plan.validate()
            for row in self.runtime_rows:
                row.validate()
            if (
                self._runtime_tensor.semantic_receipt_sha256
                != self.decision_tensor_receipt_sha256
                or self._runtime_plan.semantic_receipt_sha256
                != self.outer_inference_plan_receipt_sha256
                or self._runtime_plan.fold_index != self.fold_index
                or tuple(row.decision_session_date for row in self.runtime_rows)
                != self.origin_session_dates
                or tuple(row.security_ids for row in self.runtime_rows)
                != (self.security_ids,) * len(self.runtime_rows)
                or tuple(row.receipt_sha256 for row in self.runtime_rows)
                != self.row_receipts
            ):
                raise MassiveAdaptiveRLOuterInputsV1Error(
                    "outer input runtime replay differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterInputsExecutionV1:
    """The persisted input witness and privately environment-bound access."""

    outer_inputs: MassiveAdaptiveRLOuterInputAuthorityV1
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2

    def validate(self) -> None:
        self.outer_inputs.validate()
        self.outer_access.validate()
        bundle = self.outer_access.runtime_environment_bundle
        if (
            not self.outer_inputs.outer_forecast_authorized
            or not self.outer_access.outer_input_access_authorized
            or bundle.fold_index != self.outer_inputs.fold_index
            or bundle.primary_environment.forecast_archive is not self.outer_inputs
            or bundle.decision_session_dates != self.outer_inputs.origin_session_dates
            or self.outer_inputs.outer_access_commitment_receipt_sha256
            != self.outer_access.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer input execution lineage differs"
            )


def outer_input_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5, fold_index: int
) -> str:
    manifest.validate()
    if fold_index not in range(4):
        raise MassiveAdaptiveRLOuterInputsV1Error("outer input fold differs")
    return (
        f"adaptive-rl/{manifest.experiment_id}/outer-input-authority-v1/"
        f"fold-{fold_index}.json"
    )


def _tensor_relative_path(*, experiment_id: str, fold_index: int) -> str:
    return (
        "massive-adaptive/decision-tensor-v1/"
        f"{experiment_id}-v5-outer-fold-{fold_index}.json"
    )


def _load_or_materialize_tensor(
    *,
    root: str | Path,
    experiment_id: str,
    fold_index: int,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
    committed_at_ms: int,
    allow_materialize: bool,
) -> MassiveAdaptiveDecisionTensorV1:
    relative = _tensor_relative_path(experiment_id=experiment_id, fold_index=fold_index)
    payload = Path(root) / relative
    transaction = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in transaction)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer decision-tensor transaction is incomplete"
        )
    if not all(present):
        if not allow_materialize:
            raise MassiveAdaptiveRLOuterInputsV1Error("outer decision tensor is absent")
        return materialize_massive_adaptive_decision_tensor_v1(
            root=root,
            artifact_id=f"{experiment_id}-v5-outer-fold-{fold_index}",
            features=features,
            action_origins=action_origins,
            committed_at_ms=committed_at_ms,
        )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=time.time_ns() // 1_000_000,
    )
    return authorize_massive_adaptive_decision_tensor_v1(
        root=root,
        tensor=parse_massive_adaptive_decision_tensor_v1(
            root=root, loaded_source=loaded
        ),
        features=features,
        action_origins=action_origins,
    )


def _metadata(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    lineage: object,
    tensor: MassiveAdaptiveDecisionTensorV1,
    plan: MassiveAdaptiveRLOuterInferencePlanV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    context_origins: Sequence[object],
    rows: Sequence[MassiveAdaptiveForecastRowV2],
    runtime_sources_receipt_sha256: str,
) -> dict[str, object]:
    selected_checkpoint = cast(
        MassiveAdaptiveCheckpointV1, getattr(lineage, "selected_checkpoint")
    )
    checkpoint_source = selected_checkpoint.loaded_source.receipt.receipt_sha256
    tensor_source = tensor.loaded_source
    row_receipts = tuple(row.receipt_sha256 for row in rows)
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "outer_access_commitment_receipt_sha256": outer_access.semantic_receipt_sha256,
        "outer_access_source_receipt_sha256": _digest(
            "outer access source", outer_access.source_receipt_sha256
        ),
        "outer_access_commit_receipt_sha256": _digest(
            "outer access commit", outer_access.source_transaction_receipt_sha256
        ),
        "outer_access_committed_at_ms": _time(
            "outer access time", outer_access.source_transaction_committed_at_ms
        ),
        "fold_index": outer_access.fold_index,
        "origin_session_dates": tuple(row.decision_session_date for row in rows),
        "security_ids": tensor.security_ids,
        "row_receipts": row_receipts,
        "row_inventory_sha256": semantic_sha256(row_receipts),
        "checkpoint_choice_receipt_sha256": getattr(
            getattr(lineage, "checkpoint_choice"), "semantic_receipt_sha256"
        ),
        "selected_checkpoint_receipt_sha256": selected_checkpoint.semantic_receipt_sha256,
        "checkpoint_source_receipt_sha256": checkpoint_source,
        "model_state_receipt_sha256": selected_checkpoint.model_state_receipt_sha256,
        "training_window_plan_receipt_sha256": getattr(
            getattr(lineage, "training_window"), "semantic_receipt_sha256"
        ),
        "decision_tensor_receipt_sha256": tensor.semantic_receipt_sha256,
        "decision_tensor_source_receipt_sha256": tensor_source.receipt.receipt_sha256,
        "decision_tensor_commit_receipt_sha256": tensor_source.commit.receipt_sha256,
        "decision_tensor_committed_at_ms": tensor_source.commit.committed_at_ms,
        "outer_inference_plan_receipt_sha256": plan.semantic_receipt_sha256,
        "split_plan_receipt_sha256": plan.split_plan_receipt_sha256,
        "decision_root_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in decision_roots)
        ),
        "context_origin_inventory_sha256": semantic_sha256(
            tuple(getattr(row, "semantic_receipt_sha256") for row in context_origins)
        ),
        "runtime_sources_receipt_sha256": runtime_sources_receipt_sha256,
        "source_data_qualified": bool(
            outer_access.source_data_qualified
            and getattr(lineage, "source_data_qualified")
            and tensor.model_input_authorized
            and plan.source_data_qualified
            and tuple(row.decision_session_date for row in rows)
            == outer_access.outer_decision_session_dates
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SHA256,
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA,
    }


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterInputAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer input authority payload is not canonical JSON"
        )
    body = dict(value)
    for name in ("origin_session_dates", "security_ids", "row_receipts"):
        body[name] = tuple(cast(Sequence[object], body[name]))
    result = MassiveAdaptiveRLOuterInputAuthorityV1(
        **body,
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded,
        runtime_rows=None,
        runtime_forecasts_replayed=False,
        outer_forecast_authorized=False,
    )
    result.validate()
    return result


def run_or_resume_massive_adaptive_rl_outer_inputs_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    outer_access: MassiveAdaptiveOuterAccessCommitmentV2,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLOuterInputsExecutionV1:
    """Build and privately attach the exact outer environments after access."""

    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV5
        or type(manifest_registration)
        is not MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
        or type(outer_access) is not MassiveAdaptiveOuterAccessCommitmentV2
        or type(allow_materialize) is not bool
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer inputs require exact Manifest-V5 authorities"
        )
    manifest.validate()
    manifest_registration.validate()
    outer_access.validate()
    if (
        not manifest_registration.development_protocol_registered
        or not outer_access.outer_input_access_authorized
        or outer_access.experiment_id != manifest.experiment_id
        or outer_access.manifest_v5_receipt_sha256 != manifest.semantic_receipt_sha256
        or outer_access.manifest_v5_registration_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
        or outer_access.fold_index not in range(4)
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer inputs are not causally available from this access commitment"
        )
    frozen_policy = outer_access.frozen_policy
    initial_inputs = frozen_policy.policy_selection_authority.fold_validation_authority.release_authority.initial_validation_inputs
    runtime_sources = initial_inputs.runtime_sources_v2.base_runtime_sources_v1
    runtime_sources.validate()
    fold_index = outer_access.fold_index
    lineage = runtime_sources.fold(fold_index).supervised_lineage(fold_index)
    origins = runtime_sources.outer_origin_inputs(fold_index)
    expected_dates = outer_access.outer_decision_session_dates
    maximum_context = min(
        lineage.model_spec.maximum_context_sessions,
        MASSIVE_ADAPTIVE_MAXIMUM_CONTEXT_SESSIONS_V1,
    )
    candidate_dates = runtime_sources.split_plan.candidate_session_dates
    start = candidate_dates.index(expected_dates[0]) - maximum_context + 1
    stop = candidate_dates.index(expected_dates[-1]) + 1
    expected_tensor_dates = candidate_dates[start:stop]
    if (
        start < 0
        or origins.tensor_session_dates != expected_tensor_dates
        or not origins.source_data_qualified
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer predictor roots do not cover the causal context"
        )
    capability = (
        issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1(
            root=root, authority=manifest_registration
        )
    )
    with massive_adaptive_rl_manifest_v5_writer_scope_v1(
        root=root, capability=capability
    ):
        access_time = _time(
            "outer access time", outer_access.source_transaction_committed_at_ms
        )
        tensor = _load_or_materialize_tensor(
            root=root,
            experiment_id=manifest.experiment_id,
            fold_index=fold_index,
            features=origins.features,
            action_origins=origins.action_origins,
            committed_at_ms=max(time.time_ns() // 1_000_000, access_time) + 1,
            allow_materialize=allow_materialize,
        )
        plan = _build_outer_inference_plan_v1(
            outer_access=outer_access,
            checkpoint_choice_receipt_sha256=(
                lineage.checkpoint_choice.semantic_receipt_sha256
            ),
            selected_checkpoint=lineage.selected_checkpoint,
            decision_tensor=tensor,
            decision_roots=origins.decision_roots,
            split_plan=runtime_sources.split_plan,
            model_spec=lineage.model_spec,
        )
        rows = replay_massive_adaptive_forecast_rows_v2(
            checkpoint=lineage.selected_checkpoint,
            decision_tensor=tensor,
            plan_rows=plan.rows,
            model_spec=lineage.model_spec,
        )
        body = _metadata(
            manifest=manifest,
            outer_access=outer_access,
            lineage=lineage,
            tensor=tensor,
            plan=plan,
            decision_roots=origins.decision_roots,
            context_origins=origins.context_origins,
            rows=rows,
            runtime_sources_receipt_sha256=runtime_sources.semantic_receipt_sha256,
        )
        relative = outer_input_authority_relative_path_v1(
            manifest=manifest, fold_index=fold_index
        )
        payload = Path(root) / relative
        transaction = (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
        present = tuple(path.exists() or path.is_symlink() for path in transaction)
        if any(present) and not all(present):
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer input authority transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveRLOuterInputsV1Error(
                    "outer input authority is absent"
                )
            committed_at_ms = (
                max(
                    time.time_ns() // 1_000_000,
                    tensor.loaded_source.commit.committed_at_ms,
                    access_time,
                )
                + 1
            )
            receipt = semantic_sha256(body)
            publish_massive_source_object(
                stream=BytesIO(canonical_json_file_bytes(body)),
                root=root,
                relative_payload_path=relative,
                dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_DATASET,
                source_object_key=relative,
                requested_at_ms=committed_at_ms,
                downloaded_at_ms=committed_at_ms,
                schema_sha256=(
                    MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                ),
                entitlement_receipt_sha256=receipt,
                committed_at_ms=committed_at_ms,
                request_id=(
                    f"ADAPTIVE-RL-OUTER-INPUT-V1-{manifest.experiment_id}-"
                    f"FOLD{fold_index}"
                ),
            )
        generic = _parse(
            root=root,
            loaded=load_massive_source_bundle(
                root=root,
                relative_payload_path=relative,
                verified_at_ms=time.time_ns() // 1_000_000,
            ),
        )
        if generic.semantic_unsigned() != body:
            raise MassiveAdaptiveRLOuterInputsV1Error(
                "outer input authority does not replay"
            )
        forecast = replace(
            generic,
            runtime_rows=tuple(rows),
            runtime_forecasts_replayed=True,
            outer_forecast_authorized=generic.source_data_qualified,
            _runtime_tensor=tensor,
            _runtime_plan=plan,
        )
        forecast.validate()
    economics = manifest.base_manifest.base_manifest.base_manifest
    cost_ladder = tuple(float(value) for value in economics.cost_ladder_basis_points)
    if len(cost_ladder) != 3 or cost_ladder[1] != float(
        economics.primary_cost_basis_points
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer cost ladder differs from the scientific protocol"
        )
    outer_dates = frozenset(outer_access.outer_decision_session_dates)
    outer_decision_roots = tuple(
        row
        for row in origins.decision_roots
        if row.decision_session_date in outer_dates
    )
    outer_context_origins = tuple(
        row
        for row in origins.context_origins
        if row.decision_session_date in outer_dates
    )
    if (
        tuple(row.decision_session_date for row in outer_decision_roots)
        != outer_access.outer_decision_session_dates
        or tuple(row.decision_session_date for row in outer_context_origins)
        != outer_access.outer_decision_session_dates
    ):
        raise MassiveAdaptiveRLOuterInputsV1Error(
            "outer economic roots do not match the committed fold chronology"
        )
    environments = tuple(
        MassiveAdaptiveProfitabilityEnvV1(
            forecast_archive=forecast,
            calibration=lineage.calibration,
            inference_plan=plan,
            decision_roots=outer_decision_roots,
            context_origins=outer_context_origins,
            fill_source=runtime_sources.fill_source,
            daily_input_authority=runtime_sources.daily_input_authority,
            identity_authority=runtime_sources.identity_authority,
            economic_event_archive=runtime_sources.economic_event_archive,
            initial_capital=economics.primary_capital,
            transaction_cost_basis_points=cost,
            maximum_fill_participation=economics.maximum_fill_participation,
        )
        for cost in (*cost_ladder, float(economics.primary_cost_basis_points))
    )
    authorized_access = _authorize_massive_adaptive_outer_access_environment_v2(
        outer_access=outer_access,
        low_cost_environment=environments[0],
        primary_environment=environments[1],
        high_cost_environment=environments[2],
        fixed_control_environment=environments[3],
    )
    result = MassiveAdaptiveRLOuterInputsExecutionV1(
        outer_inputs=forecast,
        outer_access=authorized_access,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_OUTER_INPUT_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLOuterInferencePlanV1",
    "MassiveAdaptiveRLOuterInferenceRowV1",
    "MassiveAdaptiveRLOuterInputAuthorityV1",
    "MassiveAdaptiveRLOuterInputsExecutionV1",
    "MassiveAdaptiveRLOuterInputsV1Error",
    "outer_input_authority_relative_path_v1",
    "run_or_resume_massive_adaptive_rl_outer_inputs_v1",
]
