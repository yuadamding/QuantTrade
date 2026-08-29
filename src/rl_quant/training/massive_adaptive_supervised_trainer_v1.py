"""Package-owned adaptive supervised optimization and exact restart.

This trainer owns source replay, decision-root reconciliation, causal window
selection, model inference, target alignment, optimization, and checkpoint
publication.  It has no caller-provided model output or index surface.  The
canonical path is deterministic CPU float32; an explicitly named engineering
canary may exercise unqualified synthetic roots but can never produce an
authorizing checkpoint.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
    build_massive_adaptive_context_origin_authority_v1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
    build_massive_adaptive_decision_root_v1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
    authorize_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.features.massive_adaptive_target_archive_v1 import (
    MassiveAdaptiveTargetArchiveV1,
    authorize_massive_adaptive_target_archive_v1,
    materialize_massive_adaptive_target_archive_canary_v1,
)
from rl_quant.features.massive_adaptive_target_root_v1 import (
    MassiveAdaptiveTargetSourceRuntimeV1,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1,
    MassiveAdaptiveAlphaModelSpecV1,
    MassiveAdaptiveAlphaTermStructureModelV1,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.adaptive_alpha_supervised_v1 import (
    massive_adaptive_alpha_supervised_loss_v1,
)
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointStateV1,
    MassiveAdaptiveCheckpointV1,
    authorize_massive_adaptive_checkpoint_v1,
    publish_massive_adaptive_checkpoint_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
    build_massive_adaptive_split_plan_v1,
    build_massive_adaptive_split_plan_from_archive_v1,
)
from rl_quant.training.massive_adaptive_training_authority_v1 import (
    MassiveAdaptiveTrainingAuthorityV1,
    MassiveAdaptiveTrainingAuthorityV1Error,
    build_massive_adaptive_training_authority_v1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
    MassiveAdaptiveWindowRowV1,
    build_massive_adaptive_window_plan_v1,
)
from rl_quant.workflows.adaptive_alpha_training_inputs_v3 import (
    build_massive_adaptive_alpha_training_batch_v3,
)


MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SCHEMA = (
    "rl-quant.massive-adaptive-supervised-trainer-v1"
)
MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "device": "cpu",
        "dtype": "float32",
        "optimizer": "adamw",
        "scheduler": "cosine-annealing",
        "data_order": "seeded-window-permutation-per-epoch",
        "forward": "trainer-owned-source-window-to-model-call",
        "target": "source-target-v1-only",
        "checkpoint": "create-only-exact-resume-v1",
        "mixed_precision": False,
        "distributed": False,
        "outer_or_lockbox_access": False,
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveAdaptiveSupervisedTrainerV1Error(ValueError):
    """Adaptive trainer roots, configuration, or restart state differ."""


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveSupervisedTrainingConfigV1:
    seed: int = 17
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    scheduler_total_updates: int = 10_000
    deterministic_algorithms: bool = True
    intraop_threads: int = 1
    dtype: str = "float32"
    device: str = "cpu"
    mixed_precision: bool = False
    distributed: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SCHEMA
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or isinstance(self.scheduler_total_updates, bool)
            or self.scheduler_total_updates <= 0
            or not self.deterministic_algorithms
            or self.intraop_threads != 1
            or self.dtype != "float32"
            or self.device != "cpu"
            or self.mixed_precision
            or self.distributed
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_SUPERVISED_TRAINER_V1_SOURCE_SHA256
        ):
            raise MassiveAdaptiveSupervisedTrainerV1Error(
                "adaptive supervised training configuration drifted"
            )
        assert_no_adaptive_hold_semantics(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return semantic_sha256(asdict(self))


MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1 = (
    MassiveAdaptiveSupervisedTrainingConfigV1()
)


@dataclass(frozen=True, slots=True)
class _PreparedTrainingV1:
    decision_tensor: MassiveAdaptiveDecisionTensorV1
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...]
    source_targets: tuple[MassiveAdaptiveSourceTargetsV1, ...]
    target_archive: MassiveAdaptiveTargetArchiveV1
    split_plan: MassiveAdaptiveSplitPlanV1
    window_plan: MassiveAdaptiveWindowPlanV1
    training_authority: MassiveAdaptiveTrainingAuthorityV1


def _numpy_state() -> tuple[object, ...]:
    name, values, position, has_gauss, cached = np.random.get_state()
    state_values = np.asarray(values, dtype=np.int64)
    return (
        name,
        torch.from_numpy(state_values.copy()),
        int(position),
        int(has_gauss),
        float(cached),
    )


def _restore_numpy_state(value: tuple[object, ...]) -> None:
    if (
        len(value) != 5
        or not isinstance(value[1], torch.Tensor)
        or isinstance(value[2], bool)
        or not isinstance(value[2], int)
        or isinstance(value[3], bool)
        or not isinstance(value[3], int)
        or not isinstance(value[4], (int, float))
    ):
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive NumPy RNG state is malformed"
        )
    np.random.set_state(
        (
            str(value[0]),
            value[1].detach().cpu().numpy().astype(np.uint32),
            value[2],
            value[3],
            float(value[4]),
        )
    )


def _clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _prepare_training(
    *,
    root: str | Path,
    artifact_id: str,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    target_archive: MassiveAdaptiveTargetArchiveV1 | None,
    target_source_runtimes: Sequence[MassiveAdaptiveTargetSourceRuntimeV1],
    session_authority: MassiveSessionAuthority,
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    split_role: str,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1 | None,
) -> _PreparedTrainingV1:
    promoted_tensor = authorize_massive_adaptive_decision_tensor_v1(
        root=root,
        tensor=decision_tensor,
        features=features,
        action_origins=action_origins,
    )
    replayed_split = (
        build_massive_adaptive_split_plan_v1(
            candidate_session_dates=split_plan.candidate_session_dates,
            session_authority=session_authority,
        )
        if archive_freeze is None
        else build_massive_adaptive_split_plan_from_archive_v1(
            archive_freeze=archive_freeze,
            session_authority=session_authority,
        )
    )
    if replayed_split != split_plan:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive split plan does not replay from its candidate inventory"
        )
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_context = tuple(
        sorted(context_origins, key=lambda row: row.decision_session_date)
    )
    ordered_action = tuple(
        sorted(action_origins, key=lambda row: row.decision_session_date)
    )
    if not (
        len(ordered_features) == len(ordered_context) == len(ordered_action)
    ):
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive context, action, and feature inventories differ"
        )
    roots = tuple(
        build_massive_adaptive_decision_root_v1(
            context_origin=context,
            action_origin=action,
            features=feature,
        )
        for context, action, feature in zip(
            ordered_context, ordered_action, ordered_features, strict=True
        )
    )
    window_plan = build_massive_adaptive_window_plan_v1(
        decision_tensor=promoted_tensor,
        decision_roots=roots,
        split_plan=replayed_split,
        fold_index=fold_index,
        split_role=split_role,
    )
    ordered_targets = tuple(
        sorted(source_targets, key=lambda row: row.decision_session_date)
    )
    if any(
        not isinstance(row, MassiveAdaptiveSourceTargetsV1)
        for row in ordered_targets
    ):
        raise MassiveAdaptiveTrainingAuthorityV1Error(
            "adaptive training requires source-target wrappers"
        )
    promoted_target_archive = (
        materialize_massive_adaptive_target_archive_canary_v1(
            root=root,
            artifact_id=f"{artifact_id}-targets",
            decision_roots=roots,
            source_targets=ordered_targets,
            committed_at_ms=max(row.targets.built_at_ms for row in ordered_targets),
        )
        if target_archive is None
        else authorize_massive_adaptive_target_archive_v1(
            root=root,
            archive=target_archive,
            decision_roots=roots,
            source_targets=ordered_targets,
            source_runtimes=target_source_runtimes,
        )
    )
    authority = build_massive_adaptive_training_authority_v1(
        decision_tensor=promoted_tensor,
        decision_roots=roots,
        target_archive=promoted_target_archive,
        split_plan=replayed_split,
        window_plan=window_plan,
    )
    if promoted_target_archive.runtime_source_targets is None:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive target archive replay did not recover source targets"
        )
    return _PreparedTrainingV1(
        decision_tensor=promoted_tensor,
        decision_roots=roots,
        source_targets=promoted_target_archive.runtime_source_targets,
        target_archive=promoted_target_archive,
        split_plan=replayed_split,
        window_plan=window_plan,
        training_authority=authority,
    )


def _forward_window(
    *,
    model: MassiveAdaptiveAlphaTermStructureModelV1,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    row: MassiveAdaptiveWindowRowV1,
) -> object:
    runtime = decision_tensor.runtime_tensor
    if runtime is None:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive runtime tensor is absent"
        )
    index = torch.tensor(row.context_tensor_indices, dtype=torch.long)
    return model.forward_sequence(
        bars_values=runtime.bars_values.index_select(0, index).unsqueeze(0),
        bars_valid=runtime.bars_valid.index_select(0, index).unsqueeze(0),
        tape_values=runtime.tape_values.index_select(0, index).unsqueeze(0),
        tape_valid=runtime.tape_valid.index_select(0, index).unsqueeze(0),
        source_staleness=runtime.source_staleness.index_select(0, index).unsqueeze(0),
        context_membership=runtime.context_membership.index_select(
            0, index
        ).unsqueeze(0),
        action_mask=runtime.action_mask.index_select(0, index).unsqueeze(0),
    )


def _initial_runtime(
    *,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    config: MassiveAdaptiveSupervisedTrainingConfigV1,
    window_count: int,
) -> tuple[
    MassiveAdaptiveAlphaTermStructureModelV1,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.Generator,
    int,
    int,
    int,
    tuple[int, ...],
    list[float],
]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.intraop_threads)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    model = MassiveAdaptiveAlphaTermStructureModelV1(model_spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_total_updates
    )
    order_rng = torch.Generator(device="cpu")
    order_rng.manual_seed(config.seed + 1)
    order = tuple(
        int(value)
        for value in torch.randperm(window_count, generator=order_rng).tolist()
    )
    return model, optimizer, scheduler, order_rng, 0, 0, 0, order, []


def _resume_runtime(
    *,
    checkpoint: MassiveAdaptiveCheckpointV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    config: MassiveAdaptiveSupervisedTrainingConfigV1,
    window_count: int,
) -> tuple[
    MassiveAdaptiveAlphaTermStructureModelV1,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.Generator,
    int,
    int,
    int,
    tuple[int, ...],
    list[float],
]:
    state = checkpoint.runtime_state
    if state is None:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive resume checkpoint has not been root replayed"
        )
    if (
        len(state.window_order) != window_count
        or state.window_cursor >= window_count
    ):
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive resume data-order state differs"
        )
    torch.set_num_threads(config.intraop_threads)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)
    model = MassiveAdaptiveAlphaTermStructureModelV1(model_spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.scheduler_total_updates
    )
    model.load_state_dict(state.model_state, strict=True)
    optimizer.load_state_dict(state.optimizer_state)
    scheduler.load_state_dict(state.scheduler_state)
    torch.set_rng_state(state.torch_rng_state)
    if state.cuda_rng_states:
        if not torch.cuda.is_available():
            raise MassiveAdaptiveSupervisedTrainerV1Error(
                "adaptive checkpoint contains CUDA RNG state on a CPU runtime"
            )
        torch.cuda.set_rng_state_all(list(state.cuda_rng_states))
    random.setstate(state.python_rng_state)  # type: ignore[arg-type]
    _restore_numpy_state(state.numpy_rng_state)
    order_rng = torch.Generator(device="cpu")
    order_rng.set_state(state.data_order_rng_state)
    return (
        model,
        optimizer,
        scheduler,
        order_rng,
        state.update_index,
        state.epoch_index,
        state.window_cursor,
        state.window_order,
        list(state.loss_trace),
    )


def _run_and_publish(
    *,
    root: str | Path,
    artifact_id: str,
    prepared: _PreparedTrainingV1,
    model_spec: MassiveAdaptiveAlphaModelSpecV1,
    config: MassiveAdaptiveSupervisedTrainingConfigV1,
    updates: int,
    committed_at_ms: int,
    resume_checkpoint: MassiveAdaptiveCheckpointV1 | None,
    require_authorized: bool,
) -> MassiveAdaptiveCheckpointV1:
    model_spec.validate()
    config.validate()
    if isinstance(updates, bool) or updates <= 0:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive update count must be positive"
        )
    authority = prepared.training_authority
    if prepared.window_plan.split_role != "training":
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive optimizer may consume only the frozen training role"
        )
    if require_authorized and (
        not authority.development_training_authorized
        or model_spec != MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1
        or config != MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1
    ):
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "historical adaptive training requires qualified roots and frozen settings"
        )
    model_spec_receipt = model_spec.receipt_sha256
    config_receipt = config.receipt_sha256
    window_count = len(prepared.window_plan.rows)
    if resume_checkpoint is None:
        runtime = _initial_runtime(
            model_spec=model_spec, config=config, window_count=window_count
        )
    else:
        promoted_checkpoint = authorize_massive_adaptive_checkpoint_v1(
            root=root,
            checkpoint=resume_checkpoint,
            training_authority=authority,
            decision_tensor_receipt_sha256=(
                prepared.decision_tensor.semantic_receipt_sha256
            ),
            split_plan_receipt_sha256=prepared.split_plan.semantic_receipt_sha256,
            window_plan_receipt_sha256=(
                prepared.window_plan.semantic_receipt_sha256
            ),
            model_spec_receipt_sha256=model_spec_receipt,
            training_config_receipt_sha256=config_receipt,
        )
        runtime = _resume_runtime(
            checkpoint=promoted_checkpoint,
            model_spec=model_spec,
            config=config,
            window_count=window_count,
        )
    (
        model,
        optimizer,
        scheduler,
        order_rng,
        update_index,
        epoch_index,
        window_cursor,
        window_order,
        loss_trace,
    ) = runtime
    if update_index + updates > config.scheduler_total_updates:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive updates exceed the frozen scheduler horizon"
        )
    roots = {
        row.decision_session_date: row for row in prepared.decision_roots
    }
    targets = {
        row.decision_session_date: row for row in prepared.source_targets
    }
    model.train()
    for _ in range(updates):
        row = prepared.window_plan.rows[window_order[window_cursor]]
        output = _forward_window(
            model=model,
            decision_tensor=prepared.decision_tensor,
            row=row,
        )
        batch = build_massive_adaptive_alpha_training_batch_v3(
            full_window_output=output,  # type: ignore[arg-type]
            decision_tensor=prepared.decision_tensor,
            decision_root=roots[row.origin_session_date],
            source_target=targets[row.origin_session_date],
            split_plan=prepared.split_plan,
            window_plan=prepared.window_plan,
            window_row=row,
        )
        loss = massive_adaptive_alpha_supervised_loss_v1(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        optimizer.step()
        scheduler.step()
        loss_trace.append(float(loss.total.detach().cpu().item()))
        update_index += 1
        window_cursor += 1
        if window_cursor == window_count:
            epoch_index += 1
            window_cursor = 0
            window_order = tuple(
                int(value)
                for value in torch.randperm(
                    window_count, generator=order_rng
                ).tolist()
            )
    state = MassiveAdaptiveCheckpointStateV1(
        model_state=_clone_model_state(model),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        scheduler_state=copy.deepcopy(scheduler.state_dict()),
        gradient_scaler_state={},
        torch_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=tuple(
            value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()
        )
        if torch.cuda.is_available()
        else (),
        data_order_rng_state=order_rng.get_state().clone(),
        python_rng_state=random.getstate(),
        numpy_rng_state=_numpy_state(),
        update_index=update_index,
        epoch_index=epoch_index,
        window_cursor=window_cursor,
        window_order=window_order,
        loss_trace=tuple(loss_trace),
    )
    state.validate()
    generic = publish_massive_adaptive_checkpoint_v1(
        root=root,
        artifact_id=artifact_id,
        state=state,
        training_authority=authority,
        decision_tensor_receipt_sha256=(
            prepared.decision_tensor.semantic_receipt_sha256
        ),
        split_plan_receipt_sha256=prepared.split_plan.semantic_receipt_sha256,
        window_plan_receipt_sha256=prepared.window_plan.semantic_receipt_sha256,
        model_spec_receipt_sha256=model_spec_receipt,
        training_config_receipt_sha256=config_receipt,
        committed_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_checkpoint_v1(
        root=root,
        checkpoint=generic,
        training_authority=authority,
        decision_tensor_receipt_sha256=(
            prepared.decision_tensor.semantic_receipt_sha256
        ),
        split_plan_receipt_sha256=prepared.split_plan.semantic_receipt_sha256,
        window_plan_receipt_sha256=prepared.window_plan.semantic_receipt_sha256,
        model_spec_receipt_sha256=model_spec_receipt,
        training_config_receipt_sha256=config_receipt,
    )


def train_and_publish_massive_adaptive_alpha_v1(
    *,
    root: str | Path,
    artifact_id: str,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    decision_clocks: Sequence[MassiveDecisionClockAuthority],
    context_identity_authority: PITSecurityUniverseAuthority,
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    target_archive: MassiveAdaptiveTargetArchiveV1,
    target_source_runtimes: Sequence[MassiveAdaptiveTargetSourceRuntimeV1],
    session_authority: MassiveSessionAuthority,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    split_role: str,
    updates: int,
    committed_at_ms: int,
    resume_checkpoint: MassiveAdaptiveCheckpointV1 | None = None,
    model_spec: MassiveAdaptiveAlphaModelSpecV1 = (
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1
    ),
    config: MassiveAdaptiveSupervisedTrainingConfigV1 = (
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1
    ),
) -> MassiveAdaptiveCheckpointV1:
    """Run the frozen authorizing CPU trainer from live source roots."""

    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_clocks = tuple(sorted(decision_clocks, key=lambda row: row.session_date))
    clock_by_date = {row.session_date: row for row in ordered_clocks}
    if len(clock_by_date) != len(ordered_clocks) or tuple(clock_by_date) != tuple(
        row.decision_session_date for row in ordered_features
    ):
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "adaptive historical decision-clock inventory differs"
        )
    context_origins = tuple(
        build_massive_adaptive_context_origin_authority_v1(
            decision_clock=clock_by_date[feature.decision_session_date],
            session_authority=session_authority,
            identity_authority=context_identity_authority,
            features=feature,
        )
        for feature in ordered_features
    )
    prepared = _prepare_training(
        root=root,
        artifact_id=artifact_id,
        decision_tensor=decision_tensor,
        features=ordered_features,
        context_origins=context_origins,
        action_origins=action_origins,
        source_targets=source_targets,
        target_archive=target_archive,
        target_source_runtimes=target_source_runtimes,
        session_authority=session_authority,
        split_plan=split_plan,
        fold_index=fold_index,
        split_role=split_role,
        archive_freeze=archive_freeze,
    )
    return _run_and_publish(
        root=root,
        artifact_id=artifact_id,
        prepared=prepared,
        model_spec=model_spec,
        config=config,
        updates=updates,
        committed_at_ms=committed_at_ms,
        resume_checkpoint=resume_checkpoint,
        require_authorized=True,
    )


def train_and_publish_massive_adaptive_alpha_canary_v1(
    *,
    root: str | Path,
    artifact_id: str,
    decision_tensor: MassiveAdaptiveDecisionTensorV1,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    context_origins: Sequence[MassiveAdaptiveContextOriginAuthorityV1],
    action_origins: Sequence[MassiveAdaptiveOriginAuthorityV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    session_authority: MassiveSessionAuthority,
    split_plan: MassiveAdaptiveSplitPlanV1,
    fold_index: int,
    split_role: str,
    updates: int,
    committed_at_ms: int,
    resume_checkpoint: MassiveAdaptiveCheckpointV1 | None = None,
    model_spec: MassiveAdaptiveAlphaModelSpecV1 = (
        MASSIVE_ADAPTIVE_ALPHA_MODEL_SPEC_V1
    ),
    config: MassiveAdaptiveSupervisedTrainingConfigV1 = (
        MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1
    ),
) -> MassiveAdaptiveCheckpointV1:
    """Exercise exact wiring with unqualified roots; never authorize training."""

    prepared = _prepare_training(
        root=root,
        artifact_id=artifact_id,
        decision_tensor=decision_tensor,
        features=features,
        context_origins=context_origins,
        action_origins=action_origins,
        source_targets=source_targets,
        target_archive=None,
        target_source_runtimes=(),
        session_authority=session_authority,
        split_plan=split_plan,
        fold_index=fold_index,
        split_role=split_role,
        archive_freeze=None,
    )
    if prepared.training_authority.development_training_authorized:
        raise MassiveAdaptiveSupervisedTrainerV1Error(
            "the engineering canary cannot consume qualified historical roots"
        )
    return _run_and_publish(
        root=root,
        artifact_id=artifact_id,
        prepared=prepared,
        model_spec=model_spec,
        config=config,
        updates=updates,
        committed_at_ms=committed_at_ms,
        resume_checkpoint=resume_checkpoint,
        require_authorized=False,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_SUPERVISED_TRAINING_CONFIG_V1",
    "MassiveAdaptiveSupervisedTrainerV1Error",
    "MassiveAdaptiveSupervisedTrainingConfigV1",
    "train_and_publish_massive_adaptive_alpha_canary_v1",
    "train_and_publish_massive_adaptive_alpha_v1",
]
