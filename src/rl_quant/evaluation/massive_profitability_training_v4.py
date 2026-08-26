"""Package-owned deterministic training and replay-authorized checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    MassiveProfitabilityTournamentDatasetV3,
    MassiveProfitabilityTournamentDatasetV3Error,
    authorize_massive_profitability_tournament_dataset_v3,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MassiveProfitabilityTrainingConfigV1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentPlanV2,
    MassiveProfitabilityTrainingFoldV2,
    parse_massive_profitability_tournament_plan_v2,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    bind_massive_profitability_trained_run_v2,
    publish_massive_profitability_model_checkpoint_v2,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    MassiveProfitabilityModelCheckpointV3,
    authorize_massive_profitability_model_checkpoint_v3,
    bind_massive_profitability_trained_run_v3,
    publish_massive_profitability_model_checkpoint_v3,
)
from rl_quant.training.massive_profitability_training_replay_v3 import (
    train_massive_profitability_fold_replay_v3,
)


def _replay_run(
    *,
    root: str | Path,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    seed: int,
    checkpoint_v2_source_receipt_sha256: str,
    checkpoint_v2_payload_relative_path: str,
    checkpoint_v2_verified_at_ms: int,
):
    promoted = authorize_massive_profitability_tournament_dataset_v3(
        root=root,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
    )
    reloaded_plan = parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=tournament_plan.loaded_source
    )
    if promoted.runtime_dataset is None:
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "training replay requires promoted Dataset V3 tensors"
        )
    if (
        reloaded_plan.receipt_sha256 != tournament_plan.receipt_sha256
        or fold.receipt_sha256 != reloaded_plan.fold_receipts[fold.fold_index]
        or promoted.data_gate_semantic_receipt_sha256
        != reloaded_plan.data_gate_semantic_receipt_sha256
        or promoted.phase_plan_semantic_receipt_sha256
        != reloaded_plan.phase_plan_semantic_receipt_sha256
    ):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "training replay roots differ from the promoted tournament dataset"
        )
    run_v1, trace, runtime = train_massive_profitability_fold_replay_v3(
        dataset=promoted.runtime_dataset,
        tournament_plan=reloaded_plan,
        fold=fold,
        setting_id=setting_id,
        seed=seed,
        config=MassiveProfitabilityTrainingConfigV1(),
    )
    run_v2 = bind_massive_profitability_trained_run_v2(
        run_v1=run_v1,
        dataset_semantic_receipt_sha256=promoted.semantic_receipt_sha256,
        dataset_source_receipt_sha256=promoted.loaded_source.receipt_sha256,
        dataset_v2_receipt_sha256=promoted.dataset_v2_receipt_sha256,
        tournament_plan_receipt_sha256=reloaded_plan.receipt_sha256,
        tournament_plan_source_receipt_sha256=reloaded_plan.loaded_source.receipt_sha256,
        phase_plan_semantic_receipt_sha256=phase_plan.semantic_receipt_sha256,
        fold_receipt_sha256=fold.receipt_sha256,
    )
    return bind_massive_profitability_trained_run_v3(
        run_v2=run_v2,
        checkpoint_v2_source_receipt_sha256=checkpoint_v2_source_receipt_sha256,
        checkpoint_v2_payload_relative_path=checkpoint_v2_payload_relative_path,
        checkpoint_v2_verified_at_ms=checkpoint_v2_verified_at_ms,
        training_runtime=runtime,
        epoch_trace=trace,
    )


def train_and_publish_massive_profitability_fold_v4(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
    setting_id: str,
    seed: int,
    committed_at_ms: int,
) -> MassiveProfitabilityModelCheckpointV3:
    """Train, publish both compatibility and V3 checkpoints, then replay fit."""

    # First replay produces the exact state to commit. The V2 checkpoint is retained
    # only because immutable Prediction V3 consumes its safe float32 encoding.
    provisional = _replay_run(
        root=root,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
        fold=fold,
        setting_id=setting_id,
        seed=seed,
        checkpoint_v2_source_receipt_sha256="0" * 64,
        checkpoint_v2_payload_relative_path=(
            f"massive-profitability/model-checkpoint-v2/"
            f"{artifact_id}-prediction-v3-compat.json"
        ),
        checkpoint_v2_verified_at_ms=committed_at_ms,
    )
    checkpoint_v2 = publish_massive_profitability_model_checkpoint_v2(
        root=root,
        artifact_id=f"{artifact_id}-prediction-v3-compat",
        run=provisional.run_v2,
        committed_at_ms=committed_at_ms,
    )
    committed_run = bind_massive_profitability_trained_run_v3(
        run_v2=provisional.run_v2,
        checkpoint_v2_source_receipt_sha256=checkpoint_v2.loaded_source.receipt_sha256,
        checkpoint_v2_payload_relative_path=checkpoint_v2.loaded_source.payload_relative_path,
        checkpoint_v2_verified_at_ms=checkpoint_v2.loaded_source.verified_at_ms,
        training_runtime=provisional.training_runtime,
        epoch_trace=provisional.epoch_trace,
    )
    checkpoint_v3 = publish_massive_profitability_model_checkpoint_v3(
        root=root,
        artifact_id=artifact_id,
        run=committed_run,
        committed_at_ms=committed_at_ms,
    )
    return authorize_massive_profitability_checkpoint_v3_from_roots(
        root=root,
        checkpoint=checkpoint_v3,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
        fold=fold,
    )


def authorize_massive_profitability_checkpoint_v3_from_roots(
    *,
    root: str | Path,
    checkpoint: MassiveProfitabilityModelCheckpointV3,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    fold: MassiveProfitabilityTrainingFoldV2,
) -> MassiveProfitabilityModelCheckpointV3:
    """Promote a V3 checkpoint only after a complete independent training replay."""

    committed = checkpoint.run
    replayed = _replay_run(
        root=root,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
        fold=fold,
        setting_id=committed.run_v2.run_v1.setting_id,
        seed=committed.run_v2.run_v1.seed,
        checkpoint_v2_source_receipt_sha256=(
            committed.checkpoint_v2_source_receipt_sha256
        ),
        checkpoint_v2_payload_relative_path=(
            committed.checkpoint_v2_payload_relative_path
        ),
        checkpoint_v2_verified_at_ms=committed.checkpoint_v2_verified_at_ms,
    )
    return authorize_massive_profitability_model_checkpoint_v3(
        root=root, checkpoint=checkpoint, replayed_run=replayed
    )


__all__ = [
    "authorize_massive_profitability_checkpoint_v3_from_roots",
    "train_and_publish_massive_profitability_fold_v4",
]
