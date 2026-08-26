"""Package-owned training from promoted P0 tournament tensors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

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
    train_massive_profitability_fold_v2,
)
from rl_quant.training.massive_profitability_trained_run_v2 import (
    MassiveProfitabilityModelCheckpointV2,
    bind_massive_profitability_trained_run_v2,
    publish_massive_profitability_model_checkpoint_v2,
)


def train_and_publish_massive_profitability_fold_v3(
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
    config: MassiveProfitabilityTrainingConfigV1 | None = None,
    device: str | torch.device = "cpu",
) -> MassiveProfitabilityModelCheckpointV2:
    """Rebuild the committed tensors, train, bind, and publish one checkpoint."""

    training_config = (
        MassiveProfitabilityTrainingConfigV1() if config is None else config
    )
    training_config.validate()
    if not training_config.is_frozen_authorizing_contract:
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "trained run V2 requires the frozen authorizing configuration"
        )
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
    fold.validate()
    if promoted.runtime_dataset is None:
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "promoted dataset V3 has no replayed tensors"
        )
    if (
        reloaded_plan.receipt_sha256 != tournament_plan.receipt_sha256
        or promoted.data_gate_semantic_receipt_sha256
        != tournament_plan.data_gate_semantic_receipt_sha256
        or promoted.phase_plan_semantic_receipt_sha256
        != tournament_plan.phase_plan_semantic_receipt_sha256
        or fold.receipt_sha256 != tournament_plan.fold_receipts[fold.fold_index]
    ):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "training roots differ from the promoted tournament dataset"
        )
    run_v1 = train_massive_profitability_fold_v2(
        dataset=promoted.runtime_dataset,
        tournament_plan=reloaded_plan,
        fold=fold,
        setting_id=setting_id,
        seed=seed,
        config=training_config,
        device=device,
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
    return publish_massive_profitability_model_checkpoint_v2(
        root=root,
        artifact_id=artifact_id,
        run=run_v2,
        committed_at_ms=committed_at_ms,
    )


__all__ = ["train_and_publish_massive_profitability_fold_v3"]
