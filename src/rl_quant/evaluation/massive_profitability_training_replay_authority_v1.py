"""Root-bound training replay authority for the Massive P0 tournament."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_tournament_dataset_v3 import (
    MassiveProfitabilityTournamentDatasetV3,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    adapt_massive_profitability_training_fold_v2,
)
from rl_quant.evaluation.massive_profitability_training_v4 import (
    authorize_massive_profitability_checkpoint_v3_from_roots,
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
from rl_quant.models.massive_profitability_tabular_v1 import (
    MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
)
from rl_quant.training.massive_profitability_tournament_v1 import (
    MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentPlanV2,
    parse_massive_profitability_tournament_plan_v2,
)
from rl_quant.training.massive_profitability_trained_run_v3 import (
    MassiveProfitabilityModelCheckpointV3,
    parse_massive_profitability_model_checkpoint_v3,
)

MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-profitability-training-replay-authority-v1"
)
MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET = (
    "massive-profitability-training-replay-authority-v1"
)
MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA,
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "inventory": "four-folds-three-trainable-settings-five-seeds",
        "replay": "package-owned-from-dataset-gate-phase-features-targets-plan-roots",
        "training": "complete-deterministic-cpu-reexecution",
        "checkpoint": "checkpoint-v3-create-only-source",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveProfitabilityTrainingReplayAuthorityV1Error(ValueError):
    """The replay inventory differs from the root-reexecuted training runs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingReplayRowV1:
    fold_index: int
    setting_id: str
    seed: int
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    fold_receipt_sha256: str
    training_config_receipt_sha256: str
    training_runtime_receipt_sha256: str
    epoch_trace_receipt_sha256: str
    checkpoint_v3_source_receipt_sha256: str
    checkpoint_v3_payload_relative_path: str
    checkpoint_v3_verified_at_ms: int
    committed_run_v3_semantic_receipt_sha256: str
    replayed_run_v3_semantic_receipt_sha256: str
    run_v2_receipt_sha256: str
    checkpoint_v2_source_receipt_sha256: str
    replay_success: bool
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("receipt_sha256")
        return body

    def validate(self) -> None:
        if (
            not 0 <= self.fold_index < 4
            or self.setting_id not in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
            or self.seed not in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
            or not self.checkpoint_v3_payload_relative_path.startswith(
                "massive-profitability/model-checkpoint-v3/"
            )
            or not self.checkpoint_v3_payload_relative_path.endswith(".json")
            or ".." in self.checkpoint_v3_payload_relative_path
            or isinstance(self.checkpoint_v3_verified_at_ms, bool)
            or not isinstance(self.checkpoint_v3_verified_at_ms, int)
            or self.checkpoint_v3_verified_at_ms < 0
            or not self.replay_success
            or self.committed_run_v3_semantic_receipt_sha256
            != self.replayed_run_v3_semantic_receipt_sha256
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
                "training replay row V1 differs"
            )
        for value in (
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.fold_receipt_sha256,
            self.training_config_receipt_sha256,
            self.training_runtime_receipt_sha256,
            self.epoch_trace_receipt_sha256,
            self.checkpoint_v3_source_receipt_sha256,
            self.committed_run_v3_semantic_receipt_sha256,
            self.replayed_run_v3_semantic_receipt_sha256,
            self.run_v2_receipt_sha256,
            self.checkpoint_v2_source_receipt_sha256,
            self.receipt_sha256,
        ):
            _digest("training replay row V1", value)


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTrainingReplayAuthorityV1:
    dataset_semantic_receipt_sha256: str
    dataset_source_receipt_sha256: str
    data_gate_semantic_receipt_sha256: str
    phase_plan_semantic_receipt_sha256: str
    tournament_plan_receipt_sha256: str
    tournament_plan_source_receipt_sha256: str
    rows: tuple[MassiveProfitabilityTrainingReplayRowV1, ...]
    row_inventory_sha256: str
    committed_root_replay_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_root_replayed: bool
    outer_evaluation_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dataset_semantic_receipt_sha256": self.dataset_semantic_receipt_sha256,
            "dataset_source_receipt_sha256": self.dataset_source_receipt_sha256,
            "data_gate_semantic_receipt_sha256": self.data_gate_semantic_receipt_sha256,
            "phase_plan_semantic_receipt_sha256": self.phase_plan_semantic_receipt_sha256,
            "tournament_plan_receipt_sha256": self.tournament_plan_receipt_sha256,
            "tournament_plan_source_receipt_sha256": self.tournament_plan_source_receipt_sha256,
            "rows": tuple(asdict(row) for row in self.rows),
            "row_inventory_sha256": self.row_inventory_sha256,
            "committed_root_replay_qualified": self.committed_root_replay_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def validate(self) -> None:
        keys = tuple((row.fold_index, row.setting_id, row.seed) for row in self.rows)
        expected = tuple(
            (fold_index, setting_id, seed)
            for fold_index in range(4)
            for setting_id in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
            for seed in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
        )
        for row in self.rows:
            row.validate()
        if (
            self.schema
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA
            or keys != expected
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or not self.committed_root_replay_qualified
            or self.runtime_root_replayed != self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
                "training replay authority V1 identity differs"
            )
        for value in (
            self.dataset_semantic_receipt_sha256,
            self.dataset_source_receipt_sha256,
            self.data_gate_semantic_receipt_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.tournament_plan_receipt_sha256,
            self.tournament_plan_source_receipt_sha256,
            self.row_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("training replay authority V1", value)
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.dataset_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
                "training replay authority V1 committed source differs"
            )


def _row(
    *,
    checkpoint: MassiveProfitabilityModelCheckpointV3,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityTrainingReplayRowV1:
    run = checkpoint.run
    base = run.run_v2.run_v1
    body = {
        "fold_index": base.fold_index,
        "setting_id": base.setting_id,
        "seed": base.seed,
        "dataset_semantic_receipt_sha256": dataset.semantic_receipt_sha256,
        "dataset_source_receipt_sha256": dataset.loaded_source.receipt_sha256,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": tournament_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": tournament_plan.loaded_source.receipt_sha256,
        "fold_receipt_sha256": run.run_v2.fold_receipt_sha256,
        "training_config_receipt_sha256": base.training_config_receipt_sha256,
        "training_runtime_receipt_sha256": run.training_runtime.receipt_sha256,
        "epoch_trace_receipt_sha256": run.epoch_trace_receipt_sha256,
        "checkpoint_v3_source_receipt_sha256": checkpoint.loaded_source.receipt_sha256,
        "checkpoint_v3_payload_relative_path": checkpoint.loaded_source.payload_relative_path,
        "checkpoint_v3_verified_at_ms": checkpoint.loaded_source.verified_at_ms,
        "committed_run_v3_semantic_receipt_sha256": run.semantic_receipt_sha256,
        "replayed_run_v3_semantic_receipt_sha256": run.semantic_receipt_sha256,
        "run_v2_receipt_sha256": run.run_v2.run_receipt_sha256,
        "checkpoint_v2_source_receipt_sha256": run.checkpoint_v2_source_receipt_sha256,
        "replay_success": True,
    }
    result = MassiveProfitabilityTrainingReplayRowV1(
        **body, receipt_sha256=semantic_sha256(body)  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _validate_roots(
    *,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityTournamentPlanV2:
    dataset.validate()
    data_gate.validate()
    phase_plan.validate()
    tournament_plan.validate()
    if (
        dataset.data_gate_semantic_receipt_sha256 != data_gate.semantic_receipt_sha256
        or dataset.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or tournament_plan.data_gate_semantic_receipt_sha256
        != data_gate.semantic_receipt_sha256
        or tournament_plan.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority roots differ"
        )
    return tournament_plan


def _root_replayed_rows(
    *,
    root: str | Path,
    checkpoints: Sequence[MassiveProfitabilityModelCheckpointV3],
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> tuple[MassiveProfitabilityTrainingReplayRowV1, ...]:
    plan = _validate_roots(
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        tournament_plan=tournament_plan,
    )
    ordered = tuple(
        sorted(
            checkpoints,
            key=lambda value: (
                value.run.run_v2.run_v1.fold_index,
                value.run.run_v2.run_v1.setting_id,
                value.run.run_v2.run_v1.seed,
            ),
        )
    )
    expected = tuple(
        (fold_index, setting_id, seed)
        for fold_index in range(4)
        for setting_id in MASSIVE_PROFITABILITY_TRAINABLE_SETTINGS_V1
        for seed in MASSIVE_PROFITABILITY_CONFIRMATION_SEEDS_V1
    )
    actual = tuple(
        (
            value.run.run_v2.run_v1.fold_index,
            value.run.run_v2.run_v1.setting_id,
            value.run.run_v2.run_v1.seed,
        )
        for value in ordered
    )
    if actual != expected:
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority requires the exact 60-checkpoint inventory"
        )
    rows = []
    for checkpoint in ordered:
        fold_index = checkpoint.run.run_v2.run_v1.fold_index
        fold = adapt_massive_profitability_training_fold_v2(
            phase_plan.outer_folds[fold_index]
        )
        promoted = authorize_massive_profitability_checkpoint_v3_from_roots(
            root=root,
            checkpoint=checkpoint,
            dataset=dataset,
            data_gate=data_gate,
            phase_plan=phase_plan,
            features=features,
            targets=targets,
            tournament_plan=plan,
            fold=fold,
        )
        rows.append(
            _row(
                checkpoint=promoted,
                dataset=dataset,
                data_gate=data_gate,
                phase_plan=phase_plan,
                tournament_plan=plan,
            )
        )
    return tuple(rows)


def _load_checkpoints(
    *, root: str | Path, rows: Sequence[MassiveProfitabilityTrainingReplayRowV1]
) -> tuple[MassiveProfitabilityModelCheckpointV3, ...]:
    checkpoints = []
    for row in rows:
        loaded = load_massive_source_bundle(
            root=root,
            relative_payload_path=row.checkpoint_v3_payload_relative_path,
            verified_at_ms=row.checkpoint_v3_verified_at_ms,
        )
        if loaded.receipt_sha256 != row.checkpoint_v3_source_receipt_sha256:
            raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
                "training replay checkpoint source transaction differs"
            )
        checkpoints.append(
            parse_massive_profitability_model_checkpoint_v3(
                root=root, loaded_source=loaded
            )
        )
    return tuple(checkpoints)


def materialize_massive_profitability_training_replay_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
    checkpoints: Sequence[MassiveProfitabilityModelCheckpointV3],
    committed_at_ms: int,
) -> MassiveProfitabilityTrainingReplayAuthorityV1:
    """Reexecute all 60 trainings and publish their exact replay inventory."""

    reloaded_plan = parse_massive_profitability_tournament_plan_v2(
        root=root, loaded_source=tournament_plan.loaded_source
    )
    rows = _root_replayed_rows(
        root=root,
        checkpoints=checkpoints,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=reloaded_plan,
    )
    semantic = {
        "schema": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA,
        "dataset_semantic_receipt_sha256": dataset.semantic_receipt_sha256,
        "dataset_source_receipt_sha256": dataset.loaded_source.receipt_sha256,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "tournament_plan_receipt_sha256": reloaded_plan.receipt_sha256,
        "tournament_plan_source_receipt_sha256": reloaded_plan.loaded_source.receipt_sha256,
        "rows": tuple(asdict(row) for row in rows),
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "committed_root_replay_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic["semantic_receipt_sha256"] = semantic_sha256(semantic)
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority artifact ID is not path safe"
        )
    relative = f"massive-profitability/training-replay-authority-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(semantic)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=dataset.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-TRAINING-REPLAY-AUTHORITY-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_training_replay_authority_v1(
        root=root, loaded_source=loaded
    )
    if tuple(row.receipt_sha256 for row in parsed.rows) != tuple(
        row.receipt_sha256 for row in rows
    ):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "committed training replay inventory differs"
        )
    result = replace(
        parsed, runtime_root_replayed=True, outer_evaluation_authorized=True
    )
    result.validate()
    return result


def parse_massive_profitability_training_replay_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityTrainingReplayAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority V1 is not canonical JSON"
        )
    rows = tuple(
        MassiveProfitabilityTrainingReplayRowV1(**row)
        for row in payload.pop("rows")
    )
    result = MassiveProfitabilityTrainingReplayAuthorityV1(
        **payload,
        rows=rows,
        loaded_source=loaded_source,
        runtime_root_replayed=False,
        outer_evaluation_authorized=False,
    )
    result.validate()
    expected = result.semantic_unsigned() | {
        "rows": tuple(asdict(row) for row in result.rows),
        "semantic_receipt_sha256": result.semantic_receipt_sha256,
    }
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority V1 canonical bytes differ"
        )
    return result


def authorize_massive_profitability_training_replay_authority_v1(
    *,
    root: str | Path,
    replay_authority: MassiveProfitabilityTrainingReplayAuthorityV1,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    tournament_plan: MassiveProfitabilityTournamentPlanV2,
) -> MassiveProfitabilityTrainingReplayAuthorityV1:
    """Promote a generic reload only by reexecuting every frozen training run."""

    parsed = parse_massive_profitability_training_replay_authority_v1(
        root=root, loaded_source=replay_authority.loaded_source
    )
    checkpoints = _load_checkpoints(root=root, rows=parsed.rows)
    rows = _root_replayed_rows(
        root=root,
        checkpoints=checkpoints,
        dataset=dataset,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
        tournament_plan=tournament_plan,
    )
    roots = (
        dataset.semantic_receipt_sha256,
        dataset.loaded_source.receipt_sha256,
        data_gate.semantic_receipt_sha256,
        phase_plan.semantic_receipt_sha256,
        tournament_plan.receipt_sha256,
        tournament_plan.loaded_source.receipt_sha256,
    )
    committed_roots = (
        parsed.dataset_semantic_receipt_sha256,
        parsed.dataset_source_receipt_sha256,
        parsed.data_gate_semantic_receipt_sha256,
        parsed.phase_plan_semantic_receipt_sha256,
        parsed.tournament_plan_receipt_sha256,
        parsed.tournament_plan_source_receipt_sha256,
    )
    if (
        parsed.semantic_receipt_sha256 != replay_authority.semantic_receipt_sha256
        or roots != committed_roots
        or tuple(row.receipt_sha256 for row in rows)
        != tuple(row.receipt_sha256 for row in parsed.rows)
    ):
        raise MassiveProfitabilityTrainingReplayAuthorityV1Error(
            "training replay authority V1 does not reproduce from its roots"
        )
    result = replace(
        parsed, runtime_root_replayed=True, outer_evaluation_authorized=True
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_DATASET",
    "MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SCHEMA",
    "MASSIVE_PROFITABILITY_TRAINING_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MassiveProfitabilityTrainingReplayAuthorityV1",
    "MassiveProfitabilityTrainingReplayAuthorityV1Error",
    "MassiveProfitabilityTrainingReplayRowV1",
    "authorize_massive_profitability_training_replay_authority_v1",
    "materialize_massive_profitability_training_replay_authority_v1",
    "parse_massive_profitability_training_replay_authority_v1",
]
