"""Session-derived experiment manifest and training workflow for adaptive RL.

V1 registered one global tuple of PPO update indices.  That cannot describe
the expanding 126/252/378/504-session RL-fit histories.  V2 registers elapsed
market sessions instead and derives fold-local update indices from the
immutable block size.  The compatibility runner delegates execution to the
proven V1 lifecycle only after constructing a sealed fold-local manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast

import torch

from rl_quant.evaluation.massive_adaptive_profitability_env_v1 import (
    MassiveAdaptiveProfitabilityEnvV1,
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
from rl_quant.rl.massive_adaptive_ppo_policy_v1 import (
    MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256,
    MassiveAdaptivePPOConfigV1,
)
from rl_quant.training.massive_adaptive_rl_chronology_authority_v1 import (
    MassiveAdaptiveRLChronologyAuthorityV1,
)
from rl_quant.training.massive_adaptive_rl_fixed_control_registry_v1 import (
    build_massive_adaptive_rl_fixed_control_registry_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v2 import (
    MassiveAdaptiveRLTrainingForecastAuthorityV2,
)
from rl_quant.workflows.massive_adaptive_rl_v1 import (
    MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1,
    MassiveAdaptiveRLTrainingWorkflowV1,
    build_massive_adaptive_rl_experiment_manifest_v1,
    run_massive_adaptive_rl_training_workflow_v1,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v2"
)
MASSIVE_ADAPTIVE_RL_CANDIDATE_SCHEDULE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-candidate-schedule-v1"
)
MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-training-workflow-v2"
)
MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256 = semantic_sha256(
    {
        "rl_fit_sessions": "126-times-fold-index-plus-one",
        "candidate_schedule": "elapsed-sessions-derived-to-fold-local-updates",
        "registered_elapsed_sessions": (126, 252, 378, 504),
        "block_sessions": (21, 63),
        "caller_update_indices": False,
        "seed_policy": "one-canonical-predeclared-seed-no-selection",
        "outer_access": False,
        "lockbox_access": False,
        "duration_semantics": False,
    }
)


class MassiveAdaptiveRLWorkflowV2Error(ValueError):
    """A session-derived manifest, schedule, or training result differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLWorkflowV2Error(f"{name} must be a lowercase SHA-256")
    return value


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL V2 experiment ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLCandidateScheduleV1:
    fold_index: int
    rl_fit_session_count: int
    prequential_block_sessions: int
    candidate_elapsed_sessions: tuple[int, ...]
    candidate_update_indices: tuple[int, ...]
    semantic_receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_RL_CANDIDATE_SCHEDULE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        expected_fit_sessions = 126 * (self.fold_index + 1)
        expected_elapsed = tuple(
            value for value in (126, 252, 378, 504) if value <= expected_fit_sessions
        )
        expected_updates = tuple(
            value // self.prequential_block_sessions for value in expected_elapsed
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_CANDIDATE_SCHEDULE_V1_SCHEMA
            or isinstance(self.fold_index, bool)
            or self.fold_index not in range(4)
            or self.prequential_block_sessions not in {21, 63}
            or self.rl_fit_session_count != expected_fit_sessions
            or self.candidate_elapsed_sessions != expected_elapsed
            or any(
                value % self.prequential_block_sessions
                for value in self.candidate_elapsed_sessions
            )
            or self.candidate_update_indices != expected_updates
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV2Error(
                "adaptive RL session-derived candidate schedule differs"
            )
        _digest("adaptive RL candidate schedule", self.semantic_receipt_sha256)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_candidate_schedule_v1(
    *, fold_index: int, prequential_block_sessions: int
) -> MassiveAdaptiveRLCandidateScheduleV1:
    if (
        isinstance(fold_index, bool)
        or fold_index not in range(4)
        or prequential_block_sessions not in {21, 63}
    ):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL candidate schedule inputs are unsupported"
        )
    rl_fit_sessions = 126 * (fold_index + 1)
    elapsed = tuple(value for value in (126, 252, 378, 504) if value <= rl_fit_sessions)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_CANDIDATE_SCHEDULE_V1_SCHEMA,
        "fold_index": fold_index,
        "rl_fit_session_count": rl_fit_sessions,
        "prequential_block_sessions": prequential_block_sessions,
        "candidate_elapsed_sessions": elapsed,
        "candidate_update_indices": tuple(
            value // prequential_block_sessions for value in elapsed
        ),
    }
    result = MassiveAdaptiveRLCandidateScheduleV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV2:
    experiment_id: str
    fold_indices: tuple[int, ...]
    prequential_block_sessions: int
    candidate_elapsed_sessions: tuple[int, ...]
    seeds: tuple[int, ...]
    seed_policy: str
    ppo_config: MassiveAdaptivePPOConfigV1
    primary_capital: float
    cost_ladder_basis_points: tuple[float, ...]
    primary_cost_basis_points: float
    maximum_fill_participation: float
    fixed_control_registry_receipt_sha256: str
    policy_specification_sha256: str
    action_specification_sha256: str
    reward_specification_sha256: str
    benchmark_specification: str
    initial_book_specification: str
    maximum_fold_drawdown: float
    outer_gate_names: tuple[str, ...]
    fold_candidate_schedule_receipts: tuple[str, ...]
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    workflow_specification_sha256: str = MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256
    workflow_implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V2_SCHEMA

    def schedule(self, fold_index: int) -> MassiveAdaptiveRLCandidateScheduleV1:
        schedule = build_massive_adaptive_rl_candidate_schedule_v1(
            fold_index=fold_index,
            prequential_block_sessions=self.prequential_block_sessions,
        )
        if (
            schedule.semantic_receipt_sha256
            != self.fold_candidate_schedule_receipts[fold_index]
        ):
            raise MassiveAdaptiveRLWorkflowV2Error(
                "adaptive RL manifest schedule does not replay"
            )
        return schedule

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "fold_indices": self.fold_indices,
            "prequential_block_sessions": self.prequential_block_sessions,
            "candidate_elapsed_sessions": self.candidate_elapsed_sessions,
            "seeds": self.seeds,
            "seed_policy": self.seed_policy,
            "ppo_config": asdict(self.ppo_config),
            "primary_capital": self.primary_capital,
            "cost_ladder_basis_points": self.cost_ladder_basis_points,
            "primary_cost_basis_points": self.primary_cost_basis_points,
            "maximum_fill_participation": self.maximum_fill_participation,
            "fixed_control_registry_receipt_sha256": (
                self.fixed_control_registry_receipt_sha256
            ),
            "policy_specification_sha256": self.policy_specification_sha256,
            "action_specification_sha256": self.action_specification_sha256,
            "reward_specification_sha256": self.reward_specification_sha256,
            "benchmark_specification": self.benchmark_specification,
            "initial_book_specification": self.initial_book_specification,
            "maximum_fold_drawdown": self.maximum_fold_drawdown,
            "outer_gate_names": self.outer_gate_names,
            "fold_candidate_schedule_receipts": (self.fold_candidate_schedule_receipts),
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "workflow_specification_sha256": self.workflow_specification_sha256,
            "workflow_implementation_source_sha256": (
                self.workflow_implementation_source_sha256
            ),
        }

    def validate(self) -> None:
        self.ppo_config.validate()
        registry = build_massive_adaptive_rl_fixed_control_registry_v1()
        schedules = tuple(
            build_massive_adaptive_rl_candidate_schedule_v1(
                fold_index=fold_index,
                prequential_block_sessions=self.prequential_block_sessions,
            )
            for fold_index in self.fold_indices
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V2_SCHEMA
            or _artifact_id(self.experiment_id) != self.experiment_id
            or self.fold_indices != (0, 1, 2, 3)
            or self.prequential_block_sessions not in {21, 63}
            or self.candidate_elapsed_sessions != (126, 252, 378, 504)
            or len(self.seeds) != 1
            or self.seeds != tuple(sorted(set(self.seeds)))
            or any(isinstance(value, bool) or value < 0 for value in self.seeds)
            or self.seed_policy != "canonical-fixed-seed-v1"
            or self.ppo_config.seed != self.seeds[0]
            or self.primary_capital != 10_000_000.0
            or self.cost_ladder_basis_points != (10.0, 20.0, 40.0)
            or self.primary_cost_basis_points != 20.0
            or self.maximum_fill_participation != 0.02
            or self.fixed_control_registry_receipt_sha256
            != registry.semantic_receipt_sha256
            or self.policy_specification_sha256
            != MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256
            or self.action_specification_sha256
            != MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
            or self.reward_specification_sha256
            != MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
            or self.benchmark_specification != "shared-buy-and-drift-book-v1"
            or self.initial_book_specification != "all-books-cash-v1"
            or self.maximum_fold_drawdown != 0.25
            or self.outer_gate_names != MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1
            or self.fold_candidate_schedule_receipts
            != tuple(schedule.semantic_receipt_sha256 for schedule in schedules)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.workflow_specification_sha256
            != MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256
            or self.workflow_implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV2Error(
                "adaptive RL experiment manifest V2 differs"
            )
        for value in (
            self.fixed_control_registry_receipt_sha256,
            self.policy_specification_sha256,
            self.action_specification_sha256,
            self.reward_specification_sha256,
            *self.fold_candidate_schedule_receipts,
            self.protocol_receipt_sha256,
            self.workflow_specification_sha256,
            self.workflow_implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL experiment manifest V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_experiment_manifest_v2(
    *,
    experiment_id: str,
    prequential_block_sessions: int = 63,
    candidate_elapsed_sessions: tuple[int, ...] = (126, 252, 378, 504),
    seeds: tuple[int, ...] = (17,),
    ppo_config: MassiveAdaptivePPOConfigV1 | None = None,
) -> MassiveAdaptiveRLExperimentManifestV2:
    config = ppo_config or MassiveAdaptivePPOConfigV1(
        seed=seeds[0] if seeds else 0,
        rollout_length=prequential_block_sessions,
        minibatch_size=prequential_block_sessions,
    )
    registry = build_massive_adaptive_rl_fixed_control_registry_v1()
    schedules = tuple(
        build_massive_adaptive_rl_candidate_schedule_v1(
            fold_index=fold_index,
            prequential_block_sessions=prequential_block_sessions,
        )
        for fold_index in range(4)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V2_SCHEMA,
        "experiment_id": _artifact_id(experiment_id),
        "fold_indices": (0, 1, 2, 3),
        "prequential_block_sessions": prequential_block_sessions,
        "candidate_elapsed_sessions": candidate_elapsed_sessions,
        "seeds": seeds,
        "seed_policy": "canonical-fixed-seed-v1",
        "ppo_config": config,
        "primary_capital": 10_000_000.0,
        "cost_ladder_basis_points": (10.0, 20.0, 40.0),
        "primary_cost_basis_points": 20.0,
        "maximum_fill_participation": 0.02,
        "fixed_control_registry_receipt_sha256": registry.semantic_receipt_sha256,
        "policy_specification_sha256": MASSIVE_ADAPTIVE_PPO_POLICY_V1_SPEC_SHA256,
        "action_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_ACTION_SPECIFICATION_V1_SHA256
        ),
        "reward_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_REWARD_SPECIFICATION_V1_SHA256
        ),
        "benchmark_specification": "shared-buy-and-drift-book-v1",
        "initial_book_specification": "all-books-cash-v1",
        "maximum_fold_drawdown": 0.25,
        "outer_gate_names": MASSIVE_ADAPTIVE_RL_OUTER_GATE_NAMES_V1,
        "fold_candidate_schedule_receipts": tuple(
            schedule.semantic_receipt_sha256 for schedule in schedules
        ),
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "workflow_specification_sha256": MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256,
        "workflow_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLExperimentManifestV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLExperimentManifestV2(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_manifest_v2(
    path: str | Path,
) -> MassiveAdaptiveRLExperimentManifestV2:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL V2 experiment manifest is not canonical JSON"
        )
    payload = dict(value)
    for name in (
        "fold_indices",
        "candidate_elapsed_sessions",
        "seeds",
        "cost_ladder_basis_points",
        "outer_gate_names",
        "fold_candidate_schedule_receipts",
    ):
        payload[name] = tuple(cast(list[object], payload[name]))
    payload["ppo_config"] = MassiveAdaptivePPOConfigV1(
        **cast(dict[str, object], payload["ppo_config"])  # type: ignore[arg-type]
    )
    result = MassiveAdaptiveRLExperimentManifestV2(**payload)
    result.validate()
    return result


def write_massive_adaptive_rl_experiment_manifest_v2(
    *, path: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV2
) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(manifest)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL V2 experiment manifest is create-only"
        ) from error


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTrainingWorkflowV2:
    experiment_manifest_receipt_sha256: str
    candidate_schedule: MassiveAdaptiveRLCandidateScheduleV1
    compatibility_manifest_receipt_sha256: str
    runtime_workflow: MassiveAdaptiveRLTrainingWorkflowV1
    source_data_qualified: bool
    semantic_receipt_sha256: str
    development_rl_training_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_manifest_receipt_sha256": (
                self.experiment_manifest_receipt_sha256
            ),
            "candidate_schedule_receipt_sha256": (
                self.candidate_schedule.semantic_receipt_sha256
            ),
            "compatibility_manifest_receipt_sha256": (
                self.compatibility_manifest_receipt_sha256
            ),
            "runtime_workflow_receipt_sha256": (
                self.runtime_workflow.semantic_receipt_sha256
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.candidate_schedule.validate()
        self.runtime_workflow.validate()
        expected = self.source_data_qualified
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V2_SCHEMA
            or self.runtime_workflow.fold_index != self.candidate_schedule.fold_index
            or self.runtime_workflow.candidate_update_indices
            != self.candidate_schedule.candidate_update_indices
            or self.runtime_workflow.training_run.update_count
            != self.candidate_schedule.candidate_update_indices[-1]
            or self.source_data_qualified != self.runtime_workflow.source_data_qualified
            or self.development_rl_training_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256 != MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLWorkflowV2Error(
                "adaptive RL training workflow V2 differs"
            )
        for value in (
            self.experiment_manifest_receipt_sha256,
            self.compatibility_manifest_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL training workflow V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def run_massive_adaptive_rl_training_workflow_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    fold_index: int,
    seed: int,
    training_authority: MassiveAdaptiveRLTrainingForecastAuthorityV2,
    chronology_authority: MassiveAdaptiveRLChronologyAuthorityV1,
    environments: Mapping[str, MassiveAdaptiveProfitabilityEnvV1],
    artifact_root: str | Path,
    committed_at_ms: int,
    device: torch.device | str = "cpu",
) -> MassiveAdaptiveRLTrainingWorkflowV2:
    """Run one fold using only its session-derived candidate schedule."""

    manifest.validate()
    schedule = manifest.schedule(fold_index)
    training_authority.validate()
    if (
        len(training_authority.origin_session_dates) != schedule.rl_fit_session_count
        or len(training_authority.blocks)
        != schedule.rl_fit_session_count // schedule.prequential_block_sessions
        or training_authority.block_sessions != schedule.prequential_block_sessions
    ):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL training authority does not cover the derived fit prefix"
        )
    compatibility_manifest = build_massive_adaptive_rl_experiment_manifest_v1(
        experiment_id=f"{manifest.experiment_id}-v2-fold{fold_index}",
        prequential_block_sessions=manifest.prequential_block_sessions,
        candidate_update_indices=schedule.candidate_update_indices,
        seeds=manifest.seeds,
        ppo_config=manifest.ppo_config,
    )
    runtime = run_massive_adaptive_rl_training_workflow_v1(
        manifest=compatibility_manifest,
        fold_index=fold_index,
        seed=seed,
        training_authority=training_authority,
        chronology_authority=chronology_authority,
        environments=environments,
        artifact_root=artifact_root,
        committed_at_ms=committed_at_ms,
        device=device,
    )
    qualified = runtime.development_rl_training_authorized
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_TRAINING_WORKFLOW_V2_SCHEMA,
        "experiment_manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "candidate_schedule": schedule,
        "compatibility_manifest_receipt_sha256": (
            compatibility_manifest.semantic_receipt_sha256
        ),
        "runtime_workflow": runtime,
        "source_data_qualified": runtime.source_data_qualified,
        "development_rl_training_authorized": qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_WORKFLOW_V2_SPEC_SHA256,
    }
    provisional = MassiveAdaptiveRLTrainingWorkflowV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLTrainingWorkflowV2(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def _manifest_command(args: argparse.Namespace) -> int:
    config = MassiveAdaptivePPOConfigV1(
        rollout_length=args.block_sessions,
        minibatch_size=args.block_sessions,
        seed=args.seed,
    )
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id=args.experiment_id,
        prequential_block_sessions=args.block_sessions,
        seeds=(args.seed,),
        ppo_config=config,
    )
    write_massive_adaptive_rl_experiment_manifest_v2(
        path=args.output,
        manifest=manifest,
    )
    print(manifest.semantic_receipt_sha256)
    return 0


def _manifest_v3_command(args: argparse.Namespace) -> int:
    from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
        build_massive_adaptive_rl_experiment_manifest_v3,
        write_massive_adaptive_rl_experiment_manifest_v3,
    )

    config = MassiveAdaptivePPOConfigV1(
        rollout_length=args.block_sessions,
        minibatch_size=args.block_sessions,
        seed=args.seed,
    )
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=args.experiment_id,
        prequential_block_sessions=args.block_sessions,
        seeds=(args.seed,),
        ppo_config=config,
        execution_device_specification=args.device,
    )
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=args.output,
        manifest=manifest,
    )
    print(manifest.semantic_receipt_sha256)
    return 0


def _manifest_schema(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "adaptive RL experiment manifest is not canonical JSON"
        )
    return str(value.get("schema", ""))


def _validate_command(args: argparse.Namespace) -> int:
    if _manifest_schema(args.manifest) == (
        "rl-quant.massive-adaptive-rl-experiment-manifest-v3"
    ):
        from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
            load_massive_adaptive_rl_experiment_manifest_v3,
        )

        manifest_v3 = load_massive_adaptive_rl_experiment_manifest_v3(args.manifest)
        receipt = manifest_v3.semantic_receipt_sha256
    else:
        manifest_v2 = load_massive_adaptive_rl_experiment_manifest_v2(args.manifest)
        receipt = manifest_v2.semantic_receipt_sha256
    print(receipt)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if _manifest_schema(args.manifest) == (
        "rl-quant.massive-adaptive-rl-experiment-manifest-v3"
    ):
        from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
            run_massive_adaptive_rl_experiment_v2,
        )

        result_v2 = run_massive_adaptive_rl_experiment_v2(
            manifest_path=args.manifest,
            source_root=args.source_root,
            artifact_root=args.artifact_root,
            device=args.device,
            resume=args.resume,
        )
    else:
        from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v1 import (
            run_massive_adaptive_rl_experiment_v1,
        )

        result_v1 = run_massive_adaptive_rl_experiment_v1(
            manifest_path=args.manifest,
            source_root=args.source_root,
            artifact_root=args.artifact_root,
            device=args.device,
            resume=args.resume,
        )
        print(canonical_json_file_bytes(asdict(result_v1)).decode("utf-8"), end="")
        return 0 if result_v1.execution_complete else 2
    print(canonical_json_file_bytes(asdict(result_v2)).decode("utf-8"), end="")
    return 0 if result_v2.execution_complete else 2


def _verify_run_command(args: argparse.Namespace) -> int:
    if _manifest_schema(args.manifest) == (
        "rl-quant.massive-adaptive-rl-experiment-manifest-v3"
    ):
        from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
            verify_massive_adaptive_rl_experiment_v2,
        )

        result_v2 = verify_massive_adaptive_rl_experiment_v2(
            manifest_path=args.manifest,
            source_root=args.source_root,
            artifact_root=args.artifact_root,
        )
    else:
        from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v1 import (
            verify_massive_adaptive_rl_experiment_v1,
        )

        result_v1 = verify_massive_adaptive_rl_experiment_v1(
            manifest_path=args.manifest,
            source_root=args.source_root,
            artifact_root=args.artifact_root,
        )
        print(canonical_json_file_bytes(asdict(result_v1)).decode("utf-8"), end="")
        return 0
    print(canonical_json_file_bytes(asdict(result_v2)).decode("utf-8"), end="")
    return 0 if result_v2.full_verification_complete else 2


def _verify_ledger_command(args: argparse.Namespace) -> int:
    if _manifest_schema(args.manifest) != (
        "rl-quant.massive-adaptive-rl-experiment-manifest-v3"
    ):
        raise MassiveAdaptiveRLWorkflowV2Error(
            "ledger-only verification requires Manifest V3"
        )
    from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
        verify_massive_adaptive_rl_experiment_ledger_v1,
    )

    result = verify_massive_adaptive_rl_experiment_ledger_v1(
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
    )
    print(canonical_json_file_bytes(asdict(result)).decode("utf-8"), end="")
    return 0 if result.ledger_replayed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quanttrade-adaptive-rl",
        description="Massive adaptive RL immutable session-derived workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser(
        "manifest",
        help="Create one immutable session-derived experiment manifest.",
    )
    manifest.add_argument("--experiment-id", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--block-sessions", type=int, choices=(21, 63), default=63)
    manifest.add_argument("--seed", type=int, default=17)
    manifest.set_defaults(handler=_manifest_command)
    manifest_v3 = commands.add_parser(
        "manifest-v3",
        help="Create the preregistered final-profitability experiment manifest.",
    )
    manifest_v3.add_argument("--experiment-id", required=True)
    manifest_v3.add_argument("--output", required=True)
    manifest_v3.add_argument("--block-sessions", type=int, choices=(21, 63), default=63)
    manifest_v3.add_argument("--seed", type=int, default=17)
    manifest_v3.add_argument("--device", default="cpu")
    manifest_v3.set_defaults(handler=_manifest_v3_command)
    validate = commands.add_parser(
        "validate",
        help="Validate an immutable V2 manifest without opening outcomes.",
    )
    validate.add_argument("--manifest", required=True)
    validate.set_defaults(handler=_validate_command)
    for name, resume in (("run", False), ("resume", True)):
        command = commands.add_parser(
            name,
            help=(
                "Start the persisted four-fold state machine."
                if not resume
                else "Resume the persisted four-fold state machine."
            ),
        )
        command.add_argument("--manifest", required=True)
        command.add_argument("--source-root", required=True)
        command.add_argument("--artifact-root", required=True)
        command.add_argument("--device", default="cpu")
        command.set_defaults(handler=_run_command, resume=resume)
    verify = commands.add_parser(
        "verify",
        help="Deeply verify the completed experiment without advancing it.",
    )
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--source-root", required=True)
    verify.add_argument("--artifact-root", required=True)
    verify.set_defaults(handler=_verify_run_command)
    verify_ledger = commands.add_parser(
        "verify-ledger",
        help="Replay only the persisted state ledger without claiming deep verification.",
    )
    verify_ledger.add_argument("--manifest", required=True)
    verify_ledger.add_argument("--artifact-root", required=True)
    verify_ledger.set_defaults(handler=_verify_ledger_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = [
    "MassiveAdaptiveRLCandidateScheduleV1",
    "MassiveAdaptiveRLExperimentManifestV2",
    "MassiveAdaptiveRLTrainingWorkflowV2",
    "MassiveAdaptiveRLWorkflowV2Error",
    "build_massive_adaptive_rl_candidate_schedule_v1",
    "build_massive_adaptive_rl_experiment_manifest_v2",
    "load_massive_adaptive_rl_experiment_manifest_v2",
    "main",
    "run_massive_adaptive_rl_training_workflow_v2",
    "write_massive_adaptive_rl_experiment_manifest_v2",
]


if __name__ == "__main__":
    raise SystemExit(main())
