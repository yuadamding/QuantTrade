"""Create-only state ledger for the four-fold adaptive RL experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-state-v1"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "persistence": "create-only-stage-ledger",
        "resume": "last-valid-receipt",
        "failure": "durable-terminal-record",
        "caller_economics": False,
        "profitability_reporting": False,
        "live_trading": False,
    }
)


class MassiveAdaptiveRLExperimentStageV1(str, Enum):
    REGISTERED = "registered"
    SOURCE_BUNDLE_REPLAYED = "source-bundle-replayed"
    FIT_FORECASTS_AUTHORIZED = "fit-forecasts-authorized"
    PPO_AND_FIXED_CONTROLS_TRAINED = "ppo-and-fixed-controls-trained"
    INNER_VALIDATION_COMPLETED = "inner-validation-completed"
    POLICY_SELECTED_AND_FROZEN = "policy-selected-and-frozen"
    OUTER_ACCESS_COMMITTED = "outer-access-committed"
    OUTER_FORECAST_MATERIALIZED = "outer-forecast-materialized"
    PPO_AND_FC06_OUTER_LADDERS_COMPLETED = "ppo-and-fc06-outer-ladders-completed"
    FOUR_FOLD_V4_EVIDENCE_COMPLETED = "four-fold-v4-evidence-completed"
    DEVELOPMENT_REPORT_PUBLISHED = "development-report-published"
    FAILED = "failed"


MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1 = (
    MassiveAdaptiveRLExperimentStageV1.REGISTERED,
    MassiveAdaptiveRLExperimentStageV1.SOURCE_BUNDLE_REPLAYED,
    MassiveAdaptiveRLExperimentStageV1.FIT_FORECASTS_AUTHORIZED,
    MassiveAdaptiveRLExperimentStageV1.PPO_AND_FIXED_CONTROLS_TRAINED,
    MassiveAdaptiveRLExperimentStageV1.INNER_VALIDATION_COMPLETED,
    MassiveAdaptiveRLExperimentStageV1.POLICY_SELECTED_AND_FROZEN,
    MassiveAdaptiveRLExperimentStageV1.OUTER_ACCESS_COMMITTED,
    MassiveAdaptiveRLExperimentStageV1.OUTER_FORECAST_MATERIALIZED,
    MassiveAdaptiveRLExperimentStageV1.PPO_AND_FC06_OUTER_LADDERS_COMPLETED,
    MassiveAdaptiveRLExperimentStageV1.FOUR_FOLD_V4_EVIDENCE_COMPLETED,
    MassiveAdaptiveRLExperimentStageV1.DEVELOPMENT_REPORT_PUBLISHED,
)


class MassiveAdaptiveRLExperimentStateV1Error(ValueError):
    """The adaptive RL experiment state ledger is missing or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentStateV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _experiment_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "adaptive RL experiment state ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentStateV1:
    experiment_id: str
    manifest_receipt_sha256: str
    sequence_index: int
    stage: MassiveAdaptiveRLExperimentStageV1
    completed_stage_index: int
    previous_state_receipt_sha256: str | None
    stage_artifact_receipt_sha256: str
    failed_stage: MassiveAdaptiveRLExperimentStageV1 | None
    failure_code: str | None
    semantic_receipt_sha256: str
    development_profitability_reporting_authorized: bool = False
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        is_failed = self.stage is MassiveAdaptiveRLExperimentStageV1.FAILED
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SCHEMA
            or _experiment_id(self.experiment_id) != self.experiment_id
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
            or isinstance(self.completed_stage_index, bool)
            or not -1
            <= self.completed_stage_index
            < len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1)
            or (self.sequence_index == 0)
            != (self.previous_state_receipt_sha256 is None)
            or is_failed != (self.failed_stage is not None)
            or is_failed != (self.failure_code is not None)
            or not is_failed
            and self.stage
            is not MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1[
                self.completed_stage_index
            ]
            or is_failed
            and (
                not self.failure_code
                or any(
                    not (character.isalnum() or character in "-_")
                    for character in self.failure_code
                )
                or self.failed_stage is MassiveAdaptiveRLExperimentStageV1.FAILED
            )
            or self.development_profitability_reporting_authorized
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentStateV1Error(
                "adaptive RL experiment state identity or authorization differs"
            )
        for value in (
            self.manifest_receipt_sha256,
            self.stage_artifact_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL experiment state", value)
        if self.previous_state_receipt_sha256 is not None:
            _digest(
                "adaptive RL previous experiment state",
                self.previous_state_receipt_sha256,
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _state_directory(artifact_root: str | Path, experiment_id: str) -> Path:
    return Path(artifact_root) / "adaptive-rl" / experiment_id / "state-v1"


def _state_path(
    artifact_root: str | Path, state: MassiveAdaptiveRLExperimentStateV1
) -> Path:
    return _state_directory(artifact_root, state.experiment_id) / (
        f"{state.sequence_index:03d}-{state.stage.value}.json"
    )


def _write_state(
    *, artifact_root: str | Path, state: MassiveAdaptiveRLExperimentStateV1
) -> MassiveAdaptiveRLExperimentStateV1:
    state.validate()
    output = _state_path(artifact_root, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(state)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "adaptive RL experiment state is create-only"
        ) from error
    return state


def _build_state(
    *,
    experiment_id: str,
    manifest_receipt_sha256: str,
    sequence_index: int,
    stage: MassiveAdaptiveRLExperimentStageV1,
    completed_stage_index: int,
    previous_state_receipt_sha256: str | None,
    stage_artifact_receipt_sha256: str,
    failed_stage: MassiveAdaptiveRLExperimentStageV1 | None = None,
    failure_code: str | None = None,
) -> MassiveAdaptiveRLExperimentStateV1:
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SCHEMA,
        "experiment_id": experiment_id,
        "manifest_receipt_sha256": manifest_receipt_sha256,
        "sequence_index": sequence_index,
        "stage": stage,
        "completed_stage_index": completed_stage_index,
        "previous_state_receipt_sha256": previous_state_receipt_sha256,
        "stage_artifact_receipt_sha256": stage_artifact_receipt_sha256,
        "failed_stage": failed_stage,
        "failure_code": failure_code,
        "development_profitability_reporting_authorized": False,
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V1_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLExperimentStateV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def register_massive_adaptive_rl_experiment_state_v1(
    *,
    artifact_root: str | Path,
    experiment_id: str,
    manifest_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV1:
    state = _build_state(
        experiment_id=_experiment_id(experiment_id),
        manifest_receipt_sha256=_digest(
            "adaptive RL experiment manifest", manifest_receipt_sha256
        ),
        sequence_index=0,
        stage=MassiveAdaptiveRLExperimentStageV1.REGISTERED,
        completed_stage_index=0,
        previous_state_receipt_sha256=None,
        stage_artifact_receipt_sha256=manifest_receipt_sha256,
    )
    return _write_state(artifact_root=artifact_root, state=state)


def advance_massive_adaptive_rl_experiment_state_v1(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV1,
    stage: MassiveAdaptiveRLExperimentStageV1,
    stage_artifact_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV1:
    previous.validate()
    if previous.stage is MassiveAdaptiveRLExperimentStageV1.FAILED:
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "failed adaptive RL experiment state is terminal"
        )
    expected_index = previous.completed_stage_index + 1
    if (
        expected_index >= len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1)
        or stage is not MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1[expected_index]
    ):
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "adaptive RL experiment stage transition is not consecutive"
        )
    state = _build_state(
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        sequence_index=previous.sequence_index + 1,
        stage=stage,
        completed_stage_index=expected_index,
        previous_state_receipt_sha256=previous.semantic_receipt_sha256,
        stage_artifact_receipt_sha256=_digest(
            "adaptive RL stage artifact", stage_artifact_receipt_sha256
        ),
    )
    return _write_state(artifact_root=artifact_root, state=state)


def fail_massive_adaptive_rl_experiment_state_v1(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV1,
    failed_stage: MassiveAdaptiveRLExperimentStageV1,
    failure_code: str,
    failure_evidence_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV1:
    previous.validate()
    if (
        previous.stage is MassiveAdaptiveRLExperimentStageV1.FAILED
        or failed_stage is MassiveAdaptiveRLExperimentStageV1.FAILED
    ):
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "adaptive RL failed-state transition is invalid"
        )
    state = _build_state(
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        sequence_index=previous.sequence_index + 1,
        stage=MassiveAdaptiveRLExperimentStageV1.FAILED,
        completed_stage_index=previous.completed_stage_index,
        previous_state_receipt_sha256=previous.semantic_receipt_sha256,
        stage_artifact_receipt_sha256=_digest(
            "adaptive RL failure evidence", failure_evidence_receipt_sha256
        ),
        failed_stage=failed_stage,
        failure_code=failure_code,
    )
    return _write_state(artifact_root=artifact_root, state=state)


def _parse_state(payload: Mapping[str, object]) -> MassiveAdaptiveRLExperimentStateV1:
    values = dict(payload)
    values["stage"] = MassiveAdaptiveRLExperimentStageV1(str(values["stage"]))
    if values.get("failed_stage") is not None:
        values["failed_stage"] = MassiveAdaptiveRLExperimentStageV1(
            str(values["failed_stage"])
        )
    result = MassiveAdaptiveRLExperimentStateV1(**values)  # type: ignore[arg-type]
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_states_v1(
    *, artifact_root: str | Path, experiment_id: str
) -> tuple[MassiveAdaptiveRLExperimentStateV1, ...]:
    directory = _state_directory(artifact_root, _experiment_id(experiment_id))
    paths = tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()
    if not paths:
        return ()
    states = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise MassiveAdaptiveRLExperimentStateV1Error(
                "adaptive RL experiment state path is not a regular file"
            )
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
            raise MassiveAdaptiveRLExperimentStateV1Error(
                "adaptive RL experiment state is not canonical JSON"
            )
        states.append(_parse_state(cast(Mapping[str, object], value)))
    result = tuple(states)
    for index, state in enumerate(result):
        if state.sequence_index != index or (
            index > 0
            and state.previous_state_receipt_sha256
            != result[index - 1].semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLExperimentStateV1Error(
                "adaptive RL experiment state chain differs"
            )
    manifest_receipts = {state.manifest_receipt_sha256 for state in result}
    if len(manifest_receipts) != 1:
        raise MassiveAdaptiveRLExperimentStateV1Error(
            "adaptive RL experiment state manifest changed"
        )
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V1",
    "MassiveAdaptiveRLExperimentStageV1",
    "MassiveAdaptiveRLExperimentStateV1",
    "MassiveAdaptiveRLExperimentStateV1Error",
    "advance_massive_adaptive_rl_experiment_state_v1",
    "fail_massive_adaptive_rl_experiment_state_v1",
    "load_massive_adaptive_rl_experiment_states_v1",
    "register_massive_adaptive_rl_experiment_state_v1",
]
