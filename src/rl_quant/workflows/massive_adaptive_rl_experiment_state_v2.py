"""Create-only V2 state ledger for completed, blocked, and failed RL runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import cast

from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    MassiveAdaptiveRLProfitabilityReportAuthorityV1,
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
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
    validate_massive_adaptive_rl_report_against_manifest_v3,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-state-v2"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "persistence": "create-only-stage-ledger-v2",
        "resume": "blocked-is-retryable-from-last-completed-stage",
        "failure": "integrity-failure-is-terminal",
        "negative_result": "development-report-published-not-authorized",
        "positive_result": "development-report-published-authorized",
        "authorizing_report_api": (
            "manifest-v3-and-replayed-report-derived-terminal-values"
        ),
        "live_trading": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLExperimentStageV2(str, Enum):
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
    BLOCKED = "blocked"
    FAILED = "failed"


MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2 = (
    MassiveAdaptiveRLExperimentStageV2.REGISTERED,
    MassiveAdaptiveRLExperimentStageV2.SOURCE_BUNDLE_REPLAYED,
    MassiveAdaptiveRLExperimentStageV2.FIT_FORECASTS_AUTHORIZED,
    MassiveAdaptiveRLExperimentStageV2.PPO_AND_FIXED_CONTROLS_TRAINED,
    MassiveAdaptiveRLExperimentStageV2.INNER_VALIDATION_COMPLETED,
    MassiveAdaptiveRLExperimentStageV2.POLICY_SELECTED_AND_FROZEN,
    MassiveAdaptiveRLExperimentStageV2.OUTER_ACCESS_COMMITTED,
    MassiveAdaptiveRLExperimentStageV2.OUTER_FORECAST_MATERIALIZED,
    MassiveAdaptiveRLExperimentStageV2.PPO_AND_FC06_OUTER_LADDERS_COMPLETED,
    MassiveAdaptiveRLExperimentStageV2.FOUR_FOLD_V4_EVIDENCE_COMPLETED,
    MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED,
)


class MassiveAdaptiveRLExperimentStateV2Error(ValueError):
    """The adaptive RL V2 experiment ledger is inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            f"adaptive RL {name} is not path safe"
        )
    return value


def _gate_names(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))) or any(not value for value in result):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL failed gate inventory differs"
        )
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentStateV2:
    experiment_id: str
    manifest_receipt_sha256: str
    sequence_index: int
    stage: MassiveAdaptiveRLExperimentStageV2
    completed_stage_index: int
    previous_state_receipt_sha256: str | None
    stage_artifact_receipt_sha256: str
    blocked_stage: MassiveAdaptiveRLExperimentStageV2 | None
    blocker_code: str | None
    failed_stage: MassiveAdaptiveRLExperimentStageV2 | None
    failure_code: str | None
    execution_complete: bool
    profitability_report_authority_receipt_sha256: str | None
    profitability_report_receipt_sha256: str | None
    failed_gate_names: tuple[str, ...]
    development_profitability_reporting_authorized: bool
    semantic_receipt_sha256: str
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        blocked = self.stage is MassiveAdaptiveRLExperimentStageV2.BLOCKED
        failed = self.stage is MassiveAdaptiveRLExperimentStageV2.FAILED
        published = (
            self.stage
            is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
        )
        ordinary = not blocked and not failed
        next_index = self.completed_stage_index + 1
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SCHEMA
            or _identifier("experiment state ID", self.experiment_id)
            != self.experiment_id
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
            or isinstance(self.completed_stage_index, bool)
            or not -1
            <= self.completed_stage_index
            < len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2)
            or (self.sequence_index == 0)
            != (self.previous_state_receipt_sha256 is None)
            or blocked != (self.blocked_stage is not None)
            or blocked != (self.blocker_code is not None)
            or failed != (self.failed_stage is not None)
            or failed != (self.failure_code is not None)
            or ordinary
            and self.stage
            is not MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[
                self.completed_stage_index
            ]
            or blocked
            and (
                next_index >= len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2)
                or self.blocked_stage
                is not MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[next_index]
            )
            or failed
            and self.failed_stage is MassiveAdaptiveRLExperimentStageV2.FAILED
            or self.execution_complete != published
            or published
            != (
                self.profitability_report_authority_receipt_sha256 is not None
                and self.profitability_report_receipt_sha256 is not None
            )
            or not published
            and (
                self.failed_gate_names
                or self.development_profitability_reporting_authorized
            )
            or published
            and self.development_profitability_reporting_authorized
            == bool(self.failed_gate_names)
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentStateV2Error(
                "adaptive RL V2 experiment state identity or authorization differs"
            )
        _gate_names(self.failed_gate_names)
        for code in (self.blocker_code, self.failure_code):
            if code is not None:
                _identifier("state code", code)
        for value in (
            self.manifest_receipt_sha256,
            self.stage_artifact_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL V2 experiment state", value)
        for linked_value in (
            self.previous_state_receipt_sha256,
            self.profitability_report_authority_receipt_sha256,
            self.profitability_report_receipt_sha256,
        ):
            if linked_value is not None:
                _digest("adaptive RL V2 linked state", linked_value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _state_directory(artifact_root: str | Path, experiment_id: str) -> Path:
    return Path(artifact_root) / "adaptive-rl" / experiment_id / "state-v2"


def _state_path(
    artifact_root: str | Path, state: MassiveAdaptiveRLExperimentStateV2
) -> Path:
    return _state_directory(artifact_root, state.experiment_id) / (
        f"{state.sequence_index:03d}-{state.stage.value}.json"
    )


def _write_state(
    *, artifact_root: str | Path, state: MassiveAdaptiveRLExperimentStateV2
) -> MassiveAdaptiveRLExperimentStateV2:
    state.validate()
    output = _state_path(artifact_root, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(state)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL V2 experiment state is create-only"
        ) from error
    return state


def _build_state(
    *,
    previous: MassiveAdaptiveRLExperimentStateV2 | None,
    experiment_id: str,
    manifest_receipt_sha256: str,
    stage: MassiveAdaptiveRLExperimentStageV2,
    completed_stage_index: int,
    stage_artifact_receipt_sha256: str,
    blocked_stage: MassiveAdaptiveRLExperimentStageV2 | None = None,
    blocker_code: str | None = None,
    failed_stage: MassiveAdaptiveRLExperimentStageV2 | None = None,
    failure_code: str | None = None,
    profitability_report_authority_receipt_sha256: str | None = None,
    profitability_report_receipt_sha256: str | None = None,
    failed_gate_names: tuple[str, ...] = (),
    development_profitability_reporting_authorized: bool = False,
) -> MassiveAdaptiveRLExperimentStateV2:
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SCHEMA,
        "experiment_id": experiment_id,
        "manifest_receipt_sha256": manifest_receipt_sha256,
        "sequence_index": 0 if previous is None else previous.sequence_index + 1,
        "stage": stage,
        "completed_stage_index": completed_stage_index,
        "previous_state_receipt_sha256": (
            None if previous is None else previous.semantic_receipt_sha256
        ),
        "stage_artifact_receipt_sha256": stage_artifact_receipt_sha256,
        "blocked_stage": blocked_stage,
        "blocker_code": blocker_code,
        "failed_stage": failed_stage,
        "failure_code": failure_code,
        "execution_complete": (
            stage is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
        ),
        "profitability_report_authority_receipt_sha256": (
            profitability_report_authority_receipt_sha256
        ),
        "profitability_report_receipt_sha256": profitability_report_receipt_sha256,
        "failed_gate_names": _gate_names(failed_gate_names),
        "development_profitability_reporting_authorized": (
            development_profitability_reporting_authorized
        ),
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_STATE_V2_SOURCE_SHA256
        ),
    }
    result = MassiveAdaptiveRLExperimentStateV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def register_massive_adaptive_rl_experiment_state_v2(
    *, artifact_root: str | Path, experiment_id: str, manifest_receipt_sha256: str
) -> MassiveAdaptiveRLExperimentStateV2:
    result = _build_state(
        previous=None,
        experiment_id=_identifier("experiment ID", experiment_id),
        manifest_receipt_sha256=_digest(
            "adaptive RL experiment manifest", manifest_receipt_sha256
        ),
        stage=MassiveAdaptiveRLExperimentStageV2.REGISTERED,
        completed_stage_index=0,
        stage_artifact_receipt_sha256=manifest_receipt_sha256,
    )
    return _write_state(artifact_root=artifact_root, state=result)


def advance_massive_adaptive_rl_experiment_state_v2(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    stage: MassiveAdaptiveRLExperimentStageV2,
    stage_artifact_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV2:
    previous.validate()
    if previous.stage is MassiveAdaptiveRLExperimentStageV2.FAILED:
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "failed adaptive RL V2 experiment state is terminal"
        )
    expected_index = previous.completed_stage_index + 1
    if (
        stage is MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
        or expected_index >= len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2)
        or stage is not MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[expected_index]
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL V2 experiment stage transition is not consecutive"
        )
    result = _build_state(
        previous=previous,
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        stage=stage,
        completed_stage_index=expected_index,
        stage_artifact_receipt_sha256=_digest(
            "adaptive RL stage artifact", stage_artifact_receipt_sha256
        ),
    )
    return _write_state(artifact_root=artifact_root, state=result)


def block_massive_adaptive_rl_experiment_state_v2(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    blocked_stage: MassiveAdaptiveRLExperimentStageV2,
    blocker_code: str,
    blocker_evidence_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV2:
    previous.validate()
    if previous.stage is MassiveAdaptiveRLExperimentStageV2.FAILED:
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "failed adaptive RL V2 experiment state is terminal"
        )
    result = _build_state(
        previous=previous,
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        stage=MassiveAdaptiveRLExperimentStageV2.BLOCKED,
        completed_stage_index=previous.completed_stage_index,
        stage_artifact_receipt_sha256=_digest(
            "adaptive RL blocker evidence", blocker_evidence_receipt_sha256
        ),
        blocked_stage=blocked_stage,
        blocker_code=_identifier("blocker code", blocker_code),
    )
    return _write_state(artifact_root=artifact_root, state=result)


def fail_massive_adaptive_rl_experiment_state_v2(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    failed_stage: MassiveAdaptiveRLExperimentStageV2,
    failure_code: str,
    failure_evidence_receipt_sha256: str,
) -> MassiveAdaptiveRLExperimentStateV2:
    previous.validate()
    if previous.stage is MassiveAdaptiveRLExperimentStageV2.FAILED:
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "failed adaptive RL V2 experiment state is terminal"
        )
    result = _build_state(
        previous=previous,
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        stage=MassiveAdaptiveRLExperimentStageV2.FAILED,
        completed_stage_index=previous.completed_stage_index,
        stage_artifact_receipt_sha256=_digest(
            "adaptive RL failure evidence", failure_evidence_receipt_sha256
        ),
        failed_stage=failed_stage,
        failure_code=_identifier("failure code", failure_code),
    )
    return _write_state(artifact_root=artifact_root, state=result)


def publish_massive_adaptive_rl_development_report_state_v2(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    profitability_report_authority_receipt_sha256: str,
    profitability_report_receipt_sha256: str,
    failed_gate_names: Sequence[str],
    development_profitability_reporting_authorized: bool,
) -> MassiveAdaptiveRLExperimentStateV2:
    previous.validate()
    expected_index = previous.completed_stage_index + 1
    if (
        previous.stage is MassiveAdaptiveRLExperimentStageV2.FAILED
        or expected_index != len(MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2) - 1
        or MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2[expected_index]
        is not MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL report publication is not the next stage"
        )
    authority_receipt = _digest(
        "adaptive RL profitability report authority",
        profitability_report_authority_receipt_sha256,
    )
    report_receipt = _digest(
        "adaptive RL profitability report", profitability_report_receipt_sha256
    )
    result = _build_state(
        previous=previous,
        experiment_id=previous.experiment_id,
        manifest_receipt_sha256=previous.manifest_receipt_sha256,
        stage=MassiveAdaptiveRLExperimentStageV2.DEVELOPMENT_REPORT_PUBLISHED,
        completed_stage_index=expected_index,
        stage_artifact_receipt_sha256=authority_receipt,
        profitability_report_authority_receipt_sha256=authority_receipt,
        profitability_report_receipt_sha256=report_receipt,
        failed_gate_names=tuple(failed_gate_names),
        development_profitability_reporting_authorized=(
            development_profitability_reporting_authorized
        ),
    )
    return _write_state(artifact_root=artifact_root, state=result)


def publish_massive_adaptive_rl_development_report_state_v3(
    *,
    artifact_root: str | Path,
    previous: MassiveAdaptiveRLExperimentStateV2,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    report_authority: MassiveAdaptiveRLProfitabilityReportAuthorityV1,
) -> MassiveAdaptiveRLExperimentStateV2:
    """Publish a terminal state from a replayed, manifest-bound report.

    Unlike the V2 ledger primitive, this authorizing API accepts no caller
    gate inventory, report receipt, or authorization Boolean.  It derives all
    terminal values from the replayed report authority and requires the
    preceding four-fold evidence receipt to be the report's own V4 evidence
    authority.
    """

    previous.validate()
    manifest.validate()
    if (
        previous.manifest_receipt_sha256 != manifest.semantic_receipt_sha256
        or previous.experiment_id != manifest.experiment_id
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL report state belongs to another manifest"
        )
    report_authority.validate()
    validate_massive_adaptive_rl_report_against_manifest_v3(
        manifest=manifest,
        report_authority=report_authority,
    )
    if (
        previous.stage
        is not MassiveAdaptiveRLExperimentStageV2.FOUR_FOLD_V4_EVIDENCE_COMPLETED
        or previous.stage_artifact_receipt_sha256
        != report_authority.report.outer_evidence_authority_v4_receipt_sha256
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL report does not descend from the completed V4 evidence"
        )
    if (
        report_authority.runtime_report is None
        or not report_authority.runtime_report_replayed
        or not report_authority.source_data_qualified
    ):
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL profitability report is not replay authorized"
        )
    return publish_massive_adaptive_rl_development_report_state_v2(
        artifact_root=artifact_root,
        previous=previous,
        profitability_report_authority_receipt_sha256=(
            report_authority.semantic_receipt_sha256
        ),
        profitability_report_receipt_sha256=(
            report_authority.report.semantic_receipt_sha256
        ),
        failed_gate_names=report_authority.report.failed_gate_names,
        development_profitability_reporting_authorized=(
            report_authority.development_profitability_reporting_authorized
        ),
    )


def _parse_state(payload: Mapping[str, object]) -> MassiveAdaptiveRLExperimentStateV2:
    values = dict(payload)
    values["stage"] = MassiveAdaptiveRLExperimentStageV2(str(values["stage"]))
    for name in ("blocked_stage", "failed_stage"):
        if values.get(name) is not None:
            values[name] = MassiveAdaptiveRLExperimentStageV2(str(values[name]))
    values["failed_gate_names"] = tuple(cast(list[str], values["failed_gate_names"]))
    result = MassiveAdaptiveRLExperimentStateV2(**values)  # type: ignore[arg-type]
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_states_v2(
    *, artifact_root: str | Path, experiment_id: str
) -> tuple[MassiveAdaptiveRLExperimentStateV2, ...]:
    directory = _state_directory(
        artifact_root, _identifier("experiment ID", experiment_id)
    )
    paths = tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()
    if not paths:
        return ()
    states = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise MassiveAdaptiveRLExperimentStateV2Error(
                "adaptive RL V2 experiment state path is not a regular file"
            )
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
            raise MassiveAdaptiveRLExperimentStateV2Error(
                "adaptive RL V2 experiment state is not canonical JSON"
            )
        states.append(_parse_state(cast(Mapping[str, object], value)))
    result = tuple(states)
    for index, state in enumerate(result):
        if state.sequence_index != index or (
            index > 0
            and state.previous_state_receipt_sha256
            != result[index - 1].semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLExperimentStateV2Error(
                "adaptive RL V2 experiment state chain differs"
            )
    if len({state.manifest_receipt_sha256 for state in result}) != 1:
        raise MassiveAdaptiveRLExperimentStateV2Error(
            "adaptive RL V2 experiment state manifest changed"
        )
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_STAGE_ORDER_V2",
    "MassiveAdaptiveRLExperimentStageV2",
    "MassiveAdaptiveRLExperimentStateV2",
    "MassiveAdaptiveRLExperimentStateV2Error",
    "advance_massive_adaptive_rl_experiment_state_v2",
    "block_massive_adaptive_rl_experiment_state_v2",
    "fail_massive_adaptive_rl_experiment_state_v2",
    "load_massive_adaptive_rl_experiment_states_v2",
    "publish_massive_adaptive_rl_development_report_state_v2",
    "publish_massive_adaptive_rl_development_report_state_v3",
    "register_massive_adaptive_rl_experiment_state_v2",
]
