"""Preregistered validation-selection protocol for adaptive RL profitability.

Manifest V3 freezes the causal training and final economic report protocol.
V4 is a new immutable generation that additionally freezes how inner-validation
candidates are ranked, how exact ties are broken, and what happens when no
candidate passes the economic eligibility checks.  Constructing or validating
this manifest opens no validation or outer outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_ppo_v1 import (
    MassiveAdaptivePPOConfigV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
    MassiveAdaptiveRLExperimentManifestV3,
    build_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v4"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SOURCE_SHA256 = file_sha256(Path(__file__))

MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1 = (
    "primary-incremental-rl-log-wealth-descending",
    "ppo-minus-fc06-log-wealth-descending",
    "primary-strategy-active-log-wealth-descending",
    "40bp-liquidation-adjusted-return-descending",
    "maximum-drawdown-ascending",
)
MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1 = (
    "update-index-ascending",
    "checkpoint-receipt-sha256-lexicographic-ascending",
)
MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1 = (
    "primary-incremental-rl-log-wealth-strictly-positive",
    "ppo-minus-fc06-log-wealth-strictly-positive",
    "primary-strategy-active-log-wealth-strictly-positive",
    "40bp-liquidation-adjusted-return-nonnegative",
    "terminal-return-cost-ladder-low-ge-primary-ge-high",
    "maximum-drawdown-at-most-0.25",
)
MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1 = (
    "select-deterministic-top-ranked-continue-sealed-outer-diagnostic-"
    "positive-authorization-prohibited-v1"
)
MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1 = (
    "all-selected-policies-validation-eligible",
)
MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4 = tuple(
    sorted(
        (
            *MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
            *MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1,
        )
    )
)

MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256 = semantic_sha256(
    {
        "ranking_population": "every-preregistered-fold-checkpoint",
        "comparison": "lexicographic-in-declared-order",
        "ordered_metrics": MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1,
        "finite_metrics_required": True,
    }
)
MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256 = (
    semantic_sha256(
        {
            "ordered_tie_breakers": (
                MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1
            ),
            "total_order_required": True,
        }
    )
)
MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256 = semantic_sha256(
    {
        "role": "inner-validation-only",
        "candidate_ranking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_tie_breaking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "economic_eligibility_criteria": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
        ),
        "selection_when_eligible_candidates_exist": "highest-ranked-eligible",
        "selection_when_no_eligible_candidate_exists": "highest-ranked-overall",
        "no_eligible_candidate_policy": (
            MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        ),
        "selection_record": {
            "selected_candidate_validation_eligible": "derived-from-candidate",
            "validation_eligibility_failures": (
                "exact-sorted-selected-candidate-failures"
            ),
        },
        "ineligible_selection_outer_diagnostic": "continue-sealed-outer-test",
        "ineligible_selection_positive_authorization": False,
        "mandatory_final_gates": MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1,
        "mandatory_gate_aggregation": (
            "all-four-selected-candidates-economically-eligible"
        ),
        "validation_outcomes_opened_by_manifest": False,
    }
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SPEC_SHA256 = semantic_sha256(
    {
        "base_manifest": "preregistered-final-profitability-v3",
        "validation_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_tie_breaking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "no_eligible_candidate_policy": (
            MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        ),
        "validation_gates": MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1,
        "final_gates": MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4,
        "profitability_reporting": False,
        "live_trading": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLExperimentManifestV4Error(ValueError):
    """The validation-selection protocol was not preregistered exactly."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentManifestV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV4:
    base_manifest: MassiveAdaptiveRLExperimentManifestV3
    validation_selection_specification_sha256: str
    candidate_ranking_specification_sha256: str
    candidate_ranking_metric_names: tuple[str, ...]
    candidate_tie_breaking_specification_sha256: str
    candidate_tie_breaking_rule_names: tuple[str, ...]
    validation_eligibility_criteria: tuple[str, ...]
    no_eligible_candidate_policy: str
    validation_gate_names: tuple[str, ...]
    final_gate_names: tuple[str, ...]
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SCHEMA

    @property
    def experiment_id(self) -> str:
        return self.base_manifest.experiment_id

    @property
    def execution_device_specification(self) -> str:
        return self.base_manifest.execution_device_specification

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_manifest_receipt_sha256": self.base_manifest.semantic_receipt_sha256,
            "validation_selection_specification_sha256": (
                self.validation_selection_specification_sha256
            ),
            "candidate_ranking_specification_sha256": (
                self.candidate_ranking_specification_sha256
            ),
            "candidate_ranking_metric_names": self.candidate_ranking_metric_names,
            "candidate_tie_breaking_specification_sha256": (
                self.candidate_tie_breaking_specification_sha256
            ),
            "candidate_tie_breaking_rule_names": (
                self.candidate_tie_breaking_rule_names
            ),
            "validation_eligibility_criteria": self.validation_eligibility_criteria,
            "no_eligible_candidate_policy": self.no_eligible_candidate_policy,
            "validation_gate_names": self.validation_gate_names,
            "final_gate_names": self.final_gate_names,
            "profitability_reporting_authorized": False,
            "live_trading_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        if type(self.base_manifest) is not MassiveAdaptiveRLExperimentManifestV3:
            raise MassiveAdaptiveRLExperimentManifestV4Error(
                "adaptive RL experiment manifest V4 base manifest differs"
            )
        self.base_manifest.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SCHEMA
            or self.validation_selection_specification_sha256
            != MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
            or self.candidate_ranking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
            or self.candidate_ranking_metric_names
            != MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1
            or self.candidate_tie_breaking_specification_sha256
            != MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
            or self.candidate_tie_breaking_rule_names
            != MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1
            or self.validation_eligibility_criteria
            != MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
            or self.no_eligible_candidate_policy
            != MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
            or self.validation_gate_names
            != MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1
            or self.final_gate_names != MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4
            or not set(self.base_manifest.final_gate_names).issubset(
                self.final_gate_names
            )
            or not set(self.validation_gate_names).issubset(self.final_gate_names)
            or self.profitability_reporting_authorized
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentManifestV4Error(
                "adaptive RL experiment manifest V4 differs"
            )
        for value in (
            self.validation_selection_specification_sha256,
            self.candidate_ranking_specification_sha256,
            self.candidate_tie_breaking_specification_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL experiment manifest V4", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_experiment_manifest_v4(
    *,
    experiment_id: str,
    prequential_block_sessions: int = 63,
    seeds: tuple[int, ...] = (17,),
    ppo_config: MassiveAdaptivePPOConfigV1 | None = None,
    execution_device_specification: str = "cpu",
) -> MassiveAdaptiveRLExperimentManifestV4:
    base = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id=experiment_id,
        prequential_block_sessions=prequential_block_sessions,
        seeds=seeds,
        ppo_config=ppo_config,
        execution_device_specification=execution_device_specification,
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SCHEMA,
        "base_manifest": base,
        "validation_selection_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_ranking_metric_names": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1
        ),
        "candidate_tie_breaking_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
        ),
        "candidate_tie_breaking_rule_names": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1
        ),
        "validation_eligibility_criteria": (
            MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
        ),
        "no_eligible_candidate_policy": (
            MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
        ),
        "validation_gate_names": MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1,
        "final_gate_names": MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4,
        "profitability_reporting_authorized": False,
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLExperimentManifestV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLExperimentManifestV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def write_massive_adaptive_rl_experiment_manifest_v4(
    *, path: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV4
) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(manifest)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLExperimentManifestV4Error(
            "adaptive RL experiment manifest V4 is create-only"
        ) from error


def _parse_base_manifest_v2(value: object) -> MassiveAdaptiveRLExperimentManifestV2:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV4Error(
            "adaptive RL V4 base Manifest V2 is malformed"
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


def _parse_base_manifest_v3(value: object) -> MassiveAdaptiveRLExperimentManifestV3:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV4Error(
            "adaptive RL V4 base Manifest V3 is malformed"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v2(payload["base_manifest"])
    payload["final_gate_names"] = tuple(
        cast(list[str], payload["final_gate_names"])
    )
    result = MassiveAdaptiveRLExperimentManifestV3(**payload)
    result.validate()
    return result


def load_massive_adaptive_rl_experiment_manifest_v4(
    path: str | Path,
) -> MassiveAdaptiveRLExperimentManifestV4:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLExperimentManifestV4Error(
            "adaptive RL experiment manifest V4 is not canonical JSON"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest_v3(payload["base_manifest"])
    for name in (
        "candidate_ranking_metric_names",
        "candidate_tie_breaking_rule_names",
        "validation_eligibility_criteria",
        "validation_gate_names",
        "final_gate_names",
    ):
        payload[name] = tuple(cast(list[str], payload[name]))
    result = MassiveAdaptiveRLExperimentManifestV4(**payload)
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256",
    "MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V4_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4",
    "MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1",
    "MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256",
    "MassiveAdaptiveRLExperimentManifestV4",
    "MassiveAdaptiveRLExperimentManifestV4Error",
    "build_massive_adaptive_rl_experiment_manifest_v4",
    "load_massive_adaptive_rl_experiment_manifest_v4",
    "write_massive_adaptive_rl_experiment_manifest_v4",
]
