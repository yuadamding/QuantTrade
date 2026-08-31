"""Preregistered final profitability protocol for the adaptive RL experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast

from rl_quant.evaluation.massive_adaptive_outer_evidence_v1 import (
    MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1,
    MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1,
    MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1,
)
from rl_quant.evaluation.massive_adaptive_rl_profitability_report_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256,
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
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
    build_massive_adaptive_rl_experiment_manifest_v2,
)


MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-experiment-manifest-v3"
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3 = (
    "cost-ladder-monotone",
    "high-cost-mean-return-nonnegative",
    "high-cost-ppo-minus-fixed-control-nonnegative",
    "incremental-rl-lcb-positive",
    "maximum-fold-drawdown",
    "positive-incremental-folds-at-least-three",
    "positive-ppo-minus-fixed-folds-at-least-three",
    "positive-strategy-folds-at-least-three",
    "ppo-minus-fixed-control-lcb-positive",
    "primary-net-log-return-lcb-positive",
    "strategy-active-lcb-positive",
)
MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SPEC_SHA256 = semantic_sha256(
    {
        "base_manifest": "session-derived-v2",
        "final_report": MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256,
        "final_gates": MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
        "bootstrap": {
            "kind": "fold-cluster-nonwrapping-moving-block",
            "replicates": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1,
            "block_sessions": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1,
            "seed": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1,
            "lower_bound": "one-sided-95-percentile",
        },
        "annualization_sessions": 252,
        "risk_free_return": "none-net-log-return-to-volatility-v1",
        "execution_device": "manifest-bound-nonempty-string",
        "live_trading": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLExperimentManifestV3Error(ValueError):
    """The final adaptive RL profitability protocol was not preregistered."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLExperimentManifestV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLExperimentManifestV3:
    base_manifest: MassiveAdaptiveRLExperimentManifestV2
    profitability_report_specification_sha256: str
    profitability_report_implementation_source_sha256: str
    final_gate_names: tuple[str, ...]
    bootstrap_specification: str
    bootstrap_replicates: int
    bootstrap_block_sessions: int
    bootstrap_seed: int
    annualization_sessions: int
    risk_free_return_specification: str
    execution_device_specification: str
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    live_trading_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SCHEMA

    @property
    def experiment_id(self) -> str:
        return self.base_manifest.experiment_id

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_manifest_receipt_sha256": self.base_manifest.semantic_receipt_sha256,
            "profitability_report_specification_sha256": (
                self.profitability_report_specification_sha256
            ),
            "profitability_report_implementation_source_sha256": (
                self.profitability_report_implementation_source_sha256
            ),
            "final_gate_names": self.final_gate_names,
            "bootstrap_specification": self.bootstrap_specification,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_block_sessions": self.bootstrap_block_sessions,
            "bootstrap_seed": self.bootstrap_seed,
            "annualization_sessions": self.annualization_sessions,
            "risk_free_return_specification": self.risk_free_return_specification,
            "execution_device_specification": self.execution_device_specification,
            "profitability_reporting_authorized": False,
            "live_trading_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.base_manifest.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SCHEMA
            or self.profitability_report_specification_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
            or self.profitability_report_implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256
            or self.final_gate_names != MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3
            or self.bootstrap_specification
            != "fold-cluster-nonwrapping-moving-block-one-sided-95pct-v1"
            or self.bootstrap_replicates
            != MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1
            or self.bootstrap_block_sessions
            != MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1
            or self.bootstrap_seed != MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1
            or self.annualization_sessions != 252
            or self.risk_free_return_specification
            != "none-net-log-return-to-volatility-v1"
            or not self.execution_device_specification
            or self.execution_device_specification
            != self.execution_device_specification.strip()
            or self.profitability_reporting_authorized
            or self.live_trading_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLExperimentManifestV3Error(
                "adaptive RL experiment manifest V3 differs"
            )
        for value in (
            self.profitability_report_specification_sha256,
            self.profitability_report_implementation_source_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL experiment manifest V3", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_experiment_manifest_v3(
    *,
    experiment_id: str,
    prequential_block_sessions: int = 63,
    seeds: tuple[int, ...] = (17,),
    ppo_config: MassiveAdaptivePPOConfigV1 | None = None,
    execution_device_specification: str = "cpu",
) -> MassiveAdaptiveRLExperimentManifestV3:
    base = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id=experiment_id,
        prequential_block_sessions=prequential_block_sessions,
        seeds=seeds,
        ppo_config=ppo_config,
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SCHEMA,
        "base_manifest": base,
        "profitability_report_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SPEC_SHA256
        ),
        "profitability_report_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_PROFITABILITY_REPORT_AUTHORITY_V1_SOURCE_SHA256
        ),
        "final_gate_names": MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
        "bootstrap_specification": (
            "fold-cluster-nonwrapping-moving-block-one-sided-95pct-v1"
        ),
        "bootstrap_replicates": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_REPLICATES_V1,
        "bootstrap_block_sessions": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_BLOCK_SESSIONS_V1,
        "bootstrap_seed": MASSIVE_ADAPTIVE_OUTER_BOOTSTRAP_SEED_V1,
        "annualization_sessions": 252,
        "risk_free_return_specification": "none-net-log-return-to-volatility-v1",
        "execution_device_specification": execution_device_specification,
        "profitability_reporting_authorized": False,
        "live_trading_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLExperimentManifestV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLExperimentManifestV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def validate_massive_adaptive_rl_report_against_manifest_v3(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    report_authority: MassiveAdaptiveRLProfitabilityReportAuthorityV1,
) -> None:
    """Reject a final report whose preregistered protocol identity differs."""

    manifest.validate()
    report_authority.validate()
    observed_gates = tuple(
        sorted(
            set(report_authority.report.passed_gate_names)
            | set(report_authority.report.failed_gate_names)
        )
    )
    if (
        report_authority.specification_sha256
        != manifest.profitability_report_specification_sha256
        or report_authority.implementation_source_sha256
        != manifest.profitability_report_implementation_source_sha256
        or observed_gates != manifest.final_gate_names
    ):
        raise MassiveAdaptiveRLExperimentManifestV3Error(
            "adaptive RL profitability report differs from preregistered manifest"
        )


def write_massive_adaptive_rl_experiment_manifest_v3(
    *, path: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> None:
    manifest.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(manifest)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLExperimentManifestV3Error(
            "adaptive RL experiment manifest V3 is create-only"
        ) from error


def _parse_base_manifest(value: object) -> MassiveAdaptiveRLExperimentManifestV2:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLExperimentManifestV3Error(
            "adaptive RL base manifest is malformed"
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


def load_massive_adaptive_rl_experiment_manifest_v3(
    path: str | Path,
) -> MassiveAdaptiveRLExperimentManifestV3:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLExperimentManifestV3Error(
            "adaptive RL experiment manifest V3 is not canonical JSON"
        )
    payload = dict(value)
    payload["base_manifest"] = _parse_base_manifest(payload["base_manifest"])
    payload["final_gate_names"] = tuple(cast(list[str], payload["final_gate_names"]))
    result = MassiveAdaptiveRLExperimentManifestV3(**payload)
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_MANIFEST_V3_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3",
    "MassiveAdaptiveRLExperimentManifestV3",
    "MassiveAdaptiveRLExperimentManifestV3Error",
    "build_massive_adaptive_rl_experiment_manifest_v3",
    "load_massive_adaptive_rl_experiment_manifest_v3",
    "validate_massive_adaptive_rl_report_against_manifest_v3",
    "write_massive_adaptive_rl_experiment_manifest_v3",
]
