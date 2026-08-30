"""Create-only replay authority for comparator-bound adaptive RL evidence V3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_rollout_v1 import (
    MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v2 import (
    MassiveAdaptiveAuthenticatedRLOuterFoldV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v3 import (
    MassiveAdaptiveAuthenticatedRLOuterFoldV3,
    MassiveAdaptiveRLOuterEvidenceV3,
    MassiveAdaptiveRLOuterEvidenceV3Error,
    build_massive_adaptive_authenticated_rl_outer_fold_v3,
    build_massive_adaptive_rl_outer_evidence_v3,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v2 import (
    MassiveAdaptiveRLOuterPlanV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_rollout_v1 import (
    MassiveAdaptiveRLOuterRolloutAuthorityV1,
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


MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-evidence-authority-v3"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_DATASET = (
    "massive-adaptive-rl-outer-evidence-authority-v3"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA,
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
            "replay": "ppo-and-fit-selected-static-comparator",
        }
    )
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SPEC_SHA256 = semantic_sha256(
    {
        "outer_plan": "comparator-bound-v2-before-outer-access",
        "ppo": "frozen-policy-rollout-authority-v1",
        "static_comparator": "fit-selected-outer-rollout-authority-v1",
        "aggregate": "outer-evidence-v3",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(ValueError):
    """Committed V3 evidence failed exact PPO and comparator replay."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "adaptive RL V3 outer evidence artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceRecordV3:
    evidence_receipt_sha256: str
    evidence_v2_receipt_sha256: str
    authenticated_fold_inventory_sha256: str
    authenticated_fold_receipts: tuple[str, ...]
    outer_plan_v2_receipts: tuple[str, ...]
    source_data_qualified: bool
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "evidence_v2_receipt_sha256": self.evidence_v2_receipt_sha256,
            "authenticated_fold_inventory_sha256": (
                self.authenticated_fold_inventory_sha256
            ),
            "authenticated_fold_receipts": self.authenticated_fold_receipts,
            "outer_plan_v2_receipts": self.outer_plan_v2_receipts,
            "source_data_qualified": self.source_data_qualified,
            "passed_gate_names": self.passed_gate_names,
            "failed_gate_names": self.failed_gate_names,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        if (
            not self.authenticated_fold_receipts
            or len(self.authenticated_fold_receipts) != len(self.outer_plan_v2_receipts)
            or len(set(self.authenticated_fold_receipts))
            != len(self.authenticated_fold_receipts)
            or len(set(self.outer_plan_v2_receipts)) != len(self.outer_plan_v2_receipts)
            or set(self.passed_gate_names) & set(self.failed_gate_names)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
                "adaptive RL V3 outer evidence record differs"
            )
        for value in (
            self.evidence_receipt_sha256,
            self.evidence_v2_receipt_sha256,
            self.authenticated_fold_inventory_sha256,
            *self.authenticated_fold_receipts,
            *self.outer_plan_v2_receipts,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL V3 outer evidence record", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceAuthorityV3:
    record: MassiveAdaptiveRLOuterEvidenceRecordV3
    evidence_source_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_evidence: MassiveAdaptiveRLOuterEvidenceV3 | None
    runtime_folds: tuple[MassiveAdaptiveAuthenticatedRLOuterFoldV3, ...] | None
    runtime_evidence_replayed: bool
    outer_development_conclusion_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "record_receipt_sha256": self.record.semantic_receipt_sha256,
            "evidence_source_receipt_sha256": self.evidence_source_receipt_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.record.validate()
        self.loaded_source.validate()
        runtime_present = self.runtime_evidence is not None and self.runtime_folds is not None
        expected_authorized = bool(
            runtime_present
            and self.source_data_qualified
            and not self.record.failed_gate_names
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA
            or self.evidence_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.record.evidence_receipt_sha256
            or self.source_data_qualified != self.record.source_data_qualified
            or self.runtime_evidence_replayed != runtime_present
            or self.outer_development_conclusion_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
                "adaptive RL V3 outer evidence replay authority differs"
            )
        if self.runtime_evidence is not None and self.runtime_folds is not None:
            self.runtime_evidence.validate()
            rebuilt = build_massive_adaptive_rl_outer_evidence_v3(self.runtime_folds)
            if (
                rebuilt.semantic_receipt_sha256 != self.record.evidence_receipt_sha256
                or rebuilt.semantic_unsigned()
                != self.runtime_evidence.semantic_unsigned()
                or _record(rebuilt).semantic_unsigned() != self.record.semantic_unsigned()
            ):
                raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
                    "adaptive RL V3 evidence does not replay from authenticated folds"
                )
        for value in (
            self.evidence_source_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL V3 outer evidence authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _record(
    evidence: MassiveAdaptiveRLOuterEvidenceV3,
) -> MassiveAdaptiveRLOuterEvidenceRecordV3:
    evidence.validate()
    evidence_v1 = evidence.evidence_v2.evidence_v1
    body = {
        "evidence_receipt_sha256": evidence.semantic_receipt_sha256,
        "evidence_v2_receipt_sha256": evidence.evidence_v2.semantic_receipt_sha256,
        "authenticated_fold_inventory_sha256": (
            evidence.authenticated_fold_inventory_sha256
        ),
        "authenticated_fold_receipts": tuple(
            fold.semantic_receipt_sha256 for fold in evidence.authenticated_folds
        ),
        "outer_plan_v2_receipts": tuple(
            fold.outer_plan_v2_receipt_sha256 for fold in evidence.authenticated_folds
        ),
        "source_data_qualified": evidence.source_data_qualified,
        "passed_gate_names": evidence_v1.passed_gate_names,
        "failed_gate_names": evidence_v1.failed_gate_names,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLOuterEvidenceRecordV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def _payload(record: MassiveAdaptiveRLOuterEvidenceRecordV3) -> dict[str, object]:
    return {**record.semantic_unsigned(), "semantic_receipt_sha256": record.semantic_receipt_sha256}


def _parse_record(raw: bytes) -> MassiveAdaptiveRLOuterEvidenceRecordV3:
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "adaptive RL V3 outer evidence artifact is not canonical JSON"
        )
    payload = dict(cast(Mapping[str, object], value))
    for name in (
        "authenticated_fold_receipts",
        "outer_plan_v2_receipts",
        "passed_gate_names",
        "failed_gate_names",
    ):
        payload[name] = tuple(cast(Sequence[object], payload[name]))
    result = MassiveAdaptiveRLOuterEvidenceRecordV3(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def parse_massive_adaptive_rl_outer_evidence_authority_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV3:
    """Reopen committed V3 evidence without restoring any conclusion authority."""

    record = _parse_record(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    body = {
        "record": record,
        "evidence_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_data_qualified": record.source_data_qualified,
        "loaded_source": loaded_source,
        "runtime_evidence": None,
        "runtime_folds": None,
        "runtime_evidence_replayed": False,
        "outer_development_conclusion_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA,
    }
    provisional = MassiveAdaptiveRLOuterEvidenceAuthorityV3(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _rebuild_folds(
    *,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    authenticated_folds_v2: Sequence[MassiveAdaptiveAuthenticatedRLOuterFoldV2],
    ppo_outer_rollout_authorities: Sequence[MassiveAdaptiveRLOuterRolloutAuthorityV1],
    fixed_control_outer_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1
    ],
) -> tuple[MassiveAdaptiveAuthenticatedRLOuterFoldV3, ...]:
    collections = (
        outer_plans_v2,
        authenticated_folds_v2,
        ppo_outer_rollout_authorities,
        fixed_control_outer_authorities,
    )
    if not collections[0] or len({len(values) for values in collections}) != 1:
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "adaptive RL V3 replay inputs must have equal nonzero fold counts"
        )
    plans = {value.fold_index: value for value in outer_plans_v2}
    folds_v2 = {value.fold_index: value for value in authenticated_folds_v2}
    ppo = {
        value.runtime_rollout.fold_index: value
        for value in ppo_outer_rollout_authorities
        if value.runtime_rollout is not None
    }
    fixed = {
        value.runtime_rollout.fold_index: value
        for value in fixed_control_outer_authorities
        if value.runtime_rollout is not None
    }
    expected = set(range(len(outer_plans_v2)))
    if set(plans) != expected or set(folds_v2) != expected or set(ppo) != expected or set(fixed) != expected:
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "adaptive RL V3 replay fold inventories differ"
        )
    try:
        return tuple(
            build_massive_adaptive_authenticated_rl_outer_fold_v3(
                authenticated_fold_v2=folds_v2[index],
                outer_plan_v2=plans[index],
                ppo_outer_rollout_authority=ppo[index],
                fixed_control_outer_authority=fixed[index],
            )
            for index in sorted(expected)
        )
    except MassiveAdaptiveRLOuterEvidenceV3Error as exc:
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "adaptive RL V3 fold replay failed"
        ) from exc


def materialize_massive_adaptive_rl_outer_evidence_authority_v3(
    *,
    root: str | Path,
    artifact_id: str,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    authenticated_folds_v2: Sequence[MassiveAdaptiveAuthenticatedRLOuterFoldV2],
    ppo_outer_rollout_authorities: Sequence[MassiveAdaptiveRLOuterRolloutAuthorityV1],
    fixed_control_outer_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1
    ],
    committed_at_ms: int,
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV3:
    """Publish and replay one comparator-bound four-fold development result."""

    identifier = _artifact_id(artifact_id)
    folds = _rebuild_folds(
        outer_plans_v2=outer_plans_v2,
        authenticated_folds_v2=authenticated_folds_v2,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
        fixed_control_outer_authorities=fixed_control_outer_authorities,
    )
    evidence = build_massive_adaptive_rl_outer_evidence_v3(folds)
    record = _record(evidence)
    relative = f"massive-adaptive/rl-outer-evidence-v3/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(record))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=evidence.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-OUTER-EVIDENCE-V3-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_rl_outer_evidence_authority_v3(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_rl_outer_evidence_authority_v3(
        root=root,
        authority=generic,
        outer_plans_v2=outer_plans_v2,
        authenticated_folds_v2=authenticated_folds_v2,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
        fixed_control_outer_authorities=fixed_control_outer_authorities,
    )


def authorize_massive_adaptive_rl_outer_evidence_authority_v3(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLOuterEvidenceAuthorityV3,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    authenticated_folds_v2: Sequence[MassiveAdaptiveAuthenticatedRLOuterFoldV2],
    ppo_outer_rollout_authorities: Sequence[MassiveAdaptiveRLOuterRolloutAuthorityV1],
    fixed_control_outer_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterRolloutAuthorityV1
    ],
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV3:
    """Rebuild every PPO and fixed-control fold before restoring authority."""

    parsed = parse_massive_adaptive_rl_outer_evidence_authority_v3(
        root=root, loaded_source=authority.loaded_source
    )
    folds = _rebuild_folds(
        outer_plans_v2=outer_plans_v2,
        authenticated_folds_v2=authenticated_folds_v2,
        ppo_outer_rollout_authorities=ppo_outer_rollout_authorities,
        fixed_control_outer_authorities=fixed_control_outer_authorities,
    )
    evidence = build_massive_adaptive_rl_outer_evidence_v3(folds)
    record = _record(evidence)
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or record.semantic_unsigned() != parsed.record.semantic_unsigned()
        or record.semantic_receipt_sha256 != parsed.record.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV3Error(
            "committed adaptive RL V3 evidence does not replay"
        )
    result = replace(
        parsed,
        runtime_evidence=evidence,
        runtime_folds=folds,
        runtime_evidence_replayed=True,
        outer_development_conclusion_authorized=(
            record.source_data_qualified and not record.failed_gate_names
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V3_SCHEMA",
    "MassiveAdaptiveRLOuterEvidenceAuthorityV3",
    "MassiveAdaptiveRLOuterEvidenceAuthorityV3Error",
    "MassiveAdaptiveRLOuterEvidenceRecordV3",
    "authorize_massive_adaptive_rl_outer_evidence_authority_v3",
    "materialize_massive_adaptive_rl_outer_evidence_authority_v3",
    "parse_massive_adaptive_rl_outer_evidence_authority_v3",
]
