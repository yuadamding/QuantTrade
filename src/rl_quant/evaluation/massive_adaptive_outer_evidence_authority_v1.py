"""Create-only replay authority for deterministic adaptive outer evidence."""

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
from rl_quant.evaluation.massive_adaptive_outer_evidence_v1 import (
    MassiveAdaptiveOuterCostFoldV1,
    MassiveAdaptiveOuterEvidenceV1,
    build_massive_adaptive_outer_evidence_v1,
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

MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-outer-evidence-authority-v1"
)
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_DATASET = (
    "massive-adaptive-outer-evidence-authority-v1"
)
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA,
        "encoding": "canonical-json",
        "publication": "create-only-source-transaction",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "evidence": "recomputed-from-exact-four-source-derived-folds",
        "generic_reload": "nonauthorizing",
        "outer_development_conclusion": "only-when-all-frozen-gates-pass",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveOuterEvidenceAuthorityV1Error(ValueError):
    """Committed outer evidence cannot be replayed from its four fold roots."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
            "outer evidence authority artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveOuterEvidenceAuthorityV1:
    evidence: MassiveAdaptiveOuterEvidenceV1
    evidence_source_receipt_sha256: str
    fold_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_folds: tuple[MassiveAdaptiveOuterCostFoldV1, ...] | None
    runtime_evidence_replayed: bool
    outer_development_conclusion_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "evidence_receipt_sha256": self.evidence.semantic_receipt_sha256,
            "evidence_source_receipt_sha256": self.evidence_source_receipt_sha256,
            "fold_inventory_sha256": self.fold_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": (
                self.reinforcement_learning_authorized
            ),
        }

    def validate(self) -> None:
        self.evidence.validate()
        self.loaded_source.validate()
        runtime_present = self.runtime_folds is not None
        expected_authorized = bool(
            runtime_present
            and self.source_data_qualified
            and not self.evidence.failed_gate_names
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA
            or self.evidence.outer_development_conclusion_authorized
            or self.evidence_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.evidence.semantic_receipt_sha256
            or self.fold_inventory_sha256 != self.evidence.fold_inventory_sha256
            or self.source_data_qualified != self.evidence.source_data_qualified
            or self.runtime_evidence_replayed != runtime_present
            or self.outer_development_conclusion_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
                "outer evidence replay authority differs"
            )
        if self.runtime_folds is not None:
            rebuilt = build_massive_adaptive_outer_evidence_v1(self.runtime_folds)
            if (
                rebuilt.semantic_unsigned() != self.evidence.semantic_unsigned()
                or rebuilt.semantic_receipt_sha256
                != self.evidence.semantic_receipt_sha256
            ):
                raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
                    "outer evidence does not replay from runtime folds"
                )
        for value in (
            self.evidence_source_receipt_sha256,
            self.fold_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("outer evidence authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _payload(evidence: MassiveAdaptiveOuterEvidenceV1) -> dict[str, object]:
    return {
        **evidence.semantic_unsigned(),
        "semantic_receipt_sha256": evidence.semantic_receipt_sha256,
    }


def _parse_evidence(raw: bytes) -> MassiveAdaptiveOuterEvidenceV1:
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
            "outer evidence artifact is not canonical JSON"
        )
    payload = dict(cast(Mapping[str, object], value))
    for name in (
        "fold_indices",
        "fold_receipts",
        "passed_gate_names",
        "failed_gate_names",
    ):
        payload[name] = tuple(cast(Sequence[object], payload[name]))
    result = MassiveAdaptiveOuterEvidenceV1(**payload)  # type: ignore[arg-type]
    result.validate()
    return result


def parse_massive_adaptive_outer_evidence_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveOuterEvidenceAuthorityV1:
    """Reopen committed evidence without restoring its fold runtime."""

    evidence = _parse_evidence(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    body = {
        "evidence": evidence,
        "evidence_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "fold_inventory_sha256": evidence.fold_inventory_sha256,
        "source_data_qualified": evidence.source_data_qualified,
        "loaded_source": loaded_source,
        "runtime_folds": None,
        "runtime_evidence_replayed": False,
        "outer_development_conclusion_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveOuterEvidenceAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def materialize_massive_adaptive_outer_evidence_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    folds: Sequence[MassiveAdaptiveOuterCostFoldV1],
    committed_at_ms: int,
) -> MassiveAdaptiveOuterEvidenceAuthorityV1:
    """Publish one outer result and immediately replay all four folds."""

    identifier = _artifact_id(artifact_id)
    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    evidence = build_massive_adaptive_outer_evidence_v1(ordered)
    relative = f"massive-adaptive/outer-evidence-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(evidence))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=evidence.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-OUTER-EVIDENCE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_outer_evidence_authority_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_outer_evidence_authority_v1(
        root=root,
        authority=generic,
        folds=ordered,
    )


def authorize_massive_adaptive_outer_evidence_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveOuterEvidenceAuthorityV1,
    folds: Sequence[MassiveAdaptiveOuterCostFoldV1],
) -> MassiveAdaptiveOuterEvidenceAuthorityV1:
    """Recompute statistical evidence before restoring conclusion authority."""

    parsed = parse_massive_adaptive_outer_evidence_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    ordered = tuple(sorted(folds, key=lambda row: row.fold_index))
    rebuilt = build_massive_adaptive_outer_evidence_v1(ordered)
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or rebuilt.semantic_unsigned() != parsed.evidence.semantic_unsigned()
        or rebuilt.semantic_receipt_sha256 != parsed.evidence.semantic_receipt_sha256
        or tuple(row.semantic_receipt_sha256 for row in ordered)
        != parsed.evidence.fold_receipts
    ):
        raise MassiveAdaptiveOuterEvidenceAuthorityV1Error(
            "committed outer evidence does not replay"
        )
    result = replace(
        parsed,
        runtime_folds=ordered,
        runtime_evidence_replayed=True,
        outer_development_conclusion_authorized=(
            parsed.source_data_qualified and not parsed.evidence.failed_gate_names
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_OUTER_EVIDENCE_AUTHORITY_V1_SCHEMA",
    "MassiveAdaptiveOuterEvidenceAuthorityV1",
    "MassiveAdaptiveOuterEvidenceAuthorityV1Error",
    "authorize_massive_adaptive_outer_evidence_authority_v1",
    "materialize_massive_adaptive_outer_evidence_authority_v1",
    "parse_massive_adaptive_outer_evidence_authority_v1",
]
