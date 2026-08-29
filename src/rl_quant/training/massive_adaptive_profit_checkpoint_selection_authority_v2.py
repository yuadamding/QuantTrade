"""Create-only replay authority for adaptive profit checkpoint selection."""

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
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_v2 import (
    MassiveAdaptiveProfitCheckpointCandidateV2,
    MassiveAdaptiveProfitCheckpointSelectionV2,
    select_massive_adaptive_profit_checkpoint_v2,
)

MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-profit-checkpoint-selection-authority-v2"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_DATASET = (
    "massive-adaptive-profit-checkpoint-selection-authority-v2"
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA
            ),
            "encoding": "canonical-json",
            "publication": "create-only-source-transaction",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SPEC_SHA256 = (
    semantic_sha256(
        {
            "selection": "recomputed-from-source-derived-cost-ladder-candidates",
            "candidate_order": "epoch-ascending",
            "generic_reload": "nonauthorizing",
            "outer_evaluation": False,
            "profitability_reporting": False,
            "lockbox": False,
            "rl": False,
        }
    )
)


class MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(ValueError):
    """A committed checkpoint choice cannot be replayed from its candidates."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
            "checkpoint selection authority artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveProfitCheckpointSelectionAuthorityV2:
    selection: MassiveAdaptiveProfitCheckpointSelectionV2
    selection_source_receipt_sha256: str
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_candidates: tuple[MassiveAdaptiveProfitCheckpointCandidateV2, ...] | None
    runtime_selection_replayed: bool
    development_checkpoint_selection_authorized: bool
    outer_evaluation_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "selection_receipt_sha256": self.selection.semantic_receipt_sha256,
            "selection_source_receipt_sha256": self.selection_source_receipt_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "outer_evaluation_authorized": self.outer_evaluation_authorized,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": (
                self.reinforcement_learning_authorized
            ),
        }

    def validate(self) -> None:
        self.selection.validate()
        self.loaded_source.validate()
        runtime_present = self.runtime_candidates is not None
        if (
            self.schema
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA
            or self.selection.development_checkpoint_selection_authorized
            or self.selection_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.selection.semantic_receipt_sha256
            or self.candidate_inventory_sha256
            != semantic_sha256(self.selection.candidate_receipts)
            or self.source_data_qualified != self.selection.source_data_qualified
            or self.runtime_selection_replayed != runtime_present
            or self.development_checkpoint_selection_authorized
            != (runtime_present and self.source_data_qualified)
            or self.outer_evaluation_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
                "checkpoint selection replay authority differs"
            )
        for value in (
            self.selection_source_receipt_sha256,
            self.candidate_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("checkpoint selection authority", value)
        if self.runtime_candidates is not None:
            rebuilt = select_massive_adaptive_profit_checkpoint_v2(
                self.runtime_candidates
            )
            if (
                rebuilt.semantic_unsigned() != self.selection.semantic_unsigned()
                or rebuilt.semantic_receipt_sha256
                != self.selection.semantic_receipt_sha256
            ):
                raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
                    "checkpoint selection does not replay from runtime candidates"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _selection_payload(
    selection: MassiveAdaptiveProfitCheckpointSelectionV2,
) -> dict[str, object]:
    return {
        **selection.semantic_unsigned(),
        "semantic_receipt_sha256": selection.semantic_receipt_sha256,
    }


def _parse_selection_payload(raw: bytes) -> MassiveAdaptiveProfitCheckpointSelectionV2:
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
            "checkpoint selection artifact is not canonical JSON"
        )
    payload = dict(cast(Mapping[str, object], value))
    for name in (
        "candidate_epoch_indices",
        "candidate_receipts",
        "eligible_epoch_indices",
    ):
        payload[name] = tuple(cast(Sequence[object], payload[name]))
    selection = MassiveAdaptiveProfitCheckpointSelectionV2(**payload)  # type: ignore[arg-type]
    selection.validate()
    return selection


def parse_massive_adaptive_profit_checkpoint_selection_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveProfitCheckpointSelectionAuthorityV2:
    """Reopen committed selection metadata without runtime authorization."""

    selection = _parse_selection_payload(
        read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    )
    body = {
        "selection": selection,
        "selection_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "candidate_inventory_sha256": semantic_sha256(selection.candidate_receipts),
        "source_data_qualified": selection.source_data_qualified,
        "loaded_source": loaded_source,
        "runtime_candidates": None,
        "runtime_selection_replayed": False,
        "development_checkpoint_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA,
    }
    provisional = MassiveAdaptiveProfitCheckpointSelectionAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def materialize_massive_adaptive_profit_checkpoint_selection_authority_v2(
    *,
    root: str | Path,
    artifact_id: str,
    candidates: Sequence[MassiveAdaptiveProfitCheckpointCandidateV2],
    committed_at_ms: int,
) -> MassiveAdaptiveProfitCheckpointSelectionAuthorityV2:
    """Publish one immutable selection and immediately replay it."""

    identifier = _artifact_id(artifact_id)
    ordered = tuple(sorted(candidates, key=lambda row: row.epoch_index))
    selection = select_massive_adaptive_profit_checkpoint_v2(ordered)
    relative = f"massive-adaptive/profit-checkpoint-selection-v2/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_selection_payload(selection))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=selection.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-PROFIT-CHECKPOINT-SELECTION-V2-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=root,
        authority=generic,
        candidates=ordered,
    )


def authorize_massive_adaptive_profit_checkpoint_selection_authority_v2(
    *,
    root: str | Path,
    authority: MassiveAdaptiveProfitCheckpointSelectionAuthorityV2,
    candidates: Sequence[MassiveAdaptiveProfitCheckpointCandidateV2],
) -> MassiveAdaptiveProfitCheckpointSelectionAuthorityV2:
    """Recompute the winner before restoring runtime selection authority."""

    parsed = parse_massive_adaptive_profit_checkpoint_selection_authority_v2(
        root=root, loaded_source=authority.loaded_source
    )
    ordered = tuple(sorted(candidates, key=lambda row: row.epoch_index))
    rebuilt = select_massive_adaptive_profit_checkpoint_v2(ordered)
    if (
        parsed.semantic_receipt_sha256 != authority.semantic_receipt_sha256
        or rebuilt.semantic_unsigned() != parsed.selection.semantic_unsigned()
        or rebuilt.semantic_receipt_sha256
        != parsed.selection.semantic_receipt_sha256
        or tuple(row.semantic_receipt_sha256 for row in ordered)
        != parsed.selection.candidate_receipts
    ):
        raise MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error(
            "committed checkpoint selection does not replay"
        )
    result = replace(
        parsed,
        runtime_candidates=ordered,
        runtime_selection_replayed=True,
        development_checkpoint_selection_authorized=(
            parsed.source_data_qualified
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_PROFIT_CHECKPOINT_SELECTION_AUTHORITY_V2_SCHEMA",
    "MassiveAdaptiveProfitCheckpointSelectionAuthorityV2",
    "MassiveAdaptiveProfitCheckpointSelectionAuthorityV2Error",
    "authorize_massive_adaptive_profit_checkpoint_selection_authority_v2",
    "materialize_massive_adaptive_profit_checkpoint_selection_authority_v2",
    "parse_massive_adaptive_profit_checkpoint_selection_authority_v2",
]
