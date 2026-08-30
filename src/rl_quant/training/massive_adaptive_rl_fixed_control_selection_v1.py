"""Training-only selection of static adaptive compiler controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.rl.massive_adaptive_rl_action_v1 import MassiveAdaptiveRLActionV1
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyTraceV1,
)


MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_CANDIDATE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-candidate-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-selection-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-fixed-control-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-fixed-control-selection-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SOURCE_SHA256 = (
    file_sha256(Path(__file__))
)
MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": (
                MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SCHEMA
            ),
            "payload": "canonical-json-selection-and-training-candidates",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLFixedControlSelectionV1Error(ValueError):
    """Static control candidates or their training-only selection differ."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlCandidateV1:
    fold_index: int
    control_id: str
    action_receipt_sha256: str
    training_trace_receipt_sha256: str
    training_incremental_log_wealth: float
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_CANDIDATE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_CANDIDATE_V1_SCHEMA
            or self.fold_index < 0
            or not self.control_id
            or self.control_id != self.control_id.strip()
            or not math.isfinite(self.training_incremental_log_wealth)
            or not isinstance(self.source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlSelectionV1Error(
                "adaptive RL fixed-control candidate differs"
            )
        for value in (
            self.action_receipt_sha256,
            self.training_trace_receipt_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fixed-control candidate", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_fixed_control_candidate_v1(
    *,
    fold_index: int,
    control_id: str,
    action: MassiveAdaptiveRLActionV1,
    training_trace: MassiveAdaptiveRLPolicyTraceV1,
) -> MassiveAdaptiveRLFixedControlCandidateV1:
    """Bind one constant action to its complete training-only economic trace."""

    action.validate()
    training_trace.validate()
    if (
        fold_index != training_trace.fold_index
        or training_trace.evaluation_role != "training_control"
        or training_trace.transaction_cost_basis_points != 20.0
        or training_trace.frozen_targets_replayed
    ):
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "fixed control must use one primary-cost training-only trace"
        )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_CANDIDATE_V1_SCHEMA,
        "fold_index": fold_index,
        "control_id": control_id,
        "action_receipt_sha256": action.semantic_receipt_sha256,
        "training_trace_receipt_sha256": training_trace.semantic_receipt_sha256,
        "training_incremental_log_wealth": (
            training_trace.cumulative_incremental_rl_log_return
        ),
        "source_data_qualified": training_trace.source_data_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLFixedControlCandidateV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlSelectionV1:
    fold_index: int
    selected_control_id: str
    selected_action_receipt_sha256: str
    selected_candidate_receipt_sha256: str
    selected_training_trace_receipt_sha256: str
    selected_training_incremental_log_wealth: float
    candidate_inventory_sha256: str
    candidate_count: int
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_V1_SCHEMA
            or self.fold_index < 0
            or not self.selected_control_id
            or self.candidate_count <= 0
            or not math.isfinite(self.selected_training_incremental_log_wealth)
            or not isinstance(self.source_data_qualified, bool)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlSelectionV1Error(
                "adaptive RL fixed-control selection differs"
            )
        for value in (
            self.selected_action_receipt_sha256,
            self.selected_candidate_receipt_sha256,
            self.selected_training_trace_receipt_sha256,
            self.candidate_inventory_sha256,
            self.protocol_receipt_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL fixed-control selection", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def select_massive_adaptive_rl_fixed_control_v1(
    candidates: Sequence[MassiveAdaptiveRLFixedControlCandidateV1],
) -> MassiveAdaptiveRLFixedControlSelectionV1:
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    if not ordered:
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "adaptive RL fixed-control candidates are absent"
        )
    for candidate in ordered:
        candidate.validate()
    if (
        len({candidate.fold_index for candidate in ordered}) != 1
        or len({candidate.control_id for candidate in ordered}) != len(ordered)
        or len({candidate.action_receipt_sha256 for candidate in ordered})
        != len(ordered)
    ):
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "adaptive RL fixed controls span folds or are duplicated"
        )
    selected = max(
        ordered,
        key=lambda value: (
            value.training_incremental_log_wealth,
            value.control_id,
        ),
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_V1_SCHEMA,
        "fold_index": selected.fold_index,
        "selected_control_id": selected.control_id,
        "selected_action_receipt_sha256": selected.action_receipt_sha256,
        "selected_candidate_receipt_sha256": selected.semantic_receipt_sha256,
        "selected_training_trace_receipt_sha256": (
            selected.training_trace_receipt_sha256
        ),
        "selected_training_incremental_log_wealth": (
            selected.training_incremental_log_wealth
        ),
        "candidate_inventory_sha256": semantic_sha256(
            tuple(candidate.semantic_receipt_sha256 for candidate in ordered)
        ),
        "candidate_count": len(ordered),
        "source_data_qualified": all(
            candidate.source_data_qualified for candidate in ordered
        ),
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLFixedControlSelectionV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFixedControlSelectionAuthorityV1:
    selection_receipt_sha256: str
    candidate_inventory_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_selection: MassiveAdaptiveRLFixedControlSelectionV1 | None
    runtime_candidates: tuple[MassiveAdaptiveRLFixedControlCandidateV1, ...] | None
    runtime_selection_replayed: bool
    development_control_selection_authorized: bool
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    schema: str = MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
        }

    def validate(self) -> None:
        runtime = self.runtime_selection is not None and self.runtime_candidates is not None
        expected = runtime and self.source_data_qualified
        self.loaded_source.validate()
        if (
            self.schema
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SCHEMA
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.selection_receipt_sha256
            or self.runtime_selection_replayed != runtime
            or self.development_control_selection_authorized != expected
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFixedControlSelectionV1Error(
                "adaptive RL fixed-control selection authority differs"
            )
        if runtime:
            assert self.runtime_selection is not None
            assert self.runtime_candidates is not None
            self.runtime_selection.validate()
            for candidate in self.runtime_candidates:
                candidate.validate()
            if (
                self.runtime_selection.semantic_receipt_sha256
                != self.selection_receipt_sha256
                or self.runtime_selection.candidate_inventory_sha256
                != self.candidate_inventory_sha256
            ):
                raise MassiveAdaptiveRLFixedControlSelectionV1Error(
                    "adaptive RL runtime fixed-control selection differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _payload(
    selection: MassiveAdaptiveRLFixedControlSelectionV1,
    candidates: tuple[MassiveAdaptiveRLFixedControlCandidateV1, ...],
) -> dict[str, object]:
    return {
        "selection": asdict(selection),
        "candidates": tuple(asdict(candidate) for candidate in candidates),
    }


def _load(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> tuple[
    MassiveAdaptiveRLFixedControlSelectionV1,
    tuple[MassiveAdaptiveRLFixedControlCandidateV1, ...],
]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "adaptive RL fixed-control selection is not canonical JSON"
        )
    selection = MassiveAdaptiveRLFixedControlSelectionV1(**dict(value["selection"]))
    candidates = tuple(
        MassiveAdaptiveRLFixedControlCandidateV1(**dict(candidate))
        for candidate in value["candidates"]
    )
    selection.validate()
    for candidate in candidates:
        candidate.validate()
    return selection, candidates


def parse_massive_adaptive_rl_fixed_control_selection_authority_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFixedControlSelectionAuthorityV1:
    selection, _ = _load(root=root, loaded_source=loaded_source)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SCHEMA,
        "selection_receipt_sha256": selection.semantic_receipt_sha256,
        "candidate_inventory_sha256": selection.candidate_inventory_sha256,
        "source_data_qualified": selection.source_data_qualified,
        "profitability_reporting_authorized": False,
        "outer_evaluation_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    result = MassiveAdaptiveRLFixedControlSelectionAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        loaded_source=loaded_source,
        runtime_selection=None,
        runtime_candidates=None,
        runtime_selection_replayed=False,
        development_control_selection_authorized=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_fixed_control_selection_authority_v1(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLFixedControlSelectionAuthorityV1,
    candidates: Sequence[MassiveAdaptiveRLFixedControlCandidateV1],
) -> MassiveAdaptiveRLFixedControlSelectionAuthorityV1:
    parsed = parse_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=root, loaded_source=authority.loaded_source
    )
    committed_selection, committed_candidates = _load(
        root=root, loaded_source=authority.loaded_source
    )
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    rebuilt = select_massive_adaptive_rl_fixed_control_v1(ordered)
    if committed_candidates != ordered or committed_selection != rebuilt:
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "adaptive RL fixed-control selection does not replay"
        )
    result = MassiveAdaptiveRLFixedControlSelectionAuthorityV1(
        **parsed.semantic_unsigned(),  # type: ignore[arg-type]
        semantic_receipt_sha256=parsed.semantic_receipt_sha256,
        loaded_source=parsed.loaded_source,
        runtime_selection=rebuilt,
        runtime_candidates=ordered,
        runtime_selection_replayed=True,
        development_control_selection_authorized=rebuilt.source_data_qualified,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_fixed_control_selection_authority_v1(
    *,
    root: str | Path,
    artifact_id: str,
    candidates: Sequence[MassiveAdaptiveRLFixedControlCandidateV1],
    committed_at_ms: int,
) -> MassiveAdaptiveRLFixedControlSelectionAuthorityV1:
    if not artifact_id or any(
        not (character.isalnum() or character in "-_") for character in artifact_id
    ):
        raise MassiveAdaptiveRLFixedControlSelectionV1Error(
            "adaptive RL fixed-control artifact ID is not path safe"
        )
    ordered = tuple(sorted(candidates, key=lambda value: value.semantic_receipt_sha256))
    selection = select_massive_adaptive_rl_fixed_control_v1(ordered)
    relative = f"massive-adaptive/rl-fixed-control-selection-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(selection, ordered))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_FIXED_CONTROL_SELECTION_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=selection.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-FIXED-CONTROL-V1-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    generic = parse_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=root, loaded_source=loaded
    )
    return authorize_massive_adaptive_rl_fixed_control_selection_authority_v1(
        root=root,
        authority=generic,
        candidates=ordered,
    )


__all__ = [
    "MassiveAdaptiveRLFixedControlCandidateV1",
    "MassiveAdaptiveRLFixedControlSelectionAuthorityV1",
    "MassiveAdaptiveRLFixedControlSelectionV1",
    "MassiveAdaptiveRLFixedControlSelectionV1Error",
    "authorize_massive_adaptive_rl_fixed_control_selection_authority_v1",
    "build_massive_adaptive_rl_fixed_control_candidate_v1",
    "materialize_massive_adaptive_rl_fixed_control_selection_authority_v1",
    "parse_massive_adaptive_rl_fixed_control_selection_authority_v1",
    "select_massive_adaptive_rl_fixed_control_v1",
]
