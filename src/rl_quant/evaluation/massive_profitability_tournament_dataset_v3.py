"""Create-only, replay-authorized tensors for the embargoed P0 tournament."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_profitability_tournament_inputs_v2 import (
    build_massive_profitability_tournament_dataset_v2,
)
from rl_quant.features.massive_profitability_data_gate_v2 import (
    MassiveProfitabilityDataGateV2,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.features.massive_profitability_phase_plan_v2 import (
    MassiveProfitabilityPhasePlanV2,
    parse_massive_profitability_phase_plan_v2,
)
from rl_quant.features.massive_profitability_targets_v2 import (
    MassiveProfitabilityTargetsV2,
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
    MassiveProfitabilityDateTensorV1,
)
from rl_quant.training.massive_profitability_tournament_v2 import (
    MassiveProfitabilityTournamentDatasetV2,
)

MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA = (
    "rl-quant.massive-profitability-tournament-dataset-v3"
)
MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_DATASET = (
    "massive-profitability-tournament-dataset-v3"
)
MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA,
        "encoding": "canonical-json-hash-inventory",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "gate": "data-gate-v2-semantic-audit-and-snapshot",
        "phase": "reloaded-phase-plan-v2",
        "features": "exact-features-v3-inventory",
        "targets": "exact-targets-v2-inventory",
        "tensors": "package-rebuilt-per-date-source-array-hashes",
        "embargo": "maturation-only",
        "generic_reload": "nonauthorizing",
        "training": "only-after-package-replay",
        "profitability_reporting": False,
        "rl": False,
    }
)


class MassiveProfitabilityTournamentDatasetV3Error(ValueError):
    """The committed tensor inventory differs from its frozen source roots."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityTournamentDatasetV3:
    data_gate_semantic_receipt_sha256: str
    data_gate_audit_receipt_sha256: str
    data_gate_snapshot_sha256: str
    phase_plan_semantic_receipt_sha256: str
    phase_plan_source_receipt_sha256: str
    feature_receipts: tuple[str, ...]
    target_receipts: tuple[str, ...]
    feature_audit_receipts: tuple[str, ...]
    target_audit_receipts: tuple[str, ...]
    tensor_source_array_receipts: tuple[str, ...]
    entry_session_dates: tuple[str, ...]
    maturation_only_session_dates: tuple[str, ...]
    dataset_v2_receipt_sha256: str
    feature_inventory_sha256: str
    target_inventory_sha256: str
    tensor_inventory_sha256: str
    committed_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_dataset: MassiveProfitabilityTournamentDatasetV2 | None
    runtime_data_qualified: bool
    development_training_authorized: bool
    outer_prediction_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    reinforcement_learning_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "data_gate_semantic_receipt_sha256": self.data_gate_semantic_receipt_sha256,
            "data_gate_audit_receipt_sha256": self.data_gate_audit_receipt_sha256,
            "data_gate_snapshot_sha256": self.data_gate_snapshot_sha256,
            "phase_plan_semantic_receipt_sha256": self.phase_plan_semantic_receipt_sha256,
            "phase_plan_source_receipt_sha256": self.phase_plan_source_receipt_sha256,
            "feature_receipts": self.feature_receipts,
            "target_receipts": self.target_receipts,
            "tensor_source_array_receipts": self.tensor_source_array_receipts,
            "entry_session_dates": self.entry_session_dates,
            "maturation_only_session_dates": self.maturation_only_session_dates,
            "dataset_v2_receipt_sha256": self.dataset_v2_receipt_sha256,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "target_inventory_sha256": self.target_inventory_sha256,
            "tensor_inventory_sha256": self.tensor_inventory_sha256,
            "committed_data_qualified": self.committed_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": self.profitability_reporting_authorized,
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "reinforcement_learning_authorized": self.reinforcement_learning_authorized,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "feature_audit_receipts": self.feature_audit_receipts,
            "target_audit_receipts": self.target_audit_receipts,
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
            "audit_receipt_sha256": self.audit_receipt_sha256,
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA
            or not self.feature_receipts
            or len(self.feature_receipts) != len(self.target_receipts)
            or len(self.feature_receipts) != len(self.feature_audit_receipts)
            or len(self.target_receipts) != len(self.target_audit_receipts)
            or len(self.feature_receipts) != len(self.tensor_source_array_receipts)
            or self.entry_session_dates != tuple(sorted(set(self.entry_session_dates)))
            or self.maturation_only_session_dates
            != tuple(sorted(set(self.maturation_only_session_dates)))
            or set(self.entry_session_dates) & set(self.maturation_only_session_dates)
            or len(self.entry_session_dates) + len(self.maturation_only_session_dates)
            != len(self.tensor_source_array_receipts)
            or self.feature_inventory_sha256 != semantic_sha256(self.feature_receipts)
            or self.target_inventory_sha256 != semantic_sha256(self.target_receipts)
            or self.tensor_inventory_sha256
            != semantic_sha256(self.tensor_source_array_receipts)
            or not self.committed_data_qualified
            or not isinstance(self.runtime_data_qualified, bool)
            or self.runtime_data_qualified != (self.runtime_dataset is not None)
            or self.development_training_authorized != self.runtime_data_qualified
            or self.outer_prediction_authorized != self.runtime_data_qualified
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "data_gate_audit_receipt_sha256": self.data_gate_audit_receipt_sha256,
                    "phase_plan_source_receipt_sha256": self.phase_plan_source_receipt_sha256,
                    "feature_audit_receipts": self.feature_audit_receipts,
                    "target_audit_receipts": self.target_audit_receipts,
                }
            )
        ):
            raise MassiveProfitabilityTournamentDatasetV3Error(
                "tournament dataset V3 identity or authorization differs"
            )
        for value in (
            self.data_gate_semantic_receipt_sha256,
            self.data_gate_audit_receipt_sha256,
            self.data_gate_snapshot_sha256,
            self.phase_plan_semantic_receipt_sha256,
            self.phase_plan_source_receipt_sha256,
            *self.feature_receipts,
            *self.target_receipts,
            *self.feature_audit_receipts,
            *self.target_audit_receipts,
            *self.tensor_source_array_receipts,
            self.dataset_v2_receipt_sha256,
            self.feature_inventory_sha256,
            self.target_inventory_sha256,
            self.tensor_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("tournament dataset V3", value)
        if self.runtime_dataset is not None:
            self.runtime_dataset.validate()
            if (
                self.runtime_dataset.dataset_receipt_sha256
                != self.dataset_v2_receipt_sha256
                or self.runtime_dataset.data_gate_semantic_receipt_sha256
                != self.data_gate_semantic_receipt_sha256
                or self.runtime_dataset.phase_plan_semantic_receipt_sha256
                != self.phase_plan_semantic_receipt_sha256
                or self.runtime_dataset.entry_session_dates != self.entry_session_dates
                or self.runtime_dataset.maturation_only_session_dates
                != self.maturation_only_session_dates
                or tuple(row.source_array_sha256 for row in self.runtime_dataset.dates)
                != self.tensor_source_array_receipts
            ):
                raise MassiveProfitabilityTournamentDatasetV3Error(
                    "runtime tensors differ from the committed V3 inventory"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.data_gate_semantic_receipt_sha256
        ):
            raise MassiveProfitabilityTournamentDatasetV3Error(
                "tournament dataset V3 committed source differs"
            )

    def by_date(self) -> dict[str, MassiveProfitabilityDateTensorV1]:
        self.validate()
        if self.runtime_dataset is None:
            raise MassiveProfitabilityTournamentDatasetV3Error(
                "generic dataset V3 reload has no runtime tensors"
            )
        return self.runtime_dataset.by_date()


def _gate_snapshot(data_gate: MassiveProfitabilityDataGateV2) -> str:
    return semantic_sha256(
        {
            "semantic": data_gate.semantic_receipt_sha256,
            "audit": data_gate.audit_receipt_sha256,
            "gated_dates": data_gate.gated_session_dates,
            "features": data_gate.feature_receipts,
            "targets": data_gate.target_receipts,
            "passed": data_gate.data_gate_passed,
        }
    )


def materialize_massive_profitability_tournament_dataset_v3(
    *,
    root: str | Path,
    artifact_id: str,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
    committed_at_ms: int,
) -> MassiveProfitabilityTournamentDatasetV3:
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_targets = tuple(sorted(targets, key=lambda row: row.decision_session_date))
    rebuilt = build_massive_profitability_tournament_dataset_v2(
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=ordered_features,
        targets=ordered_targets,
    )
    semantic: dict[str, object] = {
        "schema": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SCHEMA,
        "data_gate_semantic_receipt_sha256": data_gate.semantic_receipt_sha256,
        "data_gate_audit_receipt_sha256": data_gate.audit_receipt_sha256,
        "data_gate_snapshot_sha256": _gate_snapshot(data_gate),
        "phase_plan_semantic_receipt_sha256": phase_plan.semantic_receipt_sha256,
        "phase_plan_source_receipt_sha256": phase_plan.loaded_source.receipt_sha256,
        "feature_receipts": tuple(
            row.semantic_receipt_sha256 for row in ordered_features
        ),
        "target_receipts": tuple(
            row.semantic_receipt_sha256 for row in ordered_targets
        ),
        "tensor_source_array_receipts": tuple(
            row.source_array_sha256 for row in rebuilt.dates
        ),
        "entry_session_dates": rebuilt.entry_session_dates,
        "maturation_only_session_dates": rebuilt.maturation_only_session_dates,
        "dataset_v2_receipt_sha256": rebuilt.dataset_receipt_sha256,
        "feature_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered_features)
        ),
        "target_inventory_sha256": semantic_sha256(
            tuple(row.semantic_receipt_sha256 for row in ordered_targets)
        ),
        "tensor_inventory_sha256": semantic_sha256(
            tuple(row.source_array_sha256 for row in rebuilt.dates)
        ),
        "committed_data_qualified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }
    semantic_receipt = semantic_sha256(semantic)
    payload = {
        **semantic,
        "feature_audit_receipts": tuple(
            row.audit_receipt_sha256 for row in ordered_features
        ),
        "target_audit_receipts": tuple(
            row.audit_receipt_sha256 for row in ordered_targets
        ),
        "semantic_receipt_sha256": semantic_receipt,
        "audit_receipt_sha256": semantic_sha256(
            {
                "semantic_receipt_sha256": semantic_receipt,
                "data_gate_audit_receipt_sha256": data_gate.audit_receipt_sha256,
                "phase_plan_source_receipt_sha256": phase_plan.loaded_source.receipt_sha256,
                "feature_audit_receipts": tuple(
                    row.audit_receipt_sha256 for row in ordered_features
                ),
                "target_audit_receipts": tuple(
                    row.audit_receipt_sha256 for row in ordered_targets
                ),
            }
        ),
    }
    if not artifact_id or any(
        not (value.isalnum() or value in "-_") for value in artifact_id
    ):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "tournament dataset V3 artifact ID is not path safe"
        )
    relative = f"massive-profitability/tournament-dataset-v3/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_TOURNAMENT_DATASET_V3_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=data_gate.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"P0-TOURNAMENT-DATASET-V3-{artifact_id}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    parsed = parse_massive_profitability_tournament_dataset_v3(
        root=root, loaded_source=loaded
    )
    return authorize_massive_profitability_tournament_dataset_v3(
        root=root,
        dataset=parsed,
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=features,
        targets=targets,
    )


def parse_massive_profitability_tournament_dataset_v3(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityTournamentDatasetV3:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "tournament dataset V3 is not canonical JSON"
        )
    for name in (
        "feature_receipts",
        "target_receipts",
        "feature_audit_receipts",
        "target_audit_receipts",
        "tensor_source_array_receipts",
        "entry_session_dates",
        "maturation_only_session_dates",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveProfitabilityTournamentDatasetV3(
        **payload,
        loaded_source=loaded_source,
        runtime_dataset=None,
        runtime_data_qualified=False,
        development_training_authorized=False,
        outer_prediction_authorized=False,
    )
    result.validate()
    expected = result.canonical_payload()
    if canonical_json_file_bytes(expected) != raw:
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "tournament dataset V3 canonical bytes differ"
        )
    return result


def authorize_massive_profitability_tournament_dataset_v3(
    *,
    root: str | Path,
    dataset: MassiveProfitabilityTournamentDatasetV3,
    data_gate: MassiveProfitabilityDataGateV2,
    phase_plan: MassiveProfitabilityPhasePlanV2,
    features: Sequence[MassiveProfitabilityOriginFeaturesV3],
    targets: Sequence[MassiveProfitabilityTargetsV2],
) -> MassiveProfitabilityTournamentDatasetV3:
    parsed = parse_massive_profitability_tournament_dataset_v3(
        root=root, loaded_source=dataset.loaded_source
    )
    reloaded_phase = parse_massive_profitability_phase_plan_v2(
        root=root, loaded_source=phase_plan.loaded_source
    )
    ordered_features = tuple(
        sorted(features, key=lambda row: row.decision_session_date)
    )
    ordered_targets = tuple(sorted(targets, key=lambda row: row.decision_session_date))
    rebuilt = build_massive_profitability_tournament_dataset_v2(
        data_gate=data_gate,
        phase_plan=phase_plan,
        features=ordered_features,
        targets=ordered_targets,
    )
    if (
        parsed.semantic_receipt_sha256 != dataset.semantic_receipt_sha256
        or not data_gate.data_gate_passed
        or parsed.data_gate_semantic_receipt_sha256 != data_gate.semantic_receipt_sha256
        or parsed.data_gate_audit_receipt_sha256 != data_gate.audit_receipt_sha256
        or parsed.data_gate_snapshot_sha256 != _gate_snapshot(data_gate)
        or reloaded_phase.semantic_receipt_sha256 != phase_plan.semantic_receipt_sha256
        or parsed.phase_plan_semantic_receipt_sha256
        != phase_plan.semantic_receipt_sha256
        or parsed.phase_plan_source_receipt_sha256
        != phase_plan.loaded_source.receipt_sha256
        or parsed.feature_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_features)
        or parsed.target_receipts
        != tuple(row.semantic_receipt_sha256 for row in ordered_targets)
        or parsed.feature_audit_receipts
        != tuple(row.audit_receipt_sha256 for row in ordered_features)
        or parsed.target_audit_receipts
        != tuple(row.audit_receipt_sha256 for row in ordered_targets)
        or parsed.tensor_source_array_receipts
        != tuple(row.source_array_sha256 for row in rebuilt.dates)
        or parsed.entry_session_dates != rebuilt.entry_session_dates
        or parsed.maturation_only_session_dates != rebuilt.maturation_only_session_dates
        or parsed.dataset_v2_receipt_sha256 != rebuilt.dataset_receipt_sha256
    ):
        raise MassiveProfitabilityTournamentDatasetV3Error(
            "dataset V3 does not replay from its frozen roots"
        )
    result = replace(
        parsed,
        runtime_dataset=rebuilt,
        runtime_data_qualified=True,
        development_training_authorized=True,
        outer_prediction_authorized=True,
    )
    result.validate()
    return result


__all__ = [
    "MassiveProfitabilityTournamentDatasetV3",
    "MassiveProfitabilityTournamentDatasetV3Error",
    "authorize_massive_profitability_tournament_dataset_v3",
    "materialize_massive_profitability_tournament_dataset_v3",
    "parse_massive_profitability_tournament_dataset_v3",
]
