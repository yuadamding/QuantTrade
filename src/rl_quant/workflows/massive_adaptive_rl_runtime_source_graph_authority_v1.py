"""Replayable runtime-source graph authority for adaptive RL experiments.

The byte-level source bundle proves that the fixed persisted files are present
and unchanged.  This authority is a separate identity for the concrete domain
objects that replayed those files.  Generic loading remains nonauthorizing;
promotion requires the same complete role-bound runtime graph again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import cast

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA,
    MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
    MassivePersistedPartitionManifestV1,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA,
    MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256,
    MassiveAdaptiveForecastCalibrationV2,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SPEC_SHA256,
    MassiveAdaptiveRLFitForecastArchiveV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA,
    MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA,
    MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SPEC_SHA256,
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
    MassiveAdaptiveFillSourceV1,
)
from rl_quant.features.massive_economic_authority_v6 import (
    MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256,
    MassiveProviderEconomicArchiveAuthorityV6,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256,
    MassiveProfitabilityDailyInputAuthorityV1,
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
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA,
    MassiveAdaptiveCausalCheckpointChoiceV1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SPEC_SHA256,
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA,
    MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SPEC_SHA256,
    MassiveAdaptiveWindowPlanV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1,
    MassiveAdaptiveRLRoleBoundSourceAuthorityV1,
    MassiveAdaptiveRLSourceAuthorityProtocol,
    MassiveAdaptiveRLSourceBundleV1,
    MassiveAdaptiveRLSourceBundleV1Error,
    authorize_massive_adaptive_rl_source_bundle_v1,
)


MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-source-graph-authority-v1"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "source_bundle": "exact-byte-level-v1-receipt",
        "roles": "complete-package-owned-role-and-fold-inventory",
        "domain_types": "package-registered-concrete-runtime-types",
        "generic_reload": "persisted-graph-replayed-nonauthorizing",
        "promotion": "complete-concrete-runtime-graph-replay",
        "qualification": "derived-from-concrete-domain-authorities",
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)


class MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(ValueError):
    """The runtime source graph is incomplete, substituted, or unreplayed."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _type_name(value: type[object] | object) -> str:
    runtime_type = value if isinstance(value, type) else type(value)
    return f"{runtime_type.__module__}.{runtime_type.__qualname__}"


def _runtime_receipt(value: object) -> str:
    for name in ("semantic_receipt_sha256", "receipt_sha256"):
        receipt = getattr(value, name, None)
        if receipt is not None:
            return _digest("adaptive RL runtime authority receipt", receipt)
    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
        "adaptive RL runtime authority exposes no root receipt"
    )


_DOMAIN_INVENTORY_ITEM_TYPES: dict[str, type[object]] = {
    "persisted-partition-inventory": MassivePersistedPartitionManifestV1,
    "training-window-inventory": MassiveAdaptiveWindowPlanV1,
    "supervised-checkpoint-inventory": MassiveAdaptiveCausalCheckpointChoiceV1,
    "calibration-inventory": MassiveAdaptiveForecastCalibrationV2,
    "fit-forecast-archive-inventory": MassiveAdaptiveRLFitForecastArchiveV1,
    "decision-root-inventory": MassiveAdaptiveDecisionRootV1,
    "context-origin-inventory": MassiveAdaptiveContextOriginAuthorityV1,
}

_DOMAIN_INVENTORY_ITEM_SCHEMAS = {
    "persisted-partition-inventory": MASSIVE_PERSISTED_PARTITION_MANIFEST_V1_SCHEMA,
    "training-window-inventory": MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SCHEMA,
    "supervised-checkpoint-inventory": (
        MASSIVE_ADAPTIVE_CAUSAL_CHECKPOINT_CHOICE_V1_SCHEMA
    ),
    "calibration-inventory": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SCHEMA,
    "fit-forecast-archive-inventory": (
        MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SCHEMA
    ),
    "decision-root-inventory": MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SCHEMA,
    "context-origin-inventory": MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SCHEMA,
}

_DOMAIN_INVENTORY_ITEM_SPECIFICATIONS: dict[str, str | None] = {
    "persisted-partition-inventory": MASSIVE_PERSISTED_PARTITION_SPEC_SHA256,
    "training-window-inventory": MASSIVE_ADAPTIVE_WINDOW_PLAN_V1_SPEC_SHA256,
    "supervised-checkpoint-inventory": None,
    "calibration-inventory": MASSIVE_ADAPTIVE_FORECAST_CALIBRATION_V2_SPEC_SHA256,
    "fit-forecast-archive-inventory": (
        MASSIVE_ADAPTIVE_RL_FIT_FORECAST_ARCHIVE_V1_SPEC_SHA256
    ),
    "decision-root-inventory": MASSIVE_ADAPTIVE_DECISION_ROOT_V1_SPEC_SHA256,
    "context-origin-inventory": (
        MASSIVE_ADAPTIVE_CONTEXT_ORIGIN_AUTHORITY_V1_SPEC_SHA256
    ),
}


def _authority_is_qualified(*, role: str, authority: object) -> bool:
    required_fields = {
        "daily-input-authority": (
            "source_transport_qualified",
            "daily_input_data_qualified",
        ),
        "fill-source-authority": (
            "source_data_qualified",
            "source_paths_replayed",
        ),
        "split-plan": (
            "candidate_source_data_qualified",
            "source_geometry_replayed",
        ),
    }.get(role, ())
    observed_fields = tuple(
        name
        for name in (
            "source_transport_qualified",
            "source_data_qualified",
            "daily_input_data_qualified",
            "source_paths_replayed",
            "source_geometry_replayed",
            "candidate_source_data_qualified",
        )
        if hasattr(authority, name)
    )
    fields = tuple(sorted(set((*required_fields, *observed_fields))))
    return all(getattr(authority, name, None) is True for name in fields)


def _item_is_qualified(item: object) -> bool:
    observed = tuple(
        getattr(item, name)
        for name in (
            "source_transport_qualified",
            "source_data_qualified",
            "daily_input_data_qualified",
            "source_paths_replayed",
            "source_windows_replayed",
            "runtime_calibration_replayed",
            "runtime_forecasts_replayed",
            "committed_source_data_qualified",
        )
        if hasattr(item, name)
    )
    return not observed or all(value is True for value in observed)


def _item_specification(item: object) -> str | None:
    value = getattr(item, "specification_sha256", None)
    if value is None:
        value = getattr(item, "partition_spec_sha256", None)
    return (
        None if value is None else _digest("source inventory item specification", value)
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLTypedAuthorityInventoryV1:
    """Concrete, role-scoped inventory for source roles containing many objects."""

    role: str
    fold_index: int | None
    runtime_schema: str
    item_type_name: str
    item_schema: str
    item_specification_sha256: str | None
    item_receipts: tuple[str, ...]
    semantic_receipt_sha256: str
    runtime_items: tuple[MassiveAdaptiveRLSourceAuthorityProtocol, ...] | None
    runtime_source_replayed: bool
    source_data_qualified: bool

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "role": self.role,
            "fold_index": self.fold_index,
            "runtime_schema": self.runtime_schema,
            "item_type_name": self.item_type_name,
            "item_schema": self.item_schema,
            "item_specification_sha256": self.item_specification_sha256,
            "item_receipts": self.item_receipts,
        }

    def validate(self) -> None:
        expected_type = _DOMAIN_INVENTORY_ITEM_TYPES.get(self.role)
        spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.get(self.role)
        runtime_present = self.runtime_items is not None
        if (
            expected_type is None
            or spec is None
            or self.runtime_schema != spec.runtime_schema
            or self.item_type_name != _type_name(expected_type)
            or self.item_schema != _DOMAIN_INVENTORY_ITEM_SCHEMAS.get(self.role)
            or self.item_specification_sha256
            != _DOMAIN_INVENTORY_ITEM_SPECIFICATIONS.get(self.role)
            or not self.item_receipts
            or len(set(self.item_receipts)) != len(self.item_receipts)
            or self.item_receipts != tuple(sorted(self.item_receipts))
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
            or self.runtime_source_replayed != runtime_present
            or self.source_data_qualified != runtime_present
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL typed source inventory identity or replay differs"
            )
        if spec.fold_scoped != (self.fold_index is not None) or (
            self.fold_index is not None and self.fold_index not in range(4)
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL typed source inventory fold differs"
            )
        for value in (*self.item_receipts, self.semantic_receipt_sha256):
            _digest("adaptive RL typed source inventory", value)
        if self.item_specification_sha256 is not None:
            _digest(
                "adaptive RL typed source inventory item specification",
                self.item_specification_sha256,
            )
        if self.runtime_items is not None:
            receipts: list[str] = []
            for item in self.runtime_items:
                if type(item) is not expected_type:
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item type differs"
                    )
                item.validate()
                if getattr(item, "schema", None) != self.item_schema:
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item schema differs"
                    )
                if _item_specification(item) != self.item_specification_sha256:
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item specification differs"
                    )
                if (
                    self.fold_index is not None
                    and hasattr(item, "fold_index")
                    and getattr(item, "fold_index") != self.fold_index
                ):
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item fold differs"
                    )
                if not _item_is_qualified(item):
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item is unqualified"
                    )
                receipts.append(_runtime_receipt(item))
            if tuple(sorted(receipts)) != self.item_receipts:
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL typed source inventory items do not replay"
                )


def build_massive_adaptive_rl_typed_authority_inventory_v1(
    *,
    role: str,
    fold_index: int | None,
    items: Sequence[MassiveAdaptiveRLSourceAuthorityProtocol],
) -> MassiveAdaptiveRLTypedAuthorityInventoryV1:
    expected_type = _DOMAIN_INVENTORY_ITEM_TYPES.get(role)
    spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.get(role)
    runtime_items = tuple(items)
    if expected_type is None or spec is None or not runtime_items:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL typed source inventory role or items differ"
        )
    for item in runtime_items:
        if type(item) is not expected_type:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL typed source inventory item type differs"
            )
        item.validate()
        if not _item_is_qualified(item):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL typed source inventory item is unqualified"
            )
    body = {
        "role": role,
        "fold_index": fold_index,
        "runtime_schema": spec.runtime_schema,
        "item_type_name": _type_name(expected_type),
        "item_schema": _DOMAIN_INVENTORY_ITEM_SCHEMAS[role],
        "item_specification_sha256": _DOMAIN_INVENTORY_ITEM_SPECIFICATIONS[role],
        "item_receipts": tuple(
            sorted(_runtime_receipt(item) for item in runtime_items)
        ),
    }
    result = MassiveAdaptiveRLTypedAuthorityInventoryV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_items=runtime_items,
        runtime_source_replayed=True,
        source_data_qualified=True,
    )
    result.validate()
    return result


_DOMAIN_RUNTIME_TYPES: dict[str, type[object]] = {
    "session-authority": MassiveSessionAuthority,
    "condition-authority": MassiveConditionAuthority,
    "persisted-partition-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "identity-authority": PITSecurityUniverseAuthority,
    "economic-event-archive": MassiveProviderEconomicArchiveAuthorityV6,
    "daily-input-authority": MassiveProfitabilityDailyInputAuthorityV1,
    "fill-source-authority": MassiveAdaptiveFillSourceV1,
    "split-plan": MassiveAdaptiveSplitPlanV1,
    "training-window-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "supervised-checkpoint-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "calibration-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "fit-forecast-archive-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "decision-root-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
    "context-origin-inventory": MassiveAdaptiveRLTypedAuthorityInventoryV1,
}

_DIRECT_DOMAIN_SPECIFICATIONS: dict[str, str | None] = {
    "session-authority": None,
    "condition-authority": None,
    "identity-authority": None,
    "economic-event-archive": MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256,
    "daily-input-authority": (
        MASSIVE_PROFITABILITY_DAILY_INPUT_AUTHORITY_V1_SPEC_SHA256
    ),
    "fill-source-authority": MASSIVE_ADAPTIVE_FILL_SOURCE_V1_SPEC_SHA256,
    "split-plan": MASSIVE_ADAPTIVE_SPLIT_PLAN_V1_SPEC_SHA256,
}


def _domain_specification(role: str) -> str:
    expected_type = _DOMAIN_RUNTIME_TYPES[role]
    role_spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[role]
    if role in _DOMAIN_INVENTORY_ITEM_TYPES:
        contract: object = {
            "item_type": _type_name(_DOMAIN_INVENTORY_ITEM_TYPES[role]),
            "item_schema": _DOMAIN_INVENTORY_ITEM_SCHEMAS[role],
            "item_specification_sha256": (_DOMAIN_INVENTORY_ITEM_SPECIFICATIONS[role]),
        }
    else:
        contract = _DIRECT_DOMAIN_SPECIFICATIONS[role]
    return semantic_sha256(
        {
            "role": role,
            "runtime_schema": role_spec.runtime_schema,
            "domain_type": _type_name(expected_type),
            "domain_contract": contract,
        }
    )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLRuntimeSourceGraphRowV1:
    role: str
    fold_index: int | None
    runtime_schema: str
    role_specification_sha256: str
    domain_type_name: str
    domain_specification_sha256: str
    domain_authority_receipt_sha256: str
    source_artifact_receipt_sha256: str
    source_artifact_file_sha256: str
    source_data_qualified: bool
    receipt_sha256: str

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.get(self.role)
        expected_type = _DOMAIN_RUNTIME_TYPES.get(self.role)
        if (
            spec is None
            or expected_type is None
            or spec.fold_scoped != (self.fold_index is not None)
            or self.fold_index is not None
            and self.fold_index not in range(4)
            or self.runtime_schema != spec.runtime_schema
            or self.role_specification_sha256 != spec.specification_sha256
            or self.domain_type_name != _type_name(expected_type)
            or self.domain_specification_sha256 != _domain_specification(self.role)
            or self.domain_authority_receipt_sha256
            != self.source_artifact_receipt_sha256
            or self.source_data_qualified is not True
            or self.receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph row differs"
            )
        for value in (
            self.role_specification_sha256,
            self.domain_specification_sha256,
            self.domain_authority_receipt_sha256,
            self.source_artifact_receipt_sha256,
            self.source_artifact_file_sha256,
            self.receipt_sha256,
        ):
            _digest("adaptive RL runtime source graph row", value)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    base_manifest_receipt_sha256: str
    source_bundle_receipt_sha256: str
    rows: tuple[MassiveAdaptiveRLRuntimeSourceGraphRowV1, ...]
    row_inventory_sha256: str
    global_authority_receipts: tuple[str, ...]
    fold_authority_receipts: tuple[tuple[str, ...], ...]
    role_schema_inventory_sha256: str
    role_specification_inventory_sha256: str
    domain_type_inventory_sha256: str
    domain_specification_inventory_sha256: str
    committed_source_data_qualified: bool
    persisted_graph_replayed: bool
    runtime_graph_replayed: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "persisted_graph_replayed",
                "runtime_graph_replayed",
                "source_data_qualified",
            }
        }

    def validate(self) -> None:
        expected_keys = tuple(
            sorted(
                (
                    (role, fold_index)
                    for role, spec in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.items()
                    for fold_index in (range(4) if spec.fold_scoped else (None,))
                ),
                key=lambda value: (value[1] is not None, value[1] or -1, value[0]),
            )
        )
        keys = tuple((row.role, row.fold_index) for row in self.rows)
        global_rows = tuple(row for row in self.rows if row.fold_index is None)
        folds = tuple(
            tuple(row for row in self.rows if row.fold_index == fold_index)
            for fold_index in range(4)
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA
            or keys != expected_keys
            or len(set(keys)) != len(keys)
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.global_authority_receipts
            != tuple(row.domain_authority_receipt_sha256 for row in global_rows)
            or self.fold_authority_receipts
            != tuple(
                tuple(row.domain_authority_receipt_sha256 for row in fold_rows)
                for fold_rows in folds
            )
            or self.role_schema_inventory_sha256
            != semantic_sha256(
                tuple(
                    (row.role, row.fold_index, row.runtime_schema) for row in self.rows
                )
            )
            or self.role_specification_inventory_sha256
            != semantic_sha256(
                tuple(
                    (row.role, row.fold_index, row.role_specification_sha256)
                    for row in self.rows
                )
            )
            or self.domain_type_inventory_sha256
            != semantic_sha256(
                tuple(
                    (row.role, row.fold_index, row.domain_type_name)
                    for row in self.rows
                )
            )
            or self.domain_specification_inventory_sha256
            != semantic_sha256(
                tuple(
                    (row.role, row.fold_index, row.domain_specification_sha256)
                    for row in self.rows
                )
            )
            or self.committed_source_data_qualified is not True
            or self.runtime_graph_replayed
            and not self.persisted_graph_replayed
            or self.source_data_qualified
            != (self.persisted_graph_replayed and self.runtime_graph_replayed)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph authority differs"
            )
        for row in self.rows:
            row.validate()
        for value in (
            self.manifest_v3_receipt_sha256,
            self.base_manifest_receipt_sha256,
            self.source_bundle_receipt_sha256,
            self.row_inventory_sha256,
            *self.global_authority_receipts,
            *(value for rows in self.fold_authority_receipts for value in rows),
            self.role_schema_inventory_sha256,
            self.role_specification_inventory_sha256,
            self.domain_type_inventory_sha256,
            self.domain_specification_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL runtime source graph authority", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _validate_domain_runtime(
    *,
    key: tuple[str, int | None],
    runtime: MassiveAdaptiveRLSourceAuthorityProtocol,
) -> MassiveAdaptiveRLRoleBoundSourceAuthorityV1:
    if type(runtime) is not MassiveAdaptiveRLRoleBoundSourceAuthorityV1:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source is not role-bound"
        )
    runtime.validate()
    if (runtime.role, runtime.fold_index) != key:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source role or fold differs"
        )
    expected_type = _DOMAIN_RUNTIME_TYPES.get(runtime.role)
    if expected_type is None or type(runtime.authority) is not expected_type:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source concrete domain type differs"
        )
    runtime.authority.validate()
    if isinstance(runtime.authority, MassiveAdaptiveRLTypedAuthorityInventoryV1):
        if (
            runtime.authority.role != runtime.role
            or runtime.authority.fold_index != runtime.fold_index
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime inventory role or fold differs"
            )
    else:
        role_spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[runtime.role]
        domain_schema = getattr(runtime.authority, "schema", None)
        expected_specification = _DIRECT_DOMAIN_SPECIFICATIONS.get(runtime.role)
        domain_specification = getattr(runtime.authority, "specification_sha256", None)
        if domain_schema is not None and domain_schema != role_spec.runtime_schema:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source domain schema differs"
            )
        if (
            expected_specification is not None
            and runtime.role != "economic-event-archive"
            and domain_specification != expected_specification
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source domain specification differs"
            )
        if not _authority_is_qualified(role=runtime.role, authority=runtime.authority):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source domain authority is unqualified"
            )
    return runtime


def _rows(
    *,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> tuple[MassiveAdaptiveRLRuntimeSourceGraphRowV1, ...]:
    artifacts = {
        (artifact.role, artifact.fold_index): artifact
        for artifact in source_bundle.artifacts
    }
    if set(runtime_sources) != set(artifacts):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph inventory is incomplete"
        )
    rows: list[MassiveAdaptiveRLRuntimeSourceGraphRowV1] = []
    for key in sorted(
        artifacts, key=lambda value: (value[1] is not None, value[1] or -1, value[0])
    ):
        runtime = _validate_domain_runtime(key=key, runtime=runtime_sources[key])
        artifact = artifacts[key]
        spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1[runtime.role]
        receipt = _runtime_receipt(runtime.authority)
        if receipt != artifact.semantic_receipt_sha256:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source differs from byte-level source bundle"
            )
        body = {
            "role": runtime.role,
            "fold_index": runtime.fold_index,
            "runtime_schema": spec.runtime_schema,
            "role_specification_sha256": spec.specification_sha256,
            "domain_type_name": _type_name(type(runtime.authority)),
            "domain_specification_sha256": _domain_specification(runtime.role),
            "domain_authority_receipt_sha256": receipt,
            "source_artifact_receipt_sha256": artifact.semantic_receipt_sha256,
            "source_artifact_file_sha256": artifact.file_sha256,
            "source_data_qualified": True,
        }
        row = MassiveAdaptiveRLRuntimeSourceGraphRowV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    return tuple(rows)


def _build_authority(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    rows: tuple[MassiveAdaptiveRLRuntimeSourceGraphRowV1, ...],
    persisted_graph_replayed: bool,
    runtime_graph_replayed: bool,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
    global_rows = tuple(row for row in rows if row.fold_index is None)
    fold_rows = tuple(
        tuple(row for row in rows if row.fold_index == fold_index)
        for fold_index in range(4)
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_receipt_sha256": manifest.base_manifest.semantic_receipt_sha256,
        "source_bundle_receipt_sha256": source_bundle.semantic_receipt_sha256,
        "rows": rows,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in rows)
        ),
        "global_authority_receipts": tuple(
            row.domain_authority_receipt_sha256 for row in global_rows
        ),
        "fold_authority_receipts": tuple(
            tuple(row.domain_authority_receipt_sha256 for row in current_rows)
            for current_rows in fold_rows
        ),
        "role_schema_inventory_sha256": semantic_sha256(
            tuple((row.role, row.fold_index, row.runtime_schema) for row in rows)
        ),
        "role_specification_inventory_sha256": semantic_sha256(
            tuple(
                (row.role, row.fold_index, row.role_specification_sha256)
                for row in rows
            )
        ),
        "domain_type_inventory_sha256": semantic_sha256(
            tuple((row.role, row.fold_index, row.domain_type_name) for row in rows)
        ),
        "domain_specification_inventory_sha256": semantic_sha256(
            tuple(
                (row.role, row.fold_index, row.domain_specification_sha256)
                for row in rows
            )
        ),
        "committed_source_data_qualified": True,
        "persisted_graph_replayed": persisted_graph_replayed,
        "runtime_graph_replayed": runtime_graph_replayed,
        "source_data_qualified": persisted_graph_replayed and runtime_graph_replayed,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def runtime_source_graph_authority_path_v1(
    *, source_root: str | Path, experiment_id: str
) -> Path:
    return (
        Path(source_root)
        / "adaptive-rl"
        / "runtime-source-graph-authority-v1"
        / f"{experiment_id}.json"
    )


def materialize_massive_adaptive_rl_runtime_source_graph_authority_v1(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
    manifest.validate()
    source_bundle.validate()
    if (
        source_bundle.experiment_id != manifest.experiment_id
        or source_bundle.manifest_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or not source_bundle.persisted_source_replayed
        or not source_bundle.runtime_source_replayed
        or not source_bundle.source_data_qualified
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph source bundle is not authorized"
        )
    rows = _rows(source_bundle=source_bundle, runtime_sources=runtime_sources)
    result = _build_authority(
        manifest=manifest,
        source_bundle=source_bundle,
        rows=rows,
        persisted_graph_replayed=False,
        runtime_graph_replayed=False,
    )
    output = runtime_source_graph_authority_path_v1(
        source_root=source_root, experiment_id=manifest.experiment_id
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(result)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority is create-only"
        ) from error
    return result


def _parse_row(value: object) -> MassiveAdaptiveRLRuntimeSourceGraphRowV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph row is malformed"
        )
    result = MassiveAdaptiveRLRuntimeSourceGraphRowV1(**dict(value))
    result.validate()
    return result


def load_massive_adaptive_rl_runtime_source_graph_authority_v1(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
    manifest.validate()
    source_bundle.validate()
    path = runtime_source_graph_authority_path_v1(
        source_root=source_root, experiment_id=manifest.experiment_id
    )
    if path.is_symlink() or not path.is_file():
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority is absent or not regular"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority cannot be decoded"
        ) from error
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority is not canonical JSON"
        )
    payload = dict(value)
    rows = payload.get("rows")
    fold_receipts = payload.get("fold_authority_receipts")
    if not isinstance(rows, list) or not isinstance(fold_receipts, list):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph inventories are malformed"
        )
    payload["rows"] = tuple(_parse_row(row) for row in rows)
    payload["global_authority_receipts"] = tuple(
        cast(list[str], payload["global_authority_receipts"])
    )
    payload["fold_authority_receipts"] = tuple(
        tuple(cast(list[str], current)) for current in fold_receipts
    )
    committed = MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1(**payload)
    committed.validate()
    if (
        committed.experiment_id != manifest.experiment_id
        or committed.manifest_v3_receipt_sha256 != manifest.semantic_receipt_sha256
        or committed.base_manifest_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
        or committed.source_bundle_receipt_sha256
        != source_bundle.semantic_receipt_sha256
        or not source_bundle.persisted_source_replayed
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority lineage differs"
        )
    result = replace(
        committed,
        persisted_graph_replayed=True,
        runtime_graph_replayed=False,
        source_data_qualified=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
    *,
    authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
    authority.validate()
    source_bundle.validate()
    try:
        authorized_bundle = authorize_massive_adaptive_rl_source_bundle_v1(
            source_bundle=source_bundle,
            runtime_sources=runtime_sources,
        )
    except MassiveAdaptiveRLSourceBundleV1Error as error:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph source bundle does not replay"
        ) from error
    if (
        not authority.persisted_graph_replayed
        or authority.source_bundle_receipt_sha256
        != authorized_bundle.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph is not persisted or source-bound"
        )
    replayed_rows = _rows(
        source_bundle=authorized_bundle,
        runtime_sources=runtime_sources,
    )
    if replayed_rows != authority.rows:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph rows do not replay"
        )
    result = replace(
        authority,
        runtime_graph_replayed=True,
        source_data_qualified=True,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1",
    "MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error",
    "MassiveAdaptiveRLRuntimeSourceGraphRowV1",
    "MassiveAdaptiveRLTypedAuthorityInventoryV1",
    "authorize_massive_adaptive_rl_runtime_source_graph_authority_v1",
    "build_massive_adaptive_rl_typed_authority_inventory_v1",
    "load_massive_adaptive_rl_runtime_source_graph_authority_v1",
    "materialize_massive_adaptive_rl_runtime_source_graph_authority_v1",
    "runtime_source_graph_authority_path_v1",
]
