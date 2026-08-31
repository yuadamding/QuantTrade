"""Replayable runtime-source graph authority for adaptive RL experiments.

The byte-level source bundle proves that the fixed persisted files are present
and unchanged.  This authority is a separate identity for the concrete domain
objects that replayed those files.  Generic loading remains nonauthorizing;
promotion requires the same complete role-bound runtime graph again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import inspect
import json
import os
from pathlib import Path
import tempfile
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
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-source-graph-replay-witness-v1"
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
        "promotion": "complete-concrete-runtime-graph-witness-retained",
        "qualification": "explicit-role-specific-domain-authority-contracts",
        "implementation_identity": "every-domain-type-source-sha256",
        "inventory_coverage": "logical-keys-and-exact-rl-fit-prefix",
        "graph_edges": "cross-authority-lineage-reconciled",
        "publication": "fsync-atomic-create-only-no-symlink-ancestors",
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


def _type_implementation_source_sha256(value: type[object]) -> str:
    source_path = inspect.getsourcefile(value)
    if source_path is None:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime authority implementation source is absent"
        )
    return file_sha256(source_path)


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
    required_fields: tuple[str, ...] | None = {
        "session-authority": (),
        "condition-authority": (),
        "identity-authority": (),
        "economic-event-archive": (),
        "persisted-partition-inventory": ("source_data_qualified",),
        "training-window-inventory": ("source_data_qualified",),
        "supervised-checkpoint-inventory": ("source_data_qualified",),
        "calibration-inventory": ("source_data_qualified",),
        "fit-forecast-archive-inventory": ("source_data_qualified",),
        "decision-root-inventory": ("source_data_qualified",),
        "context-origin-inventory": ("source_data_qualified",),
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
    }.get(role)
    return required_fields is not None and all(
        getattr(authority, name, None) is True for name in required_fields
    )


_INVENTORY_QUALIFICATION_FIELDS = {
    "persisted-partition-inventory": (),
    "training-window-inventory": ("source_windows_replayed",),
    "supervised-checkpoint-inventory": ("source_data_qualified",),
    "calibration-inventory": (
        "source_data_qualified",
        "runtime_calibration_replayed",
    ),
    "fit-forecast-archive-inventory": (
        "committed_source_data_qualified",
        "runtime_forecasts_replayed",
    ),
    "decision-root-inventory": (
        "source_data_qualified",
        "source_paths_replayed",
    ),
    "context-origin-inventory": (
        "source_data_qualified",
        "source_paths_replayed",
    ),
}


def _item_is_qualified(*, role: str, item: object) -> bool:
    fields = _INVENTORY_QUALIFICATION_FIELDS.get(role)
    if fields is None:
        return False
    return all(getattr(item, name, None) is True for name in fields)


def _item_logical_key(*, role: str, item: object) -> str:
    if role == "persisted-partition-inventory":
        value: object = getattr(item, "source_session_date", None)
    elif role == "training-window-inventory":
        rows = getattr(item, "rows", ())
        value = (
            getattr(item, "fold_index", None),
            getattr(item, "split_role", None),
            getattr(rows[0], "origin_session_date", None) if rows else None,
            getattr(rows[-1], "origin_session_date", None) if rows else None,
        )
    elif role == "supervised-checkpoint-inventory":
        value = (
            getattr(item, "fold_index", None),
            getattr(item, "selection_cutoff_session_date", None),
            getattr(item, "training_window_plan_receipt_sha256", None),
        )
    elif role == "calibration-inventory":
        value = (
            getattr(item, "fold_index", None),
            getattr(item, "checkpoint_receipt_sha256", None),
            getattr(item, "calibration_fit_stop_session_date", None),
        )
    elif role == "fit-forecast-archive-inventory":
        value = (
            getattr(item, "outer_fold_index", None),
            getattr(item, "source_fold_index", None),
            getattr(item, "block_index", None),
        )
    elif role in {"decision-root-inventory", "context-origin-inventory"}:
        value = getattr(item, "decision_session_date", None)
    else:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL typed source inventory has no logical-key contract"
        )
    if (
        value is None
        or isinstance(value, tuple)
        and any(part is None for part in value)
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL typed source inventory logical key is absent"
        )
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


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
    item_implementation_source_sha256: str
    item_schema: str
    item_specification_sha256: str | None
    item_bindings: tuple[tuple[str, str], ...]
    item_logical_keys: tuple[str, ...]
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
            "item_implementation_source_sha256": (
                self.item_implementation_source_sha256
            ),
            "item_schema": self.item_schema,
            "item_specification_sha256": self.item_specification_sha256,
            "item_bindings": self.item_bindings,
            "item_logical_keys": self.item_logical_keys,
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
            or self.item_implementation_source_sha256
            != _type_implementation_source_sha256(expected_type)
            or self.item_schema != _DOMAIN_INVENTORY_ITEM_SCHEMAS.get(self.role)
            or self.item_specification_sha256
            != _DOMAIN_INVENTORY_ITEM_SPECIFICATIONS.get(self.role)
            or not self.item_bindings
            or self.item_bindings
            != tuple(sorted(self.item_bindings, key=lambda value: value[0]))
            or tuple(binding[0] for binding in self.item_bindings)
            != self.item_logical_keys
            or tuple(sorted(binding[1] for binding in self.item_bindings))
            != self.item_receipts
            or not self.item_logical_keys
            or self.item_logical_keys != tuple(sorted(set(self.item_logical_keys)))
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
        _digest(
            "adaptive RL typed source inventory implementation",
            self.item_implementation_source_sha256,
        )
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
                if not _item_is_qualified(role=self.role, item=item):
                    raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                        "adaptive RL typed source inventory item is unqualified"
                    )
                receipts.append(_runtime_receipt(item))
            if (
                tuple(
                    sorted(
                        (
                            _item_logical_key(role=self.role, item=item),
                            _runtime_receipt(item),
                        )
                        for item in self.runtime_items
                    )
                )
                != self.item_bindings
            ):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL typed source inventory key bindings do not replay"
                )
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
        if not _item_is_qualified(role=role, item=item):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL typed source inventory item is unqualified"
            )
    item_bindings = tuple(
        sorted(
            (
                _item_logical_key(role=role, item=item),
                _runtime_receipt(item),
            )
            for item in runtime_items
        )
    )
    body = {
        "role": role,
        "fold_index": fold_index,
        "runtime_schema": spec.runtime_schema,
        "item_type_name": _type_name(expected_type),
        "item_implementation_source_sha256": (
            _type_implementation_source_sha256(expected_type)
        ),
        "item_schema": _DOMAIN_INVENTORY_ITEM_SCHEMAS[role],
        "item_specification_sha256": _DOMAIN_INVENTORY_ITEM_SPECIFICATIONS[role],
        "item_bindings": item_bindings,
        "item_logical_keys": tuple(binding[0] for binding in item_bindings),
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
            "item_implementation_source_sha256": (
                _type_implementation_source_sha256(_DOMAIN_INVENTORY_ITEM_TYPES[role])
            ),
            "item_schema": _DOMAIN_INVENTORY_ITEM_SCHEMAS[role],
            "item_specification_sha256": (_DOMAIN_INVENTORY_ITEM_SPECIFICATIONS[role]),
        }
    else:
        contract = {
            "specification_sha256": _DIRECT_DOMAIN_SPECIFICATIONS[role],
            "implementation_source_sha256": (
                _type_implementation_source_sha256(expected_type)
            ),
        }
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
    logical_coverage_inventory_sha256: str
    graph_edge_inventory_sha256: str
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
    _runtime_source_bundle: MassiveAdaptiveRLSourceBundleV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _runtime_sources: tuple[MassiveAdaptiveRLRoleBoundSourceAuthorityV1, ...] | None = (
        field(default=None, repr=False, compare=False)
    )

    @property
    def runtime_authority_receipt_sha256(self) -> str | None:
        self.validate()
        if self._runtime_source_bundle is None or self._runtime_sources is None:
            return None
        return semantic_sha256(
            {
                "schema": (
                    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V1_SCHEMA
                ),
                "persisted_graph_receipt_sha256": self.semantic_receipt_sha256,
                "source_bundle_receipt_sha256": (
                    self._runtime_source_bundle.semantic_receipt_sha256
                ),
                "runtime_authority_receipts": tuple(
                    (
                        runtime.role,
                        runtime.fold_index,
                        _runtime_receipt(runtime.authority),
                    )
                    for runtime in self._runtime_sources
                ),
                "logical_coverage_inventory_sha256": (
                    self.logical_coverage_inventory_sha256
                ),
                "graph_edge_inventory_sha256": self.graph_edge_inventory_sha256,
            }
        )

    def runtime_authority(
        self, *, role: str, fold_index: int | None
    ) -> MassiveAdaptiveRLSourceAuthorityProtocol:
        """Return one concrete witness only after the complete graph replays."""

        self.validate()
        if self._runtime_sources is None:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph has no concrete replay witness"
            )
        matches = tuple(
            runtime.authority
            for runtime in self._runtime_sources
            if runtime.role == role and runtime.fold_index == fold_index
        )
        if len(matches) != 1:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph role is absent or duplicated"
            )
        return matches[0]

    def semantic_unsigned(self) -> dict[str, object]:
        payload = asdict(
            replace(
                self,
                _runtime_source_bundle=None,
                _runtime_sources=None,
            )
        )
        payload.pop("_runtime_source_bundle")
        payload.pop("_runtime_sources")
        return {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "semantic_receipt_sha256",
                "persisted_graph_replayed",
                "runtime_graph_replayed",
                "source_data_qualified",
            }
        }

    def validate(self) -> None:
        runtime_present = (
            self._runtime_source_bundle is not None
            and self._runtime_sources is not None
        )
        partial_runtime_witness = (self._runtime_source_bundle is None) != (
            self._runtime_sources is None
        )
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
            or not isinstance(self.logical_coverage_inventory_sha256, str)
            or not isinstance(self.graph_edge_inventory_sha256, str)
            or self.committed_source_data_qualified is not True
            or partial_runtime_witness
            or self.runtime_graph_replayed != runtime_present
            or runtime_present
            and not self.persisted_graph_replayed
            or self.source_data_qualified
            != (self.persisted_graph_replayed and runtime_present)
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
        if runtime_present:
            assert self._runtime_source_bundle is not None
            assert self._runtime_sources is not None
            self._runtime_source_bundle.validate()
            runtime_sources = {
                (runtime.role, runtime.fold_index): runtime
                for runtime in self._runtime_sources
            }
            if len(runtime_sources) != len(self._runtime_sources):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL runtime source graph witness contains duplicate roles"
                )
            if (
                not self._runtime_source_bundle.runtime_source_replayed
                or not self._runtime_source_bundle.source_data_qualified
                or self._runtime_source_bundle.semantic_receipt_sha256
                != self.source_bundle_receipt_sha256
                or _rows(
                    source_bundle=self._runtime_source_bundle,
                    runtime_sources=runtime_sources,
                )
                != self.rows
            ):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL runtime source graph witness does not replay"
                )
            coverage, edges = _validate_runtime_graph_contract(
                runtime_sources=runtime_sources
            )
            if self.logical_coverage_inventory_sha256 != semantic_sha256(
                coverage
            ) or self.graph_edge_inventory_sha256 != semantic_sha256(edges):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL runtime graph coverage or edges do not replay"
                )
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
            self.logical_coverage_inventory_sha256,
            self.graph_edge_inventory_sha256,
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


def _inventory_items(
    *,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
    role: str,
    fold_index: int | None,
) -> tuple[object, ...]:
    bound = runtime_sources[(role, fold_index)]
    if not isinstance(
        bound, MassiveAdaptiveRLRoleBoundSourceAuthorityV1
    ) or not isinstance(bound.authority, MassiveAdaptiveRLTypedAuthorityInventoryV1):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime graph inventory witness is absent"
        )
    if bound.authority.runtime_items is None:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime graph inventory was not replayed"
        )
    return tuple(bound.authority.runtime_items)


def _direct_authority(
    *,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
    role: str,
) -> object:
    bound = runtime_sources[(role, None)]
    if not isinstance(bound, MassiveAdaptiveRLRoleBoundSourceAuthorityV1):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime graph direct witness is absent"
        )
    return bound.authority


def _edge(
    edges: list[tuple[str, str, str]],
    *,
    name: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected or not isinstance(observed, str):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            f"adaptive RL runtime graph edge differs: {name}"
        )
    edges.append((name, observed, str(expected)))


def _validate_runtime_graph_contract(
    *,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> tuple[tuple[object, ...], tuple[tuple[str, str, str], ...]]:
    """Validate exact RL-fit coverage and the load-bearing source graph edges."""

    split_plan = _direct_authority(
        runtime_sources=runtime_sources,
        role="split-plan",
    )
    if type(split_plan) is not MassiveAdaptiveSplitPlanV1:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime graph requires the concrete split-plan authority"
        )

    session = cast(
        MassiveSessionAuthority,
        _direct_authority(runtime_sources=runtime_sources, role="session-authority"),
    )
    condition = cast(
        MassiveConditionAuthority,
        _direct_authority(runtime_sources=runtime_sources, role="condition-authority"),
    )
    identity = cast(
        PITSecurityUniverseAuthority,
        _direct_authority(runtime_sources=runtime_sources, role="identity-authority"),
    )
    economic = cast(
        MassiveProviderEconomicArchiveAuthorityV6,
        _direct_authority(
            runtime_sources=runtime_sources,
            role="economic-event-archive",
        ),
    )
    daily = cast(
        MassiveProfitabilityDailyInputAuthorityV1,
        _direct_authority(
            runtime_sources=runtime_sources, role="daily-input-authority"
        ),
    )
    fills = cast(
        MassiveAdaptiveFillSourceV1,
        _direct_authority(
            runtime_sources=runtime_sources, role="fill-source-authority"
        ),
    )

    edges: list[tuple[str, str, str]] = []
    _edge(
        edges,
        name="split-plan/session-authority",
        observed=split_plan.session_authority_receipt_sha256,
        expected=session.receipt_sha256,
    )
    _edge(
        edges,
        name="economic-event-archive/identity-authority",
        observed=economic.identity_authority_receipt_sha256,
        expected=identity.receipt_sha256,
    )
    _edge(
        edges,
        name="daily-input/session-authority",
        observed=daily.session_authority_receipt_sha256,
        expected=session.receipt_sha256,
    )
    _edge(
        edges,
        name="daily-input/condition-authority",
        observed=daily.condition_authority_receipt_sha256,
        expected=condition.receipt_sha256,
    )
    _edge(
        edges,
        name="fill-source/daily-input",
        observed=fills.daily_input_authority_semantic_receipt_sha256,
        expected=daily.semantic_receipt_sha256,
    )
    _edge(
        edges,
        name="fill-source/session-authority",
        observed=fills.session_authority_receipt_sha256,
        expected=session.receipt_sha256,
    )
    _edge(
        edges,
        name="fill-source/condition-authority",
        observed=fills.condition_authority_receipt_sha256,
        expected=condition.receipt_sha256,
    )

    persisted_partitions = _inventory_items(
        runtime_sources=runtime_sources,
        role="persisted-partition-inventory",
        fold_index=None,
    )
    partition_dates = tuple(
        getattr(partition, "source_session_date") for partition in persisted_partitions
    )
    daily_dates = tuple(row.source_session_date for row in daily.sessions)
    if partition_dates != daily_dates:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL persisted partitions do not cover the daily-input sessions"
        )
    for partition in persisted_partitions:
        _edge(
            edges,
            name=(
                "persisted-partition/identity-authority/"
                f"{getattr(partition, 'source_session_date', '')}"
            ),
            observed=getattr(partition, "identity_authority_receipt_sha256", None),
            expected=identity.receipt_sha256,
        )

    coverage: list[object] = [
        (
            "persisted-partition-session-dates",
            partition_dates,
        )
    ]
    for fold_index in range(4):
        fold = split_plan.outer_folds[fold_index]
        expected_dates = fold.fit_session_dates[-126 * (fold_index + 1) :]
        archives = cast(
            tuple[MassiveAdaptiveRLFitForecastArchiveV1, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="fit-forecast-archive-inventory",
                fold_index=fold_index,
            ),
        )
        ordered_archives = tuple(sorted(archives, key=lambda row: row.block_index))
        block_sizes = {archive.block_sessions for archive in ordered_archives}
        if len(block_sizes) != 1:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL fit forecast block size differs within a fold"
            )
        block_sessions = next(iter(block_sizes))
        expected_block_count = len(expected_dates) // block_sessions
        if (
            tuple(archive.block_index for archive in ordered_archives)
            != tuple(range(expected_block_count))
            or any(
                archive.outer_fold_index != fold_index for archive in ordered_archives
            )
            or any(
                archive.source_fold_index
                != archive.block_index // (126 // block_sessions)
                for archive in ordered_archives
            )
            or tuple(
                date
                for archive in ordered_archives
                for date in archive.origin_session_dates
            )
            != expected_dates
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL fit forecast inventory does not cover the exact fit prefix"
            )

        decision_roots = cast(
            tuple[MassiveAdaptiveDecisionRootV1, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="decision-root-inventory",
                fold_index=fold_index,
            ),
        )
        contexts = cast(
            tuple[MassiveAdaptiveContextOriginAuthorityV1, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="context-origin-inventory",
                fold_index=fold_index,
            ),
        )
        decisions_by_date = {row.decision_session_date: row for row in decision_roots}
        contexts_by_date = {row.decision_session_date: row for row in contexts}
        if (
            tuple(sorted(decisions_by_date)) != expected_dates
            or len(decisions_by_date) != len(decision_roots)
            or tuple(sorted(contexts_by_date)) != expected_dates
            or len(contexts_by_date) != len(contexts)
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL decision or context inventory does not cover the fit prefix"
            )

        windows = cast(
            tuple[MassiveAdaptiveWindowPlanV1, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="training-window-inventory",
                fold_index=fold_index,
            ),
        )
        checkpoints = cast(
            tuple[MassiveAdaptiveCausalCheckpointChoiceV1, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="supervised-checkpoint-inventory",
                fold_index=fold_index,
            ),
        )
        calibrations = cast(
            tuple[MassiveAdaptiveForecastCalibrationV2, ...],
            _inventory_items(
                runtime_sources=runtime_sources,
                role="calibration-inventory",
                fold_index=fold_index,
            ),
        )
        window_receipts = {row.semantic_receipt_sha256 for row in windows}
        archive_window_receipts = {
            row.training_window_plan_receipt_sha256 for row in ordered_archives
        }
        selected_checkpoint_receipts = {
            row.selected_checkpoint_receipt_sha256 for row in checkpoints
        }
        archive_checkpoint_receipts = {
            row.checkpoint_receipt_sha256 for row in ordered_archives
        }
        calibration_pairs = {
            (
                row.checkpoint_receipt_sha256,
                row.training_window_plan_receipt_sha256,
            )
            for row in calibrations
        }
        archive_pairs = {
            (
                row.checkpoint_receipt_sha256,
                row.training_window_plan_receipt_sha256,
            )
            for row in ordered_archives
        }
        if (
            window_receipts != archive_window_receipts
            or selected_checkpoint_receipts != archive_checkpoint_receipts
            or calibration_pairs != archive_pairs
            or any(
                row.fold_index != fold_index or row.split_role != "training"
                for row in windows
            )
            or any(row.fold_index != fold_index for row in checkpoints)
            or any(row.fold_index != fold_index for row in calibrations)
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL training, checkpoint, calibration, and forecast coverage differs"
            )

        full_decision_inventory = semantic_sha256(
            tuple(
                decisions_by_date[date].semantic_receipt_sha256
                for date in expected_dates
            )
        )
        for archive in ordered_archives:
            origin_inventory = semantic_sha256(
                tuple(
                    decisions_by_date[date].semantic_receipt_sha256
                    for date in archive.origin_session_dates
                )
            )
            _edge(
                edges,
                name=f"fit-forecast/split-plan/{fold_index}/{archive.block_index}",
                observed=archive.split_plan_receipt_sha256,
                expected=split_plan.semantic_receipt_sha256,
            )
            _edge(
                edges,
                name=f"fit-forecast/full-decision-roots/{fold_index}/{archive.block_index}",
                observed=archive.inference_full_decision_root_inventory_sha256,
                expected=full_decision_inventory,
            )
            _edge(
                edges,
                name=f"fit-forecast/origin-decision-roots/{fold_index}/{archive.block_index}",
                observed=archive.inference_origin_decision_root_inventory_sha256,
                expected=origin_inventory,
            )
        for window in windows:
            _edge(
                edges,
                name=f"training-window/split-plan/{fold_index}/{window.semantic_receipt_sha256}",
                observed=window.split_plan_receipt_sha256,
                expected=split_plan.semantic_receipt_sha256,
            )
        for checkpoint in checkpoints:
            if checkpoint.training_window_plan_receipt_sha256 not in window_receipts:
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL checkpoint references another training-window graph"
                )
        for calibration in calibrations:
            if (
                calibration.training_window_plan_receipt_sha256 not in window_receipts
                or calibration.checkpoint_receipt_sha256
                not in selected_checkpoint_receipts
            ):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                    "adaptive RL calibration references another fit graph"
                )
        for date in expected_dates:
            context = contexts_by_date[date]
            decision = decisions_by_date[date]
            _edge(
                edges,
                name=f"context/session-authority/{fold_index}/{date}",
                observed=context.session_authority_receipt_sha256,
                expected=session.receipt_sha256,
            )
            _edge(
                edges,
                name=f"context/identity-authority/{fold_index}/{date}",
                observed=context.identity_authority_receipt_sha256,
                expected=identity.receipt_sha256,
            )
            _edge(
                edges,
                name=f"decision/session-authority/{fold_index}/{date}",
                observed=decision.session_authority_receipt_sha256,
                expected=session.receipt_sha256,
            )
            _edge(
                edges,
                name=f"decision/context-origin/{fold_index}/{date}",
                observed=decision.context_origin_receipt_sha256,
                expected=context.semantic_receipt_sha256,
            )
        coverage.append(
            (
                "rl-fit-fold",
                fold_index,
                expected_dates,
                tuple(archive.block_index for archive in ordered_archives),
                tuple(sorted(window_receipts)),
                tuple(sorted(selected_checkpoint_receipts)),
            )
        )
    return tuple(coverage), tuple(edges)


def _build_authority(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    rows: tuple[MassiveAdaptiveRLRuntimeSourceGraphRowV1, ...],
    logical_coverage: tuple[object, ...],
    graph_edges: tuple[tuple[str, str, str], ...],
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
        "logical_coverage_inventory_sha256": semantic_sha256(logical_coverage),
        "graph_edge_inventory_sha256": semantic_sha256(graph_edges),
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


def _checked_graph_path(
    *, source_root: str | Path, experiment_id: str, require_file: bool
) -> Path:
    root = Path(source_root).resolve()
    output = runtime_source_graph_authority_path_v1(
        source_root=root,
        experiment_id=experiment_id,
    )
    cursor = output
    while cursor != root:
        if root not in cursor.parents:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph escapes its source root"
            )
        if cursor.is_symlink():
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph path contains a symlink"
            )
        cursor = cursor.parent
    if require_file and not output.is_file():
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
            "adaptive RL runtime source graph authority is absent or not regular"
        )
    return output


def _write_create_only_atomic(*, output: Path, payload: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_file_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as error:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error(
                "adaptive RL runtime source graph authority is create-only"
            ) from error
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    logical_coverage, graph_edges = _validate_runtime_graph_contract(
        runtime_sources=runtime_sources
    )
    result = _build_authority(
        manifest=manifest,
        source_bundle=source_bundle,
        rows=rows,
        logical_coverage=logical_coverage,
        graph_edges=graph_edges,
        persisted_graph_replayed=False,
        runtime_graph_replayed=False,
    )
    output = _checked_graph_path(
        source_root=source_root,
        experiment_id=manifest.experiment_id,
        require_file=False,
    )
    payload = asdict(
        replace(
            result,
            _runtime_source_bundle=None,
            _runtime_sources=None,
        )
    )
    payload.pop("_runtime_source_bundle")
    payload.pop("_runtime_sources")
    _write_create_only_atomic(output=output, payload=payload)
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
    path = _checked_graph_path(
        source_root=source_root,
        experiment_id=manifest.experiment_id,
        require_file=True,
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
        _runtime_source_bundle=authorized_bundle,
        _runtime_sources=tuple(
            _validate_domain_runtime(
                key=key,
                runtime=runtime_sources[key],
            )
            for key in sorted(
                runtime_sources,
                key=lambda value: (
                    value[1] is not None,
                    value[1] or -1,
                    value[0],
                ),
            )
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V1_SCHEMA",
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
