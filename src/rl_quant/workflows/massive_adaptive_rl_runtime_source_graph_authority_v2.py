"""Validation-complete runtime-source graph generation for adaptive RL.

V2 binds the exact V2 source-bundle transaction to the witnessed V1 graph
used by the existing training implementation.  Promotion is deliberately
strict: all eight validation predictor rows and both global development-origin
inventories must exist, their artifact receipts must equal the V2 source
projections, and the V1 graph must be the one that witnessed the V1 bundle
wrapped by the V2 bundle.  This makes legacy or mixed source generations
explicit rather than silently compatible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
import json
from pathlib import Path
from typing import cast

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
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256,
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v2 import (
    MassiveAdaptiveRLSourceBundleV2,
)


MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-source-graph-authority-v2"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_DATASET = (
    "massive-adaptive-rl-runtime-source-graph-authority-v2"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-source-graph-replay-witness-v2"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SCHEMA,
            "encoding": "canonical-json-validation-complete-runtime-graph-v2",
            "generic_reload": "nonauthorizing",
        }
    )
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SPEC_SHA256 = semantic_sha256(
    {
        "source_bundle": "exact-persisted-runtime-replayed-source-bundle-v2",
        "base_graph": "exact-validation-complete-v1-runtime-witness",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256
        ),
        "validation_rows": "four-feature-plus-four-action-primary-inventories",
        "nonvalidation_projection": (
            "training-plus-global-development-origins-bound-through-source-bundle-v2"
        ),
        "legacy_or_mixed_generation": "rejected",
        "publication": "manifest-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_FEATURE_ROLE = "validation-origin-feature-inventory"
_ACTION_ROLE = "validation-origin-action-inventory"
_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(ValueError):
    """The V2 runtime-source graph is absent, legacy, or inconsistent."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def runtime_source_graph_authority_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> str:
    manifest.validate()
    return (
        "adaptive-rl/runtime-source-graph-authority-v2/"
        f"{manifest.experiment_id}-m3-{manifest.semantic_receipt_sha256}.json"
    )


def _source_transaction_exists(*, root: str | Path, relative: str) -> bool:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    if any(present) and not all(present):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 transaction is incomplete"
        )
    return all(present)


def _row_inventory(
    authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
) -> tuple[tuple[str, int | None, str, str], ...]:
    return tuple(
        (
            row.role,
            row.fold_index,
            row.domain_authority_receipt_sha256,
            row.receipt_sha256,
        )
        for row in authority.rows
    )


def _validation_row_receipts(
    inventory: Sequence[tuple[str, int | None, str, str]], *, role: str
) -> tuple[str, ...]:
    rows = tuple(row for row in inventory if row[0] == role)
    if tuple(row[1] for row in rows) != _FOLD_INDICES or len(rows) != 4:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V1 lacks validation-complete primary rows"
        )
    return tuple(row[3] for row in rows)


def _validation_domain_receipts(
    inventory: Sequence[tuple[str, int | None, str, str]], *, role: str
) -> tuple[str, ...]:
    rows = tuple(row for row in inventory if row[0] == role)
    if tuple(row[1] for row in rows) != _FOLD_INDICES or len(rows) != 4:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V1 lacks validation-complete domain rows"
        )
    return tuple(row[2] for row in rows)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    source_bundle_v2_source_receipt_sha256: str
    source_bundle_v2_commit_receipt_sha256: str
    source_bundle_v2_committed_at_ms: int
    base_source_bundle_v1_receipt_sha256: str
    base_runtime_source_graph_v1_receipt_sha256: str
    base_runtime_source_graph_v1_specification_sha256: str
    base_runtime_source_graph_v1_implementation_source_sha256: str
    base_runtime_source_graph_v1_witness_receipt_sha256: str
    row_key_receipt_inventory: tuple[tuple[str, int | None, str, str], ...]
    row_inventory_sha256: str
    validation_feature_row_receipts: tuple[str, ...]
    validation_action_row_receipts: tuple[str, ...]
    validation_feature_domain_receipts: tuple[str, ...]
    validation_action_domain_receipts: tuple[str, ...]
    logical_coverage_inventory_sha256: str
    graph_edge_inventory_sha256: str
    committed_source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_graph_replayed: bool = False
    source_data_qualified: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SCHEMA
    _source_bundle_v2: MassiveAdaptiveRLSourceBundleV2 | None = field(
        default=None, compare=False, repr=False
    )
    _base_graph_v1: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "source_bundle_v2_receipt_sha256": self.source_bundle_v2_receipt_sha256,
            "source_bundle_v2_source_receipt_sha256": (
                self.source_bundle_v2_source_receipt_sha256
            ),
            "source_bundle_v2_commit_receipt_sha256": (
                self.source_bundle_v2_commit_receipt_sha256
            ),
            "source_bundle_v2_committed_at_ms": self.source_bundle_v2_committed_at_ms,
            "base_source_bundle_v1_receipt_sha256": (
                self.base_source_bundle_v1_receipt_sha256
            ),
            "base_runtime_source_graph_v1_receipt_sha256": (
                self.base_runtime_source_graph_v1_receipt_sha256
            ),
            "base_runtime_source_graph_v1_specification_sha256": (
                self.base_runtime_source_graph_v1_specification_sha256
            ),
            "base_runtime_source_graph_v1_implementation_source_sha256": (
                self.base_runtime_source_graph_v1_implementation_source_sha256
            ),
            "base_runtime_source_graph_v1_witness_receipt_sha256": (
                self.base_runtime_source_graph_v1_witness_receipt_sha256
            ),
            "row_key_receipt_inventory": self.row_key_receipt_inventory,
            "row_inventory_sha256": self.row_inventory_sha256,
            "validation_feature_row_receipts": self.validation_feature_row_receipts,
            "validation_action_row_receipts": self.validation_action_row_receipts,
            "validation_feature_domain_receipts": (
                self.validation_feature_domain_receipts
            ),
            "validation_action_domain_receipts": (
                self.validation_action_domain_receipts
            ),
            "logical_coverage_inventory_sha256": (
                self.logical_coverage_inventory_sha256
            ),
            "graph_edge_inventory_sha256": self.graph_edge_inventory_sha256,
            "committed_source_data_qualified": self.committed_source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def source_transaction_verified(self) -> bool:
        return self._loaded_source is not None

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    @property
    def runtime_authority_receipt_sha256(self) -> str | None:
        self.validate()
        if not self.runtime_graph_replayed:
            return None
        return semantic_sha256(
            {
                "schema": (
                    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V2_SCHEMA
                ),
                "persisted_graph_v2_receipt_sha256": self.semantic_receipt_sha256,
                "source_bundle_v2_receipt_sha256": (
                    self.source_bundle_v2_receipt_sha256
                ),
                "base_graph_v1_witness_receipt_sha256": (
                    self.base_runtime_source_graph_v1_witness_receipt_sha256
                ),
                "row_inventory_sha256": self.row_inventory_sha256,
                "logical_coverage_inventory_sha256": (
                    self.logical_coverage_inventory_sha256
                ),
                "graph_edge_inventory_sha256": self.graph_edge_inventory_sha256,
            }
        )

    @property
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_graph_replayed
            and self.source_data_qualified
            and self.runtime_authority_receipt_sha256 is not None
        )

    @property
    def base_runtime_source_graph_v1(
        self,
    ) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1:
        self.validate()
        if self._base_graph_v1 is None:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 has no V1 witness"
            )
        return self._base_graph_v1

    def validate(self) -> None:
        runtime_present = (
            self._source_bundle_v2 is not None or self._base_graph_v1 is not None
        )
        if runtime_present and (
            self._source_bundle_v2 is None or self._base_graph_v1 is None
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 witness is partial"
            )
        if self._source_bundle_v2 is not None:
            self._source_bundle_v2.validate()
        if self._base_graph_v1 is not None:
            self._base_graph_v1.validate()
        if self._loaded_source is not None:
            self._loaded_source.validate()
        inventory = self.row_key_receipt_inventory
        keys = tuple((row[0], row[1]) for row in inventory)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SCHEMA
            or not self.experiment_id
            or not inventory
            or len(set(keys)) != len(keys)
            or self.row_inventory_sha256 != semantic_sha256(inventory)
            or self.validation_feature_row_receipts
            != _validation_row_receipts(inventory, role=_FEATURE_ROLE)
            or self.validation_action_row_receipts
            != _validation_row_receipts(inventory, role=_ACTION_ROLE)
            or self.validation_feature_domain_receipts
            != _validation_domain_receipts(inventory, role=_FEATURE_ROLE)
            or self.validation_action_domain_receipts
            != _validation_domain_receipts(inventory, role=_ACTION_ROLE)
            or self.committed_source_data_qualified is not True
            or self.runtime_graph_replayed != runtime_present
            or self.source_data_qualified
            != bool(self._loaded_source is not None and runtime_present)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 identity or authorization differs"
            )
        if runtime_present:
            assert self._source_bundle_v2 is not None
            assert self._base_graph_v1 is not None
            bundle = self._source_bundle_v2
            graph = self._base_graph_v1
            if (
                not bundle.development_stage_authorized
                or not graph.persisted_graph_replayed
                or not graph.runtime_graph_replayed
                or not graph.source_data_qualified
                or bundle.experiment_id != self.experiment_id
                or graph.experiment_id != self.experiment_id
                or graph.manifest_v3_receipt_sha256 != self.manifest_v3_receipt_sha256
                or bundle.semantic_receipt_sha256
                != self.source_bundle_v2_receipt_sha256
                or bundle.source_receipt_sha256
                != self.source_bundle_v2_source_receipt_sha256
                or bundle.source_transaction_receipt_sha256
                != self.source_bundle_v2_commit_receipt_sha256
                or bundle.source_transaction_committed_at_ms
                != self.source_bundle_v2_committed_at_ms
                or bundle.base_source_bundle_v1_receipt_sha256
                != self.base_source_bundle_v1_receipt_sha256
                or graph.source_bundle_receipt_sha256
                != self.base_source_bundle_v1_receipt_sha256
                or graph.semantic_receipt_sha256
                != self.base_runtime_source_graph_v1_receipt_sha256
                or graph.specification_sha256
                != self.base_runtime_source_graph_v1_specification_sha256
                or graph.implementation_source_sha256
                != self.base_runtime_source_graph_v1_implementation_source_sha256
                or graph.runtime_authority_receipt_sha256
                != self.base_runtime_source_graph_v1_witness_receipt_sha256
                or _row_inventory(graph) != inventory
            ):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                    "runtime-source graph V2 contains a mixed generation"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.source_bundle_v2_committed_at_ms
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.source_bundle_v2_receipt_sha256,
            self.source_bundle_v2_source_receipt_sha256,
            self.source_bundle_v2_commit_receipt_sha256,
            self.base_source_bundle_v1_receipt_sha256,
            self.base_runtime_source_graph_v1_receipt_sha256,
            self.base_runtime_source_graph_v1_specification_sha256,
            self.base_runtime_source_graph_v1_implementation_source_sha256,
            self.base_runtime_source_graph_v1_witness_receipt_sha256,
            self.row_inventory_sha256,
            *self.validation_feature_row_receipts,
            *self.validation_action_row_receipts,
            *self.validation_feature_domain_receipts,
            *self.validation_action_domain_receipts,
            self.logical_coverage_inventory_sha256,
            self.graph_edge_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("runtime-source graph V2", value)
        if (
            isinstance(self.source_bundle_v2_committed_at_ms, bool)
            or not isinstance(self.source_bundle_v2_committed_at_ms, int)
            or self.source_bundle_v2_committed_at_ms < 0
        ):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 source-bundle commit time differs"
            )
        for role, fold_index, domain_receipt, row_receipt in inventory:
            if not role or (
                fold_index is not None
                and (isinstance(fold_index, bool) or fold_index not in _FOLD_INDICES)
            ):
                raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                    "runtime-source graph V2 row key differs"
                )
            _digest("runtime-source graph V2 domain", domain_receipt)
            _digest("runtime-source graph V2 row", row_receipt)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle_v2: MassiveAdaptiveRLSourceBundleV2,
    runtime_source_graph_v1: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV3
        or type(source_bundle_v2) is not MassiveAdaptiveRLSourceBundleV2
        or type(runtime_source_graph_v1)
        is not MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 requires exact generation types"
        )
    manifest.validate()
    source_bundle_v2.validate()
    runtime_source_graph_v1.validate()
    source_receipt = source_bundle_v2.source_receipt_sha256
    commit_receipt = source_bundle_v2.source_transaction_receipt_sha256
    committed_at_ms = source_bundle_v2.source_transaction_committed_at_ms
    witness = runtime_source_graph_v1.runtime_authority_receipt_sha256
    if (
        not source_bundle_v2.development_stage_authorized
        or source_receipt is None
        or commit_receipt is None
        or committed_at_ms is None
        or not runtime_source_graph_v1.persisted_graph_replayed
        or not runtime_source_graph_v1.runtime_graph_replayed
        or not runtime_source_graph_v1.source_data_qualified
        or witness is None
        or source_bundle_v2.experiment_id != manifest.experiment_id
        or runtime_source_graph_v1.experiment_id != manifest.experiment_id
        or runtime_source_graph_v1.manifest_v3_receipt_sha256
        != manifest.semantic_receipt_sha256
        or runtime_source_graph_v1.source_bundle_receipt_sha256
        != source_bundle_v2.base_source_bundle_v1_receipt_sha256
        or runtime_source_graph_v1.specification_sha256
        != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SPEC_SHA256
        or runtime_source_graph_v1.implementation_source_sha256
        != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V1_SOURCE_SHA256
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "legacy or mixed runtime-source graph V1 cannot promote to V2"
        )
    inventory = _row_inventory(runtime_source_graph_v1)
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": source_bundle_v2.semantic_receipt_sha256,
        "source_bundle_v2_source_receipt_sha256": source_receipt,
        "source_bundle_v2_commit_receipt_sha256": commit_receipt,
        "source_bundle_v2_committed_at_ms": committed_at_ms,
        "base_source_bundle_v1_receipt_sha256": (
            source_bundle_v2.base_source_bundle_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_receipt_sha256": (
            runtime_source_graph_v1.semantic_receipt_sha256
        ),
        "base_runtime_source_graph_v1_specification_sha256": (
            runtime_source_graph_v1.specification_sha256
        ),
        "base_runtime_source_graph_v1_implementation_source_sha256": (
            runtime_source_graph_v1.implementation_source_sha256
        ),
        "base_runtime_source_graph_v1_witness_receipt_sha256": witness,
        "row_key_receipt_inventory": inventory,
        "row_inventory_sha256": semantic_sha256(inventory),
        "validation_feature_row_receipts": _validation_row_receipts(
            inventory, role=_FEATURE_ROLE
        ),
        "validation_action_row_receipts": _validation_row_receipts(
            inventory, role=_ACTION_ROLE
        ),
        "validation_feature_domain_receipts": _validation_domain_receipts(
            inventory, role=_FEATURE_ROLE
        ),
        "validation_action_domain_receipts": _validation_domain_receipts(
            inventory, role=_ACTION_ROLE
        ),
        "logical_coverage_inventory_sha256": (
            runtime_source_graph_v1.logical_coverage_inventory_sha256
        ),
        "graph_edge_inventory_sha256": (
            runtime_source_graph_v1.graph_edge_inventory_sha256
        ),
        "committed_source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_graph_replayed=True,
        source_data_qualified=False,
        _source_bundle_v2=source_bundle_v2,
        _base_graph_v1=runtime_source_graph_v1,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse_body(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 payload is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    rows = body.get("row_key_receipt_inventory")
    if not isinstance(rows, list):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 row inventory is malformed"
        )
    converted: list[tuple[object, object, object, object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 row is malformed"
            )
        converted.append((row[0], row[1], row[2], row[3]))
    body["row_key_receipt_inventory"] = tuple(converted)
    for name in (
        "validation_feature_row_receipts",
        "validation_action_row_receipts",
        "validation_feature_domain_receipts",
        "validation_action_domain_receipts",
    ):
        values = body.get(name)
        if not isinstance(values, list):
            raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
                "runtime-source graph V2 validation inventory is malformed"
            )
        body[name] = tuple(values)
    return body


def parse_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_graph_replayed=False,
        source_data_qualified=False,
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    verified_at_ms: int,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    return parse_massive_adaptive_rl_runtime_source_graph_authority_v2(
        root=source_root,
        loaded_source=load_massive_source_bundle(
            root=source_root,
            relative_payload_path=(
                runtime_source_graph_authority_relative_path_v2(manifest=manifest)
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *,
    authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle_v2: MassiveAdaptiveRLSourceBundleV2,
    runtime_source_graph_v1: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    authority.validate()
    expected = build_massive_adaptive_rl_runtime_source_graph_authority_v2(
        manifest=manifest,
        source_bundle_v2=source_bundle_v2,
        runtime_source_graph_v1=runtime_source_graph_v1,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != runtime_source_graph_authority_relative_path_v2(manifest=manifest)
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 does not replay"
        )
    result = replace(
        authority,
        runtime_graph_replayed=True,
        source_data_qualified=authority.committed_source_data_qualified,
        _source_bundle_v2=source_bundle_v2,
        _base_graph_v1=runtime_source_graph_v1,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    manifest.validate()
    authority.validate()
    if (
        authority._source_bundle_v2 is None
        or authority._base_graph_v1 is None
        or authority.runtime_graph_replayed is not True
        or committed_at_ms <= authority.source_bundle_v2_committed_at_ms
    ):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 is not witnessed after its source bundle"
        )
    relative = runtime_source_graph_authority_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 already exists"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=source_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-RUNTIME-SOURCE-GRAPH-V2-{authority.experiment_id}",
    )
    loaded = load_massive_source_bundle(
        root=source_root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    result = replace(
        authority,
        source_data_qualified=authority.committed_source_data_qualified,
        _loaded_source=loaded,
    )
    result.validate()
    return result


def prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle_v2: MassiveAdaptiveRLSourceBundleV2,
    runtime_source_graph_v1: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
    relative = runtime_source_graph_authority_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        return authorize_massive_adaptive_rl_runtime_source_graph_authority_v2(
            authority=load_massive_adaptive_rl_runtime_source_graph_authority_v2(
                source_root=source_root,
                manifest=manifest,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            source_bundle_v2=source_bundle_v2,
            runtime_source_graph_v1=runtime_source_graph_v1,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error(
            "runtime-source graph V2 is absent"
        )
    return materialize_massive_adaptive_rl_runtime_source_graph_authority_v2(
        source_root=source_root,
        manifest=manifest,
        authority=build_massive_adaptive_rl_runtime_source_graph_authority_v2(
            manifest=manifest,
            source_bundle_v2=source_bundle_v2,
            runtime_source_graph_v1=runtime_source_graph_v1,
        ),
        committed_at_ms=committed_at_ms,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_AUTHORITY_V2_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_GRAPH_REPLAY_WITNESS_V2_SCHEMA",
    "MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2",
    "MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2Error",
    "authorize_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "build_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "load_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "materialize_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "parse_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2",
    "runtime_source_graph_authority_relative_path_v2",
]
