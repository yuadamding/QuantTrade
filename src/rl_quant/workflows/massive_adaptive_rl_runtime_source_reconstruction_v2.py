"""Cold promotion of validation-complete adaptive-RL runtime sources to V2.

This generation deliberately retains the exact V1 objects used by the current
training implementation as private replay witnesses.  The public V2 identity
adds the missing generation boundary: it binds the V2 source bundle and graph,
an explicit V2 dependency-index transaction, and the complete per-fold
validation feature/action inventories.  Legacy or mixed V1 artifacts cannot
be promoted by name or parameter equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from rl_quant.workflows.massive_adaptive_rl_four_fold_fit_v1 import (
    MassiveAdaptiveRLFourFoldFitAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v2 import (
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_reconstruction_v1 import (
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256,
    MassiveAdaptiveRLFoldRuntimeSourcesV1,
    MassiveAdaptiveRLReplayDependencyIndexV1,
    MassiveAdaptiveRLRuntimeSourceReconstructionV1Error,
    MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable,
    MassiveAdaptiveRLRuntimeSourcesV1,
    MassiveAdaptiveRLValidationOriginInputsV1,
    load_massive_adaptive_rl_replay_dependency_index_v1,
    reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MassiveAdaptiveRLSourceBundleV1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v2 import (
    MassiveAdaptiveRLSourceBundleV2,
    prepare_or_resume_massive_adaptive_rl_source_bundle_v2,
)


MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-replay-dependency-index-v2"
)
MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_DATASET = (
    "massive-adaptive-rl-replay-dependency-index-v2"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-sources-v2"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SCHEMA,
        "encoding": "canonical-json-validation-complete-dependency-index-v2",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256 = semantic_sha256(
    {
        "base_runtime": "exact-cold-reconstructed-runtime-sources-v1-receipt",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
        ),
        "source_bundle": "exact-persisted-source-bundle-v2",
        "runtime_graph": "exact-persisted-runtime-source-graph-v2",
        "dependency_index": "exact-persisted-replay-dependency-index-v2",
        "validation_dependencies": (
            "all-fold-feature-and-action-origin-receipts-are-index-members"
        ),
        "training_compatibility": (
            "four-fold-fit-must-bind-exact-base-v1-runtime-and-witness"
        ),
        "legacy_or_mixed_generation": "rejected",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_FEATURE_ROLE = "validation-origin-feature-inventory"
_ACTION_ROLE = "validation-origin-action-inventory"
_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(ValueError):
    """The validation-complete V2 runtime generation does not replay."""


class MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
    MassiveAdaptiveRLRuntimeSourceReconstructionV2Error
):
    """Training, source, graph, or dependency generations were mixed."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def replay_dependency_index_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> str:
    manifest.validate()
    return (
        "adaptive-rl/replay-dependency-index-v2/"
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
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 transaction is incomplete"
        )
    return all(present)


def _validation_receipt_inventories(
    runtime_sources: MassiveAdaptiveRLRuntimeSourcesV1,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    feature_rows = tuple(
        tuple(row.semantic_receipt_sha256 for row in fold.validation_features)
        for fold in runtime_sources.folds
    )
    action_rows = tuple(
        tuple(row.semantic_receipt_sha256 for row in fold.validation_action_origins)
        for fold in runtime_sources.folds
    )
    if (
        tuple(fold.outer_fold_index for fold in runtime_sources.folds) != _FOLD_INDICES
        or any(not rows for rows in feature_rows)
        or any(not rows for rows in action_rows)
        or tuple(len(rows) for rows in feature_rows)
        != tuple(len(rows) for rows in action_rows)
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "runtime sources V1 lack complete validation predictor inventories"
        )
    return feature_rows, action_rows


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLReplayDependencyIndexV2:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_source_receipt_sha256: str
    runtime_source_graph_v2_commit_receipt_sha256: str
    runtime_source_graph_v2_committed_at_ms: int
    runtime_source_graph_v2_witness_receipt_sha256: str
    base_replay_dependency_index_v1_receipt_sha256: str
    base_replay_dependency_index_v1_specification_sha256: str
    base_replay_dependency_index_v1_implementation_source_sha256: str
    base_row_inventory_sha256: str
    base_object_inventory_sha256: str
    base_dependency_edge_inventory_sha256: str
    validation_feature_receipt_inventories: tuple[tuple[str, ...], ...]
    validation_action_receipt_inventories: tuple[tuple[str, ...], ...]
    validation_predictor_inventory_sha256: str
    committed_source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_index_replayed: bool = False
    source_data_qualified: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SCHEMA
    _runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2 | None = (
        field(default=None, compare=False, repr=False)
    )
    _base_index_v1: MassiveAdaptiveRLReplayDependencyIndexV1 | None = field(
        default=None, compare=False, repr=False
    )
    _base_runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1 | None = field(
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
            "runtime_source_graph_v2_receipt_sha256": (
                self.runtime_source_graph_v2_receipt_sha256
            ),
            "runtime_source_graph_v2_source_receipt_sha256": (
                self.runtime_source_graph_v2_source_receipt_sha256
            ),
            "runtime_source_graph_v2_commit_receipt_sha256": (
                self.runtime_source_graph_v2_commit_receipt_sha256
            ),
            "runtime_source_graph_v2_committed_at_ms": (
                self.runtime_source_graph_v2_committed_at_ms
            ),
            "runtime_source_graph_v2_witness_receipt_sha256": (
                self.runtime_source_graph_v2_witness_receipt_sha256
            ),
            "base_replay_dependency_index_v1_receipt_sha256": (
                self.base_replay_dependency_index_v1_receipt_sha256
            ),
            "base_replay_dependency_index_v1_specification_sha256": (
                self.base_replay_dependency_index_v1_specification_sha256
            ),
            "base_replay_dependency_index_v1_implementation_source_sha256": (
                self.base_replay_dependency_index_v1_implementation_source_sha256
            ),
            "base_row_inventory_sha256": self.base_row_inventory_sha256,
            "base_object_inventory_sha256": self.base_object_inventory_sha256,
            "base_dependency_edge_inventory_sha256": (
                self.base_dependency_edge_inventory_sha256
            ),
            "validation_feature_receipt_inventories": (
                self.validation_feature_receipt_inventories
            ),
            "validation_action_receipt_inventories": (
                self.validation_action_receipt_inventories
            ),
            "validation_predictor_inventory_sha256": (
                self.validation_predictor_inventory_sha256
            ),
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
    def development_stage_authorized(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_index_replayed
            and self.source_data_qualified
        )

    def validate(self) -> None:
        runtime_parts = (
            self._runtime_source_graph_v2,
            self._base_index_v1,
            self._base_runtime_sources_v1,
        )
        runtime_present = any(row is not None for row in runtime_parts)
        if runtime_present and any(row is None for row in runtime_parts):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "replay-dependency index V2 runtime witness is partial"
            )
        if self._runtime_source_graph_v2 is not None:
            self._runtime_source_graph_v2.validate()
        if self._base_index_v1 is not None:
            self._base_index_v1.validate()
        if self._base_runtime_sources_v1 is not None:
            self._base_runtime_sources_v1.validate()
        if self._loaded_source is not None:
            self._loaded_source.validate()
        expected_predictor_inventory = semantic_sha256(
            (
                self.validation_feature_receipt_inventories,
                self.validation_action_receipt_inventories,
            )
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SCHEMA
            or not self.experiment_id
            or len(self.validation_feature_receipt_inventories) != 4
            or len(self.validation_action_receipt_inventories) != 4
            or any(not rows for rows in self.validation_feature_receipt_inventories)
            or any(not rows for rows in self.validation_action_receipt_inventories)
            or tuple(map(len, self.validation_feature_receipt_inventories))
            != tuple(map(len, self.validation_action_receipt_inventories))
            or self.validation_predictor_inventory_sha256
            != expected_predictor_inventory
            or self.committed_source_data_qualified is not True
            or self.runtime_index_replayed != runtime_present
            or self.source_data_qualified
            != bool(self._loaded_source is not None and runtime_present)
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "replay-dependency index V2 identity or authorization differs"
            )
        if runtime_present:
            assert self._runtime_source_graph_v2 is not None
            assert self._base_index_v1 is not None
            assert self._base_runtime_sources_v1 is not None
            graph = self._runtime_source_graph_v2
            index = self._base_index_v1
            runtime = self._base_runtime_sources_v1
            features, actions = _validation_receipt_inventories(runtime)
            indexed_receipts = {row.semantic_receipt_sha256 for row in index.rows}
            if (
                not graph.development_stage_authorized
                or runtime.experiment_id != self.experiment_id
                or runtime.manifest_v3_receipt_sha256 != self.manifest_v3_receipt_sha256
                or graph.semantic_receipt_sha256
                != self.runtime_source_graph_v2_receipt_sha256
                or graph.source_receipt_sha256
                != self.runtime_source_graph_v2_source_receipt_sha256
                or graph.source_transaction_receipt_sha256
                != self.runtime_source_graph_v2_commit_receipt_sha256
                or graph.source_transaction_committed_at_ms
                != self.runtime_source_graph_v2_committed_at_ms
                or graph.runtime_authority_receipt_sha256
                != self.runtime_source_graph_v2_witness_receipt_sha256
                or index.semantic_receipt_sha256
                != self.base_replay_dependency_index_v1_receipt_sha256
                or index.specification_sha256
                != self.base_replay_dependency_index_v1_specification_sha256
                or index.implementation_source_sha256
                != self.base_replay_dependency_index_v1_implementation_source_sha256
                or index.row_inventory_sha256 != self.base_row_inventory_sha256
                or index.object_inventory_sha256 != self.base_object_inventory_sha256
                or index.dependency_edge_inventory_sha256
                != self.base_dependency_edge_inventory_sha256
                or index.source_bundle_receipt_sha256
                != graph.base_source_bundle_v1_receipt_sha256
                or index.persisted_runtime_source_graph_receipt_sha256
                != graph.base_runtime_source_graph_v1_receipt_sha256
                or index.runtime_source_graph_witness_receipt_sha256
                != graph.base_runtime_source_graph_v1_witness_receipt_sha256
                or runtime.source_bundle_receipt_sha256
                != graph.base_source_bundle_v1_receipt_sha256
                or runtime.runtime_source_graph_authority.semantic_receipt_sha256
                != graph.base_runtime_source_graph_v1_receipt_sha256
                or runtime.runtime_source_graph_authority.runtime_authority_receipt_sha256
                != graph.base_runtime_source_graph_v1_witness_receipt_sha256
                or features != self.validation_feature_receipt_inventories
                or actions != self.validation_action_receipt_inventories
                or any(
                    receipt not in indexed_receipts
                    for inventories in (features, actions)
                    for rows in inventories
                    for receipt in rows
                )
            ):
                raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
                    "replay-dependency index V2 mixes source generations"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
            or self._loaded_source.commit.committed_at_ms
            <= self.runtime_source_graph_v2_committed_at_ms
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "replay-dependency index V2 source transaction differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.source_bundle_v2_receipt_sha256,
            self.runtime_source_graph_v2_receipt_sha256,
            self.runtime_source_graph_v2_source_receipt_sha256,
            self.runtime_source_graph_v2_commit_receipt_sha256,
            self.runtime_source_graph_v2_witness_receipt_sha256,
            self.base_replay_dependency_index_v1_receipt_sha256,
            self.base_replay_dependency_index_v1_specification_sha256,
            self.base_replay_dependency_index_v1_implementation_source_sha256,
            self.base_row_inventory_sha256,
            self.base_object_inventory_sha256,
            self.base_dependency_edge_inventory_sha256,
            *(
                receipt
                for inventories in (
                    self.validation_feature_receipt_inventories,
                    self.validation_action_receipt_inventories,
                )
                for rows in inventories
                for receipt in rows
            ),
            self.validation_predictor_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("replay-dependency index V2", value)
        if (
            isinstance(self.runtime_source_graph_v2_committed_at_ms, bool)
            or not isinstance(self.runtime_source_graph_v2_committed_at_ms, int)
            or self.runtime_source_graph_v2_committed_at_ms < 0
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "replay-dependency index V2 graph commit time differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_replay_dependency_index_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    replay_dependency_index_v1: MassiveAdaptiveRLReplayDependencyIndexV1,
    runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV3
        or type(runtime_source_graph_v2)
        is not MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2
        or type(replay_dependency_index_v1)
        is not MassiveAdaptiveRLReplayDependencyIndexV1
        or type(runtime_sources_v1) is not MassiveAdaptiveRLRuntimeSourcesV1
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 requires exact generation types"
        )
    manifest.validate()
    runtime_source_graph_v2.validate()
    replay_dependency_index_v1.validate()
    runtime_sources_v1.validate()
    graph_source_receipt = runtime_source_graph_v2.source_receipt_sha256
    graph_commit_receipt = runtime_source_graph_v2.source_transaction_receipt_sha256
    graph_committed_at_ms = runtime_source_graph_v2.source_transaction_committed_at_ms
    graph_witness = runtime_source_graph_v2.runtime_authority_receipt_sha256
    if (
        not runtime_source_graph_v2.development_stage_authorized
        or graph_source_receipt is None
        or graph_commit_receipt is None
        or graph_committed_at_ms is None
        or graph_witness is None
        or replay_dependency_index_v1.experiment_id != manifest.experiment_id
        or replay_dependency_index_v1.manifest_v3_receipt_sha256
        != manifest.semantic_receipt_sha256
        or replay_dependency_index_v1.specification_sha256
        != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
        or replay_dependency_index_v1.implementation_source_sha256
        != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
        or runtime_sources_v1.experiment_id != manifest.experiment_id
        or not runtime_sources_v1.source_data_qualified
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "legacy or mixed replay-dependency index V1 cannot promote to V2"
        )
    feature_rows, action_rows = _validation_receipt_inventories(runtime_sources_v1)
    indexed_receipts = {
        row.semantic_receipt_sha256 for row in replay_dependency_index_v1.rows
    }
    if any(
        receipt not in indexed_receipts
        for inventories in (feature_rows, action_rows)
        for rows in inventories
        for receipt in rows
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "validation predictor is absent from the replay-dependency index V1"
        )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": (
            runtime_source_graph_v2.source_bundle_v2_receipt_sha256
        ),
        "runtime_source_graph_v2_receipt_sha256": (
            runtime_source_graph_v2.semantic_receipt_sha256
        ),
        "runtime_source_graph_v2_source_receipt_sha256": graph_source_receipt,
        "runtime_source_graph_v2_commit_receipt_sha256": graph_commit_receipt,
        "runtime_source_graph_v2_committed_at_ms": graph_committed_at_ms,
        "runtime_source_graph_v2_witness_receipt_sha256": graph_witness,
        "base_replay_dependency_index_v1_receipt_sha256": (
            replay_dependency_index_v1.semantic_receipt_sha256
        ),
        "base_replay_dependency_index_v1_specification_sha256": (
            replay_dependency_index_v1.specification_sha256
        ),
        "base_replay_dependency_index_v1_implementation_source_sha256": (
            replay_dependency_index_v1.implementation_source_sha256
        ),
        "base_row_inventory_sha256": replay_dependency_index_v1.row_inventory_sha256,
        "base_object_inventory_sha256": (
            replay_dependency_index_v1.object_inventory_sha256
        ),
        "base_dependency_edge_inventory_sha256": (
            replay_dependency_index_v1.dependency_edge_inventory_sha256
        ),
        "validation_feature_receipt_inventories": feature_rows,
        "validation_action_receipt_inventories": action_rows,
        "validation_predictor_inventory_sha256": semantic_sha256(
            (feature_rows, action_rows)
        ),
        "committed_source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLReplayDependencyIndexV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_index_replayed=True,
        source_data_qualified=False,
        _runtime_source_graph_v2=runtime_source_graph_v2,
        _base_index_v1=replay_dependency_index_v1,
        _base_runtime_sources_v1=runtime_sources_v1,
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
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 payload is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "validation_feature_receipt_inventories",
        "validation_action_receipt_inventories",
    ):
        rows = body.get(name)
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "replay-dependency index V2 predictor inventory is malformed"
            )
        body[name] = tuple(tuple(cast(list[str], row)) for row in rows)
    return body


def parse_massive_adaptive_rl_replay_dependency_index_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLReplayDependencyIndexV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_index_replayed=False,
        source_data_qualified=False,
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_replay_dependency_index_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    verified_at_ms: int,
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    return parse_massive_adaptive_rl_replay_dependency_index_v2(
        root=source_root,
        loaded_source=load_massive_source_bundle(
            root=source_root,
            relative_payload_path=replay_dependency_index_relative_path_v2(
                manifest=manifest
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_replay_dependency_index_v2(
    *,
    authority: MassiveAdaptiveRLReplayDependencyIndexV2,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    replay_dependency_index_v1: MassiveAdaptiveRLReplayDependencyIndexV1,
    runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    authority.validate()
    expected = build_massive_adaptive_rl_replay_dependency_index_v2(
        manifest=manifest,
        runtime_source_graph_v2=runtime_source_graph_v2,
        replay_dependency_index_v1=replay_dependency_index_v1,
        runtime_sources_v1=runtime_sources_v1,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != replay_dependency_index_relative_path_v2(manifest=manifest)
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 does not replay"
        )
    result = replace(
        authority,
        runtime_index_replayed=True,
        source_data_qualified=authority.committed_source_data_qualified,
        _runtime_source_graph_v2=runtime_source_graph_v2,
        _base_index_v1=replay_dependency_index_v1,
        _base_runtime_sources_v1=runtime_sources_v1,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_replay_dependency_index_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    authority: MassiveAdaptiveRLReplayDependencyIndexV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    manifest.validate()
    authority.validate()
    if (
        not authority.runtime_index_replayed
        or authority._runtime_source_graph_v2 is None
        or authority._base_index_v1 is None
        or authority._base_runtime_sources_v1 is None
        or committed_at_ms <= authority.runtime_source_graph_v2_committed_at_ms
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 is not witnessed after its graph"
        )
    relative = replay_dependency_index_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 already exists"
        )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=source_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-REPLAY-DEPENDENCY-INDEX-V2-{authority.experiment_id}",
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


def prepare_or_resume_massive_adaptive_rl_replay_dependency_index_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    replay_dependency_index_v1: MassiveAdaptiveRLReplayDependencyIndexV1,
    runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLReplayDependencyIndexV2:
    relative = replay_dependency_index_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        return authorize_massive_adaptive_rl_replay_dependency_index_v2(
            authority=load_massive_adaptive_rl_replay_dependency_index_v2(
                source_root=source_root,
                manifest=manifest,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            runtime_source_graph_v2=runtime_source_graph_v2,
            replay_dependency_index_v1=replay_dependency_index_v1,
            runtime_sources_v1=runtime_sources_v1,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "replay-dependency index V2 is absent"
        )
    return materialize_massive_adaptive_rl_replay_dependency_index_v2(
        source_root=source_root,
        manifest=manifest,
        authority=build_massive_adaptive_rl_replay_dependency_index_v2(
            manifest=manifest,
            runtime_source_graph_v2=runtime_source_graph_v2,
            replay_dependency_index_v1=replay_dependency_index_v1,
            runtime_sources_v1=runtime_sources_v1,
        ),
        committed_at_ms=committed_at_ms,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class MassiveAdaptiveRLRuntimeSourcesV2:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    source_bundle_v2_receipt_sha256: str
    runtime_source_graph_v2_receipt_sha256: str
    runtime_source_graph_v2_witness_receipt_sha256: str
    replay_dependency_index_v2_receipt_sha256: str
    base_runtime_sources_v1_receipt_sha256: str
    base_source_bundle_v1_receipt_sha256: str
    base_runtime_source_graph_v1_receipt_sha256: str
    base_runtime_source_graph_v1_witness_receipt_sha256: str
    base_replay_dependency_index_v1_receipt_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_feature_receipt_inventories: tuple[tuple[str, ...], ...]
    validation_action_receipt_inventories: tuple[tuple[str, ...], ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V2_SCHEMA
    _source_bundle_v2: MassiveAdaptiveRLSourceBundleV2 = field(
        compare=False, repr=False
    )
    _runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2 = field(
        compare=False, repr=False
    )
    _replay_dependency_index_v2: MassiveAdaptiveRLReplayDependencyIndexV2 = field(
        compare=False, repr=False
    )
    _base_runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1 = field(
        compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "source_bundle_v2_receipt_sha256": self.source_bundle_v2_receipt_sha256,
            "runtime_source_graph_v2_receipt_sha256": (
                self.runtime_source_graph_v2_receipt_sha256
            ),
            "runtime_source_graph_v2_witness_receipt_sha256": (
                self.runtime_source_graph_v2_witness_receipt_sha256
            ),
            "replay_dependency_index_v2_receipt_sha256": (
                self.replay_dependency_index_v2_receipt_sha256
            ),
            "base_runtime_sources_v1_receipt_sha256": (
                self.base_runtime_sources_v1_receipt_sha256
            ),
            "base_source_bundle_v1_receipt_sha256": (
                self.base_source_bundle_v1_receipt_sha256
            ),
            "base_runtime_source_graph_v1_receipt_sha256": (
                self.base_runtime_source_graph_v1_receipt_sha256
            ),
            "base_runtime_source_graph_v1_witness_receipt_sha256": (
                self.base_runtime_source_graph_v1_witness_receipt_sha256
            ),
            "base_replay_dependency_index_v1_receipt_sha256": (
                self.base_replay_dependency_index_v1_receipt_sha256
            ),
            "training_source_projection_sha256": (
                self.training_source_projection_sha256
            ),
            "validation_source_projection_sha256": (
                self.validation_source_projection_sha256
            ),
            "validation_feature_receipt_inventories": (
                self.validation_feature_receipt_inventories
            ),
            "validation_action_receipt_inventories": (
                self.validation_action_receipt_inventories
            ),
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "outer_evaluation_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    @property
    def base_runtime_sources_v1(self) -> MassiveAdaptiveRLRuntimeSourcesV1:
        self.validate()
        return self._base_runtime_sources_v1

    @property
    def source_bundle_v2(self) -> MassiveAdaptiveRLSourceBundleV2:
        self.validate()
        return self._source_bundle_v2

    @property
    def runtime_source_graph_authority_v2(
        self,
    ) -> MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2:
        self.validate()
        return self._runtime_source_graph_v2

    @property
    def replay_dependency_index_v2(self) -> MassiveAdaptiveRLReplayDependencyIndexV2:
        self.validate()
        return self._replay_dependency_index_v2

    @property
    def split_plan(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.split_plan

    @property
    def session_authority(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.session_authority

    @property
    def condition_authority(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.condition_authority

    @property
    def identity_authority(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.identity_authority

    @property
    def economic_event_archive(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.economic_event_archive

    @property
    def daily_input_authority(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.daily_input_authority

    @property
    def fill_source(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.fill_source

    @property
    def persisted_partition_manifests(self):  # type: ignore[no-untyped-def]
        return self._base_runtime_sources_v1.persisted_partition_manifests

    def fold(self, outer_fold_index: int) -> MassiveAdaptiveRLFoldRuntimeSourcesV1:
        return self._base_runtime_sources_v1.fold(outer_fold_index)

    def validation_origin_inputs(
        self, fold_index: int
    ) -> MassiveAdaptiveRLValidationOriginInputsV1:
        result = self._base_runtime_sources_v1.validation_origin_inputs(fold_index)
        expected_features = self.validation_feature_receipt_inventories[fold_index]
        expected_actions = self.validation_action_receipt_inventories[fold_index]
        if (
            tuple(row.semantic_receipt_sha256 for row in result.features)
            != expected_features
            or tuple(row.semantic_receipt_sha256 for row in result.action_origins)
            != expected_actions
        ):
            raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
                "runtime sources V2 validation-origin view differs"
            )
        return result

    def validate(self) -> None:
        self._source_bundle_v2.validate()
        self._runtime_source_graph_v2.validate()
        self._replay_dependency_index_v2.validate()
        self._base_runtime_sources_v1.validate()
        graph_witness = self._runtime_source_graph_v2.runtime_authority_receipt_sha256
        feature_rows, action_rows = _validation_receipt_inventories(
            self._base_runtime_sources_v1
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V2_SCHEMA
            or not self.experiment_id
            or not self.source_data_qualified
            or not self._source_bundle_v2.development_stage_authorized
            or not self._runtime_source_graph_v2.development_stage_authorized
            or not self._replay_dependency_index_v2.development_stage_authorized
            or not self._base_runtime_sources_v1.source_data_qualified
            or graph_witness is None
            or self.experiment_id != self._base_runtime_sources_v1.experiment_id
            or self.manifest_v3_receipt_sha256
            != self._base_runtime_sources_v1.manifest_v3_receipt_sha256
            or self.source_bundle_v2_receipt_sha256
            != self._source_bundle_v2.semantic_receipt_sha256
            or self.runtime_source_graph_v2_receipt_sha256
            != self._runtime_source_graph_v2.semantic_receipt_sha256
            or self.runtime_source_graph_v2_witness_receipt_sha256 != graph_witness
            or self.replay_dependency_index_v2_receipt_sha256
            != self._replay_dependency_index_v2.semantic_receipt_sha256
            or self.base_runtime_sources_v1_receipt_sha256
            != self._base_runtime_sources_v1.semantic_receipt_sha256
            or self.base_source_bundle_v1_receipt_sha256
            != self._source_bundle_v2.base_source_bundle_v1_receipt_sha256
            or self.base_runtime_source_graph_v1_receipt_sha256
            != self._runtime_source_graph_v2.base_runtime_source_graph_v1_receipt_sha256
            or self.base_runtime_source_graph_v1_witness_receipt_sha256
            != self._runtime_source_graph_v2.base_runtime_source_graph_v1_witness_receipt_sha256
            or self.base_replay_dependency_index_v1_receipt_sha256
            != self._replay_dependency_index_v2.base_replay_dependency_index_v1_receipt_sha256
            or self.training_source_projection_sha256
            != self._source_bundle_v2.training_source_projection_sha256
            or self.validation_source_projection_sha256
            != self._source_bundle_v2.validation_source_projection_sha256
            or self.validation_feature_receipt_inventories != feature_rows
            or self.validation_action_receipt_inventories != action_rows
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
                "runtime sources V2 identity or authorization differs"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.source_bundle_v2_receipt_sha256,
            self.runtime_source_graph_v2_receipt_sha256,
            self.runtime_source_graph_v2_witness_receipt_sha256,
            self.replay_dependency_index_v2_receipt_sha256,
            self.base_runtime_sources_v1_receipt_sha256,
            self.base_source_bundle_v1_receipt_sha256,
            self.base_runtime_source_graph_v1_receipt_sha256,
            self.base_runtime_source_graph_v1_witness_receipt_sha256,
            self.base_replay_dependency_index_v1_receipt_sha256,
            self.training_source_projection_sha256,
            self.validation_source_projection_sha256,
            *(
                receipt
                for inventories in (
                    self.validation_feature_receipt_inventories,
                    self.validation_action_receipt_inventories,
                )
                for rows in inventories
                for receipt in rows
            ),
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("runtime sources V2", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_runtime_sources_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    source_bundle_v2: MassiveAdaptiveRLSourceBundleV2,
    runtime_source_graph_v2: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2,
    replay_dependency_index_v2: MassiveAdaptiveRLReplayDependencyIndexV2,
    runtime_sources_v1: MassiveAdaptiveRLRuntimeSourcesV1,
) -> MassiveAdaptiveRLRuntimeSourcesV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV3
        or type(source_bundle_v2) is not MassiveAdaptiveRLSourceBundleV2
        or type(runtime_source_graph_v2)
        is not MassiveAdaptiveRLRuntimeSourceGraphAuthorityV2
        or type(replay_dependency_index_v2)
        is not MassiveAdaptiveRLReplayDependencyIndexV2
        or type(runtime_sources_v1) is not MassiveAdaptiveRLRuntimeSourcesV1
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "runtime sources V2 require exact generation types"
        )
    manifest.validate()
    source_bundle_v2.validate()
    runtime_source_graph_v2.validate()
    replay_dependency_index_v2.validate()
    runtime_sources_v1.validate()
    graph_witness = runtime_source_graph_v2.runtime_authority_receipt_sha256
    if (
        not source_bundle_v2.development_stage_authorized
        or not runtime_source_graph_v2.development_stage_authorized
        or not replay_dependency_index_v2.development_stage_authorized
        or not runtime_sources_v1.source_data_qualified
        or graph_witness is None
        or runtime_sources_v1.manifest_v3_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "runtime sources cannot promote to V2"
        )
    feature_rows, action_rows = _validation_receipt_inventories(runtime_sources_v1)
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "source_bundle_v2_receipt_sha256": source_bundle_v2.semantic_receipt_sha256,
        "runtime_source_graph_v2_receipt_sha256": (
            runtime_source_graph_v2.semantic_receipt_sha256
        ),
        "runtime_source_graph_v2_witness_receipt_sha256": graph_witness,
        "replay_dependency_index_v2_receipt_sha256": (
            replay_dependency_index_v2.semantic_receipt_sha256
        ),
        "base_runtime_sources_v1_receipt_sha256": (
            runtime_sources_v1.semantic_receipt_sha256
        ),
        "base_source_bundle_v1_receipt_sha256": (
            source_bundle_v2.base_source_bundle_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_receipt_sha256": (
            runtime_source_graph_v2.base_runtime_source_graph_v1_receipt_sha256
        ),
        "base_runtime_source_graph_v1_witness_receipt_sha256": (
            runtime_source_graph_v2.base_runtime_source_graph_v1_witness_receipt_sha256
        ),
        "base_replay_dependency_index_v1_receipt_sha256": (
            replay_dependency_index_v2.base_replay_dependency_index_v1_receipt_sha256
        ),
        "training_source_projection_sha256": (
            source_bundle_v2.training_source_projection_sha256
        ),
        "validation_source_projection_sha256": (
            source_bundle_v2.validation_source_projection_sha256
        ),
        "validation_feature_receipt_inventories": feature_rows,
        "validation_action_receipt_inventories": action_rows,
        "source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLRuntimeSourcesV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        _source_bundle_v2=source_bundle_v2,
        _runtime_source_graph_v2=runtime_source_graph_v2,
        _replay_dependency_index_v2=replay_dependency_index_v2,
        _base_runtime_sources_v1=runtime_sources_v1,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility(
    *,
    runtime_sources_v2: MassiveAdaptiveRLRuntimeSourcesV2,
    four_fold_fit_authority: MassiveAdaptiveRLFourFoldFitAuthorityV1,
) -> None:
    """Reject adoption unless training used the exact V1 projection under V2."""

    if (
        type(runtime_sources_v2) is not MassiveAdaptiveRLRuntimeSourcesV2
        or type(four_fold_fit_authority) is not MassiveAdaptiveRLFourFoldFitAuthorityV1
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "training compatibility requires exact authority types"
        )
    runtime_sources_v2.validate()
    four_fold_fit_authority.validate()
    if (
        not four_fold_fit_authority.development_stage_authorized
        or not four_fold_fit_authority.source_transaction_verified
        or four_fold_fit_authority.experiment_id != runtime_sources_v2.experiment_id
        or four_fold_fit_authority.manifest_v3_receipt_sha256
        != runtime_sources_v2.manifest_v3_receipt_sha256
        or four_fold_fit_authority.runtime_sources_receipt_sha256
        != runtime_sources_v2.base_runtime_sources_v1_receipt_sha256
        or four_fold_fit_authority.runtime_graph_witness_receipt_sha256
        != runtime_sources_v2.base_runtime_source_graph_v1_witness_receipt_sha256
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "four-fold fit and validation-complete V2 sources are mixed"
        )


def reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLRuntimeSourcesV2:
    """Cold-replay V1, publish/replay V2 envelopes, and return one V2 witness."""

    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV3:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV2Error(
            "runtime sources V2 require exact Manifest V3 training lineage"
        )
    manifest.validate()
    try:
        runtime_v1 = reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
            source_root=source_root,
            manifest=manifest,
        )
    except MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable:
        raise
    except MassiveAdaptiveRLRuntimeSourceReconstructionV1Error as error:
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "legacy or invalid V1 source generation must be regenerated before V2 promotion"
        ) from error
    base_graph = runtime_v1.runtime_source_graph_authority
    base_bundle = base_graph._runtime_source_bundle
    if (
        type(base_graph) is not MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1
        or type(base_bundle) is not MassiveAdaptiveRLSourceBundleV1
    ):
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "cold V1 replay did not retain the exact source-bundle witness"
        )
    source_bundle_v2 = prepare_or_resume_massive_adaptive_rl_source_bundle_v2(
        source_root=source_root,
        manifest=manifest.base_manifest,
        source_bundle_v1=base_bundle,
        committed_at_ms=committed_at_ms,
        allow_materialize=allow_materialize,
    )
    graph_commit = max(
        committed_at_ms + 1,
        cast(int, source_bundle_v2.source_transaction_committed_at_ms) + 1,
    )
    graph_v2 = prepare_or_resume_massive_adaptive_rl_runtime_source_graph_authority_v2(
        source_root=source_root,
        manifest=manifest,
        source_bundle_v2=source_bundle_v2,
        runtime_source_graph_v1=base_graph,
        committed_at_ms=graph_commit,
        allow_materialize=allow_materialize,
    )
    try:
        base_index = load_massive_adaptive_rl_replay_dependency_index_v1(
            source_root=source_root,
            manifest=manifest,
        )
    except MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable:
        raise
    except MassiveAdaptiveRLRuntimeSourceReconstructionV1Error as error:
        raise MassiveAdaptiveRLMixedRuntimeSourceGenerationError(
            "legacy or invalid V1 dependency index must be regenerated before V2 promotion"
        ) from error
    index_commit = max(
        committed_at_ms + 2,
        cast(int, graph_v2.source_transaction_committed_at_ms) + 1,
    )
    index_v2 = prepare_or_resume_massive_adaptive_rl_replay_dependency_index_v2(
        source_root=source_root,
        manifest=manifest,
        runtime_source_graph_v2=graph_v2,
        replay_dependency_index_v1=base_index,
        runtime_sources_v1=runtime_v1,
        committed_at_ms=index_commit,
        allow_materialize=allow_materialize,
    )
    return build_massive_adaptive_rl_runtime_sources_v2(
        manifest=manifest,
        source_bundle_v2=source_bundle_v2,
        runtime_source_graph_v2=graph_v2,
        replay_dependency_index_v2=index_v2,
        runtime_sources_v1=runtime_v1,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V2_SPEC_SHA256",
    "MassiveAdaptiveRLMixedRuntimeSourceGenerationError",
    "MassiveAdaptiveRLReplayDependencyIndexV2",
    "MassiveAdaptiveRLRuntimeSourceReconstructionV2Error",
    "MassiveAdaptiveRLRuntimeSourcesV2",
    "authorize_massive_adaptive_rl_replay_dependency_index_v2",
    "build_massive_adaptive_rl_replay_dependency_index_v2",
    "build_massive_adaptive_rl_runtime_sources_v2",
    "load_massive_adaptive_rl_replay_dependency_index_v2",
    "materialize_massive_adaptive_rl_replay_dependency_index_v2",
    "parse_massive_adaptive_rl_replay_dependency_index_v2",
    "prepare_or_resume_massive_adaptive_rl_replay_dependency_index_v2",
    "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v2",
    "replay_dependency_index_relative_path_v2",
    "validate_massive_adaptive_rl_runtime_sources_v2_training_compatibility",
]
