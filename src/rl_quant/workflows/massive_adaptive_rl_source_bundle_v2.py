"""Versioned validation-complete source bundle for adaptive RL.

The original source-bundle generation predated the fold-scoped validation
predictor inventories.  Those inventories were temporarily added to the V1
implementation while the validation boundary was being built.  This module
creates an explicit V2 envelope around that transitional bundle.  A V1 bundle
cannot be promoted merely because it parses: promotion requires the complete
validation feature/action role inventory and binds both the fit-only and
validation-only projections.

The V2 envelope is create-only and generic reload remains nonauthorizing.
Manifest V5 and later runners can therefore require V2 without treating older
V1 training evidence as though it had always contained validation inputs.
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
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256,
    MassiveAdaptiveRLSourceBundleV1,
    MassiveAdaptiveRLSourceBundleV1Error,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
)


MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA = (
    "rl-quant.massive-adaptive-rl-source-bundle-v2"
)
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_DATASET = "massive-adaptive-rl-source-bundle-v2"
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA,
        "encoding": "canonical-json-validation-complete-source-bundle-v2",
        "generic_reload": "nonauthorizing",
    }
)
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SPEC_SHA256 = semantic_sha256(
    {
        "base_generation": "explicit-v1-training-source-bundle-receipt",
        "base_v1_specification_sha256": (
            MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256
        ),
        "base_v1_implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256
        ),
        "roles": "complete-fixed-source-role-inventory",
        "training_projection": "all-nonvalidation-source-roles",
        "validation_projection": (
            "four-feature-and-four-action-origin-primary-inventories"
        ),
        "legacy_v1": "validation-incomplete-v1-cannot-promote",
        "mixed_generation": "rejected",
        "publication": "manifest-derived-create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "profitability_reporting": False,
        "outer_access": False,
        "lockbox_access": False,
    }
)

_VALIDATION_FEATURE_ROLE = "validation-origin-feature-inventory"
_VALIDATION_ACTION_ROLE = "validation-origin-action-inventory"
_VALIDATION_ROLES = frozenset({_VALIDATION_FEATURE_ROLE, _VALIDATION_ACTION_ROLE})
_FOLD_INDICES = (0, 1, 2, 3)


class MassiveAdaptiveRLSourceBundleV2Error(ValueError):
    """The validation-complete source generation is absent or inconsistent."""


class MassiveAdaptiveRLLegacySourceBundleV1Error(MassiveAdaptiveRLSourceBundleV2Error):
    """A fit-only or otherwise validation-incomplete V1 bundle was supplied."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def source_bundle_relative_path_v2(
    *, manifest: MassiveAdaptiveRLExperimentManifestV2
) -> str:
    manifest.validate()
    return (
        "adaptive-rl/source-bundle-v2/"
        f"{manifest.experiment_id}-m2-{manifest.semantic_receipt_sha256}.json"
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
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "adaptive RL source-bundle V2 transaction is incomplete"
        )
    return all(present)


def _artifact_inventory(
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
) -> tuple[tuple[str, int | None, str], ...]:
    return tuple(
        (row.role, row.fold_index, row.receipt_sha256)
        for row in source_bundle.artifacts
    )


def _projection_receipt(
    inventory: Sequence[tuple[str, int | None, str]], *, validation: bool
) -> str:
    return semantic_sha256(
        tuple(row for row in inventory if (row[0] in _VALIDATION_ROLES) is validation)
    )


def _validation_receipts(
    inventory: Sequence[tuple[str, int | None, str]], *, role: str
) -> tuple[str, ...]:
    rows = tuple(row for row in inventory if row[0] == role)
    if (
        tuple(row[1] for row in rows) != _FOLD_INDICES
        or len({row[1] for row in rows}) != 4
    ):
        raise MassiveAdaptiveRLLegacySourceBundleV1Error(
            "legacy source-bundle V1 lacks the complete validation predictor inventory"
        )
    return tuple(row[2] for row in rows)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLSourceBundleV2:
    experiment_id: str
    manifest_v2_receipt_sha256: str
    base_source_bundle_v1_receipt_sha256: str
    base_source_bundle_v1_specification_sha256: str
    base_source_bundle_v1_implementation_source_sha256: str
    artifact_key_receipt_inventory: tuple[tuple[str, int | None, str], ...]
    artifact_inventory_sha256: str
    training_source_projection_sha256: str
    validation_source_projection_sha256: str
    validation_feature_artifact_receipts: tuple[str, ...]
    validation_action_artifact_receipts: tuple[str, ...]
    validation_origin_feature_inventory_sha256: str
    validation_origin_action_inventory_sha256: str
    committed_source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_base_bundle_replayed: bool = False
    source_data_qualified: bool = False
    profitability_reporting_authorized: bool = False
    outer_evaluation_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA
    _base_source_bundle: MassiveAdaptiveRLSourceBundleV1 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v2_receipt_sha256": self.manifest_v2_receipt_sha256,
            "base_source_bundle_v1_receipt_sha256": (
                self.base_source_bundle_v1_receipt_sha256
            ),
            "base_source_bundle_v1_specification_sha256": (
                self.base_source_bundle_v1_specification_sha256
            ),
            "base_source_bundle_v1_implementation_source_sha256": (
                self.base_source_bundle_v1_implementation_source_sha256
            ),
            "artifact_key_receipt_inventory": self.artifact_key_receipt_inventory,
            "artifact_inventory_sha256": self.artifact_inventory_sha256,
            "training_source_projection_sha256": (
                self.training_source_projection_sha256
            ),
            "validation_source_projection_sha256": (
                self.validation_source_projection_sha256
            ),
            "validation_feature_artifact_receipts": (
                self.validation_feature_artifact_receipts
            ),
            "validation_action_artifact_receipts": (
                self.validation_action_artifact_receipts
            ),
            "validation_origin_feature_inventory_sha256": (
                self.validation_origin_feature_inventory_sha256
            ),
            "validation_origin_action_inventory_sha256": (
                self.validation_origin_action_inventory_sha256
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
            and self.runtime_base_bundle_replayed
            and self.source_data_qualified
        )

    @property
    def base_source_bundle(self) -> MassiveAdaptiveRLSourceBundleV1:
        self.validate()
        if self._base_source_bundle is None:
            raise MassiveAdaptiveRLSourceBundleV2Error(
                "source-bundle V2 has no runtime V1 witness"
            )
        return self._base_source_bundle

    def validate(self) -> None:
        runtime_present = self._base_source_bundle is not None
        persisted = self._loaded_source is not None
        inventory = self.artifact_key_receipt_inventory
        keys = tuple((row[0], row[1]) for row in inventory)
        if self._base_source_bundle is not None:
            self._base_source_bundle.validate()
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA
            or not self.experiment_id
            or not inventory
            or keys
            != tuple(
                sorted(
                    keys,
                    key=lambda value: (
                        value[1] is not None,
                        value[1] if value[1] is not None else -1,
                        value[0],
                    ),
                )
            )
            or len(set(keys)) != len(keys)
            or self.artifact_inventory_sha256 != semantic_sha256(inventory)
            or self.training_source_projection_sha256
            != _projection_receipt(inventory, validation=False)
            or self.validation_source_projection_sha256
            != _projection_receipt(inventory, validation=True)
            or self.validation_feature_artifact_receipts
            != _validation_receipts(inventory, role=_VALIDATION_FEATURE_ROLE)
            or self.validation_action_artifact_receipts
            != _validation_receipts(inventory, role=_VALIDATION_ACTION_ROLE)
            or self.committed_source_data_qualified is not True
            or self.runtime_base_bundle_replayed != runtime_present
            or self.source_data_qualified
            != bool(
                persisted and runtime_present and self.committed_source_data_qualified
            )
            or self.profitability_reporting_authorized
            or self.outer_evaluation_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLSourceBundleV2Error(
                "adaptive RL source-bundle V2 identity or authorization differs"
            )
        if runtime_present:
            assert self._base_source_bundle is not None
            base = self._base_source_bundle
            if (
                type(base) is not MassiveAdaptiveRLSourceBundleV1
                or not base.persisted_source_replayed
                or not base.runtime_source_replayed
                or not base.source_data_qualified
                or base.experiment_id != self.experiment_id
                or base.manifest_receipt_sha256 != self.manifest_v2_receipt_sha256
                or base.semantic_receipt_sha256
                != self.base_source_bundle_v1_receipt_sha256
                or base.specification_sha256
                != self.base_source_bundle_v1_specification_sha256
                or base.implementation_source_sha256
                != self.base_source_bundle_v1_implementation_source_sha256
                or _artifact_inventory(base) != inventory
                or base.validation_origin_feature_inventory_sha256
                != self.validation_origin_feature_inventory_sha256
                or base.validation_origin_action_inventory_sha256
                != self.validation_origin_action_inventory_sha256
            ):
                raise MassiveAdaptiveRLSourceBundleV2Error(
                    "source-bundle V2 runtime witness is mixed or differs"
                )
        if self._loaded_source is not None and (
            self._loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_DATASET
            or self._loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SCHEMA_SHA256
            or self._loaded_source.receipt.entitlement_receipt_sha256
            != self.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLSourceBundleV2Error(
                "source-bundle V2 source transaction differs"
            )
        for value in (
            self.manifest_v2_receipt_sha256,
            self.base_source_bundle_v1_receipt_sha256,
            self.base_source_bundle_v1_specification_sha256,
            self.base_source_bundle_v1_implementation_source_sha256,
            self.artifact_inventory_sha256,
            self.training_source_projection_sha256,
            self.validation_source_projection_sha256,
            *self.validation_feature_artifact_receipts,
            *self.validation_action_artifact_receipts,
            self.validation_origin_feature_inventory_sha256,
            self.validation_origin_action_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL source-bundle V2", value)
        for role, fold_index, receipt in inventory:
            if not role or (
                fold_index is not None
                and (isinstance(fold_index, bool) or fold_index not in _FOLD_INDICES)
            ):
                raise MassiveAdaptiveRLSourceBundleV2Error(
                    "source-bundle V2 artifact key differs"
                )
            _digest("source-bundle V2 artifact", receipt)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_source_bundle_v2(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    source_bundle_v1: MassiveAdaptiveRLSourceBundleV1,
) -> MassiveAdaptiveRLSourceBundleV2:
    if (
        type(manifest) is not MassiveAdaptiveRLExperimentManifestV2
        or type(source_bundle_v1) is not MassiveAdaptiveRLSourceBundleV1
    ):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 requires exact manifest and V1 source bundle"
        )
    manifest.validate()
    try:
        source_bundle_v1.validate()
    except MassiveAdaptiveRLSourceBundleV1Error as error:
        raise MassiveAdaptiveRLLegacySourceBundleV1Error(
            "legacy or invalid source-bundle V1 cannot promote to V2"
        ) from error
    if (
        source_bundle_v1.experiment_id != manifest.experiment_id
        or source_bundle_v1.manifest_receipt_sha256 != manifest.semantic_receipt_sha256
        or not source_bundle_v1.persisted_source_replayed
        or not source_bundle_v1.runtime_source_replayed
        or not source_bundle_v1.source_data_qualified
        or source_bundle_v1.specification_sha256
        != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256
        or source_bundle_v1.implementation_source_sha256
        != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256
    ):
        raise MassiveAdaptiveRLLegacySourceBundleV1Error(
            "legacy or unqualified source-bundle V1 cannot promote to V2"
        )
    inventory = _artifact_inventory(source_bundle_v1)
    feature_receipts = _validation_receipts(inventory, role=_VALIDATION_FEATURE_ROLE)
    action_receipts = _validation_receipts(inventory, role=_VALIDATION_ACTION_ROLE)
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v2_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_source_bundle_v1_receipt_sha256": (
            source_bundle_v1.semantic_receipt_sha256
        ),
        "base_source_bundle_v1_specification_sha256": (
            source_bundle_v1.specification_sha256
        ),
        "base_source_bundle_v1_implementation_source_sha256": (
            source_bundle_v1.implementation_source_sha256
        ),
        "artifact_key_receipt_inventory": inventory,
        "artifact_inventory_sha256": semantic_sha256(inventory),
        "training_source_projection_sha256": _projection_receipt(
            inventory, validation=False
        ),
        "validation_source_projection_sha256": _projection_receipt(
            inventory, validation=True
        ),
        "validation_feature_artifact_receipts": feature_receipts,
        "validation_action_artifact_receipts": action_receipts,
        "validation_origin_feature_inventory_sha256": (
            source_bundle_v1.validation_origin_feature_inventory_sha256
        ),
        "validation_origin_action_inventory_sha256": (
            source_bundle_v1.validation_origin_action_inventory_sha256
        ),
        "committed_source_data_qualified": True,
    }
    provisional = MassiveAdaptiveRLSourceBundleV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_base_bundle_replayed=True,
        source_data_qualified=False,
        _base_source_bundle=source_bundle_v1,
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
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 payload is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    rows = body.get("artifact_key_receipt_inventory")
    if not isinstance(rows, list):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 artifact inventory is malformed"
        )
    body["artifact_key_receipt_inventory"] = tuple(
        (
            cast(list[object], row)[0],
            cast(list[object], row)[1],
            cast(list[object], row)[2],
        )
        for row in rows
        if isinstance(row, list) and len(row) == 3
    )
    if len(cast(tuple[object, ...], body["artifact_key_receipt_inventory"])) != len(
        rows
    ):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 artifact row is malformed"
        )
    for name in (
        "validation_feature_artifact_receipts",
        "validation_action_artifact_receipts",
    ):
        values = body.get(name)
        if not isinstance(values, list):
            raise MassiveAdaptiveRLSourceBundleV2Error(
                "source-bundle V2 validation inventory is malformed"
            )
        body[name] = tuple(values)
    return body


def parse_massive_adaptive_rl_source_bundle_v2(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLSourceBundleV2:
    body = _parse_body(root=root, loaded_source=loaded_source)
    result = MassiveAdaptiveRLSourceBundleV2(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        runtime_base_bundle_replayed=False,
        source_data_qualified=False,
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_source_bundle_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    verified_at_ms: int,
) -> MassiveAdaptiveRLSourceBundleV2:
    return parse_massive_adaptive_rl_source_bundle_v2(
        root=source_root,
        loaded_source=load_massive_source_bundle(
            root=source_root,
            relative_payload_path=source_bundle_relative_path_v2(manifest=manifest),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_source_bundle_v2(
    *,
    authority: MassiveAdaptiveRLSourceBundleV2,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    source_bundle_v1: MassiveAdaptiveRLSourceBundleV1,
) -> MassiveAdaptiveRLSourceBundleV2:
    authority.validate()
    expected = build_massive_adaptive_rl_source_bundle_v2(
        manifest=manifest,
        source_bundle_v1=source_bundle_v1,
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != source_bundle_relative_path_v2(manifest=manifest)
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 does not replay from the exact V1 generation"
        )
    result = replace(
        authority,
        runtime_base_bundle_replayed=True,
        source_data_qualified=authority.committed_source_data_qualified,
        _base_source_bundle=source_bundle_v1,
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_source_bundle_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    authority: MassiveAdaptiveRLSourceBundleV2,
    committed_at_ms: int,
) -> MassiveAdaptiveRLSourceBundleV2:
    manifest.validate()
    authority.validate()
    if (
        authority._base_source_bundle is None
        or authority.runtime_base_bundle_replayed is not True
        or authority.experiment_id != manifest.experiment_id
        or authority.manifest_v2_receipt_sha256 != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLSourceBundleV2Error(
            "source-bundle V2 is not runtime witnessed"
        )
    relative = source_bundle_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        raise MassiveAdaptiveRLSourceBundleV2Error("source-bundle V2 already exists")
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=source_root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=authority.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-SOURCE-BUNDLE-V2-{authority.experiment_id}",
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


def prepare_or_resume_massive_adaptive_rl_source_bundle_v2(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    source_bundle_v1: MassiveAdaptiveRLSourceBundleV1,
    committed_at_ms: int,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLSourceBundleV2:
    relative = source_bundle_relative_path_v2(manifest=manifest)
    if _source_transaction_exists(root=source_root, relative=relative):
        return authorize_massive_adaptive_rl_source_bundle_v2(
            authority=load_massive_adaptive_rl_source_bundle_v2(
                source_root=source_root,
                manifest=manifest,
                verified_at_ms=committed_at_ms,
            ),
            manifest=manifest,
            source_bundle_v1=source_bundle_v1,
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLSourceBundleV2Error("source-bundle V2 is absent")
    return materialize_massive_adaptive_rl_source_bundle_v2(
        source_root=source_root,
        manifest=manifest,
        authority=build_massive_adaptive_rl_source_bundle_v2(
            manifest=manifest,
            source_bundle_v1=source_bundle_v1,
        ),
        committed_at_ms=committed_at_ms,
    )


__all__ = [
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_DATASET",
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V2_SPEC_SHA256",
    "MassiveAdaptiveRLLegacySourceBundleV1Error",
    "MassiveAdaptiveRLSourceBundleV2",
    "MassiveAdaptiveRLSourceBundleV2Error",
    "authorize_massive_adaptive_rl_source_bundle_v2",
    "build_massive_adaptive_rl_source_bundle_v2",
    "load_massive_adaptive_rl_source_bundle_v2",
    "materialize_massive_adaptive_rl_source_bundle_v2",
    "parse_massive_adaptive_rl_source_bundle_v2",
    "prepare_or_resume_massive_adaptive_rl_source_bundle_v2",
    "source_bundle_relative_path_v2",
]
