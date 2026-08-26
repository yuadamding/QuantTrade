"""Blind target commitment for the fixed Massive P0 lockbox dates."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_profitability_accounting_freeze_v1 import (
    MassiveProfitabilityAccountingFreezeV1,
)
from rl_quant.features.massive_profitability_archive_freeze_v1 import (
    MassiveProfitabilityArchiveFreezeV1,
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

MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA = (
    "rl-quant.massive-profitability-lockbox-target-seal-v1"
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_DATASET = (
    "massive-profitability-lockbox-target-seal-v1"
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        "format": "canonical-json-newline",
        "fields": "exact",
    }
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_SCHEMA = (
    "rl-quant.massive-profitability-lockbox-target-blob-v1"
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_DATASET = (
    "massive-profitability-lockbox-target-blob-v1"
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_SCHEMA,
        "format": "canonical-json-newline",
        "access": "separately-permissioned-root",
    }
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "dates": "exact-fixed-252-candidate-lockbox-dates",
        "public": "commitment-only-no-target-values",
        "target_blob": "separately-permissioned-root",
        "opening": "not-implemented-or-authorized-by-seal",
    }
)


class MassiveProfitabilityLockboxTargetSealV1Error(ValueError):
    """Lockbox targets are not sealed outside the training artifact root."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveProfitabilityLockboxTargetSealV1:
    lockbox_session_dates: tuple[str, ...]
    archive_freeze_semantic_receipt_sha256: str
    accounting_freeze_semantic_receipt_sha256: str
    target_semantic_receipts: tuple[str, ...]
    target_inventory_sha256: str
    sealed_blob_physical_sha256: str
    sealed_blob_source_receipt_sha256: str
    sealed_blob_commit_receipt_sha256: str
    sealed_target_count: int
    public_commitment_only: bool
    separate_permission_root_verified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    audit_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    development_training_authorized: bool
    profitability_reporting_authorized: bool
    lockbox_access_authorized: bool
    schema: str = MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "audit_receipt_sha256",
                "loaded_source",
            }
        }

    def validate(self) -> None:
        if (
            self.schema != MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA
            or len(self.lockbox_session_dates) != 252
            or self.lockbox_session_dates
            != tuple(sorted(set(self.lockbox_session_dates)))
            or len(self.target_semantic_receipts) != 252
            or self.sealed_target_count != 252
            or self.public_commitment_only is not True
            or self.separate_permission_root_verified is not True
            or self.protocol_receipt_sha256
            != MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SHA256
            or any(
                (
                    self.development_training_authorized,
                    self.profitability_reporting_authorized,
                    self.lockbox_access_authorized,
                )
            )
        ):
            raise MassiveProfitabilityLockboxTargetSealV1Error(
                "lockbox seal identity or authorization differs"
            )
        if self.target_inventory_sha256 != semantic_sha256(
            self.target_semantic_receipts
        ):
            raise MassiveProfitabilityLockboxTargetSealV1Error(
                "lockbox target inventory differs"
            )
        for value in (
            self.archive_freeze_semantic_receipt_sha256,
            self.accounting_freeze_semantic_receipt_sha256,
            *self.target_semantic_receipts,
            self.target_inventory_sha256,
            self.sealed_blob_physical_sha256,
            self.sealed_blob_source_receipt_sha256,
            self.sealed_blob_commit_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
            self.audit_receipt_sha256,
        ):
            _digest("lockbox seal", value)
        if self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveProfitabilityLockboxTargetSealV1Error(
                "lockbox seal semantic receipt differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SCHEMA_SHA256
            or self.audit_receipt_sha256
            != semantic_sha256(
                {
                    "semantic_receipt_sha256": self.semantic_receipt_sha256,
                    "loaded_source_receipt_sha256": self.loaded_source.receipt_sha256,
                }
            )
        ):
            raise MassiveProfitabilityLockboxTargetSealV1Error(
                "lockbox seal committed source differs"
            )


def materialize_massive_profitability_lockbox_target_seal_v1(
    *,
    public_root: str | Path,
    sealed_root: str | Path,
    archive_freeze: MassiveProfitabilityArchiveFreezeV1,
    accounting_freeze: MassiveProfitabilityAccountingFreezeV1,
    lockbox_targets: Sequence[MassiveProfitabilityTargetsV2],
    sealed_at_ms: int,
    entitlement_receipt_sha256: str,
    artifact_id: str,
) -> MassiveProfitabilityLockboxTargetSealV1:
    """Store target bytes in a private root and publish only their commitment."""

    archive_freeze.validate()
    accounting_freeze.validate()
    public = Path(public_root).resolve()
    private = Path(sealed_root).resolve()
    if public == private or public in private.parents or private in public.parents:
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            "lockbox target root is not separate from the public training root"
        )
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    if stat.S_IMODE(private.stat().st_mode) & 0o077:
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            "lockbox target root permits group or other access"
        )
    targets = tuple(sorted(lockbox_targets, key=lambda row: row.decision_session_date))
    dates = tuple(row.decision_session_date for row in targets)
    if (
        dates != archive_freeze.fixed_lockbox_session_dates
        or accounting_freeze.archive_freeze_semantic_receipt_sha256
        != archive_freeze.semantic_receipt_sha256
    ):
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            "lockbox target dates or accounting freeze differ"
        )
    for target in targets:
        target.validate()
        if (
            target.accounting_freeze_semantic_receipt_sha256
            != accounting_freeze.semantic_receipt_sha256
        ):
            raise MassiveProfitabilityLockboxTargetSealV1Error(
                "lockbox target uses a different accounting freeze"
            )
    target_receipts = tuple(row.semantic_receipt_sha256 for row in targets)
    blob_payload = {
        "schema": MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_SCHEMA,
        "lockbox_session_dates": dates,
        "targets": tuple(asdict(row) for row in targets),
        "target_inventory_sha256": semantic_sha256(target_receipts),
    }
    private_relative = f"massive-profitability-lockbox-target-blob-v1/{artifact_id}.json"
    sealed_receipt, sealed_commit = publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(blob_payload)),
        root=private,
        relative_payload_path=private_relative,
        dataset_id=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_DATASET,
        source_object_key=private_relative,
        requested_at_ms=sealed_at_ms,
        downloaded_at_ms=sealed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_BLOB_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "lockbox seal entitlement", entitlement_receipt_sha256
        ),
        committed_at_ms=sealed_at_ms,
    )
    payload = {
        "schema": MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA,
        "lockbox_session_dates": dates,
        "archive_freeze_semantic_receipt_sha256": (
            archive_freeze.semantic_receipt_sha256
        ),
        "accounting_freeze_semantic_receipt_sha256": (
            accounting_freeze.semantic_receipt_sha256
        ),
        "target_semantic_receipts": target_receipts,
        "target_inventory_sha256": semantic_sha256(target_receipts),
        "sealed_blob_physical_sha256": sealed_receipt.physical_sha256,
        "sealed_blob_source_receipt_sha256": sealed_receipt.receipt_sha256,
        "sealed_blob_commit_receipt_sha256": sealed_commit.receipt_sha256,
        "sealed_target_count": len(targets),
        "public_commitment_only": True,
        "separate_permission_root_verified": True,
        "protocol_receipt_sha256": MASSIVE_FINALIZED_PROFITABILITY_P0_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SHA256
        ),
        "development_training_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
    }
    relative = f"massive-profitability-lockbox-target-seal-v1/{artifact_id}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=public,
        relative_payload_path=relative,
        dataset_id=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=sealed_at_ms,
        downloaded_at_ms=sealed_at_ms,
        schema_sha256=MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        committed_at_ms=sealed_at_ms,
    )
    loaded = load_massive_source_bundle(
        root=public, relative_payload_path=relative, verified_at_ms=sealed_at_ms
    )
    return parse_massive_profitability_lockbox_target_seal_v1(
        root=public, loaded_source=loaded
    )


def parse_massive_profitability_lockbox_target_seal_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveProfitabilityLockboxTargetSealV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            "lockbox target seal is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveProfitabilityLockboxTargetSealV1Error(
            "lockbox target seal is not canonical JSON"
        )
    payload["lockbox_session_dates"] = tuple(payload["lockbox_session_dates"])
    payload["target_semantic_receipts"] = tuple(
        payload["target_semantic_receipts"]
    )
    receipt = semantic_sha256(payload)
    result = MassiveProfitabilityLockboxTargetSealV1(
        **payload,  # type: ignore[arg-type]
        semantic_receipt_sha256=receipt,
        audit_receipt_sha256=semantic_sha256(
            {
                "semantic_receipt_sha256": receipt,
                "loaded_source_receipt_sha256": loaded_source.receipt_sha256,
            }
        ),
        loaded_source=loaded_source,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_PROFITABILITY_LOCKBOX_TARGET_SEAL_V1_SCHEMA",
    "MassiveProfitabilityLockboxTargetSealV1",
    "MassiveProfitabilityLockboxTargetSealV1Error",
    "materialize_massive_profitability_lockbox_target_seal_v1",
    "parse_massive_profitability_lockbox_target_seal_v1",
]
