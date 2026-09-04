"""Create-only adoption marker for the unique Manifest-V5 writer graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
import json
from pathlib import Path
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_rl_prequential_validation_inputs_v1 import (
    initial_validation_inputs_authority_relative_path_v1,
    massive_adaptive_rl_forbidden_prequential_artifacts_v1,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA,
    MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256,
    MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1,
    MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1,
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Error,
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_artifact_root_writer_lock_v1,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    MassiveAdaptiveRLManifestV5WriterCapabilityV1,
    MassiveAdaptiveRLLegacyWriterRejectedByManifestV5,
    _issue_manifest_v5_writer_capability_v1,
    manifest_v5_registration_relative_path_v1,
    reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration,
)


MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_DATASET = (
    "massive-adaptive-rl-manifest-v5-registration-v1"
)
MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA,
            "encoding": "canonical-json-one-experiment-one-v5-writer-generation",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLManifestV5RegistrationError(ValueError):
    """Manifest V5 cannot be adopted or replayed exactly."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "adaptive RL registration experiment ID is not path safe"
        )
    return value


def _wall_clock_ms() -> int:
    value = time.time_ns() // 1_000_000
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "adaptive RL registration clock differs"
        )
    return value


def _transaction_state(*, root: str | Path, relative: str) -> tuple[bool, bool]:
    payload = Path(root) / relative
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    base_manifest_v4_receipt_sha256: str
    prequential_validation_plan_specification_sha256: str
    initial_validation_fold_indices: tuple[int, ...]
    withheld_validation_fold_indices: tuple[int, ...]
    validation_release_prerequisite_outer_fold_indices: tuple[int | None, ...]
    outer_to_validation_release_edges: tuple[tuple[int, int], ...]
    prequential_stage_sequence: tuple[str, ...]
    authority_generation_names: tuple[str, ...]
    authoritative_writer_generation: str
    diagnostic_only_continuation_required: bool
    legacy_writer_materialization_authorized: bool
    semantic_receipt_sha256: str
    runtime_manifest_replayed: bool = False
    validation_outcome_access_authorized: bool = False
    outer_access_authorized: bool = False
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA
    _runtime_manifest: MassiveAdaptiveRLExperimentManifestV5 | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_manifest_replayed",
            }
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
    def development_protocol_registered(self) -> bool:
        return bool(
            self.source_transaction_verified
            and self.runtime_manifest_replayed
            and self._runtime_manifest is not None
        )

    def validate(self) -> None:
        runtime_present = self._runtime_manifest is not None
        if self._loaded_source is not None:
            self._loaded_source.validate()
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA
            or _identifier(self.experiment_id) != self.experiment_id
            or self.initial_validation_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_INITIAL_VALIDATION_FOLDS_V1
            or self.withheld_validation_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_WITHHELD_VALIDATION_FOLDS_V1
            or self.validation_release_prerequisite_outer_fold_indices
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_PREREQUISITES_V1
            or self.outer_to_validation_release_edges
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_RELEASE_EDGES_V1
            or self.prequential_stage_sequence
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_STAGE_SEQUENCE_V1
            or self.authority_generation_names
            != MASSIVE_ADAPTIVE_RL_PREQUENTIAL_AUTHORITY_GENERATIONS_V1
            or self.authoritative_writer_generation
            != "massive-adaptive-rl-experiment-runner-v5"
            or not self.diagnostic_only_continuation_required
            or self.legacy_writer_materialization_authorized
            or self.runtime_manifest_replayed != runtime_present
            or runtime_present
            and (
                type(self._runtime_manifest)
                is not MassiveAdaptiveRLExperimentManifestV5
                or self._runtime_manifest.semantic_receipt_sha256
                != self.manifest_v5_receipt_sha256
            )
            or self.validation_outcome_access_authorized
            or self.outer_access_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self._loaded_source is not None
            and (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.manifest_v5_receipt_sha256
            )
        ):
            raise MassiveAdaptiveRLManifestV5RegistrationError(
                "adaptive RL Manifest V5 registration differs"
            )
        for name, value in self.semantic_unsigned().items():
            if name.endswith("_sha256"):
                _digest(name, value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def build_massive_adaptive_rl_manifest_v5_registration_authority_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV5:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration requires exact Manifest V5"
        )
    manifest.validate()
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_v4_receipt_sha256": (
            manifest.base_manifest.semantic_receipt_sha256
        ),
        "prequential_validation_plan_specification_sha256": (
            manifest.prequential_validation_plan_specification_sha256
        ),
        "initial_validation_fold_indices": manifest.initial_validation_fold_indices,
        "withheld_validation_fold_indices": manifest.withheld_validation_fold_indices,
        "validation_release_prerequisite_outer_fold_indices": (
            manifest.validation_release_prerequisite_outer_fold_indices
        ),
        "outer_to_validation_release_edges": (
            manifest.outer_to_validation_release_edges
        ),
        "prequential_stage_sequence": manifest.prequential_stage_sequence,
        "authority_generation_names": manifest.authority_generation_names,
        "authoritative_writer_generation": manifest.authoritative_writer_generation,
        "diagnostic_only_continuation_required": (
            manifest.diagnostic_only_continuation_required
        ),
        "legacy_writer_materialization_authorized": False,
        "validation_outcome_access_authorized": False,
        "outer_access_authorized": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLManifestV5RegistrationAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        runtime_manifest_replayed=True,
        _runtime_manifest=manifest,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _parse_registration(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration is not canonical JSON"
        )
    body = dict(cast(Mapping[str, object], value))
    for name in (
        "initial_validation_fold_indices",
        "withheld_validation_fold_indices",
        "validation_release_prerequisite_outer_fold_indices",
        "prequential_stage_sequence",
        "authority_generation_names",
    ):
        body[name] = tuple(cast(Sequence[object], body[name]))
    body["outer_to_validation_release_edges"] = tuple(
        tuple(cast(Sequence[int], row))
        for row in cast(Sequence[Sequence[int]], body["outer_to_validation_release_edges"])
    )
    result = MassiveAdaptiveRLManifestV5RegistrationAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_sha256(body),
        _loaded_source=loaded_source,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_manifest_v5_registration_authority_v1(
    *, root: str | Path, experiment_id: str, verified_at_ms: int
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    return _parse_registration(
        root=root,
        loaded_source=load_massive_source_bundle(
            root=root,
            relative_payload_path=manifest_v5_registration_relative_path_v1(
                experiment_id=experiment_id
            ),
            verified_at_ms=verified_at_ms,
        ),
    )


def authorize_massive_adaptive_rl_manifest_v5_registration_authority_v1(
    *,
    authority: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    authority.validate()
    expected = build_massive_adaptive_rl_manifest_v5_registration_authority_v1(
        manifest=manifest
    )
    if (
        authority._loaded_source is None
        or authority._loaded_source.payload_relative_path
        != manifest_v5_registration_relative_path_v1(
            experiment_id=manifest.experiment_id
        )
        or authority.semantic_unsigned() != expected.semantic_unsigned()
    ):
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration did not replay"
        )
    result = replace(
        authority,
        runtime_manifest_replayed=True,
        _runtime_manifest=manifest,
    )
    result.validate()
    return result


def issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1(
    *, authority: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1
) -> MassiveAdaptiveRLManifestV5WriterCapabilityV1:
    """Issue the only compatibility capability available before outcomes."""

    authority.validate()
    source_receipt = authority.source_receipt_sha256
    commit_receipt = authority.source_transaction_receipt_sha256
    if (
        not authority.development_protocol_registered
        or source_receipt is None
        or commit_receipt is None
    ):
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "initial-input writer capability requires replayed V5 registration"
        )
    return _issue_manifest_v5_writer_capability_v1(
        experiment_id=authority.experiment_id,
        manifest_v5_receipt_sha256=authority.manifest_v5_receipt_sha256,
        base_manifest_v4_receipt_sha256=authority.base_manifest_v4_receipt_sha256,
        registration_authority_receipt_sha256=authority.semantic_receipt_sha256,
        registration_source_receipt_sha256=source_receipt,
        registration_commit_receipt_sha256=commit_receipt,
        writer_role="initial-validation-inputs",
        allowed_fold_indices=(0, 1),
    )


def _run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1_unlocked(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    """Adopt V5 before any validation input or legacy validation evidence."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration materialization mode differs"
        )
    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV5:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration requires exact Manifest V5"
    )
    manifest.validate()
    registration_root = Path(root)
    if registration_root.is_symlink():
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration root is not a no-follow directory"
        )
    if not registration_root.exists():
        if not allow_materialize:
            raise MassiveAdaptiveRLManifestV5RegistrationError(
                "Manifest V5 registration is absent"
            )
        registration_root.mkdir(parents=True, exist_ok=True)
    if registration_root.is_symlink() or not registration_root.is_dir():
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration root is not a no-follow directory"
        )
    relative = manifest_v5_registration_relative_path_v1(
        experiment_id=manifest.experiment_id
    )
    complete, partial = _transaction_state(root=root, relative=relative)
    if partial:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration transaction is incomplete"
        )
    verified_at_ms = _wall_clock_ms()
    if complete:
        return authorize_massive_adaptive_rl_manifest_v5_registration_authority_v1(
            authority=load_massive_adaptive_rl_manifest_v5_registration_authority_v1(
                root=root,
                experiment_id=manifest.experiment_id,
                verified_at_ms=verified_at_ms,
            ),
            manifest=manifest,
        )
    initial_relative = initial_validation_inputs_authority_relative_path_v1(
        manifest=manifest.base_manifest
    )
    initial_complete, initial_partial = _transaction_state(
        root=root, relative=initial_relative
    )
    forbidden = massive_adaptive_rl_forbidden_prequential_artifacts_v1(
        root=root, manifest=manifest.base_manifest
    )
    if initial_complete or initial_partial or forbidden:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 must precede every validation input and outcome"
        )
    if not allow_materialize:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration is absent"
        )
    authority = build_massive_adaptive_rl_manifest_v5_registration_authority_v1(
        manifest=manifest
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(authority.semantic_unsigned())),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=verified_at_ms,
        downloaded_at_ms=verified_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=manifest.semantic_receipt_sha256,
        committed_at_ms=verified_at_ms,
        request_id=f"ADAPTIVE-RL-MANIFEST-V5-{manifest.experiment_id}",
    )
    return authorize_massive_adaptive_rl_manifest_v5_registration_authority_v1(
        authority=load_massive_adaptive_rl_manifest_v5_registration_authority_v1(
            root=root,
            experiment_id=manifest.experiment_id,
            verified_at_ms=verified_at_ms,
        ),
        manifest=manifest,
    )


def run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLManifestV5RegistrationAuthorityV1:
    """Adopt V5 under the experiment-global lock.

    The authoritative V5 root already owns this lock and therefore calls the
    private unlocked implementation.  Direct materializing callers must pass
    through this wrapper, closing the registration-versus-legacy-writer race.
    Read-only replay does not create a lock path.
    """

    if not allow_materialize:
        return _run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1_unlocked(
            root=root,
            manifest=manifest,
            allow_materialize=False,
        )
    try:
        with massive_adaptive_rl_artifact_root_writer_lock_v1(
            artifact_root=root
        ):
            with massive_adaptive_rl_experiment_orchestration_lock_v1(
                artifact_root=root,
                experiment_id=manifest.experiment_id,
            ):
                return _run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1_unlocked(
                    root=root,
                    manifest=manifest,
                    allow_materialize=True,
                )
    except MassiveAdaptiveRLExperimentLockV1Unavailable as error:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration is already owned"
        ) from error
    except MassiveAdaptiveRLExperimentLockV1Error as error:
        raise MassiveAdaptiveRLManifestV5RegistrationError(
            "Manifest V5 registration lock is invalid"
        ) from error


__all__ = [
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_MANIFEST_V5_REGISTRATION_V1_SPEC_SHA256",
    "MassiveAdaptiveRLLegacyWriterRejectedByManifestV5",
    "MassiveAdaptiveRLManifestV5RegistrationAuthorityV1",
    "MassiveAdaptiveRLManifestV5RegistrationError",
    "authorize_massive_adaptive_rl_manifest_v5_registration_authority_v1",
    "build_massive_adaptive_rl_manifest_v5_registration_authority_v1",
    "issue_massive_adaptive_rl_manifest_v5_initial_inputs_capability_v1",
    "_run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1_unlocked",
    "load_massive_adaptive_rl_manifest_v5_registration_authority_v1",
    "manifest_v5_registration_relative_path_v1",
    "reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration",
    "run_or_resume_massive_adaptive_rl_manifest_v5_registration_v1",
]
