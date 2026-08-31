"""Persisted composite source index for the adaptive RL experiment.

This module deliberately separates byte-level reconstruction from source
qualification.  A generic load rehashes every package-owned artifact in the
fixed source-root layout, but remains nonauthorizing until typed runtime
authorities have independently replayed and validated the same receipts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

from rl_quant.data_sources.massive.source_receipts import canonical_json_file_bytes
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLExperimentManifestV2,
)


MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-source-bundle-v1"
)
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "layout": "package-fixed-relative-paths",
        "generic_reload": "byte-replayed-nonauthorizing",
        "runtime_promotion": "typed-authority-replay-required",
        "caller_dates": False,
        "caller_checkpoint_selection": False,
        "caller_actions_or_economics": False,
        "folds": (0, 1, 2, 3),
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)

_GLOBAL_SOURCE_PATHS = {
    "session-authority": "authorities/session-authority.json",
    "condition-authority": "authorities/condition-authority.json",
    "persisted-partition-inventory": "authorities/persisted-partition-inventory.json",
    "identity-authority": "authorities/identity-authority.json",
    "economic-event-archive": "authorities/economic-event-archive.json",
    "daily-input-authority": "authorities/daily-input-authority.json",
    "fill-source-authority": "authorities/fill-source-authority.json",
    "split-plan": "authorities/adaptive-split-plan.json",
}
_FOLD_SOURCE_PATHS = {
    "training-window-inventory": "training-window-inventory.json",
    "supervised-checkpoint-inventory": "supervised-checkpoint-inventory.json",
    "calibration-inventory": "calibration-inventory.json",
    "fit-forecast-archive-inventory": "fit-forecast-archive-inventory.json",
    "decision-root-inventory": "decision-root-inventory.json",
    "context-origin-inventory": "context-origin-inventory.json",
}


class MassiveAdaptiveRLSourceBundleV1Error(ValueError):
    """The persisted adaptive RL source graph is absent or differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL source path must be a string"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL source path must be a normalized relative path"
        )
    return path.as_posix()


def _expected_paths() -> dict[tuple[str, int | None], str]:
    result: dict[tuple[str, int | None], str] = {
        (role, None): path for role, path in _GLOBAL_SOURCE_PATHS.items()
    }
    for fold_index in range(4):
        for role, name in _FOLD_SOURCE_PATHS.items():
            result[(role, fold_index)] = f"folds/fold-{fold_index}/{name}"
    return result


def _receipt_from_payload(payload: Mapping[str, object]) -> str:
    matches = tuple(
        value
        for name in ("semantic_receipt_sha256", "receipt_sha256")
        if (value := payload.get(name)) is not None
    )
    if len(matches) != 1:
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL source artifact must expose exactly one root receipt"
        )
    return _digest("adaptive RL source artifact receipt", matches[0])


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLSourceArtifactV1:
    role: str
    fold_index: int | None
    relative_path: str
    file_sha256: str
    semantic_receipt_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        expected = _expected_paths()
        if (
            (self.role, self.fold_index) not in expected
            or self.relative_path != expected[(self.role, self.fold_index)]
            or _safe_relative_path(self.relative_path) != self.relative_path
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL source artifact identity differs"
            )
        _digest("adaptive RL source artifact file", self.file_sha256)
        _digest("adaptive RL source artifact semantic", self.semantic_receipt_sha256)
        _digest("adaptive RL source artifact", self.receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLSourceBundleV1:
    experiment_id: str
    manifest_receipt_sha256: str
    artifacts: tuple[MassiveAdaptiveRLSourceArtifactV1, ...]
    artifact_inventory_sha256: str
    session_authority_receipt_sha256: str
    condition_authority_receipt_sha256: str
    persisted_partition_inventory_sha256: str
    identity_authority_receipt_sha256: str
    economic_event_archive_receipt_sha256: str
    daily_input_authority_receipt_sha256: str
    fill_source_receipt_sha256: str
    split_plan_receipt_sha256: str
    training_window_inventory_sha256: str
    supervised_checkpoint_inventory_sha256: str
    calibration_inventory_sha256: str
    fit_forecast_archive_inventory_sha256: str
    decision_root_inventory_sha256: str
    context_origin_inventory_sha256: str
    committed_source_data_qualified: bool
    persisted_source_replayed: bool
    runtime_source_replayed: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "semantic_receipt_sha256",
                "persisted_source_replayed",
                "runtime_source_replayed",
                "source_data_qualified",
            }
        }

    def validate(self) -> None:
        expected = _expected_paths()
        keys = tuple((row.role, row.fold_index) for row in self.artifacts)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SCHEMA
            or keys
            != tuple(
                sorted(
                    expected,
                    key=lambda value: (value[1] is not None, value[1] or -1, value[0]),
                )
            )
            or len(set(keys)) != len(keys)
            or self.artifact_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.artifacts))
            or not self.committed_source_data_qualified
            or self.runtime_source_replayed
            and not self.persisted_source_replayed
            or self.source_data_qualified
            != (self.persisted_source_replayed and self.runtime_source_replayed)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL source bundle identity or authorization differs"
            )
        for row in self.artifacts:
            row.validate()
        for value in (
            self.manifest_receipt_sha256,
            self.artifact_inventory_sha256,
            self.session_authority_receipt_sha256,
            self.condition_authority_receipt_sha256,
            self.persisted_partition_inventory_sha256,
            self.identity_authority_receipt_sha256,
            self.economic_event_archive_receipt_sha256,
            self.daily_input_authority_receipt_sha256,
            self.fill_source_receipt_sha256,
            self.split_plan_receipt_sha256,
            self.training_window_inventory_sha256,
            self.supervised_checkpoint_inventory_sha256,
            self.calibration_inventory_sha256,
            self.fit_forecast_archive_inventory_sha256,
            self.decision_root_inventory_sha256,
            self.context_origin_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL source bundle", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@runtime_checkable
class MassiveAdaptiveRLSourceAuthorityProtocol(Protocol):
    """Small runtime surface used to replay one persisted source artifact."""

    def validate(self) -> None: ...


def _runtime_receipt(value: MassiveAdaptiveRLSourceAuthorityProtocol) -> str:
    for name in ("semantic_receipt_sha256", "receipt_sha256"):
        receipt = getattr(value, name, None)
        if receipt is not None:
            return _digest("adaptive RL runtime source receipt", receipt)
    raise MassiveAdaptiveRLSourceBundleV1Error(
        "adaptive RL runtime source exposes no root receipt"
    )


def _runtime_is_source_qualified(
    value: MassiveAdaptiveRLSourceAuthorityProtocol,
) -> bool:
    qualification_fields = tuple(
        bool(getattr(value, name))
        for name in (
            "source_transport_qualified",
            "source_data_qualified",
            "daily_input_data_qualified",
            "fill_source_data_qualified",
            "source_windows_replayed",
            "runtime_source_replayed",
        )
        if hasattr(value, name)
    )
    return all(qualification_fields)


def _summary_receipt(
    artifacts: Mapping[tuple[str, int | None], MassiveAdaptiveRLSourceArtifactV1],
    role: str,
) -> str:
    rows = tuple(
        artifacts[(role, fold_index)].semantic_receipt_sha256 for fold_index in range(4)
    )
    return semantic_sha256(rows)


def _parse_artifact(value: object) -> MassiveAdaptiveRLSourceArtifactV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL source artifact row is not an object"
        )
    result = MassiveAdaptiveRLSourceArtifactV1(**dict(value))  # type: ignore[arg-type]
    result.validate()
    return result


def _parse_bundle(payload: Mapping[str, object]) -> MassiveAdaptiveRLSourceBundleV1:
    values = dict(payload)
    rows = values.get("artifacts")
    if not isinstance(rows, list):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL source bundle artifact inventory is malformed"
        )
    values["artifacts"] = tuple(_parse_artifact(row) for row in rows)
    result = MassiveAdaptiveRLSourceBundleV1(**values)  # type: ignore[arg-type]
    result.validate()
    return result


def _bundle_path(source_root: str | Path, experiment_id: str) -> Path:
    return (
        Path(source_root) / "adaptive-rl" / "source-bundle-v1" / f"{experiment_id}.json"
    )


def materialize_massive_adaptive_rl_source_bundle_v1(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> MassiveAdaptiveRLSourceBundleV1:
    """Publish the fixed source graph after typed authorities replay.

    Paths and roles are protocol-owned.  The caller supplies no date
    inventory, checkpoint choice, action, transition, or economic result.
    """

    manifest.validate()
    root = Path(source_root).resolve()
    expected = _expected_paths()
    if set(runtime_sources) != set(expected):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL materialization requires the complete fixed source graph"
        )
    rows: list[MassiveAdaptiveRLSourceArtifactV1] = []
    for key in sorted(
        expected, key=lambda value: (value[1] is not None, value[1] or -1, value[0])
    ):
        runtime = runtime_sources[key]
        runtime.validate()
        if not _runtime_is_source_qualified(runtime):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL runtime source is not data qualified"
            )
        relative_path = expected[key]
        path = (root / relative_path).resolve()
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL fixed source artifact is absent or not a regular file"
            )
        raw = path.read_bytes()
        value = json.loads(raw)
        runtime_receipt = _runtime_receipt(runtime)
        if (
            not isinstance(value, Mapping)
            or raw != canonical_json_file_bytes(value)
            or _receipt_from_payload(value) != runtime_receipt
        ):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL fixed source artifact differs from runtime authority"
            )
        body: dict[str, object] = {
            "role": key[0],
            "fold_index": key[1],
            "relative_path": relative_path,
            "file_sha256": file_sha256(path),
            "semantic_receipt_sha256": runtime_receipt,
        }
        row = MassiveAdaptiveRLSourceArtifactV1(
            **body,  # type: ignore[arg-type]
            receipt_sha256=semantic_sha256(body),
        )
        row.validate()
        rows.append(row)
    artifacts = tuple(rows)
    by_key = {(row.role, row.fold_index): row for row in artifacts}
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_receipt_sha256": manifest.semantic_receipt_sha256,
        "artifacts": artifacts,
        "artifact_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in artifacts)
        ),
        "session_authority_receipt_sha256": by_key[
            ("session-authority", None)
        ].semantic_receipt_sha256,
        "condition_authority_receipt_sha256": by_key[
            ("condition-authority", None)
        ].semantic_receipt_sha256,
        "persisted_partition_inventory_sha256": by_key[
            ("persisted-partition-inventory", None)
        ].semantic_receipt_sha256,
        "identity_authority_receipt_sha256": by_key[
            ("identity-authority", None)
        ].semantic_receipt_sha256,
        "economic_event_archive_receipt_sha256": by_key[
            ("economic-event-archive", None)
        ].semantic_receipt_sha256,
        "daily_input_authority_receipt_sha256": by_key[
            ("daily-input-authority", None)
        ].semantic_receipt_sha256,
        "fill_source_receipt_sha256": by_key[
            ("fill-source-authority", None)
        ].semantic_receipt_sha256,
        "split_plan_receipt_sha256": by_key[
            ("split-plan", None)
        ].semantic_receipt_sha256,
        "training_window_inventory_sha256": _summary_receipt(
            by_key, "training-window-inventory"
        ),
        "supervised_checkpoint_inventory_sha256": _summary_receipt(
            by_key, "supervised-checkpoint-inventory"
        ),
        "calibration_inventory_sha256": _summary_receipt(
            by_key, "calibration-inventory"
        ),
        "fit_forecast_archive_inventory_sha256": _summary_receipt(
            by_key, "fit-forecast-archive-inventory"
        ),
        "decision_root_inventory_sha256": _summary_receipt(
            by_key, "decision-root-inventory"
        ),
        "context_origin_inventory_sha256": _summary_receipt(
            by_key, "context-origin-inventory"
        ),
        "committed_source_data_qualified": True,
        "persisted_source_replayed": False,
        "runtime_source_replayed": False,
        "source_data_qualified": False,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLSourceBundleV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    output = _bundle_path(root, manifest.experiment_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(canonical_json_file_bytes(asdict(result)))
    except FileExistsError as error:
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL composite source bundle is create-only"
        ) from error
    return result


def load_massive_adaptive_rl_source_bundle_v1(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV2,
) -> MassiveAdaptiveRLSourceBundleV1:
    """Reopen and byte-replay the fixed persisted source graph.

    Generic reload is intentionally nonauthorizing.  Call
    :func:`authorize_massive_adaptive_rl_source_bundle_v1` with the typed
    runtime authorities before any training workflow is allowed to proceed.
    """

    manifest.validate()
    root = Path(source_root).resolve()
    path = _bundle_path(root, manifest.experiment_id)
    if path.is_symlink() or not path.is_file():
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL composite source bundle is absent or not a regular file"
        )
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL composite source bundle is not canonical JSON"
        )
    committed = _parse_bundle(value)
    if (
        committed.experiment_id != manifest.experiment_id
        or committed.manifest_receipt_sha256 != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL composite source bundle belongs to another manifest"
        )
    for row in committed.artifacts:
        artifact_path = (root / row.relative_path).resolve()
        if (
            root not in artifact_path.parents
            or artifact_path.is_symlink()
            or not artifact_path.is_file()
            or file_sha256(artifact_path) != row.file_sha256
        ):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL referenced source artifact is absent or changed"
            )
        artifact_raw = artifact_path.read_bytes()
        artifact_value = json.loads(artifact_raw)
        if (
            not isinstance(artifact_value, Mapping)
            or artifact_raw != canonical_json_file_bytes(artifact_value)
            or _receipt_from_payload(artifact_value) != row.semantic_receipt_sha256
        ):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL referenced source artifact does not replay"
            )
    result = replace(
        committed,
        persisted_source_replayed=True,
        runtime_source_replayed=False,
        source_data_qualified=False,
    )
    result.validate()
    return result


def authorize_massive_adaptive_rl_source_bundle_v1(
    *,
    source_bundle: MassiveAdaptiveRLSourceBundleV1,
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLSourceAuthorityProtocol
    ],
) -> MassiveAdaptiveRLSourceBundleV1:
    """Promote only after every typed runtime authority independently replays."""

    source_bundle.validate()
    expected = {(row.role, row.fold_index): row for row in source_bundle.artifacts}
    if not source_bundle.persisted_source_replayed or set(runtime_sources) != set(
        expected
    ):
        raise MassiveAdaptiveRLSourceBundleV1Error(
            "adaptive RL runtime source inventory is incomplete"
        )
    for key, runtime in runtime_sources.items():
        runtime.validate()
        if _runtime_receipt(runtime) != expected[
            key
        ].semantic_receipt_sha256 or not _runtime_is_source_qualified(runtime):
            raise MassiveAdaptiveRLSourceBundleV1Error(
                "adaptive RL runtime source is unqualified or differs from persisted inventory"
            )
    result = replace(
        source_bundle,
        runtime_source_replayed=True,
        source_data_qualified=True,
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_SOURCE_BUNDLE_V1_SCHEMA",
    "MassiveAdaptiveRLSourceArtifactV1",
    "MassiveAdaptiveRLSourceAuthorityProtocol",
    "MassiveAdaptiveRLSourceBundleV1",
    "MassiveAdaptiveRLSourceBundleV1Error",
    "authorize_massive_adaptive_rl_source_bundle_v1",
    "load_massive_adaptive_rl_source_bundle_v1",
    "materialize_massive_adaptive_rl_source_bundle_v1",
]
