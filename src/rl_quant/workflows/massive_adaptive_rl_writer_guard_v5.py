"""Experiment-wide Manifest-V5 writer ownership guard.

This module is intentionally dependency-light.  Legacy child materializers can
import it without importing the Manifest-V5 authority graph (and therefore
without creating circular imports).  A complete *or partial* V5 registration
transaction disables every older writer for the same experiment ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, cast

from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_artifact_root_writer_lock_v1,
    massive_adaptive_rl_experiment_materialization_lock_v1,
)


_P = ParamSpec("_P")
_R = TypeVar("_R")


class MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(ValueError):
    """A legacy writer attempted to mutate a V5-owned experiment."""


_CAPABILITY_SEAL = object()


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLManifestV5WriterCapabilityV1:
    """Narrow in-process authority for a V5-owned compatibility writer."""

    experiment_id: str
    manifest_v5_receipt_sha256: str
    base_manifest_v4_receipt_sha256: str
    registration_authority_receipt_sha256: str
    registration_source_receipt_sha256: str
    registration_commit_receipt_sha256: str
    writer_role: str
    allowed_fold_indices: tuple[int, ...]
    _seal: object = field(repr=False, compare=False)


def _digest(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "adaptive RL writer-ownership experiment ID is not path safe"
        )
    return value


def manifest_v5_registration_relative_path_v1(*, experiment_id: str) -> str:
    return (
        "adaptive-rl/"
        f"{_identifier(experiment_id)}/"
        "manifest-v5-registration-v1/registration.json"
    )


def manifest_v5_registration_transaction_state_v1(
    *, root: str | Path, experiment_id: str
) -> tuple[bool, bool]:
    payload = Path(root) / manifest_v5_registration_relative_path_v1(
        experiment_id=experiment_id
    )
    paths = (
        payload,
        payload.with_name(payload.name + ".receipt.json"),
        payload.with_name(payload.name + ".commit.json"),
    )
    present = tuple(path.exists() or path.is_symlink() for path in paths)
    return all(present), any(present) and not all(present)


def reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
    *, root: str | Path, experiment_id: str
) -> None:
    """Fail closed when any complete or partial V5 adoption marker exists."""

    complete, partial = manifest_v5_registration_transaction_state_v1(
        root=root,
        experiment_id=experiment_id,
    )
    if complete or partial:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 owns this experiment; legacy materialization is prohibited"
        )


def _issue_manifest_v5_writer_capability_v1(
    *,
    experiment_id: str,
    manifest_v5_receipt_sha256: str,
    base_manifest_v4_receipt_sha256: str,
    registration_authority_receipt_sha256: str,
    registration_source_receipt_sha256: str,
    registration_commit_receipt_sha256: str,
    writer_role: str,
    allowed_fold_indices: tuple[int, ...],
) -> MassiveAdaptiveRLManifestV5WriterCapabilityV1:
    """Issue a package-internal capability after authority replay."""

    capability = MassiveAdaptiveRLManifestV5WriterCapabilityV1(
        experiment_id=_identifier(experiment_id),
        manifest_v5_receipt_sha256=manifest_v5_receipt_sha256,
        base_manifest_v4_receipt_sha256=base_manifest_v4_receipt_sha256,
        registration_authority_receipt_sha256=(
            registration_authority_receipt_sha256
        ),
        registration_source_receipt_sha256=registration_source_receipt_sha256,
        registration_commit_receipt_sha256=registration_commit_receipt_sha256,
        writer_role=writer_role,
        allowed_fold_indices=allowed_fold_indices,
        _seal=_CAPABILITY_SEAL,
    )
    _validate_manifest_v5_writer_capability_v1(
        capability=capability,
        experiment_id=experiment_id,
        manifest_v4_receipt_sha256=base_manifest_v4_receipt_sha256,
        writer_role=writer_role,
        fold_index=None,
    )
    return capability


def _validate_manifest_v5_writer_capability_v1(
    *,
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1,
    experiment_id: str,
    manifest_v4_receipt_sha256: str,
    writer_role: str,
    fold_index: int | None,
) -> None:
    if (
        type(capability) is not MassiveAdaptiveRLManifestV5WriterCapabilityV1
        or capability._seal is not _CAPABILITY_SEAL
        or capability.experiment_id != _identifier(experiment_id)
        or not all(
            _digest(value)
            for value in (
                capability.manifest_v5_receipt_sha256,
                capability.base_manifest_v4_receipt_sha256,
                capability.registration_authority_receipt_sha256,
                capability.registration_source_receipt_sha256,
                capability.registration_commit_receipt_sha256,
            )
        )
        or capability.writer_role != writer_role
        or capability.base_manifest_v4_receipt_sha256
        != manifest_v4_receipt_sha256
        or not capability.allowed_fold_indices
        or len(set(capability.allowed_fold_indices))
        != len(capability.allowed_fold_indices)
        or any(
            isinstance(index, bool) or index not in (0, 1, 2, 3)
            for index in capability.allowed_fold_indices
        )
        or fold_index is not None
        and fold_index not in capability.allowed_fold_indices
    ):
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 writer capability differs or exceeds its fold release"
        )


def authorize_legacy_or_manifest_v5_compatibility_writer_v1(
    *,
    root: str | Path,
    experiment_id: str,
    manifest_v4_receipt_sha256: str,
    writer_role: str,
    fold_index: int | None,
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1 | None,
) -> None:
    """Permit legacy writes only before V5, or a narrowly capable V5 write."""

    complete, partial = manifest_v5_registration_transaction_state_v1(
        root=root,
        experiment_id=experiment_id,
    )
    if not complete and not partial:
        if capability is not None:
            raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                "Manifest V5 capability has no persisted registration"
            )
        return
    if partial or capability is None:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 owns this experiment; an exact writer capability is required"
        )
    _validate_manifest_v5_writer_capability_v1(
        capability=capability,
        experiment_id=experiment_id,
        manifest_v4_receipt_sha256=manifest_v4_receipt_sha256,
        writer_role=writer_role,
        fold_index=fold_index,
    )


def manifest_v5_compatibility_writer_guard_v1(
    *,
    writer_role: str,
    fold_parameter: str | None = None,
    fold_attribute_parameter: str | None = None,
    materialize_parameter: str | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Lock and authorize a compatibility materializer at publication time."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if materialize_parameter is not None and not kwargs.get(
                materialize_parameter, True
            ):
                return function(*args, **kwargs)
            root = kwargs.get("root")
            manifest = kwargs.get("manifest")
            if root is None or manifest is None:
                raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                    "guarded adaptive RL writer requires keyword-only roots"
                )
            experiment_id = _identifier(getattr(manifest, "experiment_id", None))
            manifest_receipt_value = getattr(
                manifest, "semantic_receipt_sha256", None
            )
            if not _digest(manifest_receipt_value):
                raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                    "guarded adaptive RL writer manifest receipt differs"
                )
            manifest_receipt = cast(str, manifest_receipt_value)
            fold_index: int | None = None
            if fold_parameter is not None:
                fold_index = kwargs.get(fold_parameter)  # type: ignore[assignment]
            elif fold_attribute_parameter is not None:
                authority = kwargs.get(fold_attribute_parameter)
                fold_index = getattr(authority, "fold_index", None)
            capability = kwargs.get("v5_writer_capability")
            with massive_adaptive_rl_experiment_materialization_lock_v1(
                artifact_root=root,  # type: ignore[arg-type]
                experiment_id=experiment_id,
            ):
                authorize_legacy_or_manifest_v5_compatibility_writer_v1(
                    root=root,  # type: ignore[arg-type]
                    experiment_id=experiment_id,
                    manifest_v4_receipt_sha256=manifest_receipt,
                    writer_role=writer_role,
                    fold_index=fold_index,
                    capability=capability,  # type: ignore[arg-type]
                )
                return function(*args, **kwargs)

        return guarded

    return decorate


def legacy_manifest_v5_rejecting_writer_guard_v1(
    *, materialize_parameter: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Lock an experiment-scoped legacy writer and reject V5 ownership."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if materialize_parameter is not None and not kwargs.get(
                materialize_parameter, True
            ):
                return function(*args, **kwargs)
            root = kwargs.get("root")
            manifest = kwargs.get("manifest")
            if root is None or manifest is None:
                raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                    "guarded adaptive RL writer requires keyword-only roots"
                )
            experiment_id = _identifier(getattr(manifest, "experiment_id", None))
            with massive_adaptive_rl_experiment_materialization_lock_v1(
                artifact_root=root,  # type: ignore[arg-type]
                experiment_id=experiment_id,
            ):
                reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration(
                    root=root,  # type: ignore[arg-type]
                    experiment_id=experiment_id,
                )
                return function(*args, **kwargs)

        return guarded

    return decorate


def reject_legacy_massive_adaptive_rl_writer_after_any_manifest_v5_registration(
    *, root: str | Path
) -> None:
    """Protect legacy artifacts that do not carry an experiment identity.

    Some V1 writers predate experiment-scoped lineage and therefore cannot
    prove which experiment they would mutate.  Once any experiment under the
    supplied artifact root adopts V5, those ambiguous writers fail closed.
    """

    registrations = Path(root) / "adaptive-rl"
    if not registrations.exists() or not registrations.is_dir():
        return
    for experiment in registrations.iterdir():
        if (
            not experiment.is_dir()
            or experiment.is_symlink()
            or experiment.name.startswith(".")
        ):
            continue
        complete, partial = manifest_v5_registration_transaction_state_v1(
            root=root,
            experiment_id=experiment.name,
        )
        if complete or partial:
            raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                "Manifest V5 exists under this artifact root; an unscoped legacy "
                "materializer is prohibited"
            )


def legacy_unscoped_manifest_v5_rejecting_writer_guard_v1(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Serialize and reject legacy writers without experiment-scoped lineage."""

    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        root = kwargs.get("root")
        if root is None:
            raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                "guarded adaptive RL writer requires a keyword-only root"
            )
        with massive_adaptive_rl_artifact_root_writer_lock_v1(
            artifact_root=root,  # type: ignore[arg-type]
        ):
            reject_legacy_massive_adaptive_rl_writer_after_any_manifest_v5_registration(
                root=root,  # type: ignore[arg-type]
            )
            return function(*args, **kwargs)

    return guarded


__all__ = [
    "MassiveAdaptiveRLManifestV5WriterCapabilityV1",
    "MassiveAdaptiveRLLegacyWriterRejectedByManifestV5",
    "authorize_legacy_or_manifest_v5_compatibility_writer_v1",
    "legacy_manifest_v5_rejecting_writer_guard_v1",
    "legacy_unscoped_manifest_v5_rejecting_writer_guard_v1",
    "manifest_v5_compatibility_writer_guard_v1",
    "manifest_v5_registration_relative_path_v1",
    "manifest_v5_registration_transaction_state_v1",
    "reject_legacy_massive_adaptive_rl_writer_after_any_manifest_v5_registration",
    "reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration",
]
