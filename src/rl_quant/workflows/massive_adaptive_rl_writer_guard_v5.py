"""Experiment-wide Manifest-V5 writer ownership guard.

This module is intentionally dependency-light.  Legacy child materializers can
import it without importing the Manifest-V5 authority graph (and therefore
without creating circular imports).  A complete *or partial* V5 registration
transaction disables every older writer for the same experiment ID.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import time
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
    _registration_root_resolved: str = field(repr=False, compare=False)
    _publication_roots_resolved: tuple[str, ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)


_ACTIVE_WRITER_CAPABILITY: ContextVar[
    MassiveAdaptiveRLManifestV5WriterCapabilityV1 | None
] = ContextVar("massive_adaptive_rl_manifest_v5_writer_capability", default=None)

_TRAINING_SOURCE_DIRECTORIES = frozenset(
    {
        "checkpoint-v1",
        "prequential-ppo-checkpoint-v1",
        "rl-checkpoint-v1",
        "rl-fixed-control-fit-v1",
        "rl-fixed-control-selection-v1",
        "rl-fit-forecast-archive-v1",
        "rl-fold-fit-authority-v1",
        "rl-fold-fit-inputs-v1",
        "rl-four-fold-fit-authority-v1",
        "rl-four-fold-fit-inputs-authority-v1",
    }
)
_INITIAL_INPUT_SOURCE_DIRECTORIES = frozenset(
    {
        "decision-tensor-v1",
        "forecast-archive-v2",
        "rl-prequential-initial-validation-inputs-authority-v1",
        "rl-validation-environment-registry-v2",
        "rl-validation-inputs-v1",
        "rl-validation-sources-authority-v2",
    }
)
_INITIAL_INPUT_ADAPTIVE_RL_DIRECTORIES = frozenset(
    {
        "replay-dependency-index-v2",
        "runtime-source-graph-authority-v2",
        "source-bundle-v2",
    }
)
_EXPERIMENT_SCOPED_DIRECTORIES = frozenset(
    {
        "execution-implementation-registration-v1",
        "fold-validation-v3",
        "frozen-fc06-v2",
        "frozen-policy-v2",
        "manifest-v5-registration-v1",
        "outer-access-commitment-v2",
        "outer-fold-seal-v1",
        "outer-rollout-v2",
        "policy-selection-v4",
        "prequential-state-v1",
        "profitability-report-v2",
        "state-v2",
        "validation-outcome-v3",
        "validation-release-v1",
        "walk-forward-policy-schedule-v1",
    }
)
_WRITER_ROLE_FOLD_INVENTORIES = {
    "causal-training": (0, 1, 2, 3),
    "execution-implementation-registration": (0, 1, 2, 3),
    "initial-validation-inputs": (0, 1),
}


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
    root: str | Path,
    experiment_id: str,
    manifest_v5_receipt_sha256: str,
    base_manifest_v4_receipt_sha256: str,
    registration_authority_receipt_sha256: str,
    registration_source_receipt_sha256: str,
    registration_commit_receipt_sha256: str,
    writer_role: str,
    allowed_fold_indices: tuple[int, ...],
    publication_roots: tuple[str | Path, ...] = (),
) -> MassiveAdaptiveRLManifestV5WriterCapabilityV1:
    """Issue a package-internal capability after authority replay."""

    registration_root = Path(root).resolve(strict=True)
    requested_publication_roots = (registration_root,) + tuple(
        Path(item).resolve(strict=True) for item in publication_roots
    )
    if any(not item.is_dir() for item in requested_publication_roots):
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 publication root is not a directory"
        )
    resolved_publication_roots = tuple(
        sorted({str(item) for item in requested_publication_roots})
    )
    capability = MassiveAdaptiveRLManifestV5WriterCapabilityV1(
        experiment_id=_identifier(experiment_id),
        manifest_v5_receipt_sha256=manifest_v5_receipt_sha256,
        base_manifest_v4_receipt_sha256=base_manifest_v4_receipt_sha256,
        registration_authority_receipt_sha256=(registration_authority_receipt_sha256),
        registration_source_receipt_sha256=registration_source_receipt_sha256,
        registration_commit_receipt_sha256=registration_commit_receipt_sha256,
        writer_role=writer_role,
        allowed_fold_indices=allowed_fold_indices,
        _registration_root_resolved=str(registration_root),
        _publication_roots_resolved=resolved_publication_roots,
        _seal=_CAPABILITY_SEAL,
    )
    _validate_manifest_v5_writer_capability_v1(
        root=root,
        capability=capability,
        experiment_id=experiment_id,
        manifest_v4_receipt_sha256=base_manifest_v4_receipt_sha256,
        writer_role=writer_role,
        fold_index=None,
    )
    return capability


def _validate_manifest_v5_writer_capability_v1(
    *,
    root: str | Path,
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1,
    experiment_id: str,
    manifest_v4_receipt_sha256: str,
    writer_role: str,
    fold_index: int | None,
) -> None:
    if (
        type(capability) is not MassiveAdaptiveRLManifestV5WriterCapabilityV1
        or capability._seal is not _CAPABILITY_SEAL
        or capability._registration_root_resolved
        != str(Path(root).resolve(strict=True))
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
        or capability.base_manifest_v4_receipt_sha256 != manifest_v4_receipt_sha256
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
    _validate_capability_against_persisted_registration_v1(
        root=root,
        capability=capability,
    )


def _validate_capability_against_persisted_registration_v1(
    *,
    root: str | Path,
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1,
) -> None:
    """Rebind an in-process capability to the exact persisted registration."""

    # Local import preserves the dependency-light guard module and avoids the
    # registration -> guard import cycle at module import time.
    from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
        load_massive_adaptive_rl_manifest_v5_registration_authority_v1,
    )

    registration_root = Path(capability._registration_root_resolved)
    publication_roots = tuple(
        Path(value) for value in capability._publication_roots_resolved
    )
    expected_folds = _WRITER_ROLE_FOLD_INVENTORIES.get(capability.writer_role)
    if (
        capability._seal is not _CAPABILITY_SEAL
        or capability.experiment_id != _identifier(capability.experiment_id)
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
        or expected_folds is None
        or capability.allowed_fold_indices != expected_folds
        or not registration_root.is_dir()
        or str(registration_root.resolve(strict=True))
        != capability._registration_root_resolved
        or Path(root).resolve(strict=True) != registration_root
        or not publication_roots
        or tuple(sorted(set(capability._publication_roots_resolved)))
        != capability._publication_roots_resolved
        or registration_root not in publication_roots
        or any(
            not path.is_dir() or str(path.resolve(strict=True)) != raw
            for path, raw in zip(
                publication_roots,
                capability._publication_roots_resolved,
                strict=True,
            )
        )
        or capability.writer_role != "initial-validation-inputs"
        and len(publication_roots) != 1
        or capability.writer_role == "initial-validation-inputs"
        and len(publication_roots) > 2
    ):
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 writer capability registration root differs"
        )
    try:
        authority = load_massive_adaptive_rl_manifest_v5_registration_authority_v1(
            root=registration_root,
            experiment_id=capability.experiment_id,
            verified_at_ms=time.time_ns() // 1_000_000,
        )
        authority.validate()
    except (OSError, ValueError) as error:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 writer capability registration cannot be replayed"
        ) from error
    if (
        authority.experiment_id != capability.experiment_id
        or authority.manifest_v5_receipt_sha256 != capability.manifest_v5_receipt_sha256
        or authority.base_manifest_v4_receipt_sha256
        != capability.base_manifest_v4_receipt_sha256
        or authority.semantic_receipt_sha256
        != capability.registration_authority_receipt_sha256
        or authority.source_receipt_sha256
        != capability.registration_source_receipt_sha256
        or authority.source_transaction_receipt_sha256
        != capability.registration_commit_receipt_sha256
    ):
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 writer capability is not derived from the persisted registration"
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
        root=root,
        capability=capability,
        experiment_id=experiment_id,
        manifest_v4_receipt_sha256=manifest_v4_receipt_sha256,
        writer_role=writer_role,
        fold_index=fold_index,
    )


@contextmanager
def massive_adaptive_rl_manifest_v5_writer_scope_v1(
    *,
    root: str | Path,
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1,
) -> Iterator[None]:
    """Activate one persisted-registration-derived capability for child writes."""

    _validate_capability_against_persisted_registration_v1(
        root=root,
        capability=capability,
    )
    active = _ACTIVE_WRITER_CAPABILITY.get()
    if active is not None and active is not capability:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "a different Manifest V5 writer capability is already active"
        )
    token = _ACTIVE_WRITER_CAPABILITY.set(capability)
    try:
        yield
    finally:
        _ACTIVE_WRITER_CAPABILITY.reset(token)


def authorize_massive_adaptive_rl_source_publication_v5(
    *, root: str | Path, relative_payload_path: str
) -> None:
    """Structurally guard every adaptive-RL source transaction publication."""

    parts = Path(relative_payload_path).parts
    if not parts or parts[0] not in {"adaptive-rl", "massive-adaptive"}:
        return
    active = _ACTIVE_WRITER_CAPABILITY.get()
    scoped = bool(
        parts[0] == "adaptive-rl"
        and len(parts) >= 3
        and parts[2] in _EXPERIMENT_SCOPED_DIRECTORIES
    )
    if scoped:
        experiment_id = _identifier(parts[1])
        complete, partial = manifest_v5_registration_transaction_state_v1(
            root=root,
            experiment_id=experiment_id,
        )
        if not complete and not partial:
            if active is not None:
                raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                    "Manifest V5 capability cannot publish outside its registered experiment"
                )
            return
        if partial or active is None or active.experiment_id != experiment_id:
            raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                "Manifest V5 owns this source namespace; an exact active capability is required"
            )
        _authorize_active_capability_for_publication_root_v1(
            root=root, capability=active
        )
        execution_registration = bool(
            active.writer_role == "execution-implementation-registration"
            and relative_payload_path
            == execution_registration_path_for_capability_v1(active)
        )
        training_state = bool(
            active.writer_role == "causal-training"
            and len(parts) >= 3
            and parts[2] == "state-v2"
        )
        initial_validation_release = bool(
            active.writer_role == "initial-validation-inputs"
            and relative_payload_path
            == (
                f"adaptive-rl/{active.experiment_id}/validation-release-v1/initial.json"
            )
        )
        if not any(
            (execution_registration, training_state, initial_validation_release)
        ):
            raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
                "Manifest V5 capability does not authorize this scoped source path"
            )
        return

    # Historical adaptive-rl and massive-adaptive paths do not carry an
    # experiment ID.  Once any V5 registration exists at this artifact root,
    # they may be written only inside an active V5 compatibility scope.
    if active is None:
        reject_legacy_massive_adaptive_rl_writer_after_any_manifest_v5_registration(
            root=root
        )
        return
    _authorize_active_capability_for_publication_root_v1(root=root, capability=active)
    directory = parts[1] if len(parts) >= 2 else ""
    if parts[0] == "adaptive-rl":
        allowed = (
            _INITIAL_INPUT_ADAPTIVE_RL_DIRECTORIES
            if active.writer_role == "initial-validation-inputs"
            else frozenset()
        )
    else:
        allowed = (
            _TRAINING_SOURCE_DIRECTORIES
            if active.writer_role == "causal-training"
            else _INITIAL_INPUT_SOURCE_DIRECTORIES
            if active.writer_role == "initial-validation-inputs"
            else frozenset()
        )
    if directory not in allowed:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 capability does not authorize this unscoped source path"
        )


def _authorize_active_capability_for_publication_root_v1(
    *, root: str | Path, capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1
) -> None:
    resolved_root = str(Path(root).resolve(strict=True))
    if resolved_root not in capability._publication_roots_resolved:
        raise MassiveAdaptiveRLLegacyWriterRejectedByManifestV5(
            "Manifest V5 capability does not authorize this publication root"
        )
    _validate_capability_against_persisted_registration_v1(
        root=capability._registration_root_resolved,
        capability=capability,
    )


def execution_registration_path_for_capability_v1(
    capability: MassiveAdaptiveRLManifestV5WriterCapabilityV1,
) -> str:
    return (
        "adaptive-rl/"
        f"{capability.experiment_id}/"
        "execution-implementation-registration-v1/registration.json"
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
            manifest_receipt_value = getattr(manifest, "semantic_receipt_sha256", None)
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
                if capability is None:
                    return function(*args, **kwargs)
                with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                    root=root,  # type: ignore[arg-type]
                    capability=capability,  # type: ignore[arg-type]
                ):
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
    "authorize_massive_adaptive_rl_source_publication_v5",
    "authorize_legacy_or_manifest_v5_compatibility_writer_v1",
    "legacy_manifest_v5_rejecting_writer_guard_v1",
    "legacy_unscoped_manifest_v5_rejecting_writer_guard_v1",
    "manifest_v5_compatibility_writer_guard_v1",
    "massive_adaptive_rl_manifest_v5_writer_scope_v1",
    "manifest_v5_registration_relative_path_v1",
    "manifest_v5_registration_transaction_state_v1",
    "reject_legacy_massive_adaptive_rl_writer_after_any_manifest_v5_registration",
    "reject_legacy_massive_adaptive_rl_writer_after_manifest_v5_registration",
]
