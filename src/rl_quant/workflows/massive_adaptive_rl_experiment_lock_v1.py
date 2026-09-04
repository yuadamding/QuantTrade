"""One process-wide and cross-process lock for an adaptive-RL experiment.

All authoring roots share this exact path.  Protocol-specific stage leases may
still serialize narrower work, but they cannot replace this experiment-global
writer boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator

from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256


MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256 = semantic_sha256(
    {
        "scope": "one-experiment-all-authoring-generations",
        "relative_path": (
            "adaptive-rl/<experiment-id>/orchestration-lease-v1/"
            "orchestration.lock"
        ),
        "acquisition": "nonblocking-exclusive-flock",
        "identity": "regular-file-owned-by-current-user-no-follow",
        "body_errors": "preserved",
    }
)
MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256 = semantic_sha256(
    {
        "scope": "direct-experiment-scoped-materializer",
        "underlying_lock": MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256,
        "owning_context": "reuse-exact-current-context-experiment-lock",
        "direct_call": "acquire-underlying-nonblocking-lock",
    }
)
MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256 = semantic_sha256(
    {
        "scope": "one-artifact-root-registration-and-unscoped-legacy-writers",
        "relative_path": "adaptive-rl/.writer-ownership-v1/writer.lock",
        "acquisition": "nonblocking-exclusive-flock",
        "identity": "regular-file-owned-by-current-user-no-follow",
        "body_errors": "preserved",
    }
)


class MassiveAdaptiveRLExperimentLockV1Error(ValueError):
    """The experiment-global lock path or file identity is invalid."""


class MassiveAdaptiveRLExperimentLockV1Unavailable(RuntimeError):
    """Another process currently owns the experiment-global lock."""


_ACTIVE_EXPERIMENT_LOCKS: ContextVar[tuple[str, ...]] = ContextVar(
    "massive_adaptive_rl_active_experiment_locks",
    default=(),
)
_ACTIVE_ARTIFACT_ROOT_LOCKS: ContextVar[tuple[str, ...]] = ContextVar(
    "massive_adaptive_rl_active_artifact_root_locks",
    default=(),
)


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLExperimentLockV1Error(
            "adaptive RL experiment lock ID is not path safe"
        )
    return value


def massive_adaptive_rl_experiment_lock_relative_path_v1(
    *, experiment_id: str
) -> Path:
    return (
        Path("adaptive-rl")
        / _identifier(experiment_id)
        / "orchestration-lease-v1"
        / "orchestration.lock"
    )


def _lock_identity(*, artifact_root: str | Path, experiment_id: str) -> str:
    return str(
        (
            Path(artifact_root)
            / massive_adaptive_rl_experiment_lock_relative_path_v1(
                experiment_id=experiment_id
            )
        ).resolve()
    )


def _artifact_root_lock_path(artifact_root: str | Path) -> Path:
    return (
        Path(artifact_root)
        / "adaptive-rl"
        / ".writer-ownership-v1"
        / "writer.lock"
    ).resolve()


@contextmanager
def massive_adaptive_rl_experiment_orchestration_lock_v1(
    *, artifact_root: str | Path, experiment_id: str
) -> Iterator[None]:
    """Acquire the sole authoring lock without masking body exceptions."""

    root = Path(artifact_root)
    relative = massive_adaptive_rl_experiment_lock_relative_path_v1(
        experiment_id=experiment_id
    )
    directory = root / relative.parent
    descriptor = -1
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL artifact root is not a no-follow directory"
            )
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL experiment lock directory is not a no-follow directory"
            )
        descriptor = os.open(
            directory / relative.name,
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL experiment lock identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLExperimentLockV1Unavailable(
                "adaptive RL experiment execution is already owned"
            ) from error
    except (
        MassiveAdaptiveRLExperimentLockV1Error,
        MassiveAdaptiveRLExperimentLockV1Unavailable,
    ):
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise MassiveAdaptiveRLExperimentLockV1Error(
            "adaptive RL experiment lock setup failed"
        ) from error

    token = _ACTIVE_EXPERIMENT_LOCKS.set(
        (*_ACTIVE_EXPERIMENT_LOCKS.get(), _lock_identity(
            artifact_root=artifact_root,
            experiment_id=experiment_id,
        ))
    )
    try:
        yield
    finally:
        _ACTIVE_EXPERIMENT_LOCKS.reset(token)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def massive_adaptive_rl_experiment_materialization_lock_v1(
    *, artifact_root: str | Path, experiment_id: str
) -> Iterator[None]:
    """Reuse an owning root lock, or acquire it for a direct materializer."""

    identity = _lock_identity(
        artifact_root=artifact_root,
        experiment_id=experiment_id,
    )
    if identity in _ACTIVE_EXPERIMENT_LOCKS.get():
        yield
        return
    with massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=artifact_root,
        experiment_id=experiment_id,
    ):
        yield


@contextmanager
def massive_adaptive_rl_artifact_root_writer_lock_v1(
    *, artifact_root: str | Path
) -> Iterator[None]:
    """Serialize V5 adoption against legacy writers lacking experiment lineage."""

    root = Path(artifact_root)
    lock_path = _artifact_root_lock_path(root)
    identity = str(lock_path)
    if identity in _ACTIVE_ARTIFACT_ROOT_LOCKS.get():
        yield
        return
    descriptor = -1
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL artifact root is not a no-follow directory"
            )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.parent.is_symlink() or not lock_path.parent.is_dir():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL artifact-root lock directory differs"
            )
        descriptor = os.open(
            lock_path,
            os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_RDWR,
            0o600,
        )
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid():
            raise MassiveAdaptiveRLExperimentLockV1Error(
                "adaptive RL artifact-root lock identity differs"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MassiveAdaptiveRLExperimentLockV1Unavailable(
                "adaptive RL artifact-root writer is already owned"
            ) from error
    except (
        MassiveAdaptiveRLExperimentLockV1Error,
        MassiveAdaptiveRLExperimentLockV1Unavailable,
    ):
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise MassiveAdaptiveRLExperimentLockV1Error(
            "adaptive RL artifact-root lock setup failed"
        ) from error
    token = _ACTIVE_ARTIFACT_ROOT_LOCKS.set(
        (*_ACTIVE_ARTIFACT_ROOT_LOCKS.get(), identity)
    )
    try:
        yield
    finally:
        _ACTIVE_ARTIFACT_ROOT_LOCKS.reset(token)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "MASSIVE_ADAPTIVE_RL_ARTIFACT_ROOT_WRITER_LOCK_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256",
    "MASSIVE_ADAPTIVE_RL_MATERIALIZATION_LOCK_V1_SPEC_SHA256",
    "MassiveAdaptiveRLExperimentLockV1Error",
    "MassiveAdaptiveRLExperimentLockV1Unavailable",
    "massive_adaptive_rl_experiment_lock_relative_path_v1",
    "massive_adaptive_rl_experiment_materialization_lock_v1",
    "massive_adaptive_rl_experiment_orchestration_lock_v1",
    "massive_adaptive_rl_artifact_root_writer_lock_v1",
]
