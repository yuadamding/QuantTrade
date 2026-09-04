"""One process-wide and cross-process lock for an adaptive-RL experiment.

All authoring roots share this exact path.  Protocol-specific stage leases may
still serialize narrower work, but they cannot replace this experiment-global
writer boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
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


class MassiveAdaptiveRLExperimentLockV1Error(ValueError):
    """The experiment-global lock path or file identity is invalid."""


class MassiveAdaptiveRLExperimentLockV1Unavailable(RuntimeError):
    """Another process currently owns the experiment-global lock."""


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

    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_EXPERIMENT_LOCK_V1_SPEC_SHA256",
    "MassiveAdaptiveRLExperimentLockV1Error",
    "MassiveAdaptiveRLExperimentLockV1Unavailable",
    "massive_adaptive_rl_experiment_lock_relative_path_v1",
    "massive_adaptive_rl_experiment_orchestration_lock_v1",
]
