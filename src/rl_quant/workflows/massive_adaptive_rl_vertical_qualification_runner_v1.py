"""Deterministic subprocess entry point for the fixed V5 qualification suite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import signal
import subprocess
import sys


# Operational budgets, not scientific gates. The outer budget includes source
# construction, four-fold training, economic execution, in-process replays, and
# the nested fresh-process verifier. Scheduler wall time must also leave room
# for dependency setup, timeout cleanup, and durable result capture.
MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1 = 6 * 60 * 60
MASSIVE_ADAPTIVE_RL_FRESH_REPLAY_TIMEOUT_SECONDS_V1 = 2 * 60 * 60
_QUALIFICATION_PROCESS_CLEANUP_TIMEOUT_SECONDS = 30


def _run_vertical_qualification_process_v1(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float = MASSIVE_ADAPTIVE_RL_VERTICAL_QUALIFICATION_TIMEOUT_SECONDS_V1,
) -> subprocess.CompletedProcess[bytes]:
    """Capture the suite, killing its private POSIX group on timeout/error.

    Nested verification inherits this group (including the CLI's execve).
    Killing just pytest can otherwise leave the verifier running after failed
    registration. No existing scheduler or caller process group is signaled.
    This helper never grants qualification; the registration parser owns the
    exact required-node and nonpass checks.
    """

    if os.name != "posix":
        raise OSError("V5 qualification requires POSIX process-group isolation")
    process = subprocess.Popen(
        tuple(command),
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        # Signal even if the leader has exited: a descendant can still own the
        # captured pipes and keep communicate() blocked. SIGKILL also handles a
        # stuck verifier that ignores SIGTERM. Preserve all partial evidence.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=_QUALIFICATION_PROCESS_CLEANUP_TIMEOUT_SECONDS)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    assert process.returncode is not None
    return subprocess.CompletedProcess(
        tuple(command), process.returncode, stdout=stdout, stderr=stderr
    )


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["QUANTTRADE_ADAPTIVE_RL_VERTICAL_QUALIFICATION"] = "1"
    from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
        configure_massive_adaptive_rl_deterministic_runtime_v1,
    )

    configure_massive_adaptive_rl_deterministic_runtime_v1()
    import pytest

    return int(pytest.main(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
