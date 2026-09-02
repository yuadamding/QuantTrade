from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Protocol

from rl_quant.workflows import massive_adaptive_rl_experiment_runner_v2 as runner
from rl_quant.workflows.massive_adaptive_rl_experiment_state_v2 import (
    load_massive_adaptive_rl_experiment_states_v2,
    register_massive_adaptive_rl_experiment_state_v2,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    build_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)


class _Event(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


def _hold_root_orchestration_lease(
    artifact_root: str,
    experiment_id: str,
    ready: _Event,
    release: _Event,
) -> None:
    with runner._massive_adaptive_rl_experiment_orchestration_lease_v1(
        artifact_root=artifact_root,
        experiment_id=experiment_id,
    ):
        ready.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("root orchestration lease test release timed out")


def test_concurrent_root_owner_returns_ephemeral_response_without_state_mutation(
    tmp_path: Path,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="root-orchestration-lease-canary",
        execution_device_specification="cpu",
    )
    manifest_path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(
        path=manifest_path,
        manifest=manifest,
    )
    artifact_root = tmp_path / "artifacts"
    registered = register_massive_adaptive_rl_experiment_state_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
        manifest_receipt_sha256=manifest.semantic_receipt_sha256,
    )
    before = load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    )

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    owner = context.Process(
        target=_hold_root_orchestration_lease,
        args=(
            str(artifact_root),
            manifest.experiment_id,
            ready,
            release,
        ),
    )
    owner.start()
    try:
        assert ready.wait(timeout=10.0)
        response = runner.run_massive_adaptive_rl_experiment_v2(
            manifest_path=manifest_path,
            source_root=tmp_path / "sources",
            artifact_root=artifact_root,
            device="cpu",
            resume=True,
        )
    finally:
        release.set()
        owner.join(timeout=10.0)
        if owner.is_alive():
            owner.terminate()
            owner.join(timeout=5.0)

    assert owner.exitcode == 0

    assert isinstance(response, runner.MassiveAdaptiveRLOperationalResponseV1)
    assert response.blocker_code == "execution-owned-by-another-process"
    assert response.retryable
    assert not response.ledger_mutated
    assert not response.execution_complete
    assert load_massive_adaptive_rl_experiment_states_v2(
        artifact_root=artifact_root,
        experiment_id=manifest.experiment_id,
    ) == before == (registered,)
