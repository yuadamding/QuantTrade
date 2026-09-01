from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
import rl_quant.workflows.massive_adaptive_rl_v2 as workflow_v2
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLWorkflowV2Error,
    build_massive_adaptive_rl_candidate_schedule_v1,
    build_massive_adaptive_rl_experiment_manifest_v2,
    load_massive_adaptive_rl_experiment_manifest_v2,
    main,
    run_massive_adaptive_rl_training_workflow_v2,
    write_massive_adaptive_rl_experiment_manifest_v2,
)


@pytest.mark.parametrize(
    ("block_sessions", "expected"),
    (
        (
            63,
            {
                0: ((126,), (2,)),
                1: ((126, 252), (2, 4)),
                2: ((126, 252, 378), (2, 4, 6)),
                3: ((126, 252, 378, 504), (2, 4, 6, 8)),
            },
        ),
        (
            21,
            {
                0: ((126,), (6,)),
                1: ((126, 252), (6, 12)),
                2: ((126, 252, 378), (6, 12, 18)),
                3: ((126, 252, 378, 504), (6, 12, 18, 24)),
            },
        ),
    ),
)
def test_candidate_schedule_is_derived_from_elapsed_sessions(
    block_sessions: int,
    expected: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
) -> None:
    for fold_index in range(4):
        schedule = build_massive_adaptive_rl_candidate_schedule_v1(
            fold_index=fold_index,
            prequential_block_sessions=block_sessions,
        )
        elapsed, updates = expected[fold_index]
        assert schedule.rl_fit_session_count == 126 * (fold_index + 1)
        assert schedule.candidate_elapsed_sessions == elapsed
        assert schedule.candidate_update_indices == updates


def test_manifest_v2_has_no_global_update_schedule_and_is_create_only(tmp_path) -> None:
    config = MassiveAdaptivePPOConfigV1(
        rollout_length=63,
        minibatch_size=63,
        seed=17,
    )
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="historical-2017-2025",
        prequential_block_sessions=63,
        ppo_config=config,
    )
    assert manifest.candidate_elapsed_sessions == (126, 252, 378, 504)
    assert "candidate_update_indices" not in asdict(manifest)
    assert manifest.schedule(3).candidate_update_indices == (2, 4, 6, 8)
    path = tmp_path / "manifest-v2.json"
    write_massive_adaptive_rl_experiment_manifest_v2(path=path, manifest=manifest)
    assert load_massive_adaptive_rl_experiment_manifest_v2(path) == manifest
    with pytest.raises(MassiveAdaptiveRLWorkflowV2Error, match="create-only"):
        write_massive_adaptive_rl_experiment_manifest_v2(path=path, manifest=manifest)


def test_manifest_v2_rejects_caller_changed_elapsed_schedule() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="invalid-elapsed"
    )
    changed = replace(
        manifest,
        candidate_elapsed_sessions=(63, 126),
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLWorkflowV2Error, match="manifest V2"):
        changed.validate()


@pytest.mark.parametrize(
    "changed_config",
    (
        MassiveAdaptivePPOConfigV1(
            rollout_length=21,
            minibatch_size=63,
            seed=17,
        ),
        MassiveAdaptivePPOConfigV1(
            rollout_length=63,
            minibatch_size=21,
            seed=17,
        ),
    ),
)
def test_manifest_v2_rejects_ppo_geometry_that_breaks_candidate_schedule(
    changed_config: MassiveAdaptivePPOConfigV1,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="invalid-ppo-candidate-geometry"
    )
    changed = replace(
        manifest,
        ppo_config=changed_config,
        semantic_receipt_sha256="0" * 64,
    )
    changed = replace(
        changed,
        semantic_receipt_sha256=semantic_sha256(changed.semantic_unsigned()),
    )
    with pytest.raises(MassiveAdaptiveRLWorkflowV2Error, match="manifest V2"):
        changed.validate()


def test_v2_cli_derives_schedule_without_update_arguments(tmp_path, capsys) -> None:
    path = tmp_path / "cli-manifest-v2.json"
    assert (
        main(
            [
                "manifest",
                "--experiment-id",
                "cli-session-schedule",
                "--output",
                str(path),
                "--block-sessions",
                "63",
            ]
        )
        == 0
    )
    receipt = capsys.readouterr().out.strip()
    manifest = load_massive_adaptive_rl_experiment_manifest_v2(path)
    assert receipt == manifest.semantic_receipt_sha256
    assert manifest.schedule(3).candidate_update_indices == (2, 4, 6, 8)
    assert main(["validate", "--manifest", str(path)]) == 0
    assert capsys.readouterr().out.strip() == receipt


def test_v2_training_workflow_passes_fit_authority_through_shared_protocol(
    tmp_path, monkeypatch
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v2(
        experiment_id="direct-v2-workflow"
    )
    schedule = manifest.schedule(0)
    training_authority = SimpleNamespace(
        outer_fold_index=0,
        block_sessions=63,
        blocks=(object(), object()),
        origin_session_dates=tuple(f"fit-{index:03d}" for index in range(126)),
        block_inventory_sha256=semantic_sha256("blocks"),
        source_data_qualified=True,
        semantic_receipt_sha256=semantic_sha256("v2-training-authority"),
        reinforcement_learning_authorized=True,
        validate=lambda: None,
    )
    observed: dict[str, object] = {}

    def fake_v1_runner(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            fold_index=0,
            candidate_update_indices=schedule.candidate_update_indices,
            training_run=SimpleNamespace(
                update_count=schedule.candidate_update_indices[-1]
            ),
            source_data_qualified=True,
            development_rl_training_authorized=True,
            semantic_receipt_sha256=semantic_sha256("runtime-v1-workflow"),
            validate=lambda: None,
        )

    monkeypatch.setattr(
        workflow_v2,
        "run_massive_adaptive_rl_training_workflow_v1",
        fake_v1_runner,
    )
    result = run_massive_adaptive_rl_training_workflow_v2(
        manifest=manifest,
        fold_index=0,
        seed=17,
        training_authority=cast(Any, training_authority),
        chronology_authority=cast(Any, SimpleNamespace()),
        environments=cast(Any, {}),
        artifact_root=tmp_path,
        committed_at_ms=1,
        device=torch.device("cpu"),
    )

    assert observed["training_authority"] is training_authority
    assert result.candidate_schedule == schedule
    assert result.development_rl_training_authorized
    assert not result.profitability_reporting_authorized
