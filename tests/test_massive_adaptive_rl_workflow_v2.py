from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.training.massive_adaptive_ppo_v1 import MassiveAdaptivePPOConfigV1
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLWorkflowV2Error,
    build_massive_adaptive_rl_candidate_schedule_v1,
    build_massive_adaptive_rl_experiment_manifest_v2,
    load_massive_adaptive_rl_experiment_manifest_v2,
    main,
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
