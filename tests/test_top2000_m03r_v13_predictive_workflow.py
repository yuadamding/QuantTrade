from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.workflows.top2000_m03r_v13_predictive import (
    M03RV13PredictiveWorkflowError,
    _validate_gathered_update,
    resolve_m03r_v13_completion_index,
)


def _rank_row(rank: int, *, local_count: int = 8) -> dict[str, object]:
    return {
        "update_plan_sha256": "0" * 64,
        "paired_input_sha256": "1" * 64,
        "source_array_sha256": "2" * 64,
        "step_receipt_sha256": str(rank) * 64,
        "model_state_after_sha256": "3" * 64,
        "optimizer_state_after_sha256": "4" * 64,
        "target_residual_operator_root_sha256": "5" * 64,
        "action_residual_operator_root_sha256": "6" * 64,
        "local_origin_count": local_count,
        "global_origin_count": 16,
        "distributed_gradient_synchronized": True,
    }


def test_v13_completion_index_is_exactly_two_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "1")
    assert resolve_m03r_v13_completion_index(None) == 1
    assert resolve_m03r_v13_completion_index(0) == 0
    for invalid in (-1, 2, True):
        with pytest.raises(M03RV13PredictiveWorkflowError, match="0 or 1"):
            resolve_m03r_v13_completion_index(invalid)


def test_v13_rank_update_requires_equal_state_and_complete_origin_shards() -> None:
    rows = [_rank_row(0), _rank_row(1)]
    _validate_gathered_update(rows, 2)

    drifted = [rows[0], {**rows[1], "model_state_after_sha256": "f" * 64}]
    with pytest.raises(M03RV13PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(drifted, 2)

    incomplete = [rows[0], {**rows[1], "local_origin_count": 7}]
    with pytest.raises(M03RV13PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(incomplete, 2)


def test_v13_workflow_source_is_predictive_only_and_v13_homogeneous() -> None:
    source = Path(
        "src/rl_quant/workflows/top2000_m03r_v13_predictive.py"
    ).read_text(encoding="utf-8")
    assert "top2000_m03r_v12" not in source
    assert '"economic_optimizer_updates": 0' in source
    assert '"outer_2026_accessed": False' in source
    assert "run_m03r_v13_fold_qualification" in source
    assert "load_m03r_v13_alpha_checkpoint_for_evaluation" in source
