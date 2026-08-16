from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rl_quant.training.top2000_m03r_v16_qualification_runtime as qualification
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.workflows.top2000_m03r_v16_predictive import (
    M03RV16PredictiveWorkflowError,
    _validate_gathered_update,
    resolve_m03r_v16_completion_index,
)


def _rank_row(rank: int, *, local_count: int = 22) -> dict[str, object]:
    return {
        "update_plan_sha256": "0" * 64,
        "batch_receipt_sha256": str(rank) * 64,
        "step_receipt_sha256": str(rank + 2) * 64,
        "source_array_sha256": "3" * 64,
        "selection_target_operator_root_sha256": "4" * 64,
        "action_operator_root_sha256": "5" * 64,
        "completed_updates_after": 1,
        "distributed_rank": rank,
        "local_origin_count": local_count,
        "global_origin_count": 43,
        "encoder_version_root_before": "6" * 64,
        "encoder_version_root_after": "7" * 64,
        "selection_head_version_root_before": "8" * 64,
        "selection_head_version_root_after": "9" * 64,
    }


def test_v16_completion_index_is_exactly_three_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    assert resolve_m03r_v16_completion_index(None) == 2
    assert resolve_m03r_v16_completion_index(0) == 0
    for invalid in (-1, 3, True):
        with pytest.raises(M03RV16PredictiveWorkflowError, match="drifted"):
            resolve_m03r_v16_completion_index(invalid)


def test_v16_rank_update_requires_complete_equal_mutation_evidence() -> None:
    rows = [_rank_row(0), _rank_row(1, local_count=21)]
    _validate_gathered_update(rows, 2)

    incomplete = [rows[0], {**rows[1], "local_origin_count": 20}]
    with pytest.raises(M03RV16PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(incomplete, 2)

    drifted = [
        rows[0],
        {**rows[1], "encoder_version_root_after": "a" * 64},
    ]
    with pytest.raises(M03RV16PredictiveWorkflowError, match="diverged"):
        _validate_gathered_update(drifted, 2)


def test_v16_qualification_risk_state_covers_decisions_and_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        daily_ohlcv=torch.ones((1001, 4, 5), dtype=torch.float32),
        availability=torch.ones((1001, 4), dtype=torch.bool),
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        validate_unmodified=lambda: None,
    )
    risk_source = SimpleNamespace(validate=lambda: None)
    sentinel = object()
    captured: dict[str, object] = {}

    def build(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(qualification, "build_m03r_v9_device_risk_state", build)
    geometry = render_m03r_v16_fold_geometries(1001)[0]
    result = qualification.build_m03r_v16_qualification_risk_state(
        cache,
        geometry,
        risk_source,
        SimpleNamespace(),
        SimpleNamespace(),
        device=torch.device("cpu"),
    )
    origins = captured["origin_state_indices"]
    assert result is sentinel
    assert isinstance(origins, tuple)
    assert len(origins) == 92
    assert origins[0] == geometry.qualification_origin_start_inclusive
    assert origins[-1] == geometry.qualification_origin_start_inclusive + 91
    assert captured["sequence_asset_axis_sha256"] == cache.action_hash
    assert captured["checkpoint_asset_axis_sha256"] == cache.action_hash


def test_v16_worker_source_requires_terminal_authority_and_is_predictive_only() -> None:
    source = Path(
        "src/rl_quant/workflows/top2000_m03r_v16_predictive.py"
    ).read_text(encoding="utf-8")
    assert "issue_m03r_v16_terminal_checkpoint_authority" in source
    assert "load_m03r_v16_epoch_checkpoint_for_evaluation" in source
    assert "build_m03r_v16_qualification_risk_state" in source
    assert '"economic_optimizer_updates": 0' in source
    assert '"reinforcement_learning_updates": 0' in source
    assert '"outer_2026_accessed": False' in source
    assert "top2000_m03r_v15" not in source
