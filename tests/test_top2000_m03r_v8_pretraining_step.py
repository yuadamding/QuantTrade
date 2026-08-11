from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from copy import deepcopy

import pytest
import torch

from rl_quant.training.top2000_m03r_v8_alpha_pretraining import (
    M03RV8AlphaFoldEvidence,
    M03RV8AlphaPretrainingBatch,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import (
    build_m03r_v8_alpha_pretraining_optimizer,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    M03RV8AlphaEarlyStoppingState,
    advance_m03r_v8_alpha_early_stopping,
    model_state_sha256,
    optimizer_state_sha256,
    train_m03r_v8_alpha_pretraining_update,
)


def _policy(setting_index: int = 0) -> Top2000M03RV8DevelopmentPolicy:
    torch.manual_seed(19)
    return Top2000M03RV8DevelopmentPolicy(
        setting_index,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _batch(policy: Top2000M03RV8DevelopmentPolicy) -> M03RV8AlphaPretrainingBatch:
    states = torch.randn(3, 1, 6, policy.token_dim)
    available = torch.ones(1, 6, dtype=torch.bool)
    means = []
    scales = []
    for state in states:
        output = policy.alpha_pretraining_distribution(state, available)
        means.append(output.predicted_mean.squeeze(0))
        scales.append(output.predicted_log_scale.squeeze(0))
    target = torch.randn(3, 6, 4) * 0.01
    valid = torch.ones(3, 6, 4, dtype=torch.bool)
    valid[:, 0] = False
    return M03RV8AlphaPretrainingBatch(
        predicted_mean=torch.stack(means),
        predicted_log_scale=torch.stack(scales),
        target_residual_log_return=target,
        valid=valid,
        origin_indices=torch.tensor([0, 1, 2]),
        split="training",
        fold_index=0,
        split_start_inclusive=0,
        split_stop_exclusive=70,
        source_array_sha256="a" * 64,
    )


def test_single_rank_update_mutates_only_predictive_partition() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    before = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    receipt = train_m03r_v8_alpha_pretraining_update(
        policy,
        _batch(policy),
        optimizer,
        partition,
        completed_updates=0,
        distributed_rank=0,
        distributed_world_size=1,
    )

    selected = set(partition.encoder_parameter_names) | set(
        partition.prediction_head_parameter_names
    )
    assert receipt.completed_updates_after == 1
    assert receipt.receipt_sha256
    assert receipt.model_state_before_sha256 != receipt.model_state_after_sha256
    assert any(
        not torch.equal(before[name], value)
        for name, value in policy.state_dict().items()
        if name in selected
    )
    assert all(
        torch.equal(before[name], value)
        for name, value in policy.state_dict().items()
        if name not in selected
    )


def test_nonfinite_gradient_fails_before_parameter_or_optimizer_mutation() -> None:
    policy = _policy()
    optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
    batch = _batch(policy)
    batch.predicted_mean.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    state_before = deepcopy(optimizer.state_dict())

    with pytest.raises(RuntimeError, match="non-finite"):
        train_m03r_v8_alpha_pretraining_update(
            policy,
            batch,
            optimizer,
            partition,
            completed_updates=0,
            distributed_rank=0,
            distributed_world_size=1,
        )

    assert model_state_sha256(policy) == before_model
    assert optimizer_state_sha256(optimizer) == before_optimizer
    assert optimizer.state_dict() == state_before


def _evidence(value: float) -> M03RV8AlphaFoldEvidence:
    return M03RV8AlphaFoldEvidence(
        fold_index=0,
        mean_spearman_rank_ic=(0.0, value, value - 0.01, 0.0),
        mean_top_bottom_decile_spread=(0.0, 0.01, 0.01, 0.0),
        valid_date_counts=(10, 10, 10, 10),
        source_array_sha256="b" * 64,
    )


def test_early_stopping_selects_strict_improvement_and_earliest_tie() -> None:
    state = M03RV8AlphaEarlyStoppingState()
    state = advance_m03r_v8_alpha_early_stopping(
        state,
        completed_updates=4,
        evidence=_evidence(0.03),
        model_state_sha256_value="1" * 64,
    )
    state = advance_m03r_v8_alpha_early_stopping(
        state,
        completed_updates=8,
        evidence=_evidence(0.03),
        model_state_sha256_value="2" * 64,
    )
    assert state.best_update == 4
    assert state.best_model_state_sha256 == "1" * 64
    assert state.consecutive_non_improving_evaluations == 1

    for update in (12, 16, 20):
        state = advance_m03r_v8_alpha_early_stopping(
            state,
            completed_updates=update,
            evidence=_evidence(0.02),
            model_state_sha256_value=str(update // 4) * 64,
        )
    assert state.stopped
    assert state.best_update == 4


def test_two_rank_gloo_step_finishes_with_identical_states(tmp_path) -> None:
    script = tmp_path / "two_rank_step.py"
    result_path = tmp_path / "result.json"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            import torch
            import torch.distributed as dist

            from rl_quant.training.top2000_m03r_v8_alpha_pretraining import M03RV8AlphaPretrainingBatch
            from rl_quant.training.top2000_m03r_v8_policy import Top2000M03RV8DevelopmentPolicy
            from rl_quant.training.top2000_m03r_v8_pretraining_optimizer import build_m03r_v8_alpha_pretraining_optimizer
            from rl_quant.training.top2000_m03r_v8_pretraining_step import train_m03r_v8_alpha_pretraining_update

            rank = int(os.environ["RANK"])
            torch.manual_seed(29)
            dist.init_process_group("gloo")
            policy = Top2000M03RV8DevelopmentPolicy(0, token_dim=16, raw_stock_chunk=8, activation_checkpointing=False)
            optimizer, partition = build_m03r_v8_alpha_pretraining_optimizer(policy)
            states = torch.randn(2, 1, 5, 16)
            available = torch.ones(1, 5, dtype=torch.bool)
            outputs = [policy.alpha_pretraining_distribution(row, available) for row in states]
            valid = torch.ones(2, 5, 4, dtype=torch.bool)
            valid[:, 0] = False
            batch = M03RV8AlphaPretrainingBatch(
                predicted_mean=torch.stack([row.predicted_mean.squeeze(0) for row in outputs]),
                predicted_log_scale=torch.stack([row.predicted_log_scale.squeeze(0) for row in outputs]),
                target_residual_log_return=torch.full((2, 5, 4), 0.01 * (rank + 1)),
                valid=valid,
                origin_indices=torch.tensor([rank, rank + 2]),
                split="training",
                fold_index=0,
                split_start_inclusive=0,
                split_stop_exclusive=70,
                source_array_sha256="c" * 64,
            )
            receipt = train_m03r_v8_alpha_pretraining_update(
                policy,
                batch,
                optimizer,
                partition,
                completed_updates=0,
                distributed_rank=rank,
                distributed_world_size=2,
            )
            hashes = [None, None]
            dist.all_gather_object(hashes, receipt.receipt_sha256)
            if rank == 0:
                Path({str(result_path)!r}).write_text(json.dumps(hashes))
            dist.destroy_process_group()
            """
        )
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(script),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    hashes = json.loads(result_path.read_text())
    assert hashes[0] == hashes[1]
