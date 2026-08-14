"""Single mutation boundary for one M03R-v12 predictive batch."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_PREDICTIVE_SPEC,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v12_gradient_balance import (
    M03RV12GradientBalanceReceipt,
    install_m03r_v12_balanced_gradients,
)
from rl_quant.training.top2000_m03r_v12_objective import (
    M03RV12PredictiveBatch,
    m03r_v12_predictive_loss,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v12_pretraining_optimizer import (
    M03RV12OptimizerPartition,
    validate_m03r_v12_optimizer,
)
from rl_quant.training.top2000_m03r_v12_fold import M03RV12TrainingShardPlan
from rl_quant.training.top2000_m03r_v12_schedule import M03RV12PairedInputReceipt

M03R_V12_ALPHA_STEP_SCHEMA = "rl-quant.top2000-dev.m03r-v12-alpha-step-v1"


class M03RV12AlphaStepError(ValueError):
    """The v12 alpha step failed before or during its mutation boundary."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV12AlphaStepReceipt:
    setting_index: int
    setting_id: str
    completed_updates_before: int
    completed_updates_after: int
    distributed_rank: int
    distributed_world_size: int
    batch_receipt_sha256: str
    training_shard_receipt_sha256: str
    paired_input_receipt_sha256: str
    source_array_sha256: str
    residual_operator_root_sha256: str
    optimizer_partition_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    optimizer_state_before_sha256: str
    optimizer_state_after_sha256: str
    total_loss: float
    ranking_loss: float
    economic_loss: float
    robust_regression_loss: float
    distributional_loss: float
    gradient_balance: M03RV12GradientBalanceReceipt
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    training_performed: bool = True
    qualification_evaluated_during_update: bool = False
    outer_2026_accessed: bool = False
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V12_ALPHA_STEP_SCHEMA

    def validate(self) -> None:
        self.gradient_balance.validate()
        digests = (
            self.batch_receipt_sha256,
            self.training_shard_receipt_sha256,
            self.paired_input_receipt_sha256,
            self.source_array_sha256,
            self.residual_operator_root_sha256,
            self.optimizer_partition_sha256,
            self.model_state_before_sha256,
            self.model_state_after_sha256,
            self.optimizer_state_before_sha256,
            self.optimizer_state_after_sha256,
        )
        metrics = (
            self.total_loss,
            self.ranking_loss,
            self.economic_loss,
            self.robust_regression_loss,
            self.distributional_loss,
        )
        if (
            self.setting_index not in range(3)
            or self.setting_id != M03R_V12_SETTING_IDS[self.setting_index]
            or self.completed_updates_before
            not in range(M03R_V12_PREDICTIVE_SPEC.optimizer_updates)
            or self.completed_updates_after != self.completed_updates_before + 1
            or self.distributed_world_size not in {1, 2}
            or self.distributed_rank not in range(self.distributed_world_size)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or not all(math.isfinite(value) for value in metrics)
            or self.model_state_after_sha256 == self.model_state_before_sha256
            or self.optimizer_state_after_sha256 == self.optimizer_state_before_sha256
            or self.gradient_balance.distributed_world_size
            != self.distributed_world_size
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or not self.training_performed
            or self.qualification_evaluated_during_update
            or self.outer_2026_accessed
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V12_ALPHA_STEP_SCHEMA
        ):
            raise M03RV12AlphaStepError("v12 alpha-step receipt drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


def train_m03r_v12_predictive_batch_update(
    policy: Top2000M03RV12PredictivePolicy,
    batch: M03RV12PredictiveBatch,
    optimizer: torch.optim.Optimizer,
    partition: M03RV12OptimizerPartition,
    training_shard: M03RV12TrainingShardPlan,
    paired_input: M03RV12PairedInputReceipt,
    *,
    completed_updates: int,
    distributed_rank: int,
    distributed_world_size: int,
) -> M03RV12AlphaStepReceipt:
    """Train one batch with rank/economic gradients independently bounded."""

    batch.validate()
    training_shard.validate()
    paired_input.validate()
    if (
        policy.v12_setting != batch.setting
        or isinstance(completed_updates, bool)
        or completed_updates not in range(M03R_V12_PREDICTIVE_SPEC.optimizer_updates)
        or training_shard.completed_update != completed_updates
        or paired_input.schedule_sha256 != training_shard.panel_episode_schedule_sha256
        or paired_input.fold_index != training_shard.fold_index
        or paired_input.completed_update != training_shard.completed_update
        or paired_input.episode_start != training_shard.episode_start
        or paired_input.global_origins != training_shard.global_origins
    ):
        raise M03RV12AlphaStepError("v12 policy, batch, or update cursor drifted")
    validate_m03r_v12_optimizer(policy, optimizer, partition)
    before_model = model_state_sha256(policy)
    before_optimizer = optimizer_state_sha256(optimizer)
    loss = m03r_v12_predictive_loss(batch)
    try:
        balance = install_m03r_v12_balanced_gradients(
            policy,
            optimizer,
            partition,
            loss,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        if (
            model_state_sha256(policy) != before_model
            or optimizer_state_sha256(optimizer) != before_optimizer
        ):
            raise M03RV12AlphaStepError(
                "v12 state changed before failed-gradient rejection"
            )
        raise
    after_model = model_state_sha256(policy)
    after_optimizer = optimizer_state_sha256(optimizer)
    receipt = M03RV12AlphaStepReceipt(
        setting_index=batch.setting.setting_index,
        setting_id=batch.setting.setting_id,
        completed_updates_before=completed_updates,
        completed_updates_after=completed_updates + 1,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        batch_receipt_sha256=batch.receipt_sha256,
        training_shard_receipt_sha256=training_shard.receipt_sha256,
        paired_input_receipt_sha256=paired_input.receipt_sha256,
        source_array_sha256=batch.source_array_sha256,
        residual_operator_root_sha256=_sha256(batch.residual_operator_receipt_sha256),
        optimizer_partition_sha256=partition.receipt_sha256,
        model_state_before_sha256=before_model,
        model_state_after_sha256=after_model,
        optimizer_state_before_sha256=before_optimizer,
        optimizer_state_after_sha256=after_optimizer,
        total_loss=float(loss.total.detach()),
        ranking_loss=float(loss.ranking.detach()),
        economic_loss=float(loss.economic_total.detach()),
        robust_regression_loss=float(loss.robust_regression.detach()),
        distributional_loss=float(loss.distributional.detach()),
        gradient_balance=balance,
    )
    receipt.validate()
    return receipt


__all__ = [
    "M03R_V12_ALPHA_STEP_SCHEMA",
    "M03RV12AlphaStepError",
    "M03RV12AlphaStepReceipt",
    "train_m03r_v12_predictive_batch_update",
]
