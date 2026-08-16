"""Exact-workload disposable two-H100 qualification for M03R-v16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
import torch

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTINGS,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    M03RV16FoldGeometry,
    M03RV16PanelSchedule,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_optimizer import (
    M03RV16OptimizerPartition,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    M03RV16ValidatedStructuralSlab,
)
from rl_quant.training.top2000_m03r_v16_training_runtime import (
    run_m03r_v16_pretraining_fold_update,
)
from rl_quant.training.top2000_m03r_v9_pretraining_step import (
    model_state_sha256,
    optimizer_state_sha256,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    M03RV9DeviceRiskState,
    project_m03r_v9_active_book,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    M03RV9MaterializedRiskSource,
)

M03R_V16_CAPACITY_RANK_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-exact-workload-capacity-rank-v2"
)
M03R_V16_CAPACITY_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-exact-workload-two-h100-terminal-v2"
)


class M03RV16CapacityError(ValueError):
    """The disposable V16 H100 capacity qualification failed or drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(tuple(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV16CapacityRankEvidence:
    setting_index: int
    distributed_rank: int
    distributed_world_size: int
    cuda_device_name: str
    cuda_total_memory_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    update_plan_sha256: str
    batch_receipt_sha256: str
    score_step_receipt_sha256: str
    structural_slab_receipt_sha256: str
    qualification_projection_receipt_sha256: str
    qualification_requested_active_one_way_mass: float
    qualification_projected_active_one_way_mass: float
    qualification_requested_to_executed_retention: float
    post_update_model_state_sha256: str
    post_update_optimizer_state_sha256: str
    episode_state_rows: int
    global_origin_count: int
    local_origin_count: int
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    bf16_forward_backward_executed: bool = True
    nccl_gradient_sum_executed: bool = True
    optimizer_mutation_executed: bool = True
    qualification_projection_executed: bool = True
    qualification_risk_repair_executed: bool = True
    scientific_checkpoint_published: bool = False
    disposable_output_only: bool = True
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_CAPACITY_RANK_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or self.distributed_world_size != 2
            or self.distributed_rank not in {0, 1}
            or "H100" not in self.cuda_device_name.upper()
            or self.cuda_total_memory_bytes < 75 * 1024**3
            or not 0 < self.peak_allocated_bytes <= self.peak_reserved_bytes
            or self.peak_reserved_bytes >= self.cuda_total_memory_bytes
            or self.episode_state_rows != M03R_V16_PREDICTIVE_SPEC.episode_state_rows
            or self.global_origin_count != 43
            or self.local_origin_count not in {21, 22}
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or not self.qualification_requested_active_one_way_mass > 0.0
            or not self.qualification_projected_active_one_way_mass > 0.0
            or not 0.0 < self.qualification_requested_to_executed_retention < 1.0
            or not all(
                _digest(value)
                for value in (
                    self.update_plan_sha256,
                    self.batch_receipt_sha256,
                    self.score_step_receipt_sha256,
                    self.structural_slab_receipt_sha256,
                    self.qualification_projection_receipt_sha256,
                    self.post_update_model_state_sha256,
                    self.post_update_optimizer_state_sha256,
                )
            )
            or not self.bf16_forward_backward_executed
            or not self.nccl_gradient_sum_executed
            or not self.optimizer_mutation_executed
            or not self.qualification_projection_executed
            or not self.qualification_risk_repair_executed
            or self.scientific_checkpoint_published
            or not self.disposable_output_only
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_CAPACITY_RANK_SCHEMA
        ):
            raise M03RV16CapacityError("V16 capacity rank evidence drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16CapacityTerminal:
    setting_index: int
    rank_evidence: tuple[M03RV16CapacityRankEvidence, ...]
    common_post_update_model_state_sha256: str
    common_post_update_optimizer_state_sha256: str
    maximum_peak_allocated_bytes: int
    maximum_peak_reserved_bytes: int
    minimum_unreserved_memory_bytes: int
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    exact_workload_qualified: bool = True
    scientific_checkpoint_published: bool = False
    predictive_training_authorized: bool = False
    economic_training_authorized: bool = False
    outer_2026_accessed: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_CAPACITY_TERMINAL_SCHEMA

    def validate(self) -> None:
        if len(self.rank_evidence) != 2:
            raise M03RV16CapacityError("V16 capacity requires exactly two ranks")
        for value in self.rank_evidence:
            value.validate()
        allocated = max(value.peak_allocated_bytes for value in self.rank_evidence)
        reserved = max(value.peak_reserved_bytes for value in self.rank_evidence)
        unreserved = min(
            value.cuda_total_memory_bytes - value.peak_reserved_bytes
            for value in self.rank_evidence
        )
        if (
            self.setting_index not in range(len(M03R_V16_SETTINGS))
            or tuple(value.distributed_rank for value in self.rank_evidence) != (0, 1)
            or any(
                value.setting_index != self.setting_index
                for value in self.rank_evidence
            )
            or len(
                {value.post_update_model_state_sha256 for value in self.rank_evidence}
            )
            != 1
            or len(
                {
                    value.post_update_optimizer_state_sha256
                    for value in self.rank_evidence
                }
            )
            != 1
            or self.common_post_update_model_state_sha256
            != self.rank_evidence[0].post_update_model_state_sha256
            or self.common_post_update_optimizer_state_sha256
            != self.rank_evidence[0].post_update_optimizer_state_sha256
            or self.maximum_peak_allocated_bytes != allocated
            or self.maximum_peak_reserved_bytes != reserved
            or self.minimum_unreserved_memory_bytes != unreserved
            or self.hold_target_sessions != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256 != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.minimum_unreserved_memory_bytes
            < max(
                8 * 1024**3,
                min(value.cuda_total_memory_bytes for value in self.rank_evidence)
                // 10,
            )
            or not self.exact_workload_qualified
            or self.scientific_checkpoint_published
            or self.predictive_training_authorized
            or self.economic_training_authorized
            or self.outer_2026_accessed
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_CAPACITY_TERMINAL_SCHEMA
        ):
            raise M03RV16CapacityError("V16 capacity terminal drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                **asdict(self),
                "rank_evidence": tuple(
                    value.receipt_sha256 for value in self.rank_evidence
                ),
            }
        )


def run_m03r_v16_disposable_capacity_rank(
    cache: Top2000VerifiedDevelopmentCache,
    schedule: M03RV16PanelSchedule,
    geometry: M03RV16FoldGeometry,
    risk_source: M03RV9MaterializedRiskSource,
    structural_slab: M03RV16ValidatedStructuralSlab,
    policy: Top2000M03RV16PredictivePolicy,
    optimizer: torch.optim.Optimizer,
    partition: M03RV16OptimizerPartition,
    *,
    distributed_rank: int,
    device: torch.device,
    qualification_risk_state: M03RV9DeviceRiskState,
    qualification_benchmark_weights: torch.Tensor,
    qualification_trade_mask: torch.Tensor,
    qualification_risk_asset_caps: torch.Tensor,
    qualification_risk_gross_max: torch.Tensor,
) -> M03RV16CapacityRankEvidence:
    """Execute one disposable exact-shape update and one projection probe."""

    if device.type != "cuda" or distributed_rank not in {0, 1}:
        raise M03RV16CapacityError("V16 capacity requires CUDA ranks zero and one")
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        update = run_m03r_v16_pretraining_fold_update(
            cache,
            schedule,
            geometry,
            risk_source,
            structural_slab,
            policy,
            optimizer,
            partition,
            completed_updates=0,
            distributed_rank=distributed_rank,
            distributed_world_size=2,
            device=device,
        )
    qualification_risk_state.validate()
    qualification_origin = geometry.qualification_origin_start_inclusive
    requested = qualification_benchmark_weights.clone()
    caps = qualification_risk_asset_caps.clone()
    cash = qualification_risk_state.cash_index
    eligible = torch.nonzero(
        qualification_trade_mask[0]
        & (
            torch.arange(
                qualification_trade_mask.shape[1],
                device=qualification_trade_mask.device,
            )
            != cash
        ),
        as_tuple=False,
    ).flatten()
    cash_room = float(requested[0, cash])
    if eligible.numel() == 0 or cash_room <= 2.0e-4:
        raise M03RV16CapacityError(
            "V16 capacity cannot construct a nontrivial projection probe"
        )
    asset = int(eligible[0])
    delta = min(0.01, 0.5 * cash_room)
    requested[0, asset] += delta
    requested[0, cash] -= delta
    caps[0, asset] = qualification_benchmark_weights[0, asset] + 0.25 * delta
    projection = project_m03r_v9_active_book(
        requested,
        qualification_benchmark_weights,
        qualification_trade_mask,
        caps,
        qualification_risk_gross_max,
        qualification_risk_state,
        origin_state_index=qualification_origin,
        sequence_asset_axis_sha256=cache.action_hash,
        checkpoint_asset_axis_sha256=cache.action_hash,
        expected_manifest_sha256=qualification_risk_state.manifest_sha256,
    )
    requested_mass = 0.5 * float(
        (requested - qualification_benchmark_weights).abs().sum()
    )
    projected_mass = 0.5 * float(
        (projection.projected_weights - qualification_benchmark_weights).abs().sum()
    )
    retention = float(projection.requested_to_executed_retention.squeeze())
    risk_repair_executed = not torch.equal(projection.projected_weights, requested)
    if (
        requested_mass <= 0.0
        or projected_mass <= 0.0
        or not 0.0 < retention < 1.0
        or not risk_repair_executed
    ):
        raise M03RV16CapacityError(
            "V16 capacity projection probe did not execute a nontrivial repair"
        )
    projection_receipt = _sha256(
        {
            "origin_state_index": qualification_origin,
            "risk_state_sha256": qualification_risk_state.state_sha256,
            "requested_weights_sha256": _tensor_sha256(requested),
            "projected_weights_sha256": _tensor_sha256(projection.projected_weights),
            "requested_active_one_way_mass": requested_mass,
            "projected_active_one_way_mass": projected_mass,
            "requested_to_executed_retention": retention,
            "requested_to_executed_retention_sha256": _tensor_sha256(
                projection.requested_to_executed_retention
            ),
            "risk_manifest_sha256": projection.risk_manifest_sha256,
        }
    )
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    result = M03RV16CapacityRankEvidence(
        setting_index=policy.v16_setting.setting_index,
        distributed_rank=distributed_rank,
        distributed_world_size=2,
        cuda_device_name=properties.name,
        cuda_total_memory_bytes=properties.total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        update_plan_sha256=update.update_plan.receipt_sha256,
        batch_receipt_sha256=update.batch.receipt_sha256,
        score_step_receipt_sha256=update.step.receipt_sha256,
        structural_slab_receipt_sha256=structural_slab.receipt_sha256,
        qualification_projection_receipt_sha256=projection_receipt,
        qualification_requested_active_one_way_mass=requested_mass,
        qualification_projected_active_one_way_mass=projected_mass,
        qualification_requested_to_executed_retention=retention,
        post_update_model_state_sha256=model_state_sha256(policy),
        post_update_optimizer_state_sha256=optimizer_state_sha256(optimizer),
        episode_state_rows=M03R_V16_PREDICTIVE_SPEC.episode_state_rows,
        global_origin_count=len(update.update_plan.global_origins),
        local_origin_count=len(update.update_plan.rank_origins[distributed_rank]),
    )
    result.validate()
    return result


def build_m03r_v16_capacity_terminal(
    rank_evidence: tuple[M03RV16CapacityRankEvidence, ...],
) -> M03RV16CapacityTerminal:
    rows = tuple(sorted(rank_evidence, key=lambda value: value.distributed_rank))
    if len(rows) != 2:
        raise M03RV16CapacityError("V16 capacity terminal requires two rank rows")
    result = M03RV16CapacityTerminal(
        setting_index=rows[0].setting_index,
        rank_evidence=rows,
        common_post_update_model_state_sha256=(rows[0].post_update_model_state_sha256),
        common_post_update_optimizer_state_sha256=(
            rows[0].post_update_optimizer_state_sha256
        ),
        maximum_peak_allocated_bytes=max(value.peak_allocated_bytes for value in rows),
        maximum_peak_reserved_bytes=max(value.peak_reserved_bytes for value in rows),
        minimum_unreserved_memory_bytes=min(
            value.cuda_total_memory_bytes - value.peak_reserved_bytes for value in rows
        ),
    )
    result.validate()
    return result


__all__ = [
    "M03R_V16_CAPACITY_RANK_SCHEMA",
    "M03R_V16_CAPACITY_TERMINAL_SCHEMA",
    "M03RV16CapacityError",
    "M03RV16CapacityRankEvidence",
    "M03RV16CapacityTerminal",
    "build_m03r_v16_capacity_terminal",
    "run_m03r_v16_disposable_capacity_rank",
]
