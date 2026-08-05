"""Economic endpoint and cost-ladder evaluation for canonical Hold-30 traces.

The learned ensemble is executed once at the bound 20 bp primary cost. This
module validates that trace, re-prices its immutable gross/action path at
10/20/40 bp, and compares every rung with the same bound C1 control trace.
Continuing wealth is primary; a separate score-end liquidation diagnostic
charges the risky book without mutating the continuing trace.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import torch

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.evaluation.hold30_controls import (
    HOLD30_COST_RUNGS_BPS,
    HOLD30_PRIMARY_COST_BPS,
    Hold30ControlGrossTrace,
    price_hold30_cost_ladder,
)
from rl_quant.evaluation.hold30_metrics import (
    aggregate_hold30_metrics,
    verify_hold30_metrics_digest,
)
from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.training.hold30_runtime import Hold30CanonicalTrace


HOLD30_ENDPOINT_SCHEMA = "rl-quant.hold30.economic-endpoints-v1"
HOLD30_ENDPOINT_SCORE_DECISIONS = 63
HOLD30_ENDPOINT_TOLERANCE = 1e-6
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class Hold30EndpointError(ValueError):
    """Economic inputs or endpoint accounting violate the sealed contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise Hold30EndpointError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _score_indices(mask: torch.Tensor, decisions: int) -> tuple[int, ...]:
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (decisions,)
    ):
        raise Hold30EndpointError("score_mask must be boolean [decision]")
    indices = tuple(int(value) for value in torch.where(mask.detach().cpu())[0].tolist())
    if len(indices) != HOLD30_ENDPOINT_SCORE_DECISIONS:
        raise Hold30EndpointError("economic evaluation requires exactly 63 scored decisions")
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise Hold30EndpointError("scored decisions must form one contiguous block")
    return indices


def _max_drawdown(growth: torch.Tensor) -> float:
    wealth = torch.cat((growth.new_ones(1), torch.cumprod(growth, dim=0)))
    running = torch.cummax(wealth, dim=0).values
    return float((1.0 - wealth / running).max().item())


def _sample_std(value: torch.Tensor) -> float:
    return float(value.std(unbiased=True).item()) if value.numel() > 1 else 0.0


def _risk_metrics(policy: torch.Tensor, benchmark: torch.Tensor) -> dict[str, Any]:
    active = policy - benchmark
    active_std = _sample_std(active)
    benchmark_centered = benchmark - benchmark.mean()
    benchmark_variance = float((benchmark_centered.square().sum() / (benchmark.numel() - 1)).item())
    beta = None
    if benchmark_variance > 0.0:
        covariance = float(
            (((policy - policy.mean()) * benchmark_centered).sum() / (policy.numel() - 1)).item()
        )
        beta = covariance / benchmark_variance
    information_ratio = None
    if active_std > 0.0:
        information_ratio = float(math.sqrt(252.0) * float(active.mean().item()) / active_std)
    active_log = torch.log1p(policy) - torch.log1p(benchmark)
    return {
        "mean_policy_return_daily": float(policy.mean().item()),
        "mean_benchmark_return_daily": float(benchmark.mean().item()),
        "mean_active_return_daily": float(active.mean().item()),
        "mean_active_log_return_daily": float(active_log.mean().item()),
        "active_return_volatility_daily": active_std,
        "tracking_error_annualized": active_std * math.sqrt(252.0),
        "information_ratio_annualized": information_ratio,
        "beta_to_C1": beta,
        "policy_max_drawdown": _max_drawdown(1.0 + policy),
        "active_relative_wealth_max_drawdown": _max_drawdown(torch.exp(active_log)),
    }


def _portfolio_metrics(
    policy_weights: torch.Tensor,
    benchmark_weights: torch.Tensor,
    *,
    cash_index: int,
) -> dict[str, float]:
    risky = policy_weights.clone()
    risky[:, cash_index] = 0.0
    exposure = risky.sum(dim=-1)
    conditional = risky / exposure.clamp_min(torch.finfo(risky.dtype).eps).unsqueeze(-1)
    conditional = torch.where(exposure.unsqueeze(-1) > 0.0, conditional, torch.zeros_like(risky))
    hhi = conditional.square().sum(dim=-1)
    effective = torch.where(hhi > 0.0, 1.0 / hhi, torch.zeros_like(hhi))
    sorted_risky = risky.sort(dim=-1, descending=True).values
    active_share = 0.5 * (policy_weights - benchmark_weights).abs().sum(dim=-1)
    return {
        "mean_risky_exposure": float(exposure.mean().item()),
        "mean_cash_weight": float(policy_weights[:, cash_index].mean().item()),
        "mean_active_share_vs_C1": float(active_share.mean().item()),
        "mean_conditional_risky_hhi": float(hhi.mean().item()),
        "mean_effective_risky_holdings": float(effective.mean().item()),
        "mean_top10_risky_mass": float(sorted_risky[:, :10].sum(dim=-1).mean().item()),
        "mean_top50_risky_mass": float(sorted_risky[:, :50].sum(dim=-1).mean().item()),
    }


def _action_metrics(
    trace: Hold30CanonicalTrace,
    indices: tuple[int, ...],
) -> dict[str, Any]:
    requested_constructed: list[torch.Tensor] = []
    constructed_filled: list[torch.Tensor] = []
    projection: list[torch.Tensor] = []
    cause_turnover = {
        cause: trace.transitions[0].decision_weights.new_zeros(())
        for cause in TURNOVER_CAUSES
    }
    for index in indices:
        transition = trace.transitions[index]
        requested_constructed.append(
            0.5 * (transition.requested_delta - transition.constructed_delta).abs().sum(-1)
        )
        constructed_filled.append(
            0.5 * (transition.constructed_delta - transition.filled_delta).abs().sum(-1)
        )
        projection.append(transition.projection_distance.detach())
        for cause in TURNOVER_CAUSES:
            cause_turnover[cause] += transition.turnover_by_cause[cause].detach().sum()
    requested_distance = torch.cat(requested_constructed)
    filled_distance = torch.cat(constructed_filled)
    projection_distance = torch.cat(projection)
    observations = len(indices)
    return {
        "requested_to_constructed_one_way_mean": float(requested_distance.mean().item()),
        "constructed_to_filled_one_way_mean": float(filled_distance.mean().item()),
        "constructed_to_filled_one_way_max": float(filled_distance.max().item()),
        "runtime_projection_distance_mean": float(projection_distance.mean().item()),
        "requested_constraint_binding_rate": float(
            (requested_distance > HOLD30_ENDPOINT_TOLERANCE).to(torch.float64).mean().item()
        ),
        "safety_projection_binding_rate": float(
            (filled_distance > HOLD30_ENDPOINT_TOLERANCE).to(torch.float64).mean().item()
        ),
        "turnover_by_cause": {
            cause.value: {
                "total_one_way": float(total.item()),
                "mean_one_way_per_scored_decision": float(total.item()) / observations,
            }
            for cause, total in cause_turnover.items()
        },
    }


def _learned_gross_and_turnover(
    trace: Hold30CanonicalTrace,
) -> tuple[torch.Tensor, dict[TurnoverCause, torch.Tensor]]:
    gross = torch.stack([transition.holding_return.detach() for transition in trace.transitions])
    turnover = {
        cause: torch.stack(
            [transition.turnover_by_cause[cause].detach() for transition in trace.transitions]
        )
        for cause in TURNOVER_CAUSES
    }
    if gross.ndim != 2 or gross.shape[1] != 1:
        raise Hold30EndpointError("sealed ensemble evaluation requires one portfolio path")
    expected_cost = 0.002 * sum(
        turnover.values(),
        torch.zeros_like(gross),
    )
    observed_cost = torch.stack([transition.cost.detach() for transition in trace.transitions])
    observed_net = torch.stack([transition.net_return.detach() for transition in trace.transitions])
    if not bool(
        torch.allclose(expected_cost, observed_cost, atol=1e-10, rtol=1e-10)
    ) or not bool(
        torch.allclose(gross - expected_cost, observed_net, atol=1e-10, rtol=1e-10)
    ):
        raise Hold30EndpointError("learned 20-bp trace does not reconcile to gross and turnover")
    return gross, turnover


def evaluate_hold30_endpoints(
    trace: Hold30CanonicalTrace,
    c1_trace: Hold30ControlGrossTrace,
    *,
    score_mask: torch.Tensor,
    learned_source_receipt_sha256: str,
) -> dict[str, Any]:
    """Evaluate one fold's canonical ensemble against bound C1."""

    _require_digest("learned_source_receipt_sha256", learned_source_receipt_sha256)
    if not isinstance(trace, Hold30CanonicalTrace):
        raise TypeError("trace must be Hold30CanonicalTrace")
    if not isinstance(c1_trace, Hold30ControlGrossTrace) or c1_trace.control_id != "C1":
        raise Hold30EndpointError("c1_trace must be the bound C1 control")
    count = len(trace.transitions)
    indices = _score_indices(score_mask, count)
    if c1_trace.axis_id != trace.pending_intents[0].axis_id:
        raise Hold30EndpointError("learned and C1 axes differ")
    if c1_trace.weights.shape[0] != count + 1 or c1_trace.weights.shape[1] != 1:
        raise Hold30EndpointError("C1 trace does not align with the learned portfolio path")
    if trace.boundary_states[0].ledger.cash_index != 0:
        raise Hold30EndpointError("learned trace CASH coordinate differs from bound C1")
    if not torch.equal(c1_trace.score_mask.detach().cpu(), score_mask.detach().cpu()):
        raise Hold30EndpointError("learned and C1 score masks differ")

    holding = aggregate_hold30_metrics(trace, score_origin_mask=score_mask)
    if not verify_hold30_metrics_digest(holding):
        raise AssertionError("fresh holding telemetry failed its digest")
    gross, turnover = _learned_gross_and_turnover(trace)
    c1_ladder = price_hold30_cost_ladder(c1_trace)
    learned_weights = torch.stack(
        [trace.transitions[index].post_cost_weights.detach()[0] for index in indices]
    )
    c1_weights = c1_trace.weights[torch.tensor(indices) + 1, 0]
    portfolio = _portfolio_metrics(
        learned_weights,
        c1_weights,
        cash_index=trace.boundary_states[0].ledger.cash_index,
    )
    actions = _action_metrics(trace, indices)

    rungs: dict[str, Any] = {}
    tensor_receipts: dict[str, Any] = {}
    selected = torch.tensor(indices, dtype=torch.long, device=gross.device)
    score_end_state = trace.boundary_states[indices[-1] + 1]
    learned_liquidation_turnover = float(
        (
            1.0
            - score_end_state.ledger.weights[0, score_end_state.ledger.cash_index]
        ).item()
    )
    c1_end_weights = c1_trace.weights[indices[-1] + 1, 0]
    c1_liquidation_turnover = float((1.0 - c1_end_weights[0]).item())
    for c1_rung, cost_bps in zip(c1_ladder.rungs, HOLD30_COST_RUNGS_BPS):
        rate = gross.new_tensor(cost_bps / 10_000.0)
        learned_total_turnover = sum(turnover.values(), torch.zeros_like(gross))
        learned_net = gross - rate * learned_total_turnover
        policy_scored = learned_net.index_select(0, selected)[:, 0]
        c1_scored = c1_rung.net_returns.index_select(0, selected)[:, 0]
        if bool((policy_scored <= -1.0).any()) or bool((c1_scored <= -1.0).any()):
            raise Hold30EndpointError("a cost rung reached -100%")
        active_log = torch.log1p(policy_scored) - torch.log1p(c1_scored)
        policy_wealth = torch.cumprod(1.0 + policy_scored, dim=0)[-1]
        c1_wealth = torch.cumprod(1.0 + c1_scored, dim=0)[-1]
        learned_liquidated = policy_wealth * (1.0 - rate * learned_liquidation_turnover)
        c1_liquidated = c1_wealth * (1.0 - rate * c1_liquidation_turnover)
        if learned_liquidated <= 0.0 or c1_liquidated <= 0.0:
            raise Hold30EndpointError("terminal liquidation exhausted portfolio wealth")
        rung = {
            "cost_bps": cost_bps,
            "continuing_policy_return": float((policy_wealth - 1.0).item()),
            "continuing_C1_return": float((c1_wealth - 1.0).item()),
            "continuing_active_log_wealth": float(active_log.sum().item()),
            "active_log_wealth_first_10": float(active_log[:10].sum().item()),
            "active_log_wealth_remaining": float(active_log[10:].sum().item()),
            "liquidated_policy_return": float((learned_liquidated - 1.0).item()),
            "liquidated_C1_return": float((c1_liquidated - 1.0).item()),
            "liquidated_active_log_wealth": float(
                (torch.log(learned_liquidated) - torch.log(c1_liquidated)).item()
            ),
            "learned_terminal_liquidation_turnover": learned_liquidation_turnover,
            "C1_terminal_liquidation_turnover": c1_liquidation_turnover,
            "risk": _risk_metrics(policy_scored, c1_scored),
        }
        rungs[str(cost_bps)] = rung
        tensor_receipts[str(cost_bps)] = {
            "policy_net_returns_sha256": _tensor_sha256(policy_scored),
            "C1_net_returns_sha256": _tensor_sha256(c1_scored),
            "active_log_returns_sha256": _tensor_sha256(active_log),
        }

    payload: dict[str, Any] = {
        "schema_version": HOLD30_ENDPOINT_SCHEMA,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "axis_id": c1_trace.axis_id,
        "learned_source_receipt_sha256": learned_source_receipt_sha256,
        "C1_trace_sha256": c1_trace.trace_sha256,
        "C1_cost_ladder_receipt_sha256": c1_ladder.receipt_sha256,
        "score_indices": list(indices),
        "score_mask_sha256": _tensor_sha256(score_mask),
        "cost_rungs_bps": list(HOLD30_COST_RUNGS_BPS),
        "primary_cost_bps": HOLD30_PRIMARY_COST_BPS,
        "continuing_wealth_primary": True,
        "holding_telemetry": holding,
        "portfolio": portfolio,
        "actions": actions,
        "rungs": rungs,
        "tensor_receipts": tensor_receipts,
        "scientific_qualification": False,
        "promotion_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def verify_hold30_endpoint_receipt(
    receipt: Mapping[str, Any],
    *,
    trace: Hold30CanonicalTrace | None = None,
    c1_trace: Hold30ControlGrossTrace | None = None,
    score_mask: torch.Tensor | None = None,
    learned_source_receipt_sha256: str | None = None,
) -> None:
    """Verify the receipt and optionally recompute it from live sealed traces."""

    required = {
        "schema_version",
        "protocol_generation",
        "axis_id",
        "learned_source_receipt_sha256",
        "C1_trace_sha256",
        "C1_cost_ladder_receipt_sha256",
        "score_indices",
        "score_mask_sha256",
        "cost_rungs_bps",
        "primary_cost_bps",
        "continuing_wealth_primary",
        "holding_telemetry",
        "portfolio",
        "actions",
        "rungs",
        "tensor_receipts",
        "scientific_qualification",
        "promotion_authorized",
        "receipt_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise Hold30EndpointError("endpoint receipt has partial or unknown fields")
    if (
        receipt["schema_version"] != HOLD30_ENDPOINT_SCHEMA
        or receipt["protocol_generation"] != HOLD30_PROTOCOL_GENERATION
        or receipt["cost_rungs_bps"] != list(HOLD30_COST_RUNGS_BPS)
        or receipt["primary_cost_bps"] != HOLD30_PRIMARY_COST_BPS
        or receipt["continuing_wealth_primary"] is not True
        or receipt["scientific_qualification"] is not False
        or receipt["promotion_authorized"] is not False
    ):
        raise Hold30EndpointError("endpoint receipt identity/authority fields are invalid")
    for name in (
        "axis_id",
        "learned_source_receipt_sha256",
        "C1_trace_sha256",
        "C1_cost_ladder_receipt_sha256",
        "score_mask_sha256",
        "receipt_sha256",
    ):
        _require_digest(name, receipt[name])
    if not verify_hold30_metrics_digest(receipt["holding_telemetry"]):
        raise Hold30EndpointError("holding telemetry receipt is invalid")
    score_indices = receipt["score_indices"]
    if (
        not isinstance(score_indices, list)
        or len(score_indices) != HOLD30_ENDPOINT_SCORE_DECISIONS
        or score_indices != list(range(score_indices[0], score_indices[-1] + 1))
    ):
        raise Hold30EndpointError("endpoint score indices are invalid")
    if set(receipt["rungs"]) != {"10", "20", "40"} or set(
        receipt["tensor_receipts"]
    ) != {"10", "20", "40"}:
        raise Hold30EndpointError("endpoint cost ladder is incomplete")
    for rung in receipt["tensor_receipts"].values():
        if not isinstance(rung, Mapping):
            raise Hold30EndpointError("endpoint tensor receipt is malformed")
        for digest in rung.values():
            _require_digest("endpoint tensor", digest)
    unsigned = dict(receipt)
    del unsigned["receipt_sha256"]
    if _sha256(unsigned) != receipt["receipt_sha256"]:
        raise Hold30EndpointError("endpoint receipt self-hash mismatch")
    live = (trace, c1_trace, score_mask, learned_source_receipt_sha256)
    if any(value is not None for value in live):
        if any(value is None for value in live):
            raise Hold30EndpointError(
                "live endpoint verification requires both traces, score mask, and source receipt"
            )
        assert trace is not None
        assert c1_trace is not None
        assert score_mask is not None
        assert learned_source_receipt_sha256 is not None
        recomputed = evaluate_hold30_endpoints(
            trace,
            c1_trace,
            score_mask=score_mask,
            learned_source_receipt_sha256=learned_source_receipt_sha256,
        )
        if _canonical_json(recomputed) != _canonical_json(receipt):
            raise Hold30EndpointError("live endpoint result differs from receipt")


__all__ = [
    "HOLD30_ENDPOINT_SCHEMA",
    "Hold30EndpointError",
    "evaluate_hold30_endpoints",
    "verify_hold30_endpoint_receipt",
]
