"""Fail-closed economic rebuild for Hold-30 outcome-null datasets.

The registered ``N_time`` and ``N_xs`` transforms alter ordinary risky return
outcomes only.  They intentionally do not publish a runnable sequence because
the original C1 path and every forward label are stale as soon as outcomes
change.  This module is the sole lower-layer bridge from a verified
``Hold30NullView`` to rebuilt economics:

* C1 starts from the PIT active-300 equal-weight endowment, earns each old-book
  return, then applies fill-time membership, availability, and risk repairs;
* at an explicitly receipt-bound monthly event it reconstitutes to 1/300 per
  active and tradeable name, subject to the frozen name/gross ceilings;
* every forced and scheduled one-way trade pays the row's common linear cost;
* C5 labels begin with the first return after the legal T+1 fill, follow the
  stock until a forced exit and CASH thereafter, and contain exactly thirty
  return transitions without crossing a null-transform domain.

Actor observations and every point-in-time event tensor are reused unchanged.
The builder accepts no precomputed benchmark or label tensors, so stale base
artifacts cannot be injected.  It does not discover a monthly schedule: the
caller must provide the frozen schedule explicitly, and membership changes on
unmarked dates fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

import torch

from rl_quant.datasets.hold30 import (
    Hold30DatasetError,
    Hold30DatasetSequence,
    Hold30NullDomain,
    Hold30NullView,
)


HOLD30_NULL_REBUILDER_VERSION = "hold30-outcome-null-rebuilder-v1"
HOLD30_C1_ACTIVE_COUNT = 300
HOLD30_C1_MAX_NAME_WEIGHT = 0.01
HOLD30_C5_HORIZON = 30


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _trace_digest(values: dict[str, torch.Tensor]) -> str:
    return _canonical_digest(
        {name: _tensor_digest(value) for name, value in sorted(values.items())}
    )


@dataclass(frozen=True, slots=True)
class Hold30C1RebuildTrace:
    """Rebuilt C1 books, return rows, and cause-specific execution costs."""

    weights: torch.Tensor
    holding_returns: torch.Tensor
    net_returns: torch.Tensor
    membership_turnover: torch.Tensor
    availability_turnover: torch.Tensor
    risk_turnover: torch.Tensor
    scheduled_turnover: torch.Tensor
    total_turnover: torch.Tensor
    costs: torch.Tensor

    @property
    def trace_sha256(self) -> str:
        return _trace_digest(
            {
                "weights": self.weights,
                "holding_returns": self.holding_returns,
                "net_returns": self.net_returns,
                "membership_turnover": self.membership_turnover,
                "availability_turnover": self.availability_turnover,
                "risk_turnover": self.risk_turnover,
                "scheduled_turnover": self.scheduled_turnover,
                "total_turnover": self.total_turnover,
                "costs": self.costs,
            }
        )


@dataclass(frozen=True, slots=True)
class Hold30C5Labels:
    """Thirty-session excess-log-return labels and explicit censoring."""

    values: torch.Tensor
    valid: torch.Tensor
    censored: torch.Tensor

    @property
    def labels_sha256(self) -> str:
        return _trace_digest(
            {
                "values": self.values,
                "valid": self.valid,
                "censored": self.censored,
            }
        )


@dataclass(frozen=True, slots=True)
class Hold30NullRebuildReceipt:
    """Receipt closure from source sequence and null mapping to rebuilt data."""

    builder_version: str
    source_axis_id: str
    source_provenance_receipt_id: str
    source_benchmark_sha256: str
    null_transform_id: str
    null_mapping_sha256: str
    null_output_outcomes_sha256: str
    monthly_rebalance_sha256: str
    c1_trace_sha256: str
    c5_labels_sha256: str
    transformed_provenance_receipt_id: str
    transformed_axis_id: str
    receipt_id: str


@dataclass(frozen=True, slots=True)
class Hold30NullRebuildResult:
    """A runnable transformed sequence plus independently auditable products."""

    sequence: Hold30DatasetSequence
    c1: Hold30C1RebuildTrace
    c5: Hold30C5Labels
    receipt: Hold30NullRebuildReceipt


def _domains_from_view(view: Hold30NullView) -> tuple[Hold30NullDomain, ...]:
    try:
        return tuple(Hold30NullDomain(*value) for value in view.receipt.domains)
    except (TypeError, ValueError) as exc:
        raise Hold30DatasetError("null receipt contains malformed domains") from exc


def _verify_null_view(
    base: Hold30DatasetSequence,
    view: Hold30NullView,
) -> tuple[Hold30NullDomain, ...]:
    if not isinstance(base, Hold30DatasetSequence):
        raise Hold30DatasetError("base must be a validated Hold30DatasetSequence")
    if not isinstance(view, Hold30NullView):
        raise Hold30DatasetError("view must be a Hold30NullView")
    if view.receipt.source_axis_id != base.axis_id:
        raise Hold30DatasetError("null view belongs to a different source sequence")
    if view.receipt.randomization_axis_id != base.randomization_axis_id:
        raise Hold30DatasetError("null view randomization axis does not match the source sequence")
    domains = _domains_from_view(view)
    if view.receipt.kind == "N_time":
        expected = base.n_time(view.receipt.seed, domains=domains)
    elif view.receipt.kind == "N_xs":
        expected = base.n_xs(view.receipt.seed, domains=domains)
    else:  # pragma: no cover - Hold30NullReceipt already rejects this.
        raise Hold30DatasetError("unsupported outcome-null kind")
    if expected.receipt != view.receipt:
        raise Hold30DatasetError("null receipt failed deterministic reconstruction")
    if not torch.equal(expected.source_index, view.source_index):
        raise Hold30DatasetError("null source mapping failed deterministic reconstruction")
    if not torch.equal(expected.asset_returns, view.asset_returns):
        raise Hold30DatasetError("null transformed outcomes failed deterministic reconstruction")
    return domains


def _one_way(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return 0.5 * (after - before).abs().sum(dim=-1)


def _repair_to_mask(
    weights: torch.Tensor,
    allowed: torch.Tensor,
    *,
    cash_index: int,
) -> torch.Tensor:
    target = torch.where(allowed, weights, torch.zeros_like(weights))
    target = target.clone()
    risky = torch.ones_like(target, dtype=torch.bool)
    risky[..., cash_index] = False
    target[..., cash_index] = 1.0 - torch.where(
        risky, target, torch.zeros_like(target)
    ).sum(dim=-1)
    return target


def _repair_to_risk(
    weights: torch.Tensor,
    asset_caps: torch.Tensor,
    gross_max: torch.Tensor,
    *,
    cash_index: int,
) -> torch.Tensor:
    risky = torch.ones_like(weights, dtype=torch.bool)
    risky[..., cash_index] = False
    cap = torch.minimum(
        asset_caps.clamp_min(0.0),
        weights.new_tensor(HOLD30_C1_MAX_NAME_WEIGHT),
    )
    held = torch.where(risky, weights.clamp_min(0.0), torch.zeros_like(weights))
    held = torch.minimum(held, cap)
    hard_gross = torch.minimum(
        torch.ones_like(gross_max),
        gross_max.clamp(min=0.0, max=1.0),
    )
    gross = held.sum(dim=-1)
    scale = torch.where(
        gross > hard_gross,
        hard_gross / gross.clamp_min(1e-18),
        torch.ones_like(gross),
    )
    held = held * scale.unsqueeze(-1)
    target = held.clone()
    target[..., cash_index] = 1.0 - held.sum(dim=-1)
    return target


def _monthly_equal_weight_target(
    membership: torch.Tensor,
    execution_allowed: torch.Tensor,
    asset_caps: torch.Tensor,
    gross_max: torch.Tensor,
    *,
    cash_index: int,
) -> torch.Tensor:
    risky = torch.ones_like(membership, dtype=torch.bool)
    risky[..., cash_index] = False
    active = membership & execution_allowed & risky
    target = torch.where(
        active,
        asset_caps.new_full(asset_caps.shape, 1.0 / HOLD30_C1_ACTIVE_COUNT),
        torch.zeros_like(asset_caps),
    )
    target = torch.minimum(
        target,
        torch.minimum(
            asset_caps.clamp_min(0.0),
            asset_caps.new_tensor(HOLD30_C1_MAX_NAME_WEIGHT),
        ),
    )
    gross = target.sum(dim=-1)
    allowed_gross = gross_max.clamp(min=0.0, max=1.0)
    scale = torch.where(
        gross > allowed_gross,
        allowed_gross / gross.clamp_min(1e-18),
        torch.ones_like(gross),
    )
    target = target * scale.unsqueeze(-1)
    target[..., cash_index] = 1.0 - target.sum(dim=-1)
    return target


def _validate_monthly_schedule(
    base: Hold30DatasetSequence,
    monthly_rebalance: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(monthly_rebalance, torch.Tensor)
        or monthly_rebalance.dtype != torch.bool
        or tuple(monthly_rebalance.shape) != (base.n_positions,)
    ):
        raise Hold30DatasetError("monthly_rebalance must be a boolean [position] tensor")
    schedule = monthly_rebalance.detach().to(device="cpu")
    if not bool(schedule[0]):
        raise Hold30DatasetError("C1 requires an explicit initial reconstitution at position zero")
    fill_membership = base.fill_membership.detach().to(device="cpu")
    risky_membership = fill_membership.clone()
    risky_membership[..., base.cash_index] = False
    active_counts = risky_membership.sum(dim=-1)
    if bool((active_counts != HOLD30_C1_ACTIVE_COUNT).any()):
        raise Hold30DatasetError("C1 requires exactly 300 PIT active risky members at every position")
    changes = (fill_membership[1:] != fill_membership[:-1]).any(dim=-1)
    unmarked = changes & ~schedule[1:].view(-1, 1)
    if bool(unmarked.any()):
        raise Hold30DatasetError(
            "fill membership changed outside the frozen monthly reconstitution schedule"
        )
    return schedule


def _rebuild_c1(
    base: Hold30DatasetSequence,
    transformed_returns: torch.Tensor,
    monthly_rebalance: torch.Tensor,
) -> Hold30C1RebuildTrace:
    positions, batch, assets = base.decision_state.shape[:3]
    rows = positions - 1
    weights = transformed_returns.new_zeros((positions, batch, assets))
    holding_returns = transformed_returns.new_zeros((rows, batch))
    net_returns = transformed_returns.new_zeros((rows, batch))
    membership_turnover = transformed_returns.new_zeros((rows, batch))
    availability_turnover = transformed_returns.new_zeros((rows, batch))
    risk_turnover = transformed_returns.new_zeros((rows, batch))
    scheduled_turnover = transformed_returns.new_zeros((rows, batch))
    costs = transformed_returns.new_zeros((rows, batch))
    c1_allowed = base._c1_allowed_mask()

    weights[0] = _monthly_equal_weight_target(
        base.fill_membership[0],
        c1_allowed[0],
        base.risk_asset_caps[0],
        base.risk_gross_max[0],
        cash_index=base.cash_index,
    )
    for row in range(rows):
        current = weights[row]
        holding = (current * transformed_returns[row]).sum(dim=-1)
        growth = 1.0 + holding
        if bool((growth <= 0).any()):
            raise Hold30DatasetError("C1 gross wealth reached zero during drift")
        pretrade = current * (1.0 + transformed_returns[row]) / growth.unsqueeze(-1)
        fill = row + 1

        membership_target = _repair_to_mask(
            pretrade,
            base.fill_membership[fill],
            cash_index=base.cash_index,
        )
        membership_turnover[row] = _one_way(pretrade, membership_target)
        availability_target = _repair_to_mask(
            membership_target,
            base.fill_tradability[fill],
            cash_index=base.cash_index,
        )
        availability_turnover[row] = _one_way(
            membership_target, availability_target
        )
        risk_target = _repair_to_risk(
            availability_target,
            base.risk_asset_caps[fill],
            base.risk_gross_max[fill],
            cash_index=base.cash_index,
        )
        risk_turnover[row] = _one_way(availability_target, risk_target)
        final = risk_target
        if bool(monthly_rebalance[fill]):
            final = _monthly_equal_weight_target(
                base.fill_membership[fill],
                c1_allowed[fill],
                base.risk_asset_caps[fill],
                base.risk_gross_max[fill],
                cash_index=base.cash_index,
            )
            scheduled_turnover[row] = _one_way(risk_target, final)

        total_turnover = (
            membership_turnover[row]
            + availability_turnover[row]
            + risk_turnover[row]
            + scheduled_turnover[row]
        )
        cost = base.cost_rate[row] * total_turnover
        net = holding - cost
        if bool((net <= -1.0).any()):
            raise Hold30DatasetError("C1 net return reached -100% after execution cost")
        holding_returns[row] = holding
        net_returns[row] = net
        costs[row] = cost
        weights[fill] = final

    total_turnover = (
        membership_turnover
        + availability_turnover
        + risk_turnover
        + scheduled_turnover
    )
    return Hold30C1RebuildTrace(
        weights=weights,
        holding_returns=holding_returns,
        net_returns=net_returns,
        membership_turnover=membership_turnover,
        availability_turnover=availability_turnover,
        risk_turnover=risk_turnover,
        scheduled_turnover=scheduled_turnover,
        total_turnover=total_turnover,
        costs=costs,
    )


def _rebuild_c5(
    base: Hold30DatasetSequence,
    transformed_returns: torch.Tensor,
    c1_net_returns: torch.Tensor,
    domains: tuple[Hold30NullDomain, ...],
) -> Hold30C5Labels:
    rows, batch, assets = transformed_returns.shape
    labels = transformed_returns.new_zeros((rows, batch, assets))
    valid = torch.zeros((rows, batch, assets), dtype=torch.bool, device=transformed_returns.device)
    censored = torch.zeros_like(valid)
    risky = torch.ones((batch, assets), dtype=torch.bool, device=transformed_returns.device)
    risky[..., base.cash_index] = False

    domain_stop = torch.empty(rows, dtype=torch.int64)
    for domain in domains:
        domain_stop[domain.start : domain.stop] = domain.stop

    for origin in range(rows):
        eligible = base.a_trade[origin] & risky
        if not bool(eligible.any()):
            continue
        support_stop = origin + HOLD30_C5_HORIZON + 1
        if support_stop > int(domain_stop[origin]):
            censored[origin] = eligible
            continue

        alive = eligible.clone()
        stock_log_return = transformed_returns.new_zeros((batch, assets))
        for return_row in range(origin + 1, origin + HOLD30_C5_HORIZON + 1):
            cash_return = transformed_returns[
                return_row, :, base.cash_index
            ].unsqueeze(-1)
            realized = torch.where(
                alive,
                transformed_returns[return_row],
                cash_return,
            )
            stock_log_return = stock_log_return + torch.log1p(realized)
            next_fill = return_row + 1
            alive = (
                alive
                & base.fill_membership[next_fill]
                & base.fill_tradability[next_fill]
            )

        benchmark_log_return = torch.log1p(
            c1_net_returns[origin + 1 : origin + HOLD30_C5_HORIZON + 1]
        ).sum(dim=0)
        labels[origin] = torch.where(
            eligible,
            stock_log_return - benchmark_log_return.unsqueeze(-1),
            torch.zeros_like(stock_log_return),
        )
        valid[origin] = eligible

    if bool((valid & censored).any()):
        raise AssertionError("C5 valid and censored masks overlap")
    return Hold30C5Labels(values=labels, valid=valid, censored=censored)


def rebuild_hold30_null_outcomes(
    base: Hold30DatasetSequence,
    view: Hold30NullView,
    *,
    monthly_rebalance: torch.Tensor,
) -> Hold30NullRebuildResult:
    """Verify one null view and rebuild every outcome-dependent dataset field."""

    if base.asset_returns.dtype != torch.float64:
        raise Hold30DatasetError("the Hold-30 outcome-null economic rebuild requires float64")
    domains = _verify_null_view(base, view)
    schedule = _validate_monthly_schedule(base, monthly_rebalance)
    schedule_device = schedule.to(device=base.asset_returns.device)
    c1 = _rebuild_c1(base, view.asset_returns, schedule_device)
    c5 = _rebuild_c5(base, view.asset_returns, c1.net_returns, domains)

    source_benchmark_sha256 = _trace_digest(
        {
            "weights": base.c1_benchmark_weights,
            "net_returns": base.c1_benchmark_net_returns,
        }
    )
    monthly_rebalance_sha256 = _tensor_digest(schedule)
    derived_snapshot_sha256 = _canonical_digest(
        {
            "builder_version": HOLD30_NULL_REBUILDER_VERSION,
            "source_axis_id": base.axis_id,
            "source_provenance_receipt_id": base.provenance.receipt_id,
            "null_transform_id": view.receipt.transform_id,
            "null_mapping_sha256": view.receipt.mapping_sha256,
            "null_output_outcomes_sha256": view.receipt.output_outcomes_sha256,
            "monthly_rebalance_sha256": monthly_rebalance_sha256,
        }
    )
    transformed_provenance = replace(
        base.provenance,
        data_snapshot_sha256=derived_snapshot_sha256,
        c1_benchmark_trace_sha256=c1.trace_sha256,
    )
    transformed_sequence = replace(
        base,
        asset_returns=view.asset_returns,
        c1_benchmark_weights=c1.weights,
        c1_benchmark_net_returns=c1.net_returns,
        provenance=transformed_provenance,
    )
    receipt_payload = {
        "builder_version": HOLD30_NULL_REBUILDER_VERSION,
        "source_axis_id": base.axis_id,
        "source_provenance_receipt_id": base.provenance.receipt_id,
        "source_benchmark_sha256": source_benchmark_sha256,
        "null_transform_id": view.receipt.transform_id,
        "null_mapping_sha256": view.receipt.mapping_sha256,
        "null_output_outcomes_sha256": view.receipt.output_outcomes_sha256,
        "monthly_rebalance_sha256": monthly_rebalance_sha256,
        "c1_trace_sha256": c1.trace_sha256,
        "c5_labels_sha256": c5.labels_sha256,
        "transformed_provenance_receipt_id": transformed_provenance.receipt_id,
        "transformed_axis_id": transformed_sequence.axis_id,
    }
    receipt = Hold30NullRebuildReceipt(
        **receipt_payload,
        receipt_id=_canonical_digest(receipt_payload),
    )
    return Hold30NullRebuildResult(
        sequence=transformed_sequence,
        c1=c1,
        c5=c5,
        receipt=receipt,
    )


__all__ = [
    "HOLD30_C1_ACTIVE_COUNT",
    "HOLD30_C1_MAX_NAME_WEIGHT",
    "HOLD30_C5_HORIZON",
    "HOLD30_NULL_REBUILDER_VERSION",
    "Hold30C1RebuildTrace",
    "Hold30C5Labels",
    "Hold30NullRebuildReceipt",
    "Hold30NullRebuildResult",
    "rebuild_hold30_null_outcomes",
]
