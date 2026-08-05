"""Training contracts for the pre-lockbox Hold-30 mechanism experiment.

This module owns the *optimizer geometry* of the Hold-30 direct objective.  It
deliberately does not own portfolio accounting: canonical execution and the
age/cohort ledger live in :mod:`rl_quant.envs.hold30`.  A runtime adapter must
therefore produce a no-grad canonical trace with the environment and replay
each origin through that same environment implementation.

The important distinction is between a chronological economic trace (each
calendar row is booked once) and the local origin-indexed vector-Jacobian
surrogate (daily utility rows can be reused for credit assignment).  The
surrogate below must not be described as the full derivative of chronological
wealth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.distributed as dist

from rl_quant.protocol.hold30 import resolve_hold30_setting


HOLD30_WARMUP_DECISIONS = 63
HOLD30_CREDIT_RETURNS = 30
HOLD30_SUPPORT_DECISIONS = 30
HOLD30_MAX_ORIGIN_BATCH = 32
LEGACY_GATE_TARGET_RATE = 12.0 / 252.0


@dataclass(frozen=True)
class Hold30ReplayGeometry:
    """Frozen chronological roles for one training block.

    ``n_positions`` includes the final state-only terminal observation.  The
    thirty positions immediately before it are ordinary support decisions.
    An anchor at ``t`` owns the fill/cost utility row ``t`` plus exactly thirty
    post-fill utility rows ``t+1 .. t+30`` and terminates at state ``t+31``.
    """

    warmup_decisions: int = HOLD30_WARMUP_DECISIONS
    credit_returns: int = HOLD30_CREDIT_RETURNS
    support_decisions: int = HOLD30_SUPPORT_DECISIONS
    max_origin_batch: int = HOLD30_MAX_ORIGIN_BATCH

    def __post_init__(self) -> None:
        values = (
            self.warmup_decisions,
            self.credit_returns,
            self.support_decisions,
            self.max_origin_batch,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Hold-30 geometry values must be positive integers")
        if self.credit_returns != self.support_decisions:
            raise ValueError("credit_returns and support_decisions must match in v1")

    @property
    def minimum_positions(self) -> int:
        # At least one anchor, its thirty support decisions, and terminal state.
        return self.warmup_decisions + self.support_decisions + 2

    def roles(self, n_positions: int) -> "Hold30CreditRoles":
        if isinstance(n_positions, bool) or not isinstance(n_positions, int):
            raise TypeError("n_positions must be an integer")
        if n_positions < self.minimum_positions:
            raise ValueError(
                f"Hold-30 block needs at least {self.minimum_positions} positions, got {n_positions}"
            )
        terminal = n_positions - 1
        support_start = terminal - self.support_decisions
        anchors = torch.arange(self.warmup_decisions, support_start, dtype=torch.long)
        support = torch.arange(support_start, terminal, dtype=torch.long)
        if anchors.numel() == 0:
            raise ValueError("Hold-30 block contains no loss-bearing anchors")
        offsets = torch.arange(self.credit_returns + 1, dtype=torch.long)
        utility_rows = anchors.unsqueeze(1) + offsets.unsqueeze(0)
        terminal_rows = anchors + self.credit_returns + 1
        if int(utility_rows.max()) >= terminal or int(terminal_rows.max()) > terminal:
            raise AssertionError("origin credit geometry exceeds its terminal observation")
        utility_mask = torch.zeros((anchors.numel(), n_positions), dtype=torch.bool)
        utility_mask.scatter_(1, utility_rows, True)
        return Hold30CreditRoles(
            n_positions=n_positions,
            warmup=torch.arange(0, self.warmup_decisions, dtype=torch.long),
            anchors=anchors,
            support=support,
            terminal_observation=terminal,
            utility_rows=utility_rows,
            replay_terminal_rows=terminal_rows,
            utility_mask=utility_mask,
        )

    def origin_batches(self, anchors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        anchors = torch.as_tensor(anchors, dtype=torch.long, device="cpu")
        if anchors.ndim != 1 or anchors.numel() == 0:
            raise ValueError("anchors must be a non-empty one-dimensional tensor")
        return tuple(anchors[i:i + self.max_origin_batch] for i in range(0, anchors.numel(), self.max_origin_batch))


@dataclass(frozen=True)
class Hold30CreditRoles:
    n_positions: int
    warmup: torch.Tensor
    anchors: torch.Tensor
    support: torch.Tensor
    terminal_observation: int
    utility_rows: torch.Tensor
    replay_terminal_rows: torch.Tensor
    utility_mask: torch.Tensor

    def validate(self) -> None:
        n_anchor = int(self.anchors.numel())
        if tuple(self.utility_rows.shape) != (n_anchor, HOLD30_CREDIT_RETURNS + 1):
            raise ValueError("each Hold-30 anchor must own 31 utility rows")
        if tuple(self.utility_mask.shape) != (n_anchor, self.n_positions):
            raise ValueError("utility mask shape does not match the chronology")
        if not bool((self.utility_mask.sum(1) == HOLD30_CREDIT_RETURNS + 1).all()):
            raise ValueError("every Hold-30 anchor must have one fill row and thirty return rows")
        if not torch.equal(self.utility_rows[:, 0], self.anchors):
            raise ValueError("the first utility row must be the originating action row")
        if not torch.equal(self.utility_rows[:, -1] + 1, self.replay_terminal_rows):
            raise ValueError("replay terminal must immediately follow the last credited return row")


@dataclass(frozen=True)
class Hold30CanonicalRow:
    """No-grad scalar evidence from one deployed canonical decision."""

    utility: float
    discretionary_turnover: float = 0.0
    early_sale_mass: float = 0.0
    gate: float = 0.0
    gate_entropy: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")
        if self.discretionary_turnover < 0 or self.early_sale_mass < 0:
            raise ValueError("turnover and early-sale mass cannot be negative")


@dataclass(frozen=True)
class Hold30SequenceCoefficients:
    anchor_count: int
    mean_turnover: float
    turnover_coefficient: float
    mean_gate: float
    gate_coefficient: float


@dataclass(frozen=True)
class Hold30CalendarDiagnostic:
    """Undifferentiated calendar-row objective reported from canonical rows."""

    anchor_count: int
    mean_utility: float
    mean_discretionary_turnover: float
    turnover_penalty: float
    mean_early_sale_mass: float
    early_exit_penalty: float
    value: float


@dataclass(frozen=True)
class Hold30LossContract:
    """Variant-specific terms in the frozen local surrogate."""

    mechanism: str
    lambda_turn: float = 1.0
    lambda_early: float = 0.002
    gate_entropy_coef: float = 1e-5
    gate_budget_coef: float = 1e-3
    target_turnover: float = 1.0 / 30.0

    def __post_init__(self) -> None:
        if self.mechanism not in {"H0", "H1", "H2", "H3"}:
            raise ValueError("mechanism must be H0, H1, H2, or H3")
        for name in (
            "lambda_turn",
            "lambda_early",
            "gate_entropy_coef",
            "gate_budget_coef",
            "target_turnover",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative scalar")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def for_setting(cls, setting_id: str) -> "Hold30LossContract":
        setting = resolve_hold30_setting(setting_id)
        return cls(
            setting.mechanism,
            lambda_turn=1.0 if setting.use_turnover_penalty else 0.0,
            lambda_early=0.002 if setting.use_early_exit_penalty else 0.0,
            gate_entropy_coef=1e-5 if setting.mechanism == "H0" else 0.0,
        )


@dataclass(frozen=True)
class Hold30OriginReplay:
    """Differentiable quantities from one restored origin boundary."""

    origin: int
    utility_rows: torch.Tensor
    discretionary_turnover: torch.Tensor
    early_sale_mass: torch.Tensor
    gate: torch.Tensor
    gate_entropy: torch.Tensor

    def validate(self, *, credit_returns: int = HOLD30_CREDIT_RETURNS) -> None:
        if isinstance(self.origin, bool) or not isinstance(self.origin, int) or self.origin < 0:
            raise ValueError("origin must be a non-negative integer")
        if self.utility_rows.ndim != 1 or self.utility_rows.numel() != credit_returns + 1:
            raise ValueError("origin replay must contain its fill row and exactly thirty post-fill return rows")
        for name in ("discretionary_turnover", "early_sale_mass", "gate", "gate_entropy"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(f"{name} must be a scalar tensor")
        tensors = (
            self.utility_rows,
            self.discretionary_turnover,
            self.early_sale_mass,
            self.gate,
            self.gate_entropy,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("origin replay contains non-finite values")


class Hold30ReplayAdapter(Protocol):
    """Runtime boundary between canonical environment accounting and training."""

    def canonical_pass(
        self,
        policy: torch.nn.Module,
        sequence: Any,
        roles: Hold30CreditRoles,
    ) -> tuple[Any, Sequence[Hold30CanonicalRow]]:
        """Return replay state and one no-grad row per decision position."""

    def replay_origins(
        self,
        policy: torch.nn.Module,
        sequence: Any,
        canonical_state: Any,
        origins: torch.Tensor,
        roles: Hold30CreditRoles,
    ) -> Sequence[Hold30OriginReplay]:
        """Replay independent origins while attaching only each origin action."""


def benchmark_relative_log_utility(
    policy_net_return: torch.Tensor,
    benchmark_net_return: torch.Tensor,
) -> torch.Tensor:
    """One-session relative log utility; economic equity remains policy-only."""

    policy_net_return, benchmark_net_return = torch.broadcast_tensors(policy_net_return, benchmark_net_return)
    if bool((policy_net_return <= -1).any()) or bool((benchmark_net_return <= -1).any()):
        raise ValueError("simple returns must be greater than -1 before log1p")
    return torch.log1p(policy_net_return) - torch.log1p(benchmark_net_return)


def sequence_coefficients(
    rows: Sequence[Hold30CanonicalRow],
    anchors: torch.Tensor,
    contract: Hold30LossContract,
) -> Hold30SequenceCoefficients:
    anchors = torch.as_tensor(anchors, dtype=torch.long, device="cpu")
    if anchors.ndim != 1 or anchors.numel() == 0:
        raise ValueError("anchors must be non-empty")
    if int(anchors.min()) < 0 or int(anchors.max()) >= len(rows):
        raise ValueError("anchor index falls outside canonical rows")
    selected = [rows[int(index)] for index in anchors]
    mean_turnover = sum(row.discretionary_turnover for row in selected) / len(selected)
    mean_gate = sum(row.gate for row in selected) / len(selected)
    turnover_coefficient = (
        2.0 * contract.lambda_turn * max(mean_turnover - contract.target_turnover, 0.0)
        if contract.mechanism in {"H1", "H2"}
        else 0.0
    )
    gate_coefficient = (
        contract.gate_budget_coef if contract.mechanism == "H0" and mean_gate > LEGACY_GATE_TARGET_RATE else 0.0
    )
    return Hold30SequenceCoefficients(
        anchor_count=len(selected),
        mean_turnover=mean_turnover,
        turnover_coefficient=turnover_coefficient,
        mean_gate=mean_gate,
        gate_coefficient=gate_coefficient,
    )


def calendar_diagnostic(
    rows: Sequence[Hold30CanonicalRow],
    anchors: torch.Tensor,
    contract: Hold30LossContract,
) -> Hold30CalendarDiagnostic:
    """Compute the frozen ``J_calendar`` telemetry from canonical evidence.

    This is deliberately separate from :func:`origin_surrogate`: every
    calendar row contributes once here, while origin replays reuse utility
    rows solely to form the local finite-credit vector-Jacobian surrogate.
    H0 and H3 have no duration penalties under the frozen protocol, so their
    reported value is their mean anchor utility.
    """

    anchors = torch.as_tensor(anchors, dtype=torch.long, device="cpu")
    if anchors.ndim != 1 or anchors.numel() == 0:
        raise ValueError("anchors must be non-empty")
    if int(anchors.min()) < 0 or int(anchors.max()) >= len(rows):
        raise ValueError("anchor index falls outside canonical rows")
    selected = [rows[int(index)] for index in anchors]
    count = len(selected)
    mean_utility = sum(row.utility for row in selected) / count
    mean_turnover = sum(row.discretionary_turnover for row in selected) / count
    mean_early = sum(row.early_sale_mass for row in selected) / count
    turnover_penalty = (
        contract.lambda_turn
        * max(mean_turnover - contract.target_turnover, 0.0) ** 2
        if contract.mechanism in {"H1", "H2"}
        else 0.0
    )
    early_penalty = (
        contract.lambda_early * mean_early if contract.mechanism == "H2" else 0.0
    )
    return Hold30CalendarDiagnostic(
        anchor_count=count,
        mean_utility=mean_utility,
        mean_discretionary_turnover=mean_turnover,
        turnover_penalty=turnover_penalty,
        mean_early_sale_mass=mean_early,
        early_exit_penalty=early_penalty,
        value=mean_utility - turnover_penalty - early_penalty,
    )


def origin_surrogate(
    replay: Hold30OriginReplay,
    coefficients: Hold30SequenceCoefficients,
    contract: Hold30LossContract,
) -> torch.Tensor:
    """Maximization value for one origin under the frozen local contract."""

    replay.validate()
    value = replay.utility_rows.sum()
    if contract.mechanism in {"H1", "H2"}:
        value = value - coefficients.turnover_coefficient * replay.discretionary_turnover.reshape(())
    if contract.mechanism == "H2" and contract.lambda_early:
        value = value - contract.lambda_early * replay.early_sale_mass.reshape(())
    if contract.mechanism == "H0":
        value = value - coefficients.gate_coefficient * replay.gate.reshape(())
        value = value + contract.gate_entropy_coef * replay.gate_entropy.reshape(())
    return value


def detach_tree(value: Any) -> Any:
    """Detach tensor leaves without changing the economic state's structure."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return type(value)((key, detach_tree(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value


def train_hold30_update(
    policy: torch.nn.Module,
    sequence: Any,
    adapter: Hold30ReplayAdapter,
    optimizer: torch.optim.Optimizer,
    *,
    n_positions: int,
    contract: Hold30LossContract,
    geometry: Hold30ReplayGeometry | None = None,
    grad_clip: float = 0.0,
    before_step: Callable[[torch.nn.Module], None] | None = None,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> dict[str, Any]:
    """Execute one canonical sweep and exactly one optimizer update.

    The adapter is responsible for numeric replay/canonical parity and for
    using the shared environment accounting primitive.  This function owns the
    fixed denominator, variant coefficients, independent-origin batching, and
    one-step optimizer semantics.
    """

    if distributed_world_size not in {1, 2}:
        raise ValueError("Hold-30 supports only world_size=1 or the qualified world_size=2")
    if distributed_rank not in range(distributed_world_size):
        raise ValueError("distributed_rank is outside distributed_world_size")
    distributed = distributed_world_size == 2
    if distributed and (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() != 2
        or dist.get_rank() != distributed_rank
    ):
        raise RuntimeError("world_size=2 requires a matching initialized process group")
    if distributed and before_step is not None:
        raise ValueError(
            "distributed Hold-30 owns gradient SUM reduction; before_step is unsupported"
        )
    geometry = geometry or Hold30ReplayGeometry()
    roles = geometry.roles(n_positions)
    roles.validate()
    policy.train()
    with torch.no_grad():
        canonical_state, rows = adapter.canonical_pass(policy, sequence, roles)
    if len(rows) < n_positions - 1:
        raise ValueError("canonical pass must return every decision row before the terminal observation")
    canonical_state = detach_tree(canonical_state)
    coefficients = sequence_coefficients(rows, roles.anchors, contract)
    calendar = calendar_diagnostic(rows, roles.anchors, contract)
    if distributed:
        canonical_evidence = {
            "rows": [
                (
                    row.utility,
                    row.discretionary_turnover,
                    row.early_sale_mass,
                    row.gate,
                    row.gate_entropy,
                )
                for row in rows
            ],
            "coefficients": coefficients,
        }
        gathered_evidence: list[Any] = [None, None]
        dist.all_gather_object(gathered_evidence, canonical_evidence)
        if gathered_evidence[0] != gathered_evidence[1]:
            raise RuntimeError("distributed canonical chronology/coefficients differ across ranks")
    denominator = float(coefficients.anchor_count)
    local_anchors = roles.anchors[distributed_rank::distributed_world_size]
    if local_anchors.numel() == 0:
        raise ValueError("every distributed rank must own at least one loss-bearing anchor")
    optimizer.zero_grad(set_to_none=True)
    replayed: list[int] = []
    utility_row_count = 0
    objective_total = 0.0
    for origin_batch in geometry.origin_batches(local_anchors):
        replays = tuple(adapter.replay_origins(policy, sequence, canonical_state, origin_batch, roles))
        expected = [int(value) for value in origin_batch]
        if [replay.origin for replay in replays] != expected:
            raise ValueError("origin adapter changed order, omitted, or duplicated an anchor")
        if not replays:
            raise ValueError("origin adapter returned an empty replay batch")
        batch_value: torch.Tensor | None = None
        for replay in replays:
            value = origin_surrogate(replay, coefficients, contract) / denominator
            batch_value = value if batch_value is None else batch_value + value
            replayed.append(replay.origin)
            utility_row_count += int(replay.utility_rows.numel())
            objective_total += float(value.detach())
        assert batch_value is not None
        (-batch_value).backward()
    if replayed != [int(value) for value in local_anchors]:
        raise AssertionError("not every rank-local Hold-30 anchor was replayed exactly once")
    if before_step is not None:
        before_step(policy)
    if distributed:
        for parameter in policy.parameters():
            if not parameter.requires_grad:
                continue
            used = torch.tensor(
                0 if parameter.grad is None else 1,
                dtype=torch.int64,
                device=parameter.device,
            )
            dist.all_reduce(used, op=dist.ReduceOp.SUM)
            if int(used.item()) == 0:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()
    if distributed:
        totals = torch.tensor(
            [float(utility_row_count), float(objective_total)],
            dtype=torch.float64,
            device=next(policy.parameters()).device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        utility_row_count = int(totals[0].item())
        objective_total = float(totals[1].item())
    return {
        "anchor_count": coefficients.anchor_count,
        "origin_batch_count": len(geometry.origin_batches(roles.anchors)),
        "utility_rows_replayed": utility_row_count,
        "repeated_calendar_rows": utility_row_count - coefficients.anchor_count,
        "objective": objective_total,
        "mean_turnover": coefficients.mean_turnover,
        "turnover_coefficient": coefficients.turnover_coefficient,
        "mean_gate": coefficients.mean_gate,
        "gate_coefficient": coefficients.gate_coefficient,
        "calendar_objective": calendar.value,
        "calendar_mean_utility": calendar.mean_utility,
        "calendar_turnover_penalty": calendar.turnover_penalty,
        "calendar_mean_early_sale_mass": calendar.mean_early_sale_mass,
        "calendar_early_exit_penalty": calendar.early_exit_penalty,
        "optimizer_steps": 1,
        "distributed_world_size": distributed_world_size,
        "origin_shard_policy": "strided-rank-mod-world-size",
        "utility_mask": roles.utility_mask,
    }


__all__ = [
    "HOLD30_CREDIT_RETURNS",
    "HOLD30_MAX_ORIGIN_BATCH",
    "HOLD30_SUPPORT_DECISIONS",
    "HOLD30_WARMUP_DECISIONS",
    "Hold30CanonicalRow",
    "Hold30CalendarDiagnostic",
    "Hold30CreditRoles",
    "Hold30LossContract",
    "Hold30OriginReplay",
    "Hold30ReplayAdapter",
    "Hold30ReplayGeometry",
    "Hold30SequenceCoefficients",
    "benchmark_relative_log_utility",
    "calendar_diagnostic",
    "detach_tree",
    "origin_surrogate",
    "sequence_coefficients",
    "train_hold30_update",
]
