"""Causal validation evidence for the nonpromotable TOP2000 M03R-v7 panel.

The only scored observations are the fixed 63 decisions owned by one of the
six development folds.  A policy may consume causal history before that
window and unscored holding/label support after it, but neither region enters
the reported return vector.  Seed models are evaluated separately for model
diagnostics.  Investment evidence is produced by a fresh chronological run of
one output-space five-seed ensemble; seed return paths are never averaged or
treated as independent histories.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import nn

from rl_quant.envs.hold30 import TURNOVER_CAUSES, TurnoverCause
from rl_quant.models.daily_policy import Hold30Intent
from rl_quant.models.hold30_exit_action_v6 import (
    M03R_V6_HOLD_ACTION_INDEX,
    M03RV6ExitAction,
    straight_through_m03r_v6_exit_action,
)
from rl_quant.models.hold30_hazard import bound_hold30_hazard_residual
from rl_quant.protocol.hold30_alpha_m03r_v7_top2000_dev import (
    M03R_TOP2000_DEV_DATA_ROLE,
    M03R_TOP2000_DEV_DESIGN_ID,
    M03R_TOP2000_DEV_PROTOCOL_GENERATION,
    M03R_TOP2000_DEV_PROTOCOL_SHA256,
)
from rl_quant.training.hold30 import Hold30ReplayGeometry
from rl_quant.training.hold30_runtime import (
    Hold30CanonicalTrace,
    Hold30ChronologicalRuntime,
    Hold30DecisionStateProvider,
    Hold30Sequence,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_SEEDS,
    TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS,
    Top2000M03RV7ActionBuilder,
    Top2000M03RV7DevelopmentFold,
    Top2000M03RV7DevelopmentPolicy,
    bind_top2000_m03r_v7_runtime_sequence,
)

TOP2000_M03R_V7_SEED_VALIDATION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-seed-validation-v1"
)
TOP2000_M03R_V7_FOLD_ENSEMBLE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-fold-ensemble-v1"
)
TOP2000_M03R_V7_VALIDATION_TRACE_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v7-validation-trace-v1"
)
TOP2000_M03R_V7_ENSEMBLE_RULE = (
    "five-seed-output-space-mean-entry-alpha-aux-median-risk-hazard-v1"
)


class Top2000M03RV7ValidationError(RuntimeError):
    """Validation chronology, member evidence, or receipt binding is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Top2000M03RV7ValidationError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash one exact detached tensor including dtype and shape."""

    if not isinstance(value, torch.Tensor):
        raise Top2000M03RV7ValidationError("tensor hash requires a tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Top2000M03RV7ValidationTraceEvidence:
    """Exact 63-decision arrays plus reproducible scalar diagnostics."""

    policy_net_returns: torch.Tensor
    benchmark_net_returns: torch.Tensor
    active_log_returns: torch.Tensor
    total_one_way_turnover: torch.Tensor
    discretionary_one_way_turnover: torch.Tensor
    forced_one_way_turnover: torch.Tensor
    discretionary_sold_notional_by_age: torch.Tensor
    terminal_risky_notional_by_age: torch.Tensor
    score_transition_start: int
    score_transition_stop_exclusive: int

    def __post_init__(self) -> None:
        rows = TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
        for name in (
            "policy_net_returns",
            "benchmark_net_returns",
            "active_log_returns",
            "total_one_way_turnover",
            "discretionary_one_way_turnover",
            "forced_one_way_turnover",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (rows,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise Top2000M03RV7ValidationError(
                    f"{name} must be a finite floating [{rows}] tensor"
                )
        for name in (
            "discretionary_sold_notional_by_age",
            "terminal_risky_notional_by_age",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (61,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())
            ):
                raise Top2000M03RV7ValidationError(
                    f"{name} must be a finite nonnegative [61] tensor"
                )
        if (
            self.score_transition_stop_exclusive - self.score_transition_start
            != rows
        ):
            raise Top2000M03RV7ValidationError(
                "validation trace must score exactly 63 transitions"
            )
        expected_active = torch.log1p(self.policy_net_returns) - torch.log1p(
            self.benchmark_net_returns
        )
        if not bool(
            torch.allclose(
                self.active_log_returns,
                expected_active,
                atol=1e-7,
                rtol=1e-7,
            )
        ):
            raise Top2000M03RV7ValidationError(
                "active log returns do not reconcile to policy and benchmark"
            )
        if bool(
            (
                self.total_one_way_turnover
                + 1.0e-7
                < self.discretionary_one_way_turnover
                + self.forced_one_way_turnover
            ).any()
        ):
            raise Top2000M03RV7ValidationError(
                "turnover components exceed total one-way turnover"
            )

    def array_sha256s(self) -> dict[str, str]:
        return {
            name: tensor_sha256(getattr(self, name))
            for name in (
                "policy_net_returns",
                "benchmark_net_returns",
                "active_log_returns",
                "total_one_way_turnover",
                "discretionary_one_way_turnover",
                "forced_one_way_turnover",
                "discretionary_sold_notional_by_age",
                "terminal_risky_notional_by_age",
            )
        }

    @property
    def trace_sha256(self) -> str:
        return _sha256(
            {
                "schema": TOP2000_M03R_V7_VALIDATION_TRACE_SCHEMA,
                "score_transition_start": self.score_transition_start,
                "score_transition_stop_exclusive": (
                    self.score_transition_stop_exclusive
                ),
                "arrays": self.array_sha256s(),
            }
        )

    def metrics(self) -> dict[str, float | int | None]:
        policy = self.policy_net_returns.to(torch.float64)
        benchmark = self.benchmark_net_returns.to(torch.float64)
        active = self.active_log_returns.to(torch.float64)
        policy_std = policy.std(unbiased=False)
        active_std = active.std(unbiased=False)
        sold = self.discretionary_sold_notional_by_age.to(torch.float64)
        terminal = self.terminal_risky_notional_by_age.to(torch.float64)
        ages = torch.arange(61, dtype=torch.float64)

        sold_total = float(sold.sum())
        sale_mean_age = (
            None if sold_total <= 0 else float((sold * ages).sum() / sold.sum())
        )
        sale_median_age: int | None = None
        if sold_total > 0:
            sale_median_age = int(
                torch.searchsorted(sold.cumsum(0), sold.sum() * 0.5).item()
            )
        terminal_total = float(terminal.sum())
        terminal_mean_age = (
            None
            if terminal_total <= 0
            else float((terminal * ages).sum() / terminal.sum())
        )
        return {
            "decision_count": int(policy.numel()),
            "policy_cumulative_net_return": float(torch.expm1(torch.log1p(policy).sum())),
            "benchmark_cumulative_net_return": float(
                torch.expm1(torch.log1p(benchmark).sum())
            ),
            "cumulative_active_log_return": float(active.sum()),
            "annualized_policy_mean_return": float(policy.mean() * 252.0),
            "annualized_policy_volatility": float(policy_std * math.sqrt(252.0)),
            "annualized_policy_sharpe_zero_cash": (
                None
                if float(policy_std) <= 0.0
                else float(policy.mean() / policy_std * math.sqrt(252.0))
            ),
            "annualized_active_log_return": float(active.mean() * 252.0),
            "annualized_tracking_error": float(active_std * math.sqrt(252.0)),
            "annualized_information_ratio": (
                None
                if float(active_std) <= 0.0
                else float(active.mean() / active_std * math.sqrt(252.0))
            ),
            "mean_total_one_way_turnover": float(
                self.total_one_way_turnover.mean()
            ),
            "mean_discretionary_one_way_turnover": float(
                self.discretionary_one_way_turnover.mean()
            ),
            "mean_forced_one_way_turnover": float(
                self.forced_one_way_turnover.mean()
            ),
            "discretionary_sold_notional": sold_total,
            "discretionary_sold_notional_younger_than_30": float(sold[:30].sum()),
            "notional_weighted_discretionary_sale_age": sale_mean_age,
            "median_discretionary_sale_age": sale_median_age,
            "terminal_risky_notional": terminal_total,
            "terminal_notional_weighted_age": terminal_mean_age,
        }

    def artifact_payload(self) -> dict[str, Any]:
        return {
            "schema": TOP2000_M03R_V7_VALIDATION_TRACE_SCHEMA,
            "score_transition_start": self.score_transition_start,
            "score_transition_stop_exclusive": self.score_transition_stop_exclusive,
            "policy_net_returns": self.policy_net_returns.detach().cpu(),
            "benchmark_net_returns": self.benchmark_net_returns.detach().cpu(),
            "active_log_returns": self.active_log_returns.detach().cpu(),
            "total_one_way_turnover": self.total_one_way_turnover.detach().cpu(),
            "discretionary_one_way_turnover": (
                self.discretionary_one_way_turnover.detach().cpu()
            ),
            "forced_one_way_turnover": self.forced_one_way_turnover.detach().cpu(),
            "discretionary_sold_notional_by_age": (
                self.discretionary_sold_notional_by_age.detach().cpu()
            ),
            "terminal_risky_notional_by_age": (
                self.terminal_risky_notional_by_age.detach().cpu()
            ),
            "array_sha256": self.array_sha256s(),
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class Top2000M03RV7SeedValidationReceipt:
    setting_index: int
    setting_id: str
    fold_index: int
    seed: int
    fold_receipt_sha256: str
    sequence_receipt_sha256: str
    checkpoint_file_sha256: str
    model_state_sha256: str
    validation_trace_artifact_sha256: str
    validation_trace_sha256: str
    array_sha256: dict[str, str]
    metrics: dict[str, float | int | None]
    validation_global_decision_start: int
    validation_global_decision_stop_exclusive: int
    first_validation_date: str
    last_validation_date: str
    checkpoint_selection_rule: str = (
        "frozen-final-optimizer-update-no-validation-selection-v1"
    )
    evaluation_autograd_enabled: bool = False
    protocol_sha256: str = M03R_TOP2000_DEV_PROTOCOL_SHA256
    protocol_generation: str = M03R_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_TOP2000_DEV_DESIGN_ID
    data_role: str = M03R_TOP2000_DEV_DATA_ROLE
    development_only: bool = True
    future_selected_universe: bool = True
    outer_evaluation_authorized: bool = False
    promotion_eligible: bool = False
    schema: str = TOP2000_M03R_V7_SEED_VALIDATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "fold_receipt_sha256",
            "sequence_receipt_sha256",
            "checkpoint_file_sha256",
            "model_state_sha256",
            "validation_trace_artifact_sha256",
            "validation_trace_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            self.schema != TOP2000_M03R_V7_SEED_VALIDATION_SCHEMA
            or self.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256
            or self.protocol_generation != M03R_TOP2000_DEV_PROTOCOL_GENERATION
            or self.design_id != M03R_TOP2000_DEV_DESIGN_ID
            or self.data_role != M03R_TOP2000_DEV_DATA_ROLE
            or self.seed not in TOP2000_M03R_V7_DEV_SEEDS
            or self.checkpoint_selection_rule
            != "frozen-final-optimizer-update-no-validation-selection-v1"
            or self.evaluation_autograd_enabled
            or self.validation_global_decision_stop_exclusive
            - self.validation_global_decision_start
            != TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
            or not self.development_only
            or not self.future_selected_universe
            or self.outer_evaluation_authorized
            or self.promotion_eligible
        ):
            raise Top2000M03RV7ValidationError(
                "seed validation receipt identity or development-only gate drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class Top2000M03RV7FoldEnsembleReceipt:
    setting_index: int
    setting_id: str
    fold_index: int
    fold_receipt_sha256: str
    ordered_seeds: tuple[int, ...]
    seed_validation_receipt_sha256s: tuple[str, ...]
    member_checkpoint_file_sha256s: tuple[str, ...]
    member_model_state_sha256s: tuple[str, ...]
    sequence_receipt_sha256: str
    validation_trace_artifact_sha256: str
    validation_trace_sha256: str
    array_sha256: dict[str, str]
    metrics: dict[str, float | int | None]
    validation_global_decision_start: int
    validation_global_decision_stop_exclusive: int
    first_validation_date: str
    last_validation_date: str
    ensemble_rule: str = TOP2000_M03R_V7_ENSEMBLE_RULE
    protocol_sha256: str = M03R_TOP2000_DEV_PROTOCOL_SHA256
    protocol_generation: str = M03R_TOP2000_DEV_PROTOCOL_GENERATION
    design_id: str = M03R_TOP2000_DEV_DESIGN_ID
    data_role: str = M03R_TOP2000_DEV_DATA_ROLE
    development_only: bool = True
    future_selected_universe: bool = True
    outer_evaluation_authorized: bool = False
    promotion_eligible: bool = False
    seeds_are_independent_return_paths: bool = False
    chronological_return_path_count: int = 1
    evaluation_autograd_enabled: bool = False
    schema: str = TOP2000_M03R_V7_FOLD_ENSEMBLE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "fold_receipt_sha256",
            "sequence_receipt_sha256",
            "validation_trace_artifact_sha256",
            "validation_trace_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "seed_validation_receipt_sha256s",
            "member_checkpoint_file_sha256s",
            "member_model_state_sha256s",
        ):
            values = getattr(self, name)
            if len(values) != 5:
                raise Top2000M03RV7ValidationError(
                    f"{name} must bind exactly five members"
                )
            for index, value in enumerate(values):
                _require_sha256(f"{name}[{index}]", value)
        if (
            self.schema != TOP2000_M03R_V7_FOLD_ENSEMBLE_SCHEMA
            or self.ensemble_rule != TOP2000_M03R_V7_ENSEMBLE_RULE
            or self.protocol_sha256 != M03R_TOP2000_DEV_PROTOCOL_SHA256
            or self.protocol_generation != M03R_TOP2000_DEV_PROTOCOL_GENERATION
            or self.design_id != M03R_TOP2000_DEV_DESIGN_ID
            or self.data_role != M03R_TOP2000_DEV_DATA_ROLE
            or self.ordered_seeds != TOP2000_M03R_V7_DEV_SEEDS
            or self.validation_global_decision_stop_exclusive
            - self.validation_global_decision_start
            != TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
            or not self.development_only
            or not self.future_selected_universe
            or self.outer_evaluation_authorized
            or self.promotion_eligible
            or self.seeds_are_independent_return_paths
            or self.chronological_return_path_count != 1
            or self.evaluation_autograd_enabled
        ):
            raise Top2000M03RV7ValidationError(
                "fold ensemble receipt identity or inference semantics drifted"
            )

    @property
    def receipt_sha256(self) -> str:
        return _sha256(asdict(self))


def model_state_sha256(policy: nn.Module) -> str:
    """Content-bind model tensors independent of torch serialization metadata."""

    rows = []
    for name, value in sorted(policy.state_dict().items()):
        rows.append((name, tensor_sha256(value)))
    return _sha256(rows)


def _median_optional(
    intents: Sequence[Hold30Intent],
    name: str,
) -> torch.Tensor | None:
    values = [getattr(intent, name) for intent in intents]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise Top2000M03RV7ValidationError(
            f"ensemble members disagree on optional field {name}"
        )
    return torch.stack(cast(list[torch.Tensor], values)).median(dim=0).values


def _mean_optional(
    intents: Sequence[Hold30Intent],
    name: str,
) -> torch.Tensor | None:
    values = [getattr(intent, name) for intent in intents]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise Top2000M03RV7ValidationError(
            f"ensemble members disagree on optional field {name}"
        )
    return torch.stack(cast(list[torch.Tensor], values)).mean(dim=0)


def _aggregate_exit_action(
    intents: Sequence[Hold30Intent],
    available: torch.Tensor,
) -> M03RV6ExitAction | None:
    actions = [intent.exit_action_v6 for intent in intents]
    if all(action is None for action in actions):
        return None
    if any(action is None for action in actions):
        raise Top2000M03RV7ValidationError(
            "ensemble members disagree on the v6 exit-action surface"
        )
    populated = cast(list[M03RV6ExitAction], actions)
    enabled = populated[0].exact_hold_atom_enabled
    if any(action.exact_hold_atom_enabled != enabled for action in populated):
        raise Top2000M03RV7ValidationError(
            "ensemble members disagree on the exact-hold atom"
        )
    risky = available.bool().clone()
    risky[:, 0] = False
    logits = torch.stack([action.logits for action in populated]).median(dim=0).values
    logits = torch.where(risky.unsqueeze(-1), logits, torch.zeros_like(logits))
    soft, decision = straight_through_m03r_v6_exit_action(
        logits,
        allow_exact_hold_atom=enabled,
    )
    unavailable = ~risky
    if bool(unavailable.any()):
        sentinel = torch.zeros(3, device=logits.device, dtype=logits.dtype)
        sentinel[M03R_V6_HOLD_ACTION_INDEX] = 1.0
        soft = torch.where(unavailable.unsqueeze(-1), sentinel, soft)
        decision = torch.where(unavailable.unsqueeze(-1), sentinel, decision)
    return M03RV6ExitAction(
        logits=logits,
        soft_probabilities=soft,
        decision_st=decision,
        risky_available=risky,
        exact_hold_atom_enabled=enabled,
    )


def aggregate_top2000_m03r_v7_intents(
    intents: Sequence[Hold30Intent],
    available: torch.Tensor,
) -> Hold30Intent:
    """Aggregate exactly five raw output intents before one execution step."""

    if len(intents) != 5:
        raise Top2000M03RV7ValidationError(
            "TOP2000 output-space ensemble requires exactly five intents"
        )
    entry = _mean_optional(intents, "entry_scores")
    raw_hazard = _median_optional(intents, "raw_hazard_residual")
    hazard = (
        None
        if raw_hazard is None
        else bound_hold30_hazard_residual(raw_hazard, mode="smooth_tanh")
    )
    if hazard is not None:
        assert raw_hazard is not None
        risky = available.bool().clone()
        risky[:, 0] = False
        raw_hazard = torch.where(risky, raw_hazard, torch.zeros_like(raw_hazard))
        hazard = torch.where(risky, hazard, torch.full_like(hazard, -12.0))
    return Hold30Intent(
        entry_scores=entry,
        target_logits=_mean_optional(intents, "target_logits"),
        gate=_median_optional(intents, "gate"),
        hazard_residual=hazard,
        raw_hazard_residual=raw_hazard,
        exact_hold_probability=_median_optional(intents, "exact_hold_probability"),
        exact_hold_logit=_median_optional(intents, "exact_hold_logit"),
        exact_hold_soft_probability=_median_optional(
            intents, "exact_hold_soft_probability"
        ),
        exact_hold_decision_st=_median_optional(intents, "exact_hold_decision_st"),
        exposure_residual=_median_optional(intents, "exposure_residual"),
        alpha_mean_30d=_mean_optional(intents, "alpha_mean_30d"),
        alpha_downside_30d=_median_optional(intents, "alpha_downside_30d"),
        active_risk_scale=_median_optional(intents, "active_risk_scale"),
        signal_confidence=_median_optional(intents, "signal_confidence"),
        uncalibrated_signal_confidence_logit=_median_optional(
            intents, "uncalibrated_signal_confidence_logit"
        ),
        benchmark_derisk_request=_median_optional(
            intents, "benchmark_derisk_request"
        ),
        total_risk_overlay=_median_optional(intents, "total_risk_overlay"),
        auxiliary_alpha_mean=_mean_optional(intents, "auxiliary_alpha_mean"),
        exit_action_v6=_aggregate_exit_action(intents, available),
    )


class Top2000M03RV7OutputSpaceEnsemblePolicy(nn.Module):
    """Five complete encoders whose decision outputs are aggregated once."""

    episode_factor_loadings: torch.Tensor
    episode_factor_constraint_pinv: torch.Tensor
    state_provider_compatibility_id = (
        Top2000M03RV7DevelopmentPolicy.state_provider_compatibility_id
    )

    def __init__(self, members: Sequence[Top2000M03RV7DevelopmentPolicy]) -> None:
        super().__init__()
        if len(members) != 5:
            raise Top2000M03RV7ValidationError(
                "fold ensemble requires exactly five member policies"
            )
        setting_ids = {member.setting.setting_id for member in members}
        token_dims = {member.token_dim for member in members}
        if len(setting_ids) != 1 or len(token_dims) != 1:
            raise Top2000M03RV7ValidationError(
                "ensemble members must share setting identity and architecture"
            )
        self.members = nn.ModuleList(members)
        self.setting = members[0].setting
        self._member_token_dim = members[0].token_dim
        self.register_buffer(
            "episode_factor_loadings",
            torch.empty((0, 0), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "episode_factor_constraint_pinv",
            torch.empty((0, 0), dtype=torch.float32),
            persistent=False,
        )

    @property
    def token_dim(self) -> int:
        # Used only to create an unused sequence placeholder.  The state
        # provider returns [batch,asset,member,token] and runtime accepts its
        # leading [batch,asset] contract.
        return self._member_token_dim

    def _typed_members(self) -> tuple[Top2000M03RV7DevelopmentPolicy, ...]:
        return tuple(
            cast(Top2000M03RV7DevelopmentPolicy, member)
            for member in self.members
        )

    def bind_episode_factor_loadings(self, loadings: torch.Tensor) -> None:
        members = self._typed_members()
        for member in members:
            member.bind_episode_factor_loadings(loadings)
        self.episode_factor_loadings = (
            members[0].episode_factor_loadings.detach().clone()
        )
        self.episode_factor_constraint_pinv = (
            members[0].episode_factor_constraint_pinv.detach().clone()
        )

    def encode_episode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.stack(
            [
                member.encode_episode(*args, **kwargs)
                for member in self._typed_members()
            ],
            dim=-2,
        )

    def hold30_intent(
        self,
        state_t: torch.Tensor,
        prev_weights: torch.Tensor,
        available: torch.Tensor,
        age_summaries: torch.Tensor | None = None,
    ) -> Hold30Intent:
        expected = (
            prev_weights.shape[0],
            prev_weights.shape[1],
            5,
            self._member_token_dim,
        )
        if tuple(state_t.shape) != expected:
            raise Top2000M03RV7ValidationError(
                f"ensemble decision state must have shape {expected}"
            )
        intents = tuple(
            member.hold30_intent(
                state_t[:, :, index],
                prev_weights,
                available,
                age_summaries,
            )
            for index, member in enumerate(self._typed_members())
        )
        return aggregate_top2000_m03r_v7_intents(intents, available)


def build_top2000_m03r_v7_validation_runtime(
    policy: Top2000M03RV7DevelopmentPolicy | Top2000M03RV7OutputSpaceEnsemblePolicy,
    *,
    state_provider: Hold30DecisionStateProvider,
) -> Hold30ChronologicalRuntime:
    """Build validation with the exact execution transform used in training."""

    return Hold30ChronologicalRuntime(
        "H2",
        action_builder=Top2000M03RV7ActionBuilder(policy),
        state_provider=state_provider,
        require_trainable_state_provider=False,
    )


def evaluate_top2000_m03r_v7_validation_trace(
    policy: Top2000M03RV7DevelopmentPolicy | Top2000M03RV7OutputSpaceEnsemblePolicy,
    sequence: Hold30Sequence,
    *,
    score_transition_start: int,
    score_transition_stop_exclusive: int,
) -> Top2000M03RV7ValidationTraceEvidence:
    """Execute a detached causal chronology and score exactly 63 transitions."""

    if (
        score_transition_stop_exclusive - score_transition_start
        != TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
        or score_transition_start < 0
        or score_transition_stop_exclusive > sequence.n_positions - 1
    ):
        raise Top2000M03RV7ValidationError(
            "validation score bounds must select exactly 63 available transitions"
        )
    bound, provider = bind_top2000_m03r_v7_runtime_sequence(
        sequence,
        cast(Top2000M03RV7DevelopmentPolicy, policy),
    )
    runtime = build_top2000_m03r_v7_validation_runtime(
        policy,
        state_provider=provider,
    )
    roles = Hold30ReplayGeometry(
        warmup_decisions=63,
        label_support_decisions=63,
        max_origin_batch=1,
    ).roles(sequence.n_positions)
    policy.eval()
    with torch.no_grad():
        trace, _rows = runtime.canonical_pass(policy, bound, roles)
    if not isinstance(trace, Hold30CanonicalTrace):
        raise Top2000M03RV7ValidationError(
            "canonical evaluator returned an unexpected trace"
        )
    scored = trace.transitions[
        score_transition_start:score_transition_stop_exclusive
    ]
    if len(scored) != TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS:
        raise Top2000M03RV7ValidationError("validation trace was truncated")

    def stack(name: str) -> torch.Tensor:
        return torch.stack(
            [getattr(transition, name).mean() for transition in scored]
        ).detach().to(device="cpu", dtype=torch.float64)

    policy_returns = stack("net_return")
    benchmark_returns = stack("benchmark_net_return")
    discretionary = torch.stack(
        [transition.discretionary_accounting.turnover.mean() for transition in scored]
    ).detach().to(device="cpu", dtype=torch.float64)
    cause_rows = {
        cause: torch.stack(
            [transition.turnover_by_cause[cause].mean() for transition in scored]
        ).detach().to(device="cpu", dtype=torch.float64)
        for cause in TURNOVER_CAUSES
    }
    total = sum(cause_rows.values(), torch.zeros_like(discretionary))
    forced = sum(
        (
            value
            for cause, value in cause_rows.items()
            if cause not in {TurnoverCause.DISCRETIONARY, TurnoverCause.STARTUP}
        ),
        torch.zeros_like(discretionary),
    )
    sold = torch.stack(
        [
            transition.discretionary_accounting.sold_value_by_age.sum(dim=(0, 1))
            for transition in scored
        ]
    ).sum(0).detach().to(device="cpu", dtype=torch.float64)
    # Holding telemetry belongs to the scored window boundary, not the final
    # label-support state 63 sessions later.  The latter is retained only to
    # make h63 auxiliary targets causal and complete.
    score_boundary = trace.boundary_states[score_transition_stop_exclusive]
    terminal = score_boundary.ledger.economic_value.sum(dim=(0, 1)).detach()
    terminal = terminal.to(device="cpu", dtype=torch.float64)
    return Top2000M03RV7ValidationTraceEvidence(
        policy_net_returns=policy_returns,
        benchmark_net_returns=benchmark_returns,
        active_log_returns=torch.log1p(policy_returns) - torch.log1p(benchmark_returns),
        total_one_way_turnover=total,
        discretionary_one_way_turnover=discretionary,
        forced_one_way_turnover=forced,
        discretionary_sold_notional_by_age=sold,
        terminal_risky_notional_by_age=terminal,
        score_transition_start=score_transition_start,
        score_transition_stop_exclusive=score_transition_stop_exclusive,
    )


def validate_fold_score_bounds(
    fold: Top2000M03RV7DevelopmentFold,
    *,
    sequence_global_state_start: int,
    sequence_state_rows: int,
) -> tuple[int, int]:
    """Translate immutable global fold decisions into one local trace slice."""

    start = fold.validation_decision_start - sequence_global_state_start
    stop = fold.validation_decision_stop_exclusive - sequence_global_state_start
    if (
        start != 251
        or stop - start != TOP2000_M03R_V7_DEV_VALIDATION_DECISIONS
        or stop > sequence_state_rows - 1
    ):
        raise Top2000M03RV7ValidationError(
            "validation chronology must carry 251 causal context transitions and "
            "score the exact fixed 63-decision fold window"
        )
    return start, stop


__all__ = [
    "TOP2000_M03R_V7_ENSEMBLE_RULE",
    "TOP2000_M03R_V7_FOLD_ENSEMBLE_SCHEMA",
    "TOP2000_M03R_V7_SEED_VALIDATION_SCHEMA",
    "TOP2000_M03R_V7_VALIDATION_TRACE_SCHEMA",
    "Top2000M03RV7FoldEnsembleReceipt",
    "Top2000M03RV7OutputSpaceEnsemblePolicy",
    "Top2000M03RV7SeedValidationReceipt",
    "Top2000M03RV7ValidationError",
    "Top2000M03RV7ValidationTraceEvidence",
    "aggregate_top2000_m03r_v7_intents",
    "build_top2000_m03r_v7_validation_runtime",
    "evaluate_top2000_m03r_v7_validation_trace",
    "model_state_sha256",
    "tensor_sha256",
    "validate_fold_score_bounds",
]
