from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch
import torch.nn.functional as F


ACTION_CONFIDENCE_FIELD_NAMES = (
    "valid_action",
    "q_mean",
    "q_std_epistemic",
    "q_std_total",
    "q_lcb_05",
    "q_ucb_95",
    "p_positive",
    "profit_confidence",
    "p_beats_cash",
    "p_best",
    "p_best_member_vote",
    "p_best_draw",
    "selection_confidence",
    "advantage_mean",
    "advantage_lcb",
    "rank",
    "confidence",
)


@dataclass(frozen=True)
class ActionConfidenceConfig:
    method: str = "ensemble_residual"
    hurdle_bps: float = 2.0
    interval_alpha: float = 0.05
    min_calibration_rows: int = 1_000
    ood_penalty: bool = True
    confidence_beta_best: float = 0.5
    confidence_beta_positive: float = 0.5
    ood_lambda: float = 1.0
    q_value_scale: float = 10_000.0
    p_best_draws: int = 512
    p_best_draw_batch_rows: int = 512
    p_best_draw_batch_size: int = 64
    p_best_draw_seed: int = 17

    def validate(self) -> None:
        if self.hurdle_bps < 0:
            raise ValueError("hurdle_bps must be non-negative.")
        if not 0.0 < self.interval_alpha < 0.5:
            raise ValueError("interval_alpha must be in (0, 0.5).")
        if self.min_calibration_rows < 1:
            raise ValueError("min_calibration_rows must be positive.")
        if self.confidence_beta_best < 0 or self.confidence_beta_positive < 0:
            raise ValueError("confidence beta values must be non-negative.")
        if self.ood_lambda < 0:
            raise ValueError("ood_lambda must be non-negative.")
        if self.q_value_scale <= 0:
            raise ValueError("q_value_scale must be positive.")
        if self.p_best_draws <= 0:
            raise ValueError("p_best_draws must be positive.")
        if self.p_best_draw_batch_rows <= 0:
            raise ValueError("p_best_draw_batch_rows must be positive.")
        if self.p_best_draw_batch_size <= 0:
            raise ValueError("p_best_draw_batch_size must be positive.")

    @property
    def hurdle_return(self) -> float:
        return self.hurdle_bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionConfidenceOutput:
    valid_actions: torch.Tensor
    q_mean: torch.Tensor
    q_std_epistemic: torch.Tensor
    q_std_total: torch.Tensor
    q_lcb: torch.Tensor
    q_ucb: torch.Tensor
    p_positive: torch.Tensor
    profit_confidence: torch.Tensor
    p_beats_cash: torch.Tensor
    p_best: torch.Tensor
    p_best_member_vote: torch.Tensor
    p_best_draw: torch.Tensor
    selection_confidence: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_lcb: torch.Tensor
    rank: torch.Tensor
    confidence: torch.Tensor
    ood_score: torch.Tensor
    field_names: tuple[str, ...] = field(default=ACTION_CONFIDENCE_FIELD_NAMES)

    def as_tensor(self) -> torch.Tensor:
        tensors = [
            self.valid_actions.float(),
            self.q_mean,
            self.q_std_epistemic,
            self.q_std_total,
            self.q_lcb,
            self.q_ucb,
            self.p_positive,
            self.profit_confidence,
            self.p_beats_cash,
            self.p_best,
            self.p_best_member_vote,
            self.p_best_draw,
            self.selection_confidence,
            self.advantage_mean,
            self.advantage_lcb,
            self.rank,
            self.confidence,
        ]
        expected_shape = tuple(self.valid_actions.shape)
        for index, tensor in enumerate(tensors):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Action confidence tensor {self.field_names[index] if index < len(self.field_names) else index!r} "
                    f"has shape {tuple(tensor.shape)}, expected {expected_shape}."
                )
        if len(tensors) != len(self.field_names):
            raise ValueError("Action confidence field count does not match ACTION_CONFIDENCE_FIELD_NAMES.")
        if self.ood_score.numel() != expected_shape[0]:
            raise ValueError("ood_score length must match confidence rows.")
        return torch.stack(tensors, dim=-1)


def _ensure_member_axis(q_values: torch.Tensor) -> torch.Tensor:
    values = q_values.float()
    if values.ndim == 2:
        return values.unsqueeze(0)
    if values.ndim == 3:
        return values
    raise ValueError("q_values must have shape [rows, actions] or [members, rows, actions].")


def _validate_matrix_shape(name: str, value: torch.Tensor, expected: torch.Size | tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(expected):
        raise ValueError(f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}.")


def _realized_net_returns(
    action_returns: torch.Tensor,
    *,
    action_target_weights: torch.Tensor | None,
    action_cost_bps: torch.Tensor | None,
) -> torch.Tensor:
    net = action_returns.float()
    weights = action_target_weights.float() if action_target_weights is not None else torch.ones_like(net)
    net = net * weights
    if action_cost_bps is not None:
        net = net - action_cost_bps.float() / 10_000.0 * weights.abs()
    return net


def _normal_prob_greater(mean: torch.Tensor, std: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    safe_std = std.float().clamp_min(1e-8)
    threshold_tensor = torch.as_tensor(threshold, dtype=mean.dtype, device=mean.device)
    z = (mean - threshold_tensor) / (safe_std * math.sqrt(2.0))
    return (0.5 * (1.0 + torch.erf(z))).clamp(0.0, 1.0)


def _ece_score(probability: torch.Tensor, outcome: torch.Tensor, *, bins: int = 10) -> float:
    valid = torch.isfinite(probability) & torch.isfinite(outcome.float())
    if not bool(valid.any().item()):
        return float("nan")
    probs = probability[valid].float().clamp(0.0, 1.0)
    labels = outcome[valid].float()
    ece = torch.tensor(0.0, dtype=torch.float32)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            in_bin = (probs >= lower) & (probs <= upper)
        else:
            in_bin = (probs >= lower) & (probs < upper)
        if bool(in_bin.any().item()):
            weight = in_bin.float().mean()
            ece = ece + weight * (probs[in_bin].mean() - labels[in_bin].mean()).abs()
    return float(ece.item())


def _masked_nan(values: torch.Tensor, valid_actions: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~valid_actions, float("nan"))


class ActionConfidenceCalibrator:
    def __init__(self, config: ActionConfidenceConfig | None = None) -> None:
        self.config = config or ActionConfidenceConfig()
        self.config.validate()
        self.global_residual_std = torch.tensor(float("nan"), dtype=torch.float32)
        self.action_residual_std: torch.Tensor | None = None
        self.metrics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.ood_method = "none"
        self.ood_penalty_active = False
        self.fitted = False

    def _append_warning_once(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def fit(
        self,
        q_values: torch.Tensor,
        realized_returns: torch.Tensor,
        valid_actions: torch.Tensor,
        *,
        action_target_weights: torch.Tensor | None = None,
        action_cost_bps: torch.Tensor | None = None,
    ) -> "ActionConfidenceCalibrator":
        q_members = _ensure_member_axis(q_values)
        _validate_matrix_shape("realized_returns", realized_returns, q_members.shape[1:])
        _validate_matrix_shape("valid_actions", valid_actions, q_members.shape[1:])
        if action_target_weights is not None:
            _validate_matrix_shape("action_target_weights", action_target_weights, q_members.shape[1:])
        if action_cost_bps is not None:
            _validate_matrix_shape("action_cost_bps", action_cost_bps, q_members.shape[1:])
        q_mean = q_members.mean(dim=0) / float(self.config.q_value_scale)
        valid = valid_actions.bool() & torch.isfinite(realized_returns.float()) & torch.isfinite(q_mean)
        realized_net = _realized_net_returns(
            realized_returns,
            action_target_weights=action_target_weights,
            action_cost_bps=action_cost_bps,
        )
        residual = realized_net - q_mean
        valid_residual = valid & torch.isfinite(residual)
        residual_values = residual[valid_residual]
        if residual_values.numel() < 2:
            raise ValueError("At least two valid calibration residuals are required.")
        if residual_values.numel() < self.config.min_calibration_rows:
            self.warnings.append(
                f"calibration_rows_below_minimum:{int(residual_values.numel())}<{self.config.min_calibration_rows}"
            )
        global_std = residual_values.std(unbiased=False).clamp_min(1e-8)
        action_std = torch.full((q_mean.shape[1],), float(global_std.item()), dtype=torch.float32)
        for action in range(q_mean.shape[1]):
            values = residual[:, action][valid_residual[:, action]]
            if values.numel() >= 2:
                action_std[action] = values.std(unbiased=False).clamp_min(1e-8)
        self.global_residual_std = global_std.detach().cpu()
        self.action_residual_std = action_std.detach().cpu()
        sigma = action_std.to(q_mean.device).expand_as(q_mean)
        p_positive = _normal_prob_greater(q_mean, sigma, self.config.hurdle_return)
        outcome = realized_net > self.config.hurdle_return
        coverage_alpha = self.config.interval_alpha
        z_value = float(NormalDist().inv_cdf(1.0 - coverage_alpha))
        q_lcb = q_mean - z_value * sigma
        q_ucb = q_mean + z_value * sigma
        covered = (realized_net >= q_lcb) & (realized_net <= q_ucb)
        brier = ((p_positive[valid_residual] - outcome[valid_residual].float()) ** 2).mean()
        self.metrics = {
            "calibration_rows": int(residual_values.numel()),
            "global_residual_std": float(global_std.item()),
            "brier_p_positive": float(brier.item()),
            "ece_p_positive": _ece_score(p_positive[valid_residual], outcome[valid_residual]),
            "interval_alpha": float(coverage_alpha),
            "interval_coverage": float(covered[valid_residual].float().mean().item()),
            "hurdle_bps": float(self.config.hurdle_bps),
            # These metrics are computed on the SAME residuals used to fit sigma, so they are
            # in-sample and optimistic by construction (coverage sits near nominal, ECE/Brier
            # are best-case). Treat them as fit diagnostics, not out-of-sample calibration.
            "in_sample_optimistic": True,
            "interval_coverage_basis": "residual_only_sigma_in_sample",
        }
        self.fitted = True
        return self

    def predict(
        self,
        q_values: torch.Tensor,
        valid_actions: torch.Tensor,
        *,
        ood_score: torch.Tensor | None = None,
    ) -> ActionConfidenceOutput:
        if not self.fitted or self.action_residual_std is None:
            raise ValueError("ActionConfidenceCalibrator.fit must be called before predict.")
        q_members = _ensure_member_axis(q_values)
        _validate_matrix_shape("valid_actions", valid_actions, q_members.shape[1:])
        valid = valid_actions.bool()
        q_members_return = q_members / float(self.config.q_value_scale)
        q_mean = q_members_return.mean(dim=0)
        q_std_epistemic = q_members_return.std(dim=0, unbiased=False)
        if self.action_residual_std.numel() != q_mean.shape[1]:
            raise ValueError("Fitted residual action count does not match q_values action dimension.")
        residual_std = self.action_residual_std.to(q_mean.device).expand_as(q_mean)
        q_std_total = torch.sqrt(q_std_epistemic.square() + residual_std.square()).clamp_min(1e-8)
        z_value = float(NormalDist().inv_cdf(1.0 - self.config.interval_alpha))
        q_lcb = q_mean - z_value * q_std_total
        q_ucb = q_mean + z_value * q_std_total
        p_positive = _normal_prob_greater(q_mean, q_std_total, self.config.hurdle_return)
        cash_mean = q_mean[:, :1]
        cash_std = q_std_total[:, :1]
        # NOTE: diff_std assumes zero correlation between the action and CASH (no covariance
        # term), so the difference variance is overstated and p_beats_cash is pulled toward
        # 0.5 for correlated instruments. p_best_draw similarly assumes diagonal covariance.
        # These are documented as independence-assuming heuristics in manifest().
        diff_std = torch.sqrt(q_std_total.square() + cash_std.square()).clamp_min(1e-8)
        p_beats_cash = _normal_prob_greater(q_mean - cash_mean, diff_std, self.config.hurdle_return)
        # CASH-vs-CASH is an undefined self-comparison; expose NaN instead of a confusing
        # sub-0.5 "cash beats cash" value (CASH is action index 0 by contract).
        cash_self_mask = torch.zeros_like(p_beats_cash, dtype=torch.bool)
        cash_self_mask[:, 0] = True
        p_beats_cash = p_beats_cash.masked_fill(cash_self_mask, float("nan"))
        p_best_member_vote = self._p_best_member_vote(q_members_return, valid)
        if q_members_return.shape[0] == 1:
            self._append_warning_once("p_best_member_vote_is_argmax_indicator_with_single_member")
        p_best_draw = self._p_best_draw(q_mean, q_std_total, valid)
        p_best = p_best_draw
        advantage_mean, advantage_lcb = self._advantages(q_mean, q_lcb, q_ucb, valid)
        rank = self._rank(q_mean, valid)
        if ood_score is None:
            ood = torch.zeros(q_mean.shape[0], dtype=q_mean.dtype, device=q_mean.device)
            self.ood_method = "none"
            self.ood_penalty_active = False
            if self.config.ood_penalty:
                self._append_warning_once("ood_penalty_configured_but_no_ood_score_supplied")
        else:
            ood = ood_score.to(device=q_mean.device, dtype=q_mean.dtype).flatten()
            if ood.numel() != q_mean.shape[0]:
                raise ValueError("ood_score length must match q_values rows.")
            self.ood_method = "external_score"
            self.ood_penalty_active = bool(self.config.ood_penalty)
        if self.config.ood_penalty:
            ood_penalty = torch.exp(-float(self.config.ood_lambda) * ood).clamp(0.0, 1.0)
        else:
            ood_penalty = torch.ones_like(ood)
        profit_confidence = p_positive
        selection_confidence = p_best_draw.clamp(0.0, 1.0) * ood_penalty[:, None]
        confidence = selection_confidence.pow(float(self.config.confidence_beta_best)) * profit_confidence.clamp(
            0.0,
            1.0,
        ).pow(float(self.config.confidence_beta_positive))
        p_best = p_best.masked_fill(~valid, 0.0)
        p_best_member_vote = p_best_member_vote.masked_fill(~valid, 0.0)
        p_best_draw = p_best_draw.masked_fill(~valid, 0.0)
        selection_confidence = selection_confidence.masked_fill(~valid, float("nan"))
        return ActionConfidenceOutput(
            valid_actions=valid.detach().cpu(),
            q_mean=_masked_nan(q_mean, valid).detach().cpu(),
            q_std_epistemic=_masked_nan(q_std_epistemic, valid).detach().cpu(),
            q_std_total=_masked_nan(q_std_total, valid).detach().cpu(),
            q_lcb=_masked_nan(q_lcb, valid).detach().cpu(),
            q_ucb=_masked_nan(q_ucb, valid).detach().cpu(),
            p_positive=_masked_nan(p_positive, valid).detach().cpu(),
            profit_confidence=_masked_nan(profit_confidence, valid).detach().cpu(),
            p_beats_cash=_masked_nan(p_beats_cash, valid).detach().cpu(),
            p_best=p_best.detach().cpu(),
            p_best_member_vote=p_best_member_vote.detach().cpu(),
            p_best_draw=p_best_draw.detach().cpu(),
            selection_confidence=selection_confidence.detach().cpu(),
            advantage_mean=_masked_nan(advantage_mean, valid).detach().cpu(),
            advantage_lcb=_masked_nan(advantage_lcb, valid).detach().cpu(),
            rank=_masked_nan(rank, valid).detach().cpu(),
            confidence=_masked_nan(confidence, valid).detach().cpu(),
            ood_score=ood.detach().cpu(),
        )

    @staticmethod
    def _p_best_member_vote(q_members: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = q_members.masked_fill(~valid.unsqueeze(0), -float("inf"))
        winners = masked.argmax(dim=-1)
        p_best = torch.zeros_like(q_members[0])
        valid_rows = valid.any(dim=1)
        for member in range(q_members.shape[0]):
            winner = winners[member]
            src = torch.ones((winner.numel(), 1), dtype=p_best.dtype, device=p_best.device)
            p_best.scatter_add_(dim=1, index=winner.unsqueeze(1), src=src)
        p_best = p_best / max(float(q_members.shape[0]), 1.0)
        return p_best.masked_fill(~valid_rows[:, None], 0.0).masked_fill(~valid, 0.0)

    def _p_best_draw(self, q_mean: torch.Tensor, q_std_total: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        rows, actions = q_mean.shape
        p_best = torch.zeros_like(q_mean)
        generator = torch.Generator(device=q_mean.device)
        generator.manual_seed(int(self.config.p_best_draw_seed))
        row_batch = int(self.config.p_best_draw_batch_rows)
        draw_batch = int(self.config.p_best_draw_batch_size)
        total_draws = int(self.config.p_best_draws)
        for row_start in range(0, rows, row_batch):
            row_end = min(row_start + row_batch, rows)
            mean_chunk = q_mean[row_start:row_end]
            std_chunk = q_std_total[row_start:row_end]
            valid_chunk = valid[row_start:row_end]
            counts = torch.zeros_like(mean_chunk)
            draws_done = 0
            while draws_done < total_draws:
                current_draws = min(draw_batch, total_draws - draws_done)
                noise = torch.randn(
                    (current_draws, row_end - row_start, actions),
                    device=q_mean.device,
                    dtype=q_mean.dtype,
                    generator=generator,
                )
                samples = mean_chunk.unsqueeze(0) + noise * std_chunk.unsqueeze(0)
                samples = samples.masked_fill(~valid_chunk.unsqueeze(0), -float("inf"))
                winners = samples.argmax(dim=-1)
                counts = counts + F.one_hot(winners, num_classes=actions).to(dtype=q_mean.dtype).sum(dim=0)
                draws_done += current_draws
            row_valid = valid_chunk.any(dim=1)
            p_best[row_start:row_end] = (counts / float(total_draws)).masked_fill(~row_valid[:, None], 0.0)
        return p_best.masked_fill(~valid, 0.0)

    @staticmethod
    def _advantages(
        q_mean: torch.Tensor,
        q_lcb: torch.Tensor,
        q_ucb: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = torch.full_like(q_mean, float("nan"))
        advantage_lcb = torch.full_like(q_mean, float("nan"))
        for action in range(q_mean.shape[1]):
            other_valid = valid.clone()
            other_valid[:, action] = False
            has_other = other_valid.any(dim=1)
            max_other_mean = q_mean.masked_fill(~other_valid, -float("inf")).max(dim=1).values
            max_other_ucb = q_ucb.masked_fill(~other_valid, -float("inf")).max(dim=1).values
            advantage[:, action] = torch.where(has_other, q_mean[:, action] - max_other_mean, torch.nan)
            advantage_lcb[:, action] = torch.where(has_other, q_lcb[:, action] - max_other_ucb, torch.nan)
        return advantage, advantage_lcb

    @staticmethod
    def _rank(q_mean: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = q_mean.masked_fill(~valid, -float("inf"))
        order = masked.argsort(dim=1, descending=True)
        ranks = torch.empty_like(q_mean)
        rank_values = torch.arange(1, q_mean.shape[1] + 1, dtype=q_mean.dtype, device=q_mean.device)
        ranks.scatter_(dim=1, index=order, src=rank_values.expand(q_mean.shape[0], -1))
        return ranks.masked_fill(~valid, float("nan"))

    def manifest(
        self,
        *,
        split_name: str,
        ensemble_size: int,
        calibration_split: str,
        uses_test_for_calibration: bool = False,
        uses_checkpoint_selection_for_calibration: bool = False,
    ) -> dict[str, Any]:
        lower_quantile = float(self.config.interval_alpha)
        upper_quantile = float(1.0 - self.config.interval_alpha)
        confidence_reportability_errors: list[str] = []
        if uses_test_for_calibration:
            confidence_reportability_errors.append("test_split_used_for_confidence_calibration")
        if uses_checkpoint_selection_for_calibration:
            confidence_reportability_errors.append("calibration_split_reused_for_checkpoint_selection")
        warnings = list(dict.fromkeys(self.warnings))
        if ensemble_size <= 1 and "p_best_member_vote_is_argmax_indicator_with_single_member" not in warnings:
            warnings.append("p_best_member_vote_is_argmax_indicator_with_single_member")
        return {
            "schema_version": "action_confidence_v2",
            "confidence_method": self.config.method,
            "ensemble_size": int(ensemble_size),
            "split": split_name,
            "calibration_split": calibration_split,
            "uses_test_for_calibration": bool(uses_test_for_calibration),
            "uses_checkpoint_selection_for_calibration": bool(uses_checkpoint_selection_for_calibration),
            "confidence_reportable": not confidence_reportability_errors,
            "confidence_reportability_errors": confidence_reportability_errors,
            "hurdle_bps": float(self.config.hurdle_bps),
            "interval_alpha": float(self.config.interval_alpha),
            "interval_quantiles": {
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "central_interval_coverage_target": upper_quantile - lower_quantile,
                "interval_assumption": "normal_residual",
                "legacy_field_names": {
                    "q_lcb_05": "lower quantile when interval_alpha=0.05",
                    "q_ucb_95": "upper quantile when interval_alpha=0.05",
                },
            },
            "confidence_semantics": {
                "p_positive": "normal-residual estimate of P(weighted net return > hurdle)",
                "profit_confidence": "alias of p_positive",
                "p_best": "backward-compatible alias of p_best_draw",
                "p_best_member_vote": "fraction of ensemble members that rank the action first; with one member this is an argmax indicator",
                "p_best_draw": "Monte Carlo probability the action is best under independent normal predictive draws",
                "selection_confidence": "p_best_draw after any configured OOD penalty",
                "confidence": "selection_confidence^beta_best * profit_confidence^beta_positive",
                "p_beats_cash": "P(action net return - CASH net return > hurdle); NaN for the CASH self-comparison",
            },
            "independence_assumption": (
                "p_best_draw and p_beats_cash assume cross-action independence (diagonal residual "
                "covariance). For correlated instruments (e.g. QQQ/SPY/TQQQ or any action vs CASH) "
                "the difference variance is overstated and these probabilities are biased toward 0.5; "
                "treat them as upper-variance heuristics rather than calibrated probabilities until a "
                "residual covariance is estimated and used for correlated draws."
            ),
            "calibration_metrics_basis": (
                "calibration_metrics in this manifest are IN-SAMPLE (computed on the residuals used to "
                "fit sigma) and optimistic; fit-time interval_coverage uses residual-only sigma whereas "
                "predict-time q_lcb/q_ucb use sqrt(epistemic^2 + residual^2), so reported coverage does "
                "not describe the published bands when ensemble_size > 1."
            ),
            "p_best_method": "predictive_residual_draws",
            "p_best_member_vote_semantics": "ensemble_argmax_vote_fraction",
            "ood_method": self.ood_method,
            "ood_penalty_configured": bool(self.config.ood_penalty),
            "ood_penalty_active": bool(self.ood_penalty_active),
            "field_names": list(ACTION_CONFIDENCE_FIELD_NAMES),
            "config": self.config.to_dict(),
            "calibration_metrics": dict(self.metrics),
            "warnings": warnings,
        }


def save_action_confidence_npz(
    path: str | Path,
    confidence: ActionConfidenceOutput,
    *,
    row_indices: torch.Tensor,
    decision_timestamps: list[str],
    action_names: list[str],
    manifest: dict[str, Any],
) -> None:
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows, actions = confidence.valid_actions.shape
    if row_indices.numel() != rows:
        raise ValueError("row_indices length must match confidence rows.")
    if len(decision_timestamps) != rows:
        raise ValueError("decision_timestamps length must match confidence rows.")
    if len(action_names) != actions:
        raise ValueError("action_names length must match confidence actions.")
    arrays = {
        "row_indices": row_indices.detach().cpu().long().numpy(),
        "decision_timestamps": np.asarray(decision_timestamps, dtype=str),
        "action_names": np.asarray(action_names, dtype=str),
        "field_names": np.asarray(confidence.field_names, dtype=str),
        "confidence_tensor": confidence.as_tensor().numpy().astype("float32"),
        "valid_actions": confidence.valid_actions.numpy().astype(bool),
        "q_mean": confidence.q_mean.numpy().astype("float32"),
        "q_std_epistemic": confidence.q_std_epistemic.numpy().astype("float32"),
        "q_std_total": confidence.q_std_total.numpy().astype("float32"),
        "q_lcb_05": confidence.q_lcb.numpy().astype("float32"),
        "q_ucb_95": confidence.q_ucb.numpy().astype("float32"),
        "p_positive": confidence.p_positive.numpy().astype("float32"),
        "profit_confidence": confidence.profit_confidence.numpy().astype("float32"),
        "p_beats_cash": confidence.p_beats_cash.numpy().astype("float32"),
        "p_best": confidence.p_best.numpy().astype("float32"),
        "p_best_member_vote": confidence.p_best_member_vote.numpy().astype("float32"),
        "p_best_draw": confidence.p_best_draw.numpy().astype("float32"),
        "selection_confidence": confidence.selection_confidence.numpy().astype("float32"),
        "advantage_mean": confidence.advantage_mean.numpy().astype("float32"),
        "advantage_lcb": confidence.advantage_lcb.numpy().astype("float32"),
        "rank": confidence.rank.numpy().astype("float32"),
        "confidence": confidence.confidence.numpy().astype("float32"),
        "ood_score": confidence.ood_score.numpy().astype("float32"),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True), dtype=str),
    }
    np.savez_compressed(target, **arrays)
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
