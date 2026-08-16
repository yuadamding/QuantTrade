"""Stage-2 DAILY cross-sectional policy WITH cross-day memory -- the ``daily_raw`` path.

This is the day-level redesign. It addresses the two structural gaps of the generic daily mode: (1) the only
profit-trained raw-second representation saw just the last block of each day, and (2) the policy had no learned
memory across days (only the carried portfolio weight). Here:

  * FullDayRawEncoder -- a TRAINABLE two-tier causal transformer over the WHOLE RTH session of raw 1s bars (not
    the last block) -> a per-stock end-of-day embedding. Profit gradients shape it. Stock-day-local, per-field
    normalization keeps price and volume on meaningful independent scales without batch/future-day coupling.
  * CrossDayTemporalEncoder -- a CAUSAL transformer over the DAY axis (per stock, shared weights), windowed to a
    `lookback` of prior days, so the policy can compute multi-day patterns (reversal/momentum/vol) from the
    sequence of daily embeddings. This is the learned cross-day memory BPTT alone cannot provide.
  * DailyCrossSectionPolicy -- fuses the FROZEN Stage-1 context (detached), the trainable full-day raw embedding,
    and raw news into a per-day per-stock token; runs the temporal encoder; then a per-day cross-sectional
    set-transformer emits long-only target weights + an act-gate. The portfolio carry / turnover / T+1 credit
    happen in the rollout (rl_quant.training.daily_policy), which carries the position across the whole episode.

The frozen context enters as plain detached tensors -- no gradient reaches the Stage-1 encoder (the context/policy
split holds). Only this module's parameters are trained by the PnL objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.utils.checkpoint
from torch import nn

from rl_quant.models.context_encoder import _CausalBlock, _sinusoidal
from rl_quant.models.decision_policy import _NewsAggregator
from rl_quant.models.hold30_alpha import (
    Hold30AlphaHead,
    Hold30AlphaHeadConfig,
    M03RV6ConfidenceLifecycleStage,
)
from rl_quant.models.hold30_confidence_v6 import (
    M03RV6StandaloneConfidenceConfig,
    M03RV6StandaloneConfidenceHead,
)
from rl_quant.models.hold30_exit_action_v6 import (
    M03RV6ExitAction,
    M03RV6ExitActionHead,
    validate_m03r_v6_exit_action_protocol,
)
from rl_quant.models.hold30_hazard import (
    HOLD30_HAZARD_BOUND_MODES,
    HOLD30_HAZARD_MAX,
    HOLD30_HAZARD_MIN,
    Hold30HazardBoundMode,
    bound_hold30_hazard_residual,
    clip_hold30_hazard_residual as _clip_hold30_hazard_residual,
    straight_through_exact_hold_decision,
)
from rl_quant.protocol.hold_target import (
    DEFAULT_HOLD_TARGET_SPEC,
    LEGACY_HOLD30_TARGET_SPEC,
    HoldTargetSpec,
    hold_release_hazard as _hold_release_hazard,
)
from rl_quant.protocol.constraints import project_capped_risky_simplex
from rl_quant.protocol.hold30 import (
    HOLD30_MECH8_SETTINGS,
    resolve_hold30_setting,
)
from rl_quant.protocol.hold30_alpha_m03r import (
    M03R_SETTING_IDS as M03R_V4_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r import (
    resolve_m03r_setting as resolve_m03r_v4_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    M03R_SETTING_IDS as M03R_V5_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v5 import (
    resolve_m03r_v5_setting,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    M03R_SETTING_IDS as M03R_V6_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v6 import (
    resolve_m03r_v6_setting,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_MECH8_SETTINGS,
    resolve_hold30_alpha_setting,
)
from rl_quant.protocol.hold30_m03r_confidence import (
    M03RConfidenceCalibrationManifest,
)

HOLD30_AGE_CAP = 60
HOLD30_AGE_SUMMARY_DIM = 5


@dataclass(frozen=True)
class Hold30ModelSwitches:
    """Model-facing switches for the frozen eight-setting Hold-30 screen.

    Loss-only switches are retained here so a checkpoint can bind the complete
    setting identity even though the model consumes only ``use_age_input`` and
    ``use_exposure_timing``.  The training registry remains authoritative for
    optimizer/loss construction.
    """

    setting_id: str
    mechanism: Literal["H0", "H1", "H2", "H3"]
    use_age_input: bool
    use_exposure_timing: bool
    use_early_exit_penalty: bool
    use_turnover_penalty: bool
    use_alpha_head: bool = False
    use_uncertainty: bool = False
    use_total_risk_overlay: bool = False
    use_direct_sharpe: bool = False
    use_confidence_scaled_active_risk: bool = False
    use_three_way_exit_action: bool = False
    allow_exact_hold_atom: bool = False


_HOLD30_MODEL_SWITCHES = {
    setting.setting_id: Hold30ModelSwitches(
        setting.setting_id,
        setting.mechanism,  # type: ignore[arg-type]
        setting.use_position_age,
        setting.use_exposure_timing,
        setting.use_early_exit_penalty,
        setting.use_turnover_penalty,
    )
    for setting in HOLD30_MECH8_SETTINGS
}
_HOLD30_MODEL_SWITCHES.update(
    {
        setting.setting_id: Hold30ModelSwitches(
            setting.setting_id,
            "H0" if setting.mechanism == "legacy-scalar-gate" else "H2",
            setting.age_aware,
            False,
            setting.age_aware,
            setting.age_aware,
            use_alpha_head=setting.supervised_residual_alpha_heads,
            use_uncertainty=setting.uncertainty_downside_heads,
            use_total_risk_overlay=setting.sharpe_mode == "separate-total-risk-overlay",
            use_direct_sharpe=setting.sharpe_mode == "direct-two-pass-gradient",
        )
        for setting in HOLD30_ALPHA_MECH8_SETTINGS
    }
)
_HOLD30_MODEL_SWITCHES.update(
    {
        setting_id: Hold30ModelSwitches(
            setting_id=setting_id,
            mechanism="H2",
            use_age_input=setting.age_aware_holding,
            use_exposure_timing=False,
            use_early_exit_penalty=setting.age_aware_holding,
            use_turnover_penalty=setting.age_aware_holding,
            use_alpha_head=setting.residual_alpha_heads,
            use_uncertainty=setting.uncertainty_scaled_sizing,
            use_total_risk_overlay=(
                setting.sharpe_mode == "separate-total-risk-overlay"
            ),
            use_direct_sharpe=(setting.sharpe_mode == "direct-two-pass-gradient"),
        )
        for setting_id in M03R_V4_SETTING_IDS
        for setting in (resolve_m03r_v4_setting(setting_id),)
    }
)
_HOLD30_M03R_V5_MODEL_SWITCHES = {
    setting_id: Hold30ModelSwitches(
        setting_id=setting_id,
        mechanism="H2",
        use_age_input=setting.age_aware_holding,
        use_exposure_timing=False,
        use_early_exit_penalty=setting.age_aware_holding,
        use_turnover_penalty=setting.age_aware_holding,
        use_alpha_head=setting.residual_alpha_heads,
        use_uncertainty=setting.use_downside_adjusted_stock_score,
        use_total_risk_overlay=(setting.sharpe_mode == "separate-total-risk-overlay"),
        use_direct_sharpe=(setting.sharpe_mode == "direct-two-pass-gradient"),
    )
    for setting_id in M03R_V5_SETTING_IDS
    for setting in (resolve_m03r_v5_setting(setting_id),)
}
_HOLD30_M03R_V6_MODEL_SWITCHES = {
    setting_id: Hold30ModelSwitches(
        setting_id=setting_id,
        mechanism="H2",
        use_age_input=setting.age_aware_holding,
        use_exposure_timing=False,
        use_early_exit_penalty=setting.age_aware_holding,
        use_turnover_penalty=setting.age_aware_holding,
        use_alpha_head=setting.residual_alpha_heads,
        use_uncertainty=setting.use_downside_adjusted_stock_score,
        use_total_risk_overlay=(setting.sharpe_mode == "separate-total-risk-overlay"),
        use_direct_sharpe=(setting.sharpe_mode == "direct-two-pass-gradient"),
        use_confidence_scaled_active_risk=(
            setting.use_confidence_scaled_active_risk_budget
        ),
        use_three_way_exit_action=(setting.exit_hazard_mode == "learned-age-aware"),
        allow_exact_hold_atom=(
            setting.exit_hazard_mode == "learned-age-aware"
            and setting.exact_hold_action_supported
        ),
    )
    for setting_id in M03R_V6_SETTING_IDS
    for setting in (resolve_m03r_v6_setting(setting_id),)
}
HOLD30_V2_MODEL_SETTING_IDS = tuple(
    setting.setting_id for setting in HOLD30_MECH8_SETTINGS
)
HOLD30_ALPHA_MODEL_SETTING_IDS = tuple(
    setting.setting_id for setting in HOLD30_ALPHA_MECH8_SETTINGS
)
HOLD30_M03R_MODEL_SETTING_IDS = M03R_V4_SETTING_IDS
HOLD30_M03R_V4_MODEL_SETTING_IDS = M03R_V4_SETTING_IDS
HOLD30_M03R_V5_MODEL_SETTING_IDS = M03R_V5_SETTING_IDS
HOLD30_M03R_V6_MODEL_SETTING_IDS = M03R_V6_SETTING_IDS
# Backward-compatible V2 public inventory; V3 has a disjoint explicit export.
HOLD30_MODEL_SETTING_IDS = HOLD30_V2_MODEL_SETTING_IDS


def resolve_hold30_model_switches(setting_id: str) -> Hold30ModelSwitches:
    """Return the legacy/V3 model contract for a registered setting.

    Shared M03R IDs resolve to the frozen v4 contract here solely for backward
    compatibility. New M03R v5 callers must use the generation-qualified
    :func:`resolve_hold30_m03r_v5_model_switches` resolver.
    """

    # Resolve through the protocol first so every artifact-producing surface
    # shares one alias-rejection/error contract.
    if setting_id in _HOLD30_MODEL_SWITCHES and setting_id.startswith("hold30a-"):
        alpha_setting = resolve_hold30_alpha_setting(setting_id)
        return _HOLD30_MODEL_SWITCHES[alpha_setting.setting_id]
    if setting_id in M03R_V4_SETTING_IDS:
        m03r_setting = resolve_m03r_v4_setting(setting_id)
        return _HOLD30_MODEL_SWITCHES[m03r_setting.setting_id]
    hold30_setting = resolve_hold30_setting(setting_id)
    return _HOLD30_MODEL_SWITCHES[hold30_setting.setting_id]


def resolve_hold30_m03r_v5_model_switches(setting_id: str) -> Hold30ModelSwitches:
    """Return only the exact generation-qualified M03R v5 model contract."""

    registered = resolve_m03r_v5_setting(setting_id)
    return _HOLD30_M03R_V5_MODEL_SWITCHES[registered.setting_id]


def resolve_hold30_m03r_v6_model_switches(setting_id: str) -> Hold30ModelSwitches:
    """Return only the exact generation-qualified M03R v6 model contract."""

    validate_m03r_v6_exit_action_protocol()
    registered = resolve_m03r_v6_setting(setting_id)
    return _HOLD30_M03R_V6_MODEL_SWITCHES[registered.setting_id]


@dataclass(frozen=True)
class Hold30Intent:
    """Raw decision-time intent; the execution adapter owns portfolio construction.

    H0/H1 populate ``target_logits`` and ``gate``. H2 populates entry,
    hazard, and exposure outputs. H3 populates only ``entry_scores``. Keeping
    the raw intent separate prevents fill-time masks or repaired holdings from
    leaking into the actor.
    """

    entry_scores: torch.Tensor | None = None
    target_logits: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    hazard_residual: torch.Tensor | None = None
    raw_hazard_residual: torch.Tensor | None = None
    exact_hold_probability: torch.Tensor | None = None
    exact_hold_logit: torch.Tensor | None = None
    exact_hold_soft_probability: torch.Tensor | None = None
    exact_hold_decision_st: torch.Tensor | None = None
    exposure_residual: torch.Tensor | None = None
    alpha_mean_30d: torch.Tensor | None = None
    alpha_downside_30d: torch.Tensor | None = None
    active_risk_scale: torch.Tensor | None = None
    signal_confidence: torch.Tensor | None = None
    uncalibrated_signal_confidence_logit: torch.Tensor | None = None
    benchmark_derisk_request: torch.Tensor | None = None
    total_risk_overlay: torch.Tensor | None = None
    auxiliary_alpha_mean: torch.Tensor | None = None
    exit_action_v6: M03RV6ExitAction | None = None


def hold30_age_prior_logit(age: torch.Tensor) -> torch.Tensor:
    """Reference age clock ``beta(a)`` for post-return fill-time ages."""

    if not age.is_floating_point():
        age = age.to(dtype=torch.float32)
    return -2.0 + (age.clamp(min=0.0, max=float(HOLD30_AGE_CAP)) - 30.0) / 4.0


def clip_hold30_hazard_residual(raw_hazard: torch.Tensor) -> torch.Tensor:
    """Preserve the historical public import while delegating to its owner."""

    return _clip_hold30_hazard_residual(raw_hazard)


def hold_age_prior_logit(
    age: torch.Tensor,
    *,
    hold_spec: HoldTargetSpec = DEFAULT_HOLD_TARGET_SPEC,
) -> torch.Tensor:
    """Generic soft-holding age clock; new callers default to three sessions."""

    hold_spec.validate()
    if not age.is_floating_point():
        age = age.to(dtype=torch.float32)
    return (
        -2.0
        + (
            age.clamp(min=0.0, max=float(hold_spec.age_cap_sessions))
            - hold_spec.calibrated_release_location
        )
        / hold_spec.release_transition_width_sessions
    )


def hold_release_hazard(
    age: torch.Tensor,
    hazard_residual: torch.Tensor,
    *,
    hold_spec: HoldTargetSpec = DEFAULT_HOLD_TARGET_SPEC,
    exact_hold_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generic normalized release hazard under one immutable hold target."""

    return _hold_release_hazard(
        age,
        hazard_residual,
        hold_spec=hold_spec,
        exact_hold_probability=exact_hold_probability,
    )


def hold30_release_hazard(
    age: torch.Tensor,
    hazard_residual: torch.Tensor,
    *,
    exact_hold_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized cohort-release hazard from the RFC.

    ``age`` and ``hazard_residual`` follow ordinary PyTorch broadcasting.
    ``hazard_residual=-12`` returns exact zero for every age; zero residual
    follows the approximately 30-session reference release clock.  A later
    research generation may also provide an exact-hold mixture probability.
    The expected release is then multiplied by ``1-p_hold``; ``p_hold=1`` is a
    separate exact hold atom rather than hazard-logit saturation.
    """

    return hold_release_hazard(
        age,
        hazard_residual,
        hold_spec=LEGACY_HOLD30_TARGET_SPEC,
        exact_hold_probability=exact_hold_probability,
    )


def hold_proposed_release(
    age_notional: torch.Tensor,
    hazard_residual: torch.Tensor,
    *,
    hold_spec: HoldTargetSpec = DEFAULT_HOLD_TARGET_SPEC,
    exact_hold_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return gross proposed release under one immutable holding target."""

    hold_spec.validate()
    age_bins = hold_spec.age_cap_sessions + 1
    if age_notional.ndim < 2 or age_notional.shape[-1] != age_bins:
        raise ValueError(
            f"age_notional must end in {age_bins} age bins; got {tuple(age_notional.shape)}"
        )
    if hazard_residual.shape != age_notional.shape[:-1]:
        raise ValueError(
            "hazard_residual must match age_notional without its age-bin axis; "
            f"got {tuple(hazard_residual.shape)} and {tuple(age_notional.shape)}"
        )
    if (
        exact_hold_probability is not None
        and exact_hold_probability.shape != hazard_residual.shape
    ):
        raise ValueError(
            "exact_hold_probability must match hazard_residual; "
            f"got {tuple(exact_hold_probability.shape)} and "
            f"{tuple(hazard_residual.shape)}"
        )
    ages = torch.arange(age_bins, device=age_notional.device, dtype=age_notional.dtype)
    hazards = hold_release_hazard(
        ages,
        hazard_residual.unsqueeze(-1).to(dtype=age_notional.dtype),
        hold_spec=hold_spec,
        exact_hold_probability=(
            None
            if exact_hold_probability is None
            else exact_hold_probability.unsqueeze(-1).to(dtype=age_notional.dtype)
        ),
    )
    return (age_notional * hazards).sum(dim=-1)


def hold30_proposed_release(
    age_notional: torch.Tensor,
    hazard_residual: torch.Tensor,
    *,
    exact_hold_probability: torch.Tensor | None = None,
) -> torch.Tensor:
    """Legacy Hold-30 proposed release; historical callers remain target 30."""

    return hold_proposed_release(
        age_notional,
        hazard_residual,
        hold_spec=LEGACY_HOLD30_TARGET_SPEC,
        exact_hold_probability=exact_hold_probability,
    )


def exact_hold30_intent(reference: torch.Tensor) -> Hold30Intent:
    """Construct the finite H2 neutral action for a ``[..., asset]`` reference.

    Entry scores are intentionally zero: when release is exactly zero and
    risky exposure is unchanged they are irrelevant to the executed delta.
    """

    if reference.ndim < 2 or not reference.is_floating_point():
        raise ValueError("reference must be a floating-point [..., asset] tensor")
    return Hold30Intent(
        entry_scores=torch.zeros_like(reference),
        hazard_residual=torch.full_like(reference, HOLD30_HAZARD_MIN),
        raw_hazard_residual=torch.full_like(reference, HOLD30_HAZARD_MIN),
        exact_hold_probability=torch.ones_like(reference),
        exposure_residual=torch.zeros(
            reference.shape[:-1], device=reference.device, dtype=reference.dtype
        ),
    )


class FullDayRawEncoder(nn.Module):
    """Trainable two-tier causal encoder over a full RTH session of raw 1s bars -> per-stock end-of-day embedding.

    Tier-1 attends locally within `block_seconds` blocks; tier-2 attends causally over the block summaries; the
    last valid block's context is the day embedding. Unlike the frozen Stage-1 context encoder, profit gradients
    update this. Its normalization never couples separate stocks or days."""

    pos1: torch.Tensor
    pos2: torch.Tensor

    def __init__(
        self,
        *,
        bar_feature_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        feedforward_dim: int,
        dropout: float,
        block_seconds: int,
        max_seconds: int,
        grad_checkpoint: bool = False,
        raw_norm: str = "instance",
        stock_chunk: int = 0,
    ) -> None:
        super().__init__()
        d = d_model
        if d % n_heads:
            raise ValueError(f"raw d_model {d} must be divisible by n_heads {n_heads}")
        if raw_norm not in ("instance", "level"):
            raise ValueError(
                f"raw_norm must be 'instance' or 'level', got {raw_norm!r}"
            )
        self.block_seconds = int(block_seconds)
        self.grad_checkpoint = grad_checkpoint
        self.raw_norm = raw_norm
        self.stock_chunk = int(
            stock_chunk
        )  # >0: encode the stock axis in chunks (bit-identical: every norm here
        #                                       is per-(stock,day); huge universes need it for activation memory)
        t1 = max(1, n_layers // 2)
        t2 = max(1, n_layers - t1)
        self.input_proj = nn.Linear(bar_feature_dim, d)
        self.register_buffer(
            "pos1", _sinusoidal(self.block_seconds, d), persistent=False
        )
        self.register_buffer(
            "pos2",
            _sinusoidal(max_seconds // max(1, self.block_seconds) + 2, d),
            persistent=False,
        )
        self.tier1 = nn.ModuleList(
            [_CausalBlock(d, n_heads, feedforward_dim, dropout) for _ in range(t1)]
        )
        self.tier2 = nn.ModuleList(
            [_CausalBlock(d, n_heads, feedforward_dim, dropout) for _ in range(t2)]
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.d_model = d

    def forward(self, bars: torch.Tensor, bar_mask: torch.Tensor) -> torch.Tensor:
        """bars [B,A,S,F] raw OHLCV (session-aligned), bar_mask [B,A,S] -> [B,A,d] end-of-day per-stock embedding.
        `stock_chunk>0` encodes the (weight-shared, per-stock-normalized) stock axis in chunks -- bit-identical,
        bounded activation memory; with grad_checkpoint each chunk is checkpointed (backward recomputes one chunk)."""
        A = bars.shape[1]
        ck = self.stock_chunk if self.stock_chunk and 0 < self.stock_chunk < A else A
        if ck >= A:
            return self._encode_stocks(bars, bar_mask)
        outs = []
        for lo in range(0, A, ck):
            bc, mc = bars[:, lo : lo + ck], bar_mask[:, lo : lo + ck]
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                outs.append(
                    torch.utils.checkpoint.checkpoint(
                        self._encode_stocks, bc, mc, use_reentrant=False
                    )
                )
            else:
                outs.append(self._encode_stocks(bc, mc))
        return torch.cat(outs, dim=1)  # [B,A,d]

    def _encode_stocks(
        self, bars: torch.Tensor, bar_mask: torch.Tensor
    ) -> torch.Tensor:
        B, A, S, _feature_dim = bars.shape
        d = self.d_model
        bl = self.block_seconds
        nB = S // bl
        if nB <= 0:
            raise ValueError(
                f"FullDayRawEncoder needs at least one {bl}s block; got S={S}"
            )
        bars = bars[:, :, : nB * bl]
        bar_mask = bar_mask[:, :, : nB * bl].bool()
        # Per-(stock,day) normalization over that stock-day's valid seconds. BOTH modes have NO coupling across the
        # batch/day axis (a future day cannot affect a past day's normalization -> strictly causal) and use only
        # day-d's own bars (PIT-clean for the END-OF-DAY embedding). They differ in what they preserve:
        m = bar_mask.unsqueeze(-1).to(bars.dtype)  # [B,A,Sd,1]
        cnt = m.sum(dim=2).clamp_min(1.0)  # [B,A,1]
        if self.raw_norm == "instance":
            # per-FIELD standardize: affine-invariant, so the day's intraday move MAGNITUDE is whitened away (only
            # the vol-normalized SHAPE survives) -- bad for a cross-sectional RETURN policy, kept for back-compat.
            mean = (bars * m).sum(dim=2) / cnt  # [B,A,F]
            var = ((bars - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / cnt
            normed = ((bars - mean.unsqueeze(2)) / (var.unsqueeze(2) + 1e-5).sqrt()) * m
        else:  # "level": magnitude-preserving (the daily_raw default)
            # Prices -> deviation from the day's mean CLOSE expressed in RETURN units (divide by the price level, do
            # NOT divide by std): multiplicatively scale-INVARIANT (a $5 and a $500 name are comparable; splits don't
            # matter) yet magnitude-SENSITIVE (a +5% day reads ~10x a +0.5% day -- the cross-sectional signal the
            # instance norm destroyed). Volume -> centered log1p (relative intraday volume; absolute level isn't a
            # return signal). This is INPUT NORMALIZATION of raw OHLCV, not an engineered feature column.
            price, vol = bars[..., :4], bars[..., 4:]
            anchor = ((bars[..., 3:4] * m).sum(dim=2) / cnt).clamp_min(
                1e-2
            )  # [B,A,1] mean close level
            price_n = (price - anchor.unsqueeze(2)) / anchor.unsqueeze(
                2
            )  # ~ price/anchor - 1 (return units)
            vlog = torch.log1p(vol.clamp_min(0.0))
            vol_n = vlog - (vlog * m).sum(dim=2, keepdim=True) / cnt.unsqueeze(
                2
            )  # centered log-volume
            normed = torch.cat([price_n, vol_n], dim=-1) * m
        x = self.input_proj(normed).reshape(B * A * nB, bl, d)
        # Keep autocast's BF16 projection in BF16.  Adding the persistent FP32 positional buffer directly would
        # promote the full raw-session activation, all transformer residuals, and the returned day embedding to
        # FP32.  The small positional slice is the value that should follow the compute dtype.
        x = x + self.pos1[:bl].to(dtype=x.dtype).view(1, bl, d)
        bm1 = bar_mask.reshape(B * A * nB, bl)

        def packed_last(
            rows: torch.Tensor,
            valid: torch.Tensor,
            blocks: nn.ModuleList,
            norm: nn.Module,
        ) -> torch.Tensor:
            """Encode ragged causal rows and retain only their last valid state.

            Grouping by valid length lets SDPA use its native causal kernel. In contrast, combining a key-padding
            mask with causality materializes an ``[N, heads, S, S]`` mask; at full-session universe sizes that mask
            alone can consume multiple GiB. Positional embeddings are already attached before compaction, so a
            gap's absolute time remains represented even though invalid query/key slots are not carried through
            attention. Empty rows remain exactly zero.
            """
            counts = valid.sum(-1)
            result = torch.zeros(rows.shape[0], d, dtype=rows.dtype, device=rows.device)
            for length_value in counts.unique(sorted=True).tolist():
                length = int(length_value)
                if length <= 0:
                    continue
                selected = counts == length
                selected_rows, selected_valid = rows[selected], valid[selected]
                if length == rows.shape[1]:
                    packed = selected_rows
                else:
                    positions = (
                        torch.arange(rows.shape[1], device=rows.device)
                        .expand(selected_rows.shape[0], -1)[selected_valid]
                        .reshape(selected_rows.shape[0], length)
                    )
                    packed = torch.gather(
                        selected_rows, 1, positions.unsqueeze(-1).expand(-1, -1, d)
                    )
                for block in blocks:
                    if self.grad_checkpoint and self.training:
                        packed = torch.utils.checkpoint.checkpoint(
                            lambda value, layer=block: layer(value, None),
                            packed,
                            use_reentrant=False,
                        )
                    else:
                        packed = block(packed, None)
                # CUDA autocast LayerNorm returns FP32; indexed assignment requires an explicit compute-dtype cast.
                result[selected] = norm(packed[:, -1]).to(dtype=result.dtype)
            return result

        summ = packed_last(x, bm1, self.tier1, self.norm1).reshape(B * A, nB, d)
        block_has = bm1.any(-1).reshape(B * A, nB)
        summ = summ * block_has.unsqueeze(-1).to(dtype=summ.dtype)
        h = summ + self.pos2[:nB].to(dtype=summ.dtype).unsqueeze(0)
        # Only the final day state is consumed, so tier 2 can use the same ragged-last path and avoid retaining a
        # padded full-session output. Missing blocks still keep their absolute ``pos2`` timestamp.
        day = packed_last(h, block_has, self.tier2, self.norm2).reshape(B, A, d)
        return day * block_has.any(-1).reshape(B, A, 1).to(
            dtype=day.dtype
        )  # zero for stocks absent all day


class CrossDayTemporalEncoder(nn.Module):
    """CAUSAL transformer over the DAY axis (per stock, shared weights) -> learned multi-day memory.

    Input [B,T,A,d] sequence of per-day per-stock embeddings -> [B,T,A,d] where position t attends only to days
    0..t (strictly causal -- no future leak). Per stock, so the representation is permutation-equivariant across
    the action axis. The effective memory horizon is bounded by the training episode length / eval window
    (`daily_lookback`), not by a hard attention band -- attending to all in-window prior days is correct and lets
    the model weight recent vs distant days itself."""

    pos: torch.Tensor

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        feedforward_dim: int,
        dropout: float,
        max_days: int,
        attention_lookback: int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "pos", _sinusoidal(max_days + 2, d_model), persistent=False
        )
        self.blocks = nn.ModuleList(
            [
                _CausalBlock(d_model, n_heads, feedforward_dim, dropout)
                for _ in range(max(1, n_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        if attention_lookback is not None and (
            isinstance(attention_lookback, bool)
            or not isinstance(attention_lookback, int)
            or attention_lookback <= 0
        ):
            raise ValueError("attention_lookback must be a positive integer or None")
        self.attention_lookback = attention_lookback

    def forward(
        self, seq: torch.Tensor, day_valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        """seq [B,T,A,d] -> [B,T,A,d]. day_valid [B,T,A] (a stock has a real embedding that day) -> absent days are
        masked as attention KEYS (a not-yet-listed stock never feeds the memory); the causal order is in _CausalBlock."""
        B, T, A, d = seq.shape
        if T > self.pos.shape[0]:
            raise ValueError(
                f"episode/eval length {T} exceeds temporal max_days {self.pos.shape[0]}"
            )
        x = seq.permute(0, 2, 1, 3).reshape(B * A, T, d)
        x = x + self.pos[:T].to(dtype=x.dtype).unsqueeze(
            0
        )  # [B*A, T, d], no BF16 -> FP32 promotion
        kpm = (
            (~day_valid.bool()).permute(0, 2, 1).reshape(B * A, T)
            if day_valid is not None
            else None
        )
        for blk in self.blocks:
            x = blk(x, kpm, causal_lookback=self.attention_lookback)
        # Standalone CUDA LayerNorm returns FP32 under autocast.  Keep its FP32 internal reduction, then restore
        # the BF16 residual dtype so the full [B,T,A,d] state and allocator input do not double in size.
        x = self.norm(x).to(dtype=x.dtype)
        return x.reshape(B, A, T, d).permute(0, 2, 1, 3)  # [B,T,A,d]


@dataclass(frozen=True, slots=True)
class Hold30TwoSpeedContextContract:
    """Explicit fast/slow session contract for later Hold-30 generations.

    Existing v2/v3 configurations do not populate this contract and retain
    their exact semantics.  M03R-like generations can bind the human meaning
    of the historical ``raw_recent_days`` and ``daily_lookback`` fields without
    introducing a second, contradictory source of truth.
    """

    fast_raw_context_sessions: int
    slow_context_sessions: int

    def __post_init__(self) -> None:
        for name, value in (
            ("fast_raw_context_sessions", self.fast_raw_context_sessions),
            ("slow_context_sessions", self.slow_context_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fast_raw_context_sessions > self.slow_context_sessions:
            raise ValueError(
                "fast_raw_context_sessions cannot exceed slow_context_sessions"
            )

    def validate_model_geometry(
        self,
        *,
        raw_recent_days: int,
        daily_lookback: int,
        max_days: int,
    ) -> None:
        if raw_recent_days != self.fast_raw_context_sessions:
            raise ValueError(
                "raw_recent_days must equal fast_raw_context_sessions under the "
                "explicit Hold-30 context contract"
            )
        if daily_lookback != self.slow_context_sessions:
            raise ValueError(
                "daily_lookback must equal slow_context_sessions under the "
                "explicit Hold-30 context contract"
            )
        if max_days < self.slow_context_sessions:
            raise ValueError(
                "max_days must cover the complete slow_context_sessions window"
            )


@dataclass
class DailyCrossSectionConfig:
    context_dim: int  # frozen Stage-1 per-stock/market context width (d_model)
    bar_feature_dim: int = 5
    raw_policy_dim: int = 128  # trainable full-day raw encoder width
    raw_policy_layers: int = 2
    raw_policy_heads: int = 4
    raw_block_seconds: int = 300
    session_seconds: int = 23400
    news_raw_dim: int = 1
    max_news: int = 32
    news_embed_dim: int = 32
    token_dim: int = 256  # per-day per-stock token + temporal/allocator width
    temporal_layers: int = 2
    temporal_heads: int = 4
    daily_lookback: int = 60
    # Optional model-enforced rolling attention window.  ``None`` preserves
    # every legacy caller, which already supplies externally windowed input.
    # Development adapters that feed a longer chronology must bind this
    # explicitly to prevent a short-context ablation from seeing older rows.
    temporal_attention_lookback: int | None = None
    max_days: int = 256
    alloc_layers: int = 2
    alloc_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.0
    temperature: float = 1.0
    max_stock_weight: float = 1.0
    gate_init_bias: float = 2.0
    grad_checkpoint: bool = False
    raw_norm: str = "level"  # full-day raw input norm: "level" (magnitude-preserving) | "instance" (whitened)
    raw_recent_days: int = 0
    #                                  window get the (expensive, trainable) full-day raw encode; older days'
    #                                  tokens carry frozen ctx + news + the past-return channel only (has_raw=0).
    #                                  Extends the cross-day memory to e.g. 252d at ~the 42d raw compute.
    #                                  0 = every day raw (the original behavior).
    # Development-only daily-aggregate adapters can opt into a lightweight
    # raw-day token at every context row while retaining the distinct 42-day
    # full-intraday contract in their outer protocol.  Legacy and canonical
    # raw-bar routes remain byte-for-byte unchanged by the default.
    encode_aggregated_daily_ohlcv_all_days: bool = False
    raw_stock_chunk: int = 0
    #                                  many stocks (bit-identical -- all its norms are per-(stock,day)); REQUIRED
    #                                  for huge universes (TOP2000: ~512/chunk on an 80GB H100). 0 = single pass.
    hold30_setting: str | None = None
    #                                  frozen V2 IDs in HOLD30_MODEL_SETTING_IDS or V3 IDs in
    #                                  HOLD30_ALPHA_MODEL_SETTING_IDS.
    age_summary_dim: int = HOLD30_AGE_SUMMARY_DIM
    # Result-moving V3 score coefficient. It deliberately has no default;
    # uncertainty settings fail closed until the manifest supplies it.
    alpha_downside_penalty_kappa: float | None = None
    alpha_active_log_scale_bounds: tuple[float, float] | None = None
    alpha_uncertainty_log_scale_bounds: tuple[float, float] | None = None
    # Opt-in post-v3 contracts. Defaults preserve every frozen v2/v3 model.
    hold30_mechanism_generation: Literal[
        "v2-v3-frozen", "m03r-v1", "m03r-v2", "m03r-v3"
    ] = "v2-v3-frozen"
    hold30_fast_raw_context_sessions: int | None = None
    hold30_slow_context_sessions: int | None = None
    hold30_hazard_bound_mode: Hold30HazardBoundMode = "hard_clip"
    hold30_exact_hold_mixture: bool = False
    hold30_exact_hold_logit_bias: float | None = None
    hold30_fixed_hazard_residual: float | None = None
    alpha_confidence_calibration_manifest_sha256: str | None = None
    alpha_confidence_calibration_manifest: M03RConfidenceCalibrationManifest | None = (
        None
    )
    alpha_confidence_calibration_seed: int | None = None
    alpha_confidence_calibration_checkpoint_sha256: str | None = None
    alpha_confidence_calibration_model_state_sha256: str | None = None
    alpha_confidence_calibration_source_score_array_sha256: str | None = None
    alpha_confidence_calibration_source_target_array_sha256: str | None = None
    alpha_m03r_v6_confidence_stage: M03RV6ConfidenceLifecycleStage | None = None
    alpha_confidence_calibration_fit_evidence: object | None = None


class DailyCrossSectionPolicy(nn.Module):
    """Long-only daily cross-sectional policy with cross-day memory. See module docstring.

    encode_episode(): frozen context (detached) + trainable full-day raw + raw news -> per-day per-stock token ->
    causal cross-day temporal state [B,T,A,token_dim]. step(): per-day cross-sectional set-transformer over the
    temporal state + carried weight -> long-only target weights + act-gate. The frozen context never receives a
    gradient (it arrives as a plain tensor)."""

    def __init__(self, config: DailyCrossSectionConfig) -> None:
        super().__init__()
        if not 0 < float(config.max_stock_weight) <= 1:
            raise ValueError("max_stock_weight must lie in (0, 1]")
        if isinstance(config.age_summary_dim, bool) or int(config.age_summary_dim) <= 0:
            raise ValueError("age_summary_dim must be a positive integer")
        if config.hold30_hazard_bound_mode not in HOLD30_HAZARD_BOUND_MODES:
            raise ValueError(
                "hold30_hazard_bound_mode must be 'hard_clip' or 'smooth_tanh'"
            )
        if config.hold30_mechanism_generation not in {
            "v2-v3-frozen",
            "m03r-v1",
            "m03r-v2",
            "m03r-v3",
        }:
            raise ValueError(
                "hold30_mechanism_generation must be v2-v3-frozen, m03r-v1, "
                "m03r-v2, or m03r-v3"
            )
        if not isinstance(config.hold30_exact_hold_mixture, bool):
            raise TypeError("hold30_exact_hold_mixture must be boolean")
        if config.hold30_mechanism_generation != "m03r-v3" and (
            config.alpha_m03r_v6_confidence_stage is not None
            or config.alpha_confidence_calibration_fit_evidence is not None
        ):
            raise ValueError(
                "the v6 confidence lifecycle and fit evidence require m03r-v3"
            )
        if config.hold30_exact_hold_mixture:
            if (
                config.hold30_exact_hold_logit_bias is None
                or isinstance(config.hold30_exact_hold_logit_bias, bool)
                or not math.isfinite(float(config.hold30_exact_hold_logit_bias))
            ):
                raise ValueError(
                    "an exact-hold mixture requires a finite "
                    "hold30_exact_hold_logit_bias"
                )
        elif config.hold30_exact_hold_logit_bias is not None:
            raise ValueError(
                "hold30_exact_hold_logit_bias is forbidden when the exact-hold "
                "mixture is disabled"
            )
        if config.hold30_fixed_hazard_residual is not None:
            fixed = config.hold30_fixed_hazard_residual
            if (
                isinstance(fixed, bool)
                or not math.isfinite(float(fixed))
                or not HOLD30_HAZARD_MIN <= float(fixed) <= HOLD30_HAZARD_MAX
            ):
                raise ValueError(
                    "hold30_fixed_hazard_residual must be finite and lie in [-12,12]"
                )
            if config.hold30_exact_hold_mixture:
                raise ValueError(
                    "fixed-hazard comparator cannot also learn an exact-hold mixture"
                )
        context_values = (
            config.hold30_fast_raw_context_sessions,
            config.hold30_slow_context_sessions,
        )
        if (context_values[0] is None) != (context_values[1] is None):
            raise ValueError(
                "hold30 fast and slow context session fields must be supplied together"
            )
        if config.hold30_mechanism_generation == "v2-v3-frozen":
            if (
                context_values[0] is not None
                or config.hold30_hazard_bound_mode != "hard_clip"
                or config.hold30_exact_hold_mixture
                or config.hold30_fixed_hazard_residual is not None
            ):
                raise ValueError(
                    "post-v3 context or hazard options require the explicit "
                    "m03r-v1 mechanism generation"
                )
        else:
            if context_values[0] is None:
                raise ValueError(
                    f"{config.hold30_mechanism_generation} requires explicit fast "
                    "and slow context sessions"
                )
            if config.hold30_hazard_bound_mode != "smooth_tanh":
                raise ValueError(
                    f"{config.hold30_mechanism_generation} requires the "
                    "smooth_tanh hazard bound"
                )
        self.hold30_context_contract: Hold30TwoSpeedContextContract | None = None
        if context_values[0] is not None:
            assert context_values[1] is not None
            self.hold30_context_contract = Hold30TwoSpeedContextContract(
                int(context_values[0]),
                int(context_values[1]),
            )
            self.hold30_context_contract.validate_model_geometry(
                raw_recent_days=int(config.raw_recent_days),
                daily_lookback=int(config.daily_lookback),
                max_days=int(config.max_days),
            )
        self.config = config
        is_m03r_v4 = config.hold30_mechanism_generation == "m03r-v1"
        is_m03r_v5 = config.hold30_mechanism_generation == "m03r-v2"
        is_m03r_v6 = config.hold30_mechanism_generation == "m03r-v3"
        if is_m03r_v6:
            if config.hold30_setting is None:
                raise ValueError("m03r-v3 requires an exact M03R v6 setting identity")
            self.hold30_switches = _HOLD30_M03R_V6_MODEL_SWITCHES.get(
                config.hold30_setting
            )
            if self.hold30_switches is None:
                resolve_m03r_v6_setting(config.hold30_setting)
                raise AssertionError("unreachable M03R v6 setting resolution")
        elif is_m03r_v5:
            if config.hold30_setting is None:
                raise ValueError("m03r-v2 requires an exact M03R v5 setting identity")
            self.hold30_switches = _HOLD30_M03R_V5_MODEL_SWITCHES.get(
                config.hold30_setting
            )
            if self.hold30_switches is None:
                resolve_m03r_v5_setting(config.hold30_setting)
                raise AssertionError("unreachable M03R v5 setting resolution")
        else:
            self.hold30_switches = (
                resolve_hold30_model_switches(config.hold30_setting)
                if config.hold30_setting is not None
                else None
            )
        if is_m03r_v4:
            if config.hold30_setting not in M03R_V4_SETTING_IDS:
                raise ValueError(
                    "m03r-v1 requires an exact M03R setting identity from v4"
                )
            assert config.hold30_setting is not None
            resolve_m03r_v4_setting(config.hold30_setting)
        v6_confidence_binding_present = any(
            value is not None
            for value in (
                config.alpha_m03r_v6_confidence_stage,
                config.alpha_confidence_calibration_fit_evidence,
                config.alpha_confidence_calibration_manifest_sha256,
                config.alpha_confidence_calibration_manifest,
                config.alpha_confidence_calibration_seed,
                config.alpha_confidence_calibration_checkpoint_sha256,
                config.alpha_confidence_calibration_model_state_sha256,
                config.alpha_confidence_calibration_source_score_array_sha256,
                config.alpha_confidence_calibration_source_target_array_sha256,
            )
        )
        if (
            is_m03r_v6
            and self.hold30_switches is not None
            and not self.hold30_switches.use_confidence_scaled_active_risk
            and v6_confidence_binding_present
        ):
            raise ValueError(
                "a v6 setting without confidence-scaled active risk cannot bind "
                "the confidence lifecycle or calibration evidence"
            )
        if not (is_m03r_v4 or is_m03r_v5 or is_m03r_v6) and config.hold30_setting in (
            set(M03R_V4_SETTING_IDS)
            | set(M03R_V5_SETTING_IDS)
            | set(M03R_V6_SETTING_IDS)
        ):
            raise ValueError(
                "an M03R setting identity requires its explicit M03R mechanism generation"
            )
        if is_m03r_v4 or is_m03r_v5 or is_m03r_v6:
            assert config.hold30_setting is not None
            m03r_setting = (
                resolve_m03r_v4_setting(config.hold30_setting)
                if is_m03r_v4
                else resolve_m03r_v5_setting(config.hold30_setting)
                if is_m03r_v5
                else resolve_m03r_v6_setting(config.hold30_setting)
            )
            expected_slow = int(m03r_setting.slow_context_trading_sessions)
            if config.hold30_slow_context_sessions != expected_slow:
                raise ValueError(
                    "M03R slow context must match its exact registered setting"
                )
            fixed_expected = m03r_setting.exit_hazard_mode == "fixed-hold30-prior"
            if is_m03r_v6 and config.hold30_exact_hold_mixture:
                raise ValueError(
                    "M03R v6 uses its mutually exclusive three-way action head; "
                    "the frozen v4/v5 exact-hold mixture must remain disabled"
                )
            if (
                not is_m03r_v6
                and not fixed_expected
                and not config.hold30_exact_hold_mixture
            ):
                raise ValueError(
                    "learned M03R exit settings require the hard exact-hold "
                    "branch and an explicit initialization bias"
                )
            if fixed_expected and config.hold30_fixed_hazard_residual != 0.0:
                raise ValueError(
                    "A08-fixed-exit-hazard requires fixed residual 0.0 (the "
                    "30-session structural prior)"
                )
            if not fixed_expected and config.hold30_fixed_hazard_residual is not None:
                raise ValueError("fixed hazard is exclusive to A08-fixed-exit-hazard")
        self.raw_encoder = FullDayRawEncoder(
            bar_feature_dim=config.bar_feature_dim,
            d_model=config.raw_policy_dim,
            n_heads=config.raw_policy_heads,
            n_layers=config.raw_policy_layers,
            feedforward_dim=config.raw_policy_dim * 2,
            dropout=config.dropout,
            block_seconds=config.raw_block_seconds,
            max_seconds=config.session_seconds,
            grad_checkpoint=config.grad_checkpoint,
            raw_norm=config.raw_norm,
            stock_chunk=config.raw_stock_chunk,
        )
        self.news_agg = _NewsAggregator(config.news_raw_dim, config.news_embed_dim)
        # per-day per-stock token: [market | per-stock frozen ctx | full-day raw | news | past_ret | past_valid |
        # has_raw]. past_ret = the stock's OWN 1-day close-to-close return for that day (PIT: known at EOD) -- the
        # raw close series under a scale-invariant normalization, so the cross-day temporal encoder can compute
        # momentum/reversal over its window; has_raw flags whether the raw component is real or a two-speed zero.
        tok_in = (
            config.context_dim * 2 + config.raw_policy_dim + config.news_embed_dim + 3
        )
        self.token_proj = nn.Linear(tok_in, config.token_dim)
        self.temporal = CrossDayTemporalEncoder(
            d_model=config.token_dim,
            n_heads=config.temporal_heads,
            n_layers=config.temporal_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
            max_days=config.max_days,
            attention_lookback=config.temporal_attention_lookback,
        )
        # allocator: cross-sectional set-transformer over [temporal state | prev weight] per day
        self.alloc_in = nn.Linear(config.token_dim + 1, config.token_dim)
        self.cash_bias = nn.Parameter(torch.zeros(config.token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.alloc_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(
            layer, num_layers=config.alloc_layers, enable_nested_tensor=False
        )
        self.score = nn.Sequential(
            nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1)
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1)
        )
        gate_bias = config.gate_init_bias
        if self.hold30_switches is not None:
            if self.hold30_switches.mechanism == "H0":
                gate_bias = 2.0
            elif self.hold30_switches.mechanism == "H1":
                gate_bias = -3.3844844191
        nn.init.constant_(self.gate_head[-1].bias, gate_bias)

        # Hold-30 heads are created only for a registered Hold-30 setting. The
        # legacy/default module therefore retains its exact parameter names,
        # shapes, initialization order, and ``step`` behavior.
        self.entry_head: nn.Module | None = None
        self.hazard_features: nn.Module | None = None
        self.hazard_head: nn.Linear | None = None
        self.exact_hold_head: nn.Linear | None = None
        self.exit_action_head_v6: M03RV6ExitActionHead | None = None
        self.exposure_head: nn.Module | None = None
        self.alpha_head: Hold30AlphaHead | None = None
        self.standalone_confidence_head_v6: M03RV6StandaloneConfidenceHead | None = None
        if self.hold30_switches is not None:
            if (
                self.hold30_switches.use_confidence_scaled_active_risk
                and not self.hold30_switches.use_alpha_head
            ):
                stage = config.alpha_m03r_v6_confidence_stage
                if stage is None:
                    raise ValueError(
                        "M02 confidence-scaled active risk requires an explicit "
                        "v6 confidence lifecycle stage"
                    )
                assert config.hold30_setting is not None
                self.standalone_confidence_head_v6 = M03RV6StandaloneConfidenceHead(
                    M03RV6StandaloneConfidenceConfig(
                        setting_id=config.hold30_setting,
                        hidden_dim=config.token_dim,
                        lifecycle_stage=stage,
                        calibration_manifest_sha256=(
                            config.alpha_confidence_calibration_manifest_sha256
                        ),
                        calibration_manifest=(
                            config.alpha_confidence_calibration_manifest
                        ),
                        calibration_seed=config.alpha_confidence_calibration_seed,
                        calibration_checkpoint_sha256=(
                            config.alpha_confidence_calibration_checkpoint_sha256
                        ),
                        calibration_model_state_sha256=(
                            config.alpha_confidence_calibration_model_state_sha256
                        ),
                        calibration_source_score_array_sha256=(
                            config.alpha_confidence_calibration_source_score_array_sha256
                        ),
                        calibration_source_target_array_sha256=(
                            config.alpha_confidence_calibration_source_target_array_sha256
                        ),
                        calibration_fit_evidence=(
                            config.alpha_confidence_calibration_fit_evidence
                        ),
                    )
                )
            if self.hold30_switches.use_alpha_head:
                assert config.hold30_setting is not None
                self.alpha_head = Hold30AlphaHead(
                    Hold30AlphaHeadConfig(
                        setting_id=config.hold30_setting,
                        hidden_dim=config.token_dim,
                        age_summary_dim=int(config.age_summary_dim),
                        downside_penalty_kappa=config.alpha_downside_penalty_kappa,
                        active_log_scale_bounds=config.alpha_active_log_scale_bounds,
                        uncertainty_log_scale_bounds=(
                            config.alpha_uncertainty_log_scale_bounds
                        ),
                        hazard_bound_mode=config.hold30_hazard_bound_mode,
                        exact_hold_mixture=config.hold30_exact_hold_mixture,
                        exact_hold_logit_bias=config.hold30_exact_hold_logit_bias,
                        fixed_hazard_residual=config.hold30_fixed_hazard_residual,
                        confidence_calibration_manifest_sha256=(
                            config.alpha_confidence_calibration_manifest_sha256
                        ),
                        confidence_calibration_manifest=(
                            config.alpha_confidence_calibration_manifest
                        ),
                        confidence_calibration_seed=(
                            config.alpha_confidence_calibration_seed
                        ),
                        confidence_calibration_checkpoint_sha256=(
                            config.alpha_confidence_calibration_checkpoint_sha256
                        ),
                        confidence_calibration_model_state_sha256=(
                            config.alpha_confidence_calibration_model_state_sha256
                        ),
                        confidence_calibration_source_score_array_sha256=(
                            config.alpha_confidence_calibration_source_score_array_sha256
                        ),
                        confidence_calibration_source_target_array_sha256=(
                            config.alpha_confidence_calibration_source_target_array_sha256
                        ),
                        m03r_v6_confidence_stage=(
                            config.alpha_m03r_v6_confidence_stage
                        ),
                        confidence_calibration_fit_evidence=(
                            config.alpha_confidence_calibration_fit_evidence
                        ),
                        mechanism_generation=(
                            "v3-frozen"
                            if config.hold30_mechanism_generation == "v2-v3-frozen"
                            else config.hold30_mechanism_generation
                        ),
                    )
                )
            elif self.hold30_switches.mechanism in ("H0", "H1"):
                self._init_hold30_output(self.score[-1])
                nn.init.orthogonal_(self.gate_head[-1].weight, gain=1e-3)
            else:
                self.entry_head = nn.Sequential(
                    nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, 1)
                )
                self._init_hold30_output(self.entry_head[-1])
            if (
                self.hold30_switches.mechanism == "H2"
                and not self.hold30_switches.use_alpha_head
            ):
                hazard_input_dim = config.token_dim + 1
                if self.hold30_switches.use_age_input:
                    hazard_input_dim += int(config.age_summary_dim)
                self.hazard_features = nn.Sequential(
                    nn.Linear(hazard_input_dim, config.token_dim),
                    nn.GELU(),
                    nn.LayerNorm(config.token_dim),
                )
                if config.hold30_fixed_hazard_residual is None:
                    self.hazard_head = nn.Linear(config.token_dim, 1)
                    self._init_hold30_output(self.hazard_head)
                if config.hold30_exact_hold_mixture:
                    self.exact_hold_head = nn.Linear(config.token_dim, 1)
                    self._init_hold30_output(self.exact_hold_head)
                    assert config.hold30_exact_hold_logit_bias is not None
                    nn.init.constant_(
                        self.exact_hold_head.bias,
                        float(config.hold30_exact_hold_logit_bias),
                    )
                if self.hold30_switches.use_three_way_exit_action:
                    self.exit_action_head_v6 = M03RV6ExitActionHead(
                        config.token_dim,
                        allow_exact_hold_atom=(
                            self.hold30_switches.allow_exact_hold_atom
                        ),
                    )
                if self.hold30_switches.use_exposure_timing:
                    self.exposure_head = nn.Sequential(
                        nn.LayerNorm(config.token_dim),
                        nn.Linear(config.token_dim, 1),
                    )
                    self._init_hold30_output(self.exposure_head[-1])
        self.temperature = config.temperature
        self.token_dim = config.token_dim

    @staticmethod
    def _init_hold30_output(linear: nn.Linear) -> None:
        """Frozen small-output initialization for every Hold-30 raw intent head."""

        nn.init.orthogonal_(linear.weight, gain=1e-3)
        nn.init.zeros_(linear.bias)

    def _raw_day(self, day_bars_fn, t):
        """Full-day raw embedding for day t across the batch: day_bars_fn(t) -> (bars [B,A,S,F], mask [B,A,S]).
        The raw encoder is per-(stock,day) independent (instance-norm, no batch coupling), so encoding day-by-day
        is bit-identical to encoding the [B*T] reshape at once."""
        bars_t, mask_t = day_bars_fn(t)
        return self.raw_encoder(bars_t, mask_t)  # [B,A,dr]

    def _raw_day_mask(self, T: int) -> list[bool]:
        """Two-speed assignment for a length-T episode: the last `raw_recent_days` days get the trainable raw
        encode (all days if raw_recent_days<=0)."""
        if self.config.encode_aggregated_daily_ohlcv_all_days:
            return [True] * T
        r = self.config.raw_recent_days
        return [True] * T if r <= 0 else [t >= T - r for t in range(T)]

    def _episode_tokens(
        self,
        market,
        per_stock,
        day_bars_fn,
        news_raw,
        news_mask,
        past_ret,
        past_ret_valid,
        raw_day_mask,
        reload_ckpt,
    ):
        """Build the per-day per-stock TOKENS (everything BEFORE the cross-day temporal encoder): frozen context +
        (two-speed) trainable full-day raw + news + the past-return channel -> tok [B,T,A,token_dim].
        day_bars_fn(t) yields day-t bars/mask (a tensor slice in-RAM, or a lazy disk load when streaming);
        raw_day_mask[t] selects which days get the raw encode (False -> zeros + has_raw=0: the day contributes
        frozen ctx/news/past-return only -- and its bars are NEVER loaded, the two-speed compute saving). When
        `reload_ckpt` and training, each raw day's encode is checkpointed so backward RE-LOADS + recomputes it."""
        B, T, A, dc = per_stock.shape
        ckpt = reload_ckpt and self.training
        dr = self.config.raw_policy_dim
        # A BF16 frozen context is already quantized to BF16. Under BF16
        # autocast, keep every token component in that dtype before the giant
        # concatenation; Linear would cast the FP32 concatenation back to BF16
        # anyway. Outside autocast, promote locally to FP32 so BF16 context
        # remains a valid explicit-AMP-off fallback with FP32 module weights.
        low_precision_context = per_stock.dtype in (torch.float16, torch.bfloat16)
        assembly_dtype = (
            per_stock.dtype
            if low_precision_context
            and torch.is_autocast_enabled(per_stock.device.type)
            else torch.float32
        )
        raw_days = []
        for t in range(T):
            if not raw_day_mask[t]:
                raw_days.append(
                    torch.zeros(B, A, dr, device=per_stock.device, dtype=assembly_dtype)
                )
            elif ckpt:
                raw_days.append(
                    torch.utils.checkpoint.checkpoint(
                        self._raw_day, day_bars_fn, t, use_reentrant=False
                    ).to(assembly_dtype)
                )
            else:
                raw_days.append(self._raw_day(day_bars_fn, t).to(assembly_dtype))
        raw = torch.stack(raw_days, dim=1)  # [B,T,A,dr]
        news = self.news_agg(
            news_raw.reshape(B * T, A, news_raw.shape[3], news_raw.shape[4]),
            news_mask.reshape(B * T, A, news_mask.shape[3]),
        ).reshape(B, T, A, -1)
        news = news.to(assembly_dtype)  # [B,T,A,ne]
        mkt = market.to(assembly_dtype).unsqueeze(2).expand(B, T, A, dc)
        per_stock = per_stock.to(assembly_dtype)
        flag = torch.tensor(raw_day_mask, device=per_stock.device, dtype=assembly_dtype)
        flag = flag.view(1, T, 1, 1).expand(B, T, A, 1)
        # Fixed input scaling to ~unit variance (daily moves are ~2%, the other token channels are ~unit scale;
        # unscaled, the momentum channel would start ~100x under-weighted into token_proj). A constant, applied
        # identically everywhere -- input normalization, not a learned/engineered feature.
        pr = (past_ret * 50.0).unsqueeze(-1).to(assembly_dtype)
        pv = past_ret_valid.unsqueeze(-1).to(assembly_dtype)
        return self.token_proj(
            torch.cat([mkt, per_stock, raw, news, pr, pv, flag], dim=-1)
        )  # [B,T,A,token_dim]

    def temporal_state(self, tok, avail):
        """Run the CAUSAL cross-day memory over a (possibly windowed) token slice. tok [B,W,A,token_dim],
        avail [B,W,A] -> [B,W,A,token_dim]. For a rolling EVAL the caller slices the last `daily_lookback` days so
        the memory horizon (and positional range) matches what TRAINING exercised (episode_len), not the full
        split -- otherwise eval runs the temporal encoder at sequence positions/contexts it never saw in training."""
        return self.temporal(tok, day_valid=avail.bool())

    def encode_episode(
        self,
        market,
        per_stock,
        bars,
        bar_mask,
        news_raw,
        news_mask,
        avail,
        past_ret,
        past_ret_valid,
    ):
        """In-RAM encode: bars/bar_mask are pre-stacked [B,T,A,S,F]/[B,T,A,S]. -> temporal_state [B,T,A,token_dim].
        Two-speed: only the last `raw_recent_days` days are raw-encoded (all, if 0)."""
        T = per_stock.shape[1]
        tok = self._episode_tokens(
            market,
            per_stock,
            lambda t: (bars[:, t], bar_mask[:, t]),
            news_raw,
            news_mask,
            past_ret,
            past_ret_valid,
            self._raw_day_mask(T),
            reload_ckpt=False,
        )
        return self.temporal_state(tok, avail)

    def encode_episode_streaming(
        self,
        market,
        per_stock,
        day_bars_fn,
        news_raw,
        news_mask,
        avail,
        n_days,
        past_ret,
        past_ret_valid,
    ):
        """Streaming encode: day_bars_fn(t) lazily loads day-t bars/mask [B,A,S,F]/[B,A,S] from disk; backward
        reloads + recomputes per day (reload_ckpt) so the whole episode's bars are never resident. Two-speed days
        outside `raw_recent_days` never load their bars at all."""
        tok = self._episode_tokens(
            market,
            per_stock,
            day_bars_fn,
            news_raw,
            news_mask,
            past_ret,
            past_ret_valid,
            self._raw_day_mask(n_days),
            reload_ckpt=True,
        )
        return self.temporal_state(tok, avail)

    def encode_tokens_dual(
        self,
        market,
        per_stock,
        day_bars_fn,
        news_raw,
        news_mask,
        past_ret,
        past_ret_valid,
        *,
        raw_start: int = 0,
    ):
        """EVAL token variants for a rolling scored suffix.

        ``tok_noraw`` is built for every day. ``tok_raw`` runs the expensive
        full-day raw encoder only on ``[raw_start, T)``; callers must choose a
        start covering the union of every scored decision's raw window. Earlier
        ``tok_raw`` rows equal the no-raw variant and are never selected. This
        preserves scored outputs while avoiding raw encoding of long input-only
        burn-in prefixes. ``raw_start=0`` retains the full legacy behavior.
        """
        T = per_stock.shape[1]
        if (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, int)
            or not 0 <= raw_start <= T
        ):
            raise ValueError(
                f"raw_start must be an integer in [0, {T}], got {raw_start!r}"
            )
        tok_raw = self._episode_tokens(
            market,
            per_stock,
            day_bars_fn,
            news_raw,
            news_mask,
            past_ret,
            past_ret_valid,
            [t >= raw_start for t in range(T)],
            reload_ckpt=False,
        )
        tok_noraw = self._episode_tokens(
            market,
            per_stock,
            day_bars_fn,
            news_raw,
            news_mask,
            past_ret,
            past_ret_valid,
            [False] * T,
            reload_ckpt=False,
        )
        return tok_raw, tok_noraw

    def _allocator_hidden(self, state_t, prev_weights, available):
        """Shared cross-sectional allocator state and normalized availability mask."""

        _batch, A, _ = state_t.shape
        # Portfolio accounting remains FP32, but the previous-weight feature is immediately consumed by an
        # autocast Linear.  Cast that one column before concatenation so it cannot promote the much wider BF16
        # temporal state to FP32 merely to be cast back inside alloc_in.
        prev_feature = prev_weights.unsqueeze(-1).to(dtype=state_t.dtype)
        tok = self.alloc_in(torch.cat([state_t, prev_feature], dim=-1))
        cash_marker = (
            (torch.arange(A, device=tok.device) == 0).to(dtype=tok.dtype).view(1, A, 1)
        )
        tok = tok + self.cash_bias.to(dtype=tok.dtype) * cash_marker
        kpm = ~available.bool()
        kpm = kpm.clone()
        kpm[:, 0] = False  # CASH always available
        h = self.attn(tok, src_key_padding_mask=kpm)
        return h, kpm

    def _gate(self, h, kpm):
        """Portfolio-wide gate from an allocator state, with stable wide reductions."""

        avail = (~kpm).to(dtype=h.dtype).unsqueeze(-1)
        summary = (h * avail).sum(dim=1, dtype=torch.float32) / avail.sum(
            dim=1, dtype=torch.float32
        ).clamp_min(1.0)  # stable wide reduction; only [B,d] stays FP32
        return torch.sigmoid(self.gate_head(summary).squeeze(-1))

    def step(self, state_t, prev_weights, available):
        """Legacy one-day absolute allocation API, retained without Hold-30 semantics.

        ``state_t`` is ``[B,A,token_dim]`` and ``prev_weights`` / ``available``
        are ``[B,A]``. Returns requested long-only simplex weights and one
        portfolio-wide gate. New Hold-30 runtimes use :meth:`hold30_intent`.
        """

        h, kpm = self._allocator_hidden(state_t, prev_weights, available)
        scores = self.score(h).squeeze(-1) / self.temperature
        scores = scores.masked_fill(kpm, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # requested long-only simplex
        weights = project_capped_risky_simplex(
            weights,
            ~kpm,
            max_risky_weight=self.config.max_stock_weight,
            cash_index=0,
        )
        return weights, self._gate(h, kpm)

    def hold30_intent(
        self, state_t, prev_weights, available, age_summaries=None
    ) -> Hold30Intent:
        """Emit one registered Hold-30 decision-time raw intent.

        ``age_summaries`` has shape ``[B,A,5]`` and order
        ``(mean_age/60, frac_lt10, frac_lt20, frac_lt30, frac_ge30)``.
        Only H2 settings whose switch enables age consume it. Entry scores are
        computed from market state with a zero previous-weight feature, making
        them exactly invariant to holdings and age. Portfolio construction,
        fill-time remasking, and cohort release remain execution-layer work.
        """

        switches = self.hold30_switches
        if switches is None:
            raise RuntimeError(
                "hold30_intent requires DailyCrossSectionConfig.hold30_setting"
            )
        if state_t.ndim != 3 or state_t.shape[-1] != self.token_dim:
            raise ValueError(
                f"state_t must have shape [B,A,{self.token_dim}]; got {tuple(state_t.shape)}"
            )
        B, A, _ = state_t.shape
        expected = (B, A)
        if tuple(prev_weights.shape) != expected or tuple(available.shape) != expected:
            raise ValueError(
                f"prev_weights and available must both have shape {expected}; "
                f"got {tuple(prev_weights.shape)} and {tuple(available.shape)}"
            )
        if A < 1:
            raise ValueError(
                "Hold-30 intent requires a CASH coordinate at asset index 0"
            )

        if switches.mechanism in ("H0", "H1"):
            hidden, kpm = self._allocator_hidden(state_t, prev_weights, available)
            return Hold30Intent(
                target_logits=self.score(hidden).squeeze(-1).float(),
                gate=self._gate(hidden, kpm).float(),
            )

        # H2 entry and H3 sleeve scores are market-only by construction: the
        # previous-weight feature is exactly zero, and age never enters this
        # cross-sectional path.
        market_hidden, kpm = self._allocator_hidden(
            state_t, torch.zeros_like(prev_weights), available
        )
        if switches.use_alpha_head:
            if self.alpha_head is None:
                raise RuntimeError("registered alpha setting is missing its head")
            expected_age = (B, A, int(self.config.age_summary_dim))
            if age_summaries is None or tuple(age_summaries.shape) != expected_age:
                actual = None if age_summaries is None else tuple(age_summaries.shape)
                raise ValueError(
                    f"age_summaries must have shape {expected_age}; got {actual}"
                )
            output = self.alpha_head(
                market_hidden,
                prev_weights,
                age_summaries,
                available.bool(),
            )
            return Hold30Intent(
                entry_scores=output.risk_adjusted_score,
                hazard_residual=output.hazard_residual,
                raw_hazard_residual=output.raw_hazard_residual,
                exact_hold_probability=output.exact_hold_probability,
                exact_hold_logit=output.exact_hold_logit,
                exact_hold_soft_probability=output.exact_hold_soft_probability,
                exact_hold_decision_st=output.exact_hold_decision_st,
                exposure_residual=torch.zeros_like(output.active_risk_scale),
                alpha_mean_30d=output.mean_30d,
                alpha_downside_30d=output.downside_30d,
                active_risk_scale=output.active_risk_scale,
                signal_confidence=output.signal_confidence,
                uncalibrated_signal_confidence_logit=(
                    output.uncalibrated_signal_confidence_logit
                ),
                benchmark_derisk_request=output.benchmark_derisk_request,
                total_risk_overlay=output.total_risk_overlay,
                auxiliary_alpha_mean=output.auxiliary_mean,
                exit_action_v6=output.exit_action_v6,
            )
        if self.entry_head is None:
            raise RuntimeError("registered H2/H3 setting is missing its entry head")
        risky_available = ~kpm
        risky_available = risky_available.clone()
        risky_available[:, 0] = False
        entry = self.entry_head(market_hidden).squeeze(-1)
        entry = torch.where(risky_available, entry, torch.zeros_like(entry)).float()
        if switches.mechanism == "H3":
            return Hold30Intent(entry_scores=entry)

        if self.hazard_features is None:
            raise RuntimeError("registered H2 setting is missing its hazard features")
        hazard_parts = [
            market_hidden,
            prev_weights.to(dtype=market_hidden.dtype).unsqueeze(-1),
        ]
        if switches.use_age_input:
            expected_age = (B, A, int(self.config.age_summary_dim))
            if age_summaries is None or tuple(age_summaries.shape) != expected_age:
                actual = None if age_summaries is None else tuple(age_summaries.shape)
                raise ValueError(
                    f"age_summaries must have shape {expected_age}; got {actual}"
                )
            hazard_parts.append(
                age_summaries.to(device=state_t.device, dtype=market_hidden.dtype)
            )
        hazard_hidden = self.hazard_features(torch.cat(hazard_parts, dim=-1))
        if self.config.hold30_fixed_hazard_residual is None:
            if self.hazard_head is None:
                raise RuntimeError("registered H2 setting is missing its hazard head")
            raw_hazard = self.hazard_head(hazard_hidden).squeeze(-1)
            hazard = bound_hold30_hazard_residual(
                raw_hazard,
                mode=self.config.hold30_hazard_bound_mode,
            )
        else:
            raw_hazard = torch.full_like(
                entry,
                float(self.config.hold30_fixed_hazard_residual),
            )
            hazard = raw_hazard
        exact_hold: torch.Tensor | None = None
        exact_hold_logit: torch.Tensor | None = None
        exact_hold_soft_probability: torch.Tensor | None = None
        if self.exact_hold_head is not None:
            exact_hold_logit = self.exact_hold_head(hazard_hidden).squeeze(-1)
            exact_hold_soft_probability = torch.sigmoid(exact_hold_logit)
            exact_hold = straight_through_exact_hold_decision(exact_hold_logit)
            exact_hold_logit = torch.where(
                risky_available,
                exact_hold_logit,
                torch.zeros_like(exact_hold_logit),
            ).float()
            exact_hold_soft_probability = torch.where(
                risky_available,
                exact_hold_soft_probability,
                torch.ones_like(exact_hold_soft_probability),
            ).float()
            exact_hold = torch.where(
                risky_available,
                exact_hold,
                torch.ones_like(exact_hold),
            ).float()
        exit_action_v6 = (
            None
            if self.exit_action_head_v6 is None
            else self.exit_action_head_v6(
                hazard_hidden,
                risky_available,
                cash_index=0,
            )
        )
        raw_hazard = torch.where(
            risky_available,
            raw_hazard,
            torch.zeros_like(raw_hazard),
        ).float()
        hazard = torch.where(
            risky_available,
            hazard,
            torch.full_like(hazard, HOLD30_HAZARD_MIN),
        ).float()

        if switches.use_exposure_timing:
            if self.exposure_head is None:
                raise RuntimeError("registered H2 setting is missing its exposure head")
            mask = risky_available.to(dtype=hazard_hidden.dtype).unsqueeze(-1)
            pooled = (hazard_hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(
                dim=1, dtype=torch.float32
            ).clamp_min(1.0)
            exposure = self.exposure_head(pooled).squeeze(-1).float()
        else:
            exposure = torch.zeros(B, device=state_t.device, dtype=torch.float32)
        confidence_output = (
            None
            if self.standalone_confidence_head_v6 is None
            else self.standalone_confidence_head_v6(
                market_hidden,
                available.bool(),
                cash_index=0,
            )
        )
        return Hold30Intent(
            entry_scores=entry,
            hazard_residual=hazard,
            raw_hazard_residual=raw_hazard,
            exact_hold_probability=(
                None
                if self.config.hold30_mechanism_generation == "m03r-v2"
                else exact_hold
            ),
            exact_hold_logit=(
                exact_hold_logit
                if self.config.hold30_mechanism_generation == "m03r-v2"
                else None
            ),
            exact_hold_soft_probability=(
                exact_hold_soft_probability
                if self.config.hold30_mechanism_generation == "m03r-v2"
                else None
            ),
            exact_hold_decision_st=(
                exact_hold
                if self.config.hold30_mechanism_generation == "m03r-v2"
                else None
            ),
            exposure_residual=exposure,
            exit_action_v6=exit_action_v6,
            active_risk_scale=(
                None
                if confidence_output is None
                else confidence_output.active_risk_scale
            ),
            signal_confidence=(
                None
                if confidence_output is None
                else confidence_output.signal_confidence
            ),
            uncalibrated_signal_confidence_logit=(
                None
                if confidence_output is None
                else confidence_output.uncalibrated_logit
            ),
            benchmark_derisk_request=(
                None
                if confidence_output is None
                else confidence_output.benchmark_derisk_request
            ),
        )


class DailyForwardHead(nn.Module):
    """Daily SSL pretext head: from each stock's per-day context predict its next-H-day cross-sectionally
    demeaned close-to-close return (the daily relative-value target). Trained jointly with Stage-1, then discarded."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, per_stock: torch.Tensor) -> torch.Tensor:
        return self.net(per_stock).squeeze(-1)
