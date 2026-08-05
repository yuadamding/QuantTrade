"""A series of model + training-strategy DESIGNS for the Phase-1 two-stage, EVENT-TIMED framework.

Each design fully specifies BOTH transformers' architecture AND the training strategy/setup for BOTH stages:

  ARCHITECTURE
    context (two-tier causal transformer over RAW seconds): session_seconds (full RTH session encoded once),
      block_seconds (tier-1 block length = the candidate/decision cadence), d_model / enc_layers / enc_heads.
    policy: raw-second policy encoder (raw_policy_dim / raw_policy_layers / raw_policy_heads) feeding the
      set-transformer (policy_token_dim / policy_layers / policy_heads).
  TRAINING STRATEGY / SETUP (per stage where it differs)
    budget: ssl_steps, policy_steps, ssl_batch_size (DAYS per SSL micro-batch), ssl_accum (grad-accum),
      batch_days (episodes per policy micro-batch), policy_accum (policy grad-accum).
    optimization: ssl_lr / pol_lr, ssl_weight_decay / pol_weight_decay, ssl_warmup_frac / pol_warmup_frac,
      schedule ('cosine' warmup->decay, or 'constant'), grad_clip, amp (bf16 autocast), grad_checkpoint.
    policy objective: cost (turnover), risk_lambda (downside), entropy_coef (allocation exploration),
      temperature (allocation sharpness), max_actions_per_day (soft trade budget) + budget_lambda (its penalty),
      and the CASH-basin / label-accounting knobs -- gate_init_bias (start trading), gate_entropy_coef
      (gate exploration), missing_label_penalty, friction_warmup_frac (ramp cost+budget 0->full so the edge is
      learned before friction bites), and ssl_perstock_coef (Stage-1 cross-sectional pretext weight).

EVENT-TIMED: the policy is NOT on a fixed decision clock. The encoder turns each full session into a context at
EVERY `block_seconds` block (78 blocks/day at 300s); the policy chooses WHEN to trade (a per-block act-gate) under
a SOFT per-day budget of ~`max_actions_per_day` trades, and trades execute T+1. So the candidate grid is the
encoder's blocks -- there is no separate per-candidate storage; one full-session encode yields every context.

The series no longer varies "lookback": the two-tier hierarchy reaches the WHOLE session by design, so every real
design encodes the full RTH session and instead varies (context arch x policy raw/set arch x block cadence x
training strategy x trade budget). `large` is the smaller MINIMUM/floor. Full-session SSL is dominated by the tier-1
activations; `grad_checkpoint` (recompute tier-1 in backward) + `amp` (bf16) keep ONE day/micro-batch within an
80 GB H100 up to ~d512, and `ssl_accum` builds the effective target-batch at fixed peak VRAM. Verify with
nvidia-smi and tune ssl_accum / ssl_batch_size. The 2xH100 sweep runs two designs at once (one per GPU).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Phase1Design:
    name: str
    note: str
    # --- architecture ---
    session_seconds: int             # full RTH session encoded once per day (one raw token/second)
    block_seconds: int               # tier-1 block = candidate/decision cadence (300s -> 78 blocks/day)
    d_model: int
    enc_layers: int
    enc_heads: int
    policy_token_dim: int
    policy_layers: int
    policy_heads: int
    # --- training budget ---
    ssl_steps: int                   # SSL OPTIMIZER steps (each = ssl_accum micro-batches)
    policy_steps: int
    ssl_batch_size: int              # DAYS per SSL micro-batch (a day = one full session)
    batch_days: int                  # days per policy step
    raw_policy_dim: int = 64          # trainable Stage-2 raw-second encoder width (profit-gradient path)
    raw_policy_layers: int = 1
    raw_policy_heads: int = 4
    ssl_accum: int = 8               # grad-accum: effective SSL target-batch = ssl_batch_size * ssl_accum days
    policy_accum: int = 1            # grad-accum: effective policy batch = batch_days * policy_accum episodes
    # --- training strategy / setup (defaults; designs override to vary) ---
    dropout: float = 0.1
    ssl_lr: float = 2e-4
    ssl_weight_decay: float = 1e-2
    ssl_warmup_frac: float = 0.05
    pol_lr: float = 3e-4
    pol_weight_decay: float = 3e-2
    pol_warmup_frac: float = 0.05
    schedule: str = "cosine"          # 'cosine' (warmup->cosine decay) | 'constant'
    grad_clip: float = 1.0
    cost: float = 5e-4
    risk_lambda: float = 0.1
    entropy_coef: float = 0.0
    temperature: float = 1.0
    max_stock_weight: float = 1.0    # hard cap per non-CASH target weight; CASH remains unrestricted
    # -- objective-shaping penalties are calibrated in RETURN UNITS (the per-step net-return scale ~1e-4..1e-3).
    #    The 2026-06-29 loss-scale audit showed the old values (budget 0.1, gate_ent 1e-3, missing 1.0) were
    #    2-4 ORDERS OF MAGNITUDE above the return term: optimizing the loss with ZERO edge reproduced the trained
    #    run exactly (gate pinned at max_actions/nB, cash 0.9+) and no achievable signal (even IC=1) could move it.
    #    Rule of thumb: a penalty's marginal per-trade hurdle should sit near the trading cost, not 1000x above it.
    max_actions_per_day: float = 5.0  # SOFT action budget per rollout: intraday rollout=day; daily_raw=episode
    budget_lambda: float = 1e-3       # rate penalty: marginal trade beyond budget must expect ~budget_lambda
    #                                   (~10bp/trade = 2x the 5bp cost) -- a soft budget, not a hard clamp (was 0.1
    #                                   = a 9.5%-per-5-min hurdle no signal can clear)
    gate_init_bias: float = 2.0       # initial act-gate logit (sigmoid(2)=0.88): start TRADING, not in CASH
    gate_entropy_coef: float = 1e-5   # Bernoulli gate-entropy bonus: keeps the gate off exactly 0 (the only
    #                                   gradient path out of the CASH basin) but subordinate to returns (was 1e-3
    #                                   = ~100x the return term at equilibrium, positioning the gate by itself)
    missing_label_penalty: float = 1e-3 # steers off chronically label-missing names without dominating allocation
    #                                   learning (was 1.0 = ~90x any realistic return gradient -> the alloc head
    #                                   trained as a missing-label classifier)
    friction_warmup_frac: float = 0.3 # ramp turnover cost + budget penalty 0->full over this frac of policy_steps
    ssl_perstock_coef: float = 1.0    # weight of the per-stock cross-sectional SSL pretext (relative-value signal)
    horizon_mode: str = "intraday"    # "intraday" (trade 5-min blocks within a day) | "daily" (hold ACROSS days)
    episode_len: int = 21             # daily mode: days per episode (the RANGE of ability -- max holdable span)
    episode_stride: int = 0           # daily mode: train sliding-window stride (0=non-overlap; small=>more samples)
    bptt_window: int = 1              # truncated-BPTT span: credit a held position's returns to the decision that
    #                                   set it over this many steps (1=myopic 1-step; >1 needed to LEARN long holds)
    label_horizon_days: int = 21      # daily_raw: close-to-close forward-return horizon H (per-decision credit signal)
    auxiliary_horizons: tuple[int, ...] = ()  # daily_raw: ordered auxiliary label horizons. Empty retains the
    #                                   legacy single-label contract at ``label_horizon_days``.
    scored_tail_days: int | None = None  # daily_raw: controlled/scored suffix length. ``None`` retains the
    #                                   legacy caller-supplied score-tail behavior.
    target_holding_days: int | None = None  # daily_raw: soft notional holding-duration target; ``None`` means the
    #                                   legacy design has no explicit holding-duration mandate.
    target_discretionary_turnover: float | None = None  # target mean one-way discretionary turnover per decision.
    terminal_liquidate: bool = True   # report/run liquidation at a real terminal boundary. Hold-30 uses continuing
    #                                   wealth and records optional liquidation separately.
    daily_lookback: int = 60          # daily_raw: learned cross-day MEMORY window. EFFECTIVE horizon = min(this,
    #                                   episode_len): training episodes are episode_len long, and eval bounds its
    #                                   rolling temporal window to episode_len to match (the position CARRY can still
    #                                   hold longer -- continuous eval rides positions across windows).
    exec_delay: int = 1               # daily_raw: exactly one day. Its two-book rollout models one pending target
    #                                   (decide EOD d, execute close d+1); longer delays need an explicit order queue.
    raw_norm: str = "level"           # daily_raw full-day raw input norm: "level" preserves intraday RETURN
    #                                   magnitude (the cross-sectional signal); "instance" whitens it away (legacy)
    context_storage_dtype: str = "float32"  # frozen EOD context kept between stages; bfloat16 is an explicit
    #                                   large-universe storage/communication + AMP token-assembly choice
    raw_recent_days: int = 0          # daily_raw TWO-SPEED tokens: >0 -> only the last this-many days of an
    #                                   episode/eval window get the trainable full-day raw encode; older days feed
    #                                   the cross-day memory as frozen ctx + news + past-return channel. Extends
    #                                   reach (e.g. 252d, enough to COMPUTE 12-month momentum) at ~this-many days'
    #                                   raw compute. 0 = every day raw.
    ssl_daily_coef: float = 1.0       # daily_raw: weight of the DAILY next-H-day SSL pretext (decoupled from
    #                                   ssl_perstock_coef so the noisy intraday next-block pretext can be zeroed
    #                                   while keeping the daily relative-value target)
    bar_seconds: int = 1              # bar GRID resolution: 1 = raw 1-second tokens; >1 = the loader RESAMPLES to
    #                                   bar_seconds-OHLCV (open=first/high=max/low=min/close=last/volume=sum per
    #                                   slot -- the same raw fields on a coarser grid, computed at load time).
    #                                   T+1 labels / day open+close / PIT joins stay on raw 1-second timestamps.
    #                                   60 cuts per-day encode tokens ~60x: the lever that fits TOP2000 training
    #                                   inside ONE DAY on 4xH100. Must divide block_seconds.
    enc_stock_chunk: int = 0          # Stage-1 encoder stock-axis chunk (numerically identical: exact at eval /
    #                                   dropout=0; with dropout>0 the RNG is consumed per-chunk, so train-mode is
    #                                   statistically -- not bit -- equivalent; the chunk value is part of the
    #                                   context identity). REQUIRED for huge universes: one un-chunked TOP2000 day
    #                                   is a ~0.5TB tier-1 activation at the raw 1-second grid. The safe chunk also
    #                                   depends on grid, width, and LOCAL micro-batch; measure the exact geometry.
    #                                   0 = single pass.
    raw_stock_chunk: int = 0          # Stage-2 full-day raw encoder stock chunk (same equivalence caveat).
    #                                   Its safe value also depends on grid, width, and local episode batch.
    #                                   0 = single pass.
    amp: bool = False                 # bf16 autocast (frees ~44% activation -> bigger batch at same VRAM)
    grad_checkpoint: bool = False     # recompute tier-1 in backward (needed for full-session SSL at d>=384)
    min_gpus: int = 1                 # GPUs to give this setting (data-parallel). Set 2 if peak VRAM > one card
    #                                   (~80GB H100); the sweep then launches it via torchrun across that many GPUs.

    def __post_init__(self) -> None:
        if self.d_model % self.enc_heads:
            raise ValueError(f"{self.name}: enc_heads {self.enc_heads} must divide d_model {self.d_model}")
        if self.policy_token_dim % self.policy_heads:
            raise ValueError(f"{self.name}: policy_heads {self.policy_heads} must divide "
                             f"policy_token_dim {self.policy_token_dim}")
        if self.raw_policy_dim % self.raw_policy_heads:
            raise ValueError(f"{self.name}: raw_policy_heads {self.raw_policy_heads} must divide "
                             f"raw_policy_dim {self.raw_policy_dim}")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"{self.name}: schedule must be 'cosine' or 'constant'")
        if self.horizon_mode not in ("intraday", "daily", "daily_raw"):
            raise ValueError(f"{self.name}: horizon_mode must be 'intraday', 'daily', or 'daily_raw'")
        if self.context_storage_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"{self.name}: context_storage_dtype must be 'float32' or 'bfloat16'")
        if self.label_horizon_days < 1 or self.daily_lookback < 1 or self.exec_delay < 1:
            raise ValueError(f"{self.name}: need label_horizon_days>=1, daily_lookback>=1, exec_delay>=1")
        if not isinstance(self.terminal_liquidate, bool):
            raise TypeError(f"{self.name}: terminal_liquidate must be a bool")
        if any(
            isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1
            for horizon in self.auxiliary_horizons
        ):
            raise ValueError(f"{self.name}: auxiliary_horizons must contain positive integers")
        if tuple(sorted(set(self.auxiliary_horizons))) != self.auxiliary_horizons:
            raise ValueError(f"{self.name}: auxiliary_horizons must be strictly increasing and unique")
        if self.auxiliary_horizons and self.label_horizon_days not in self.auxiliary_horizons:
            raise ValueError(f"{self.name}: auxiliary_horizons must include label_horizon_days")
        if self.scored_tail_days is not None and (
            isinstance(self.scored_tail_days, bool)
            or not isinstance(self.scored_tail_days, int)
            or not 1 <= self.scored_tail_days <= self.episode_len
        ):
            raise ValueError(f"{self.name}: scored_tail_days must be in [1, episode_len] or None")
        if self.target_holding_days is not None and (
            isinstance(self.target_holding_days, bool)
            or not isinstance(self.target_holding_days, int)
            or self.target_holding_days < 1
        ):
            raise ValueError(f"{self.name}: target_holding_days must be a positive integer or None")
        if self.target_discretionary_turnover is not None and (
            isinstance(self.target_discretionary_turnover, bool)
            or not 0.0 < self.target_discretionary_turnover <= 1.0
        ):
            raise ValueError(f"{self.name}: target_discretionary_turnover must lie in (0, 1] or be None")
        holding_fields = (
            self.scored_tail_days,
            self.target_holding_days,
            self.target_discretionary_turnover,
        )
        if any(value is not None for value in holding_fields):
            if self.horizon_mode != "daily_raw":
                raise ValueError(f"{self.name}: explicit holding fields require horizon_mode='daily_raw'")
            if any(value is None for value in holding_fields):
                raise ValueError(
                    f"{self.name}: scored_tail_days, target_holding_days, and "
                    "target_discretionary_turnover must be configured together"
                )
            assert self.scored_tail_days is not None and self.target_holding_days is not None
            if self.scored_tail_days < self.target_holding_days + self.exec_delay:
                raise ValueError(
                    f"{self.name}: scored_tail_days must cover target_holding_days plus exec_delay"
                )
            if self.bptt_window < self.target_holding_days:
                raise ValueError(f"{self.name}: bptt_window must be >= target_holding_days")
            if self.label_horizon_days != self.target_holding_days:
                raise ValueError(f"{self.name}: label_horizon_days must equal target_holding_days")
            if self.terminal_liquidate:
                raise ValueError(f"{self.name}: explicit holding designs require terminal_liquidate=False")
        if self.horizon_mode == "daily_raw" and self.exec_delay != 1:
            raise ValueError(
                f"{self.name}: daily_raw supports exec_delay=1 only; longer delays require a pending-order queue"
            )
        if self.horizon_mode != "daily_raw" and self.policy_accum != 1:
            raise ValueError(f"{self.name}: policy_accum is supported by daily_raw training only")
        if self.episode_len <= 1:
            raise ValueError(f"{self.name}: episode_len must be > 1")
        if self.raw_recent_days < 0 or self.raw_recent_days > self.episode_len:
            raise ValueError(f"{self.name}: raw_recent_days must be in [0, episode_len]")
        if self.ssl_daily_coef < 0:
            raise ValueError(f"{self.name}: ssl_daily_coef must be >= 0")
        if self.enc_stock_chunk < 0 or self.raw_stock_chunk < 0:
            raise ValueError(f"{self.name}: stock chunks must be >= 0 (0 = single pass)")
        if self.episode_stride < 0:
            raise ValueError(f"{self.name}: episode_stride must be >= 0")
        if self.bptt_window < 1:
            raise ValueError(f"{self.name}: bptt_window must be >= 1")
        if self.temperature <= 0:
            raise ValueError(f"{self.name}: temperature must be > 0")
        if not 0 < self.max_stock_weight <= 1:
            raise ValueError(f"{self.name}: max_stock_weight must lie in (0, 1]")
        if self.session_seconds % self.block_seconds:
            raise ValueError(f"{self.name}: block_seconds {self.block_seconds} must divide "
                             f"session_seconds {self.session_seconds}")
        if self.bar_seconds < 1 or self.block_seconds % self.bar_seconds:
            raise ValueError(f"{self.name}: bar_seconds {self.bar_seconds} must be >=1 and divide "
                             f"block_seconds {self.block_seconds}")
        if self.max_actions_per_day <= 0 or self.budget_lambda < 0 or self.missing_label_penalty < 0:
            raise ValueError(f"{self.name}: need max_actions_per_day>0, budget_lambda>=0, missing_label_penalty>=0")
        if self.min_gpus < 1:
            raise ValueError(f"{self.name}: min_gpus must be >= 1")
        for f in ("session_seconds", "block_seconds", "ssl_steps", "policy_steps", "ssl_batch_size",
                  "ssl_accum", "batch_days", "policy_accum", "raw_policy_dim", "raw_policy_layers",
                  "raw_policy_heads"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{self.name}: {f} must be positive")


FULL = 23400  # full RTH session (09:30->16:00) in seconds; 78 blocks at 300s
_SERIES = [
    # tiny: CPU smoke / CI only (short session, 4 blocks of 30s).
    Phase1Design("tiny", "smoke/CI only", session_seconds=120, block_seconds=30, d_model=24, enc_layers=1,
                 enc_heads=2, policy_token_dim=24, policy_layers=1, policy_heads=2, ssl_steps=40, policy_steps=60,
                 raw_policy_dim=24, raw_policy_heads=2, ssl_batch_size=2, ssl_accum=1, batch_days=4, dropout=0.0,
                 max_actions_per_day=2.0),

    # large: the MINIMUM -- full session, modest model, standard 300s blocks (78/day), budget ~5.
    Phase1Design("large", "MINIMUM: full session, d256/4L, 300s blocks, budget 5", session_seconds=FULL,
                 block_seconds=300, d_model=256, enc_layers=4, enc_heads=8, policy_token_dim=256, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_days=32,
                 grad_checkpoint=True),

    # ===== variety: context arch x policy arch x block cadence x training strategy x trade budget =====
    Phase1Design("wide", "WIDE d512/8L, full session, 300s blocks, budget 5; bf16", session_seconds=FULL,
                 block_seconds=300, d_model=512, enc_layers=8, enc_heads=8, policy_token_dim=512, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_days=48,
                 amp=True, grad_checkpoint=True),

    Phase1Design("deep", "DEEP-NARROW d384/16L, full session, budget 3; warmup-heavy, clip 0.5, lr 1.5e-4",
                 session_seconds=FULL, block_seconds=300, d_model=384, enc_layers=16, enc_heads=8,
                 policy_token_dim=384, policy_layers=6, policy_heads=8, ssl_steps=3500, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=16, batch_days=48, ssl_lr=1.5e-4, ssl_warmup_frac=0.10,
                 pol_warmup_frac=0.10, grad_clip=0.5, max_actions_per_day=3.0, amp=True, grad_checkpoint=True),

    Phase1Design("balanced", "BALANCED d512/10L, full session, budget 5; calibrated entropy, temp 1.5, cost 1e-3",
                 session_seconds=FULL, block_seconds=300, d_model=512, enc_layers=10, enc_heads=8,
                 policy_token_dim=512, policy_layers=6, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=16, batch_days=64, entropy_coef=1e-5, temperature=1.5, cost=1e-3,
                 amp=True, grad_checkpoint=True),

    Phase1Design("coarse_blocks", "COARSE 600s blocks (39/day), d384/8L, full session; constant lr, risk 0.2",
                 session_seconds=FULL, block_seconds=600, d_model=384, enc_layers=8, enc_heads=8,
                 policy_token_dim=512, policy_layers=4, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=8, batch_days=48, schedule="constant", pol_weight_decay=5e-2,
                 risk_lambda=0.2, amp=True, grad_checkpoint=True),

    Phase1Design("active", "ACTIVE budget 8 (looser budget_lambda 5e-4), d512/10L, full session; bf16, entropy",
                 session_seconds=FULL, block_seconds=300, d_model=512, enc_layers=10, enc_heads=8,
                 policy_token_dim=640, policy_layers=6, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=16, batch_days=64, amp=True, grad_checkpoint=True, ssl_lr=2.5e-4,
                 entropy_coef=2e-5, max_actions_per_day=8.0, budget_lambda=5e-4),

    # ===== LONGER-HORIZON experiments (coarser blocks => decision cadence AND T+1 hold both lengthen) =====
    # NB: the IC probe found price-based cross-sectional signal ~0 at ALL horizons (5min..daily) in TOP50, so
    # these mainly test whether covariates/news carry edge at a longer hold; expect ~null on price alone.
    Phase1Design("h30m", "30-MIN horizon: 1800s blocks (13/day), d384/8L, full session; budget 4", session_seconds=FULL,
                 block_seconds=1800, d_model=384, enc_layers=8, enc_heads=8, policy_token_dim=384, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_days=48,
                 max_actions_per_day=4.0, amp=True, grad_checkpoint=True),

    Phase1Design("h65m", "65-MIN horizon: 3900s blocks (6/day), d384/8L, full session; budget 3", session_seconds=FULL,
                 block_seconds=3900, d_model=384, enc_layers=8, enc_heads=8, policy_token_dim=384, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_days=48,
                 max_actions_per_day=3.0, amp=True, grad_checkpoint=True),

    # ===== CROSS-DAY (daily cross-sectional): hold positions ACROSS days, scored on open->open T+1 returns =====
    # This is where documented cross-sectional equity predictability (daily reversal/momentum, fundamentals) lives.
    # The encoder still summarizes each full session (300s blocks); the policy decides once/day from the END-OF-DAY
    # context and carries positions across `episode_len`-day episodes. budget off (turnover cost regulates); the
    # intraday per-stock SSL is off (intraday cross-section is dead) -- per_stock still fuses covariates/fundamentals.
    Phase1Design("daily_xs", "DAILY cross-sectional, hold across days (21d episodes), d512/8L; bf16", session_seconds=FULL,
                 block_seconds=300, d_model=512, enc_layers=8, enc_heads=8, policy_token_dim=512, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_days=16,
                 horizon_mode="daily", episode_len=21, bptt_window=21, budget_lambda=0.0, ssl_perstock_coef=0.0,
                 amp=True, grad_checkpoint=True),

    # LONG-RANGE cross-sectional: 180-day episodes = the RANGE of ability (positions CAN persist up to 180 days,
    # but the policy chooses each hold's length -- it is NOT forced to hold 180d). Turnover cost (not a sparse
    # budget) regulates frequency; truncated BPTT (window 30) lets a held position's multi-day returns credit the
    # decision that set it, so long holds are LEARNABLE. Overlapping train windows (stride 20) keep enough samples.
    Phase1Design("daily_long", "LONG-RANGE daily: 180d episodes (range), free hold length, BPTT 30, d512/8L; bf16",
                 session_seconds=FULL, block_seconds=300, d_model=512, enc_layers=8, enc_heads=8,
                 policy_token_dim=512, policy_layers=4, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=8, batch_days=8, horizon_mode="daily", episode_len=180,
                 episode_stride=20, bptt_window=30, budget_lambda=0.0, ssl_perstock_coef=0.0,
                 amp=True, grad_checkpoint=True),

    # ===== DAILY_RAW: the day-level redesign (learn a day strategy from the FULL raw second-bar day) =====
    # Structural upgrades over `daily`: (1) a TRAINABLE full-day two-tier raw encoder (profit gradients shape the
    # WHOLE session, not just the last block); (2) a CAUSAL cross-day temporal encoder -> learned multi-day memory
    # (reversal/momentum/vol), which BPTT alone cannot provide; (3) a DAILY per-stock SSL target (next-H-day
    # cross-sectional close-to-close return) instead of the intraday one; (4) CONTINUOUS chronological eval +
    # terminal liquidation; (5) realistic cost from step 1 (friction_warmup=0). Long-only; label = close[d+1+H] /
    # close[d+1] - 1 (H=label_horizon_days), PIT-clean (execute one day after the EOD decision). The gate + carry
    # let positions HOLD up to daily_lookback days; episodes are kept moderate because the trainable full-day raw
    # encode is ~episode_len full-session forwards/step (grad_checkpoint bounds the memory). Tune episode_len /
    # batch_days / policy_steps to the compute budget.
    Phase1Design("daily_raw", "DAILY_RAW: full-day trainable raw + cross-day memory, H=21 close-to-close; d384/6L",
                 session_seconds=FULL, block_seconds=300, d_model=384, enc_layers=6, enc_heads=8,
                 policy_token_dim=256, policy_layers=3, policy_heads=8, ssl_steps=3000, policy_steps=4000,
                 ssl_batch_size=1, ssl_accum=8, batch_days=6, raw_policy_dim=128, raw_policy_layers=2,
                 raw_policy_heads=8, horizon_mode="daily_raw", episode_len=42, episode_stride=5, bptt_window=42,
                 label_horizon_days=21, daily_lookback=42, exec_delay=1, budget_lambda=0.0, ssl_perstock_coef=1.0,
                 friction_warmup_frac=0.0, cost=5e-4, temperature=0.5, raw_norm="level", amp=True,
                 grad_checkpoint=True),

    # 252-day reach: the ONLY signal that survived the IC probes (12-month momentum, needs ~252d of history) is
    # invisible to a 42d memory. Two-speed tokens keep the raw compute at ~42 days while the cross-day memory +
    # the past-return channel span a full year; the noisy intraday per-stock SSL pretext is OFF (its demeaned
    # next-block target has IC~0 -- pure gradient noise), the DAILY relative-value pretext stays on.
    # 1-MINUTE bars (bar_seconds=60): same raw OHLCV fields on a coarser grid, resampled at load time; labels and
    # PIT joins stay 1s-accurate. Cuts the day-encode ~60x -> this design completes in HOURS on one H100 (the
    # 1-second variant needed ~31 GPU-h for SSL alone). The daily thesis lives at 21d horizons -- intra-5-minute
    # microstructure carried ~no signal in the IC probes, so the coarser grid costs little information.
    Phase1Design("daily_raw_252", "DAILY_RAW 252d reach: two-speed tokens (raw last 42d), H=21; d384/6L; 1-min bars",
                 session_seconds=FULL, block_seconds=300, bar_seconds=60, d_model=384, enc_layers=6, enc_heads=8,
                 policy_token_dim=256, policy_layers=3, policy_heads=8, ssl_steps=3000, policy_steps=4000,
                 ssl_batch_size=1, ssl_accum=8, batch_days=2, raw_policy_dim=128, raw_policy_layers=2,
                 raw_policy_heads=8, horizon_mode="daily_raw", episode_len=252, episode_stride=15, bptt_window=42,
                 label_horizon_days=21, daily_lookback=252, exec_delay=1, raw_recent_days=42,
                 budget_lambda=0.0, ssl_perstock_coef=0.0, ssl_daily_coef=1.0,
                 friction_warmup_frac=0.0, cost=5e-4, temperature=0.5, raw_norm="level", amp=True,
                 grad_checkpoint=True),

    # ===== TOP2000, one H100 per setting (four-setting pool), with optimizer budgets preserved by accumulation.
    # Sizing rationale (2026-07 workflow, probe-grounded): the two stages have OPPOSITE binding constraints.
    #   Stage-1 is DATA-RICH (~1.5M train stock-days, params A-independent) -> d512/8L (~25.5M params, ~60:1
    #   sample:param). Stage-2 is OVERFITTING-BOUND (effective samples = ~840 train DAYS regardless of A; H=21
    #   overlap deflates to eff_n ~ 40) -> decision core SMALL (tok96/2L ~ 0.3M) + heavy decoupled weight decay;
    #   per-stock raw encoder raw128/2L (stock-day-rich, A-independent).
    # H100 MEMORY/THROUGHPUT: an RTX 3080 BF16+checkpoint slope probe at the configured LOCAL SSL batch of three
    # grew by ~0.080 GiB allocated / ~0.103 GiB reserved per unchunked stock. Extrapolating an all-at-once
    # 2001-action pass is therefore far above the launcher's 75 GiB ceiling. A 640-stock chunk bounds the dominant
    # recompute near 52/66 GiB plus the small full-axis outputs while retaining large tensor-core work units. The
    # largest Stage-2 variant projected near 70 GiB reserved unchunked; a 1024-stock raw chunk gives two large,
    # balanced passes and ample backward headroom. Runtime telemetry remains authoritative across CUDA builds.
    # One distinct 36-day SSL draw is partitioned into twelve three-day accumulation passes, preserving the former
    # four-rank effective batch without multiplying one H100's peak activation. Stage 2 likewise accumulates four
    # disjoint two-episode micro-batches before one optimizer step, preserving the former global batch of eight,
    # 716 optimizer updates,
    # and about 120k monthly scored-tail exposures. Four H100s run independent settings concurrently.
    Phase1Design("daily_raw_top2000", "TOP2000 H100 primary: 1m bars, d512/8L context, 252d memory, monthly budget",
                 session_seconds=FULL, block_seconds=300, bar_seconds=60, d_model=512, enc_layers=8, enc_heads=8,
                 policy_token_dim=96, policy_layers=2, policy_heads=6, ssl_steps=1000, policy_steps=716,
                 ssl_batch_size=3, ssl_accum=12, batch_days=2, policy_accum=4,
                 raw_policy_dim=128, raw_policy_layers=2,
                 raw_policy_heads=8, horizon_mode="daily_raw", episode_len=252, episode_stride=21, bptt_window=21,
                 label_horizon_days=21, daily_lookback=252, exec_delay=1, raw_recent_days=42,
                 budget_lambda=1e-3, ssl_perstock_coef=0.0, ssl_daily_coef=1.0, pol_weight_decay=0.3,
                 max_actions_per_day=12.0, friction_warmup_frac=0.0, cost=1e-3, temperature=0.5,
                 max_stock_weight=0.01, raw_norm="level", context_storage_dtype="bfloat16", amp=True,
                 grad_checkpoint=True, enc_stock_chunk=640, raw_stock_chunk=1024, min_gpus=1),
]

# ===== TOP50 / 4xH100 one-seed screening =====================================
#
# TOP50 has only ~840 training dates.  Four-way data parallelism would give a
# rank only ~210 contiguous dates and silently destroy a 252-day episode.
# These settings therefore remain ONE-GPU jobs; the pool runs four independent
# settings concurrently.  The paired seed makes one-factor comparisons less
# noisy, while the sweep-level multiple-testing correction still counts every
# setting searched.
#
# For the 60-second designs, 8-day SSL micro-batches x 1 accumulation preserve
# the original effective batch of eight while replacing eight small forwards
# with one tensor-core-friendly forward.  The policy budgets keep approximately
# 120k newly-scored-date draws per run after score-tail de-duplication:
#   42d: 8 * stride5  * 3000 = 120k
#  126d: 8 * stride10 * 1500 = 120k
#  252d: 8 * stride15 * 1000 = 120k
# Activation checkpointing is retained as a safety rail until the H100 peak
# telemetry proves a setting has enough headroom to disable it.  Model width is
# varied in explicit ablations only; unused VRAM is not evidence for a larger
# hypothesis class.
_daily_252 = next(d for d in _SERIES if d.name == "daily_raw_252")
_top50_h100_base = replace(
    _daily_252,
    name="top50_h100_252",
    note="TOP50 H100 primary: 60s grid, 252d memory, equal-exposure B8 screening",
    ssl_batch_size=8,
    ssl_accum=1,
    batch_days=8,
    policy_steps=1000,
    max_stock_weight=0.10,
)

_TOP50_H100_SERIES = [
    # Core causal grid/memory dose-response.  These four are the primary,
    # interpretable comparison and are ordered first so the scheduler can fill
    # four H100s immediately after cache construction.
    replace(
        _top50_h100_base,
        name="top50_h100_42_1s",
        note="TOP50 H100 grid control: 1s grid, 42d memory/raw, equal-exposure B8",
        bar_seconds=1,
        episode_len=42,
        episode_stride=5,
        bptt_window=42,
        daily_lookback=42,
        raw_recent_days=0,
        ssl_batch_size=2,
        ssl_accum=4,
        policy_steps=3000,
        ssl_perstock_coef=0.0,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_42_60s",
        note="TOP50 H100 fast control: 60s grid, 42d memory/raw, equal-exposure B8",
        episode_len=42,
        episode_stride=5,
        bptt_window=42,
        daily_lookback=42,
        raw_recent_days=0,
        policy_steps=3000,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_126",
        note="TOP50 H100 memory dose: 60s grid, 126d memory, raw recent 42d",
        episode_len=126,
        episode_stride=10,
        bptt_window=42,
        daily_lookback=126,
        raw_recent_days=42,
        policy_steps=1500,
    ),
    _top50_h100_base,

    # Representation/capacity ablations.  Only these settings alter the model
    # size, keeping the primary memory comparison at the audited base capacity.
    replace(
        _top50_h100_base,
        name="top50_h100_252_small",
        note="TOP50 capacity ablation: d256/4L, raw64/1L, token128/2L",
        d_model=256,
        enc_layers=4,
        enc_heads=8,
        raw_policy_dim=64,
        raw_policy_layers=1,
        raw_policy_heads=4,
        policy_token_dim=128,
        policy_layers=2,
        policy_heads=8,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_large",
        note="TOP50 capacity ablation: d512/8L, raw192/3L, token384/4L",
        d_model=512,
        enc_layers=8,
        enc_heads=8,
        raw_policy_dim=192,
        raw_policy_layers=3,
        raw_policy_heads=8,
        policy_token_dim=384,
        policy_layers=4,
        policy_heads=8,
    ),

    # Sampling-resolution ablations around the 60-second primary.  The 15s
    # variant tests whether sub-minute shape helps; 300s removes within-block
    # detail while preserving the same 78 daily context blocks.
    replace(
        _top50_h100_base,
        name="top50_h100_252_15s",
        note="TOP50 resolution ablation: 15s grid, 252d memory",
        bar_seconds=15,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_300s",
        note="TOP50 resolution ablation: one bar per 5m block, 252d memory",
        bar_seconds=300,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_block1m",
        note="TOP50 context-cadence ablation: 1m blocks on a 1m grid",
        block_seconds=60,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_block15m",
        note="TOP50 context-cadence ablation: 15m blocks on a 1m grid",
        block_seconds=900,
    ),

    # Auxiliary-label and economic sensitivity.  The policy reward remains the
    # canonical one-day transition; H changes only Stage-1's forward target.
    replace(
        _top50_h100_base,
        name="top50_h100_252_h5",
        note="TOP50 auxiliary-horizon ablation: H=5d",
        label_horizon_days=5,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_h63",
        note="TOP50 auxiliary-horizon ablation: H=63d",
        label_horizon_days=63,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_risk0",
        note="TOP50 objective sensitivity: no downside penalty",
        risk_lambda=0.0,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_risk25",
        note="TOP50 objective sensitivity: downside penalty 0.25",
        risk_lambda=0.25,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_cost2bp",
        note="TOP50 friction sensitivity: 2bp per one-way turnover",
        cost=2e-4,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_cost10bp",
        note="TOP50 friction sensitivity: 10bp per one-way turnover",
        cost=1e-3,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_temp1",
        note="TOP50 concentration sensitivity: allocation temperature 1.0",
        temperature=1.0,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_cap20",
        note="TOP50 concentration sensitivity: 20% risky-name cap",
        max_stock_weight=0.20,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_uncapped",
        note="TOP50 signal-only concentration control: risky-name cap disabled",
        max_stock_weight=1.0,
    ),
    replace(
        _top50_h100_base,
        name="top50_h100_252_raw21",
        note="TOP50 raw-history sensitivity: trainable raw recent 21d",
        raw_recent_days=21,
    ),
]
_SERIES.extend(_TOP50_H100_SERIES)

# ===== TOP2000 / four independent H100s, one seed per setting ===============
#
# Each TOP2000 setting uses one H100; the pool runs up to four settings at once.
# All settings retain the full 252-day episode and differ by one declared factor.
# The first ten form the core policy/optimizer/trading screen and share one
# expensive Stage-1 context exactly.  Wide-only bar/block/label ablations are
# intentionally last because each requires a distinct cache and/or SSL run.
#
# `max_actions_per_day` is legacy terminology.  In daily_raw, the loss divides
# it by episode length, so it is the target number of reallocations per 252-day
# training episode.  The primary value 12 is monthly-ish; 5 and 52 are explicit
# slow/weekly sensitivities.  Cost settings are robustness studies and must not
# be ranked against each other on their differently costed net-return metric.
_top2000_h100_base = next(d for d in _SERIES if d.name == "daily_raw_top2000")

# Hold-30 is deliberately registered outside the future-selected TOP2000
# sweeps.  The v2 workflow must bind a monthly point-in-time active-300 axis;
# this design borrows only the already-proven compact H100 tensor geometry.  Its
# stateful chronological runtime consumes ``scored_tail_days`` and the explicit
# holding contract; feeding this design to the legacy cash-reset sweep would
# violate that contract.  Two H100s are the declared per-setting allocation.
_daily_raw_pit300_hold30 = replace(
    _top2000_h100_base,
    name="daily_raw_pit300_hold30",
    note="PIT-300 Hold-30 v2: 63d controlled credit, explicit 30-session soft holding contract",
    # This is a mechanism screen over fewer than two thousand independent
    # pre-lockbox dates, not a capacity sweep.  Do not inherit TOP2000's
    # d512/8L Stage-1 encoder: the v1 specification freezes width-128 causal
    # context blocks, two layers, four heads, and a width-256 feed-forward
    # sublayer for every mechanism.
    d_model=128,
    enc_layers=2,
    enc_heads=4,
    dropout=0.0,
    policy_steps=128,
    pol_lr=1e-4,
    pol_weight_decay=1e-4,
    schedule="constant",
    grad_clip=0.5,
    cost=2e-3,
    raw_policy_dim=64,
    raw_policy_layers=2,
    raw_policy_heads=4,
    policy_token_dim=128,
    policy_layers=2,
    policy_heads=4,
    episode_stride=0,
    bptt_window=63,
    label_horizon_days=30,
    # V1/v2 has one 30-session representation target.  Multi-horizon heads
    # remain a later registered ablation and cannot enter this mechanism screen.
    auxiliary_horizons=(),
    scored_tail_days=63,
    target_holding_days=30,
    target_discretionary_turnover=1.0 / 30.0,
    daily_lookback=63,
    terminal_liquidate=False,
    budget_lambda=0.0,
    gate_init_bias=-3.3844844191,
    gate_entropy_coef=0.0,
    min_gpus=2,
)

_TOP2000_H100_VARIANTS = [
    replace(
        _top2000_h100_base,
        name="top2000_h100_policy_small",
        note="TOP2000 policy-capacity ablation: raw64/1L and token64/1L",
        raw_policy_dim=64,
        raw_policy_layers=1,
        raw_policy_heads=4,
        policy_token_dim=64,
        policy_layers=1,
        policy_heads=4,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_policy_large",
        note="TOP2000 policy-capacity ablation: raw192/3L and token192/3L",
        raw_policy_dim=192,
        raw_policy_layers=3,
        raw_policy_heads=8,
        policy_token_dim=192,
        policy_layers=3,
        policy_heads=8,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_lr1e4",
        note="TOP2000 policy learning-rate ablation: 1e-4",
        pol_lr=1e-4,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_lr6e4",
        note="TOP2000 policy learning-rate ablation: 6e-4",
        pol_lr=6e-4,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_bptt5",
        note="TOP2000 credit-horizon ablation: five-day truncated BPTT within the 21d scored tail",
        bptt_window=5,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_actions26",
        note="TOP2000 turnover-budget ablation: 26 target reallocations per 252d episode",
        max_actions_per_day=26.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_actions5",
        note="TOP2000 turnover-budget ablation: five target reallocations per 252d episode",
        max_actions_per_day=5.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_actions52",
        note="TOP2000 turnover-budget ablation: 52 target reallocations per 252d episode",
        max_actions_per_day=52.0,
    ),
    # Net return already charges one-way turnover. Put the cost-only control in
    # the core screen so the scheduler tests the objective-aligned alternative
    # before secondary concentration sensitivities.
    replace(
        _top2000_h100_base,
        name="top2000_h100_budget0",
        note="TOP2000 turnover-control ablation: transaction cost only (no soft gate-rate penalty)",
        budget_lambda=0.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_cap005",
        note="TOP2000 concentration ablation: 0.5% hard risky-name cap",
        max_stock_weight=0.005,
    ),

    # Wide policy/objective sensitivities.  These still reuse the primary
    # frozen context; only Stage 2 is rerun.
    replace(
        _top2000_h100_base,
        name="top2000_h100_raw21",
        note="TOP2000 raw-history ablation: trainable raw encoder on recent 21d",
        raw_recent_days=21,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_raw84",
        note="TOP2000 raw-history ablation: trainable raw encoder on recent 84d",
        raw_recent_days=84,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_cap02",
        note="TOP2000 concentration ablation: 2% hard risky-name cap",
        max_stock_weight=0.02,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_uncapped",
        note="TOP2000 signal-only concentration control: risky-name cap disabled",
        max_stock_weight=1.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_risk0",
        note="TOP2000 downside-objective ablation: no downside penalty",
        risk_lambda=0.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_risk25",
        note="TOP2000 downside-objective ablation: downside penalty 0.25",
        risk_lambda=0.25,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_cost5bp",
        note="TOP2000 friction sensitivity: 5bp per one-way turnover",
        cost=5e-4,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_cost20bp",
        note="TOP2000 friction sensitivity: 20bp per one-way turnover",
        cost=2e-3,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_temp1",
        note="TOP2000 allocation-temperature ablation: temperature 1.0",
        temperature=1.0,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_entropy1e5",
        note="TOP2000 allocation-entropy ablation: coefficient 1e-5",
        entropy_coef=1e-5,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_wd10",
        note="TOP2000 policy regularization ablation: AdamW decay 0.10",
        pol_weight_decay=0.10,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_wd60",
        note="TOP2000 policy regularization ablation: AdamW decay 0.60",
        pol_weight_decay=0.60,
    ),

    # Expensive representation/target sensitivities.  These alter Stage 1 and
    # therefore intentionally form separate context-cache groups.
    replace(
        _top2000_h100_base,
        name="top2000_h100_bar300",
        note="TOP2000 resolution ablation: one 5-minute bar per context block",
        bar_seconds=300,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_block15m",
        note="TOP2000 context-cadence ablation: 15-minute blocks on the 1-minute grid",
        block_seconds=900,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_h5",
        note="TOP2000 auxiliary-target ablation: five-day relative return",
        label_horizon_days=5,
    ),
    replace(
        _top2000_h100_base,
        name="top2000_h100_h63",
        note="TOP2000 auxiliary-target ablation: 63-day relative return",
        label_horizon_days=63,
    ),
]
_TOP2000_H100_SERIES = [_top2000_h100_base, *_TOP2000_H100_VARIANTS]
_SERIES.extend(_TOP2000_H100_VARIANTS)
_SERIES.append(_daily_raw_pit300_hold30)

DESIGNS: dict[str, Phase1Design] = {d.name: d for d in _SERIES}

DEFAULT_DESIGN = "daily_raw"
# The old intraday TOP50 sweep is a documented null regime.  The current default
# is a broad, ONE-SEED screening study intended for four independent H100 jobs.
# It is exploratory: promote no winner until a fresh point-in-time universe,
# walk-forward selection, and multi-seed confirmation have succeeded.
TOP50_H100_CORE_SWEEP = [d.name for d in _TOP50_H100_SERIES[:4]]
TOP50_H100_WIDE_SWEEP = [d.name for d in _TOP50_H100_SERIES]
TOP2000_H100_CORE_SWEEP = [d.name for d in _TOP2000_H100_SERIES[:10]]
TOP2000_H100_WIDE_SWEEP = [d.name for d in _TOP2000_H100_SERIES]
HOLD30_BASE_DESIGN = _daily_raw_pit300_hold30.name
SWEEP = TOP2000_H100_WIDE_SWEEP
# Longer-horizon probes (run explicitly with --design; not in the default sweep).
HORIZON_SWEEP = ["h30m", "h65m"]
