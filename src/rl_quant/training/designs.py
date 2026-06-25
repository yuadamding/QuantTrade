"""A series of model + training-strategy DESIGNS for the Phase-1 two-stage, EVENT-TIMED framework.

Each design fully specifies BOTH transformers' architecture AND the training strategy/setup for BOTH stages:

  ARCHITECTURE
    context (two-tier causal transformer over RAW seconds): session_seconds (full RTH session encoded once),
      block_seconds (tier-1 block length = the candidate/decision cadence), d_model / enc_layers / enc_heads.
    policy (set-transformer): policy_token_dim / policy_layers / policy_heads.
  TRAINING STRATEGY / SETUP (per stage where it differs)
    budget: ssl_steps, policy_steps, ssl_batch_size (DAYS per SSL micro-batch), ssl_accum (grad-accum),
      batch_days (days per policy step).
    optimization: ssl_lr / pol_lr, ssl_weight_decay / pol_weight_decay, ssl_warmup_frac / pol_warmup_frac,
      schedule ('cosine' warmup->decay, or 'constant'), grad_clip, amp (bf16 autocast), grad_checkpoint.
    policy objective: cost (turnover), risk_lambda (downside), entropy_coef (allocation exploration),
      temperature (allocation sharpness), max_actions_per_day (soft trade budget) + budget_lambda (its penalty),
      and the CASH-basin escape knobs -- gate_init_bias (start trading), gate_entropy_coef (gate exploration),
      friction_warmup_frac (ramp cost+budget 0->full so the edge is learned before friction bites), and
      ssl_perstock_coef (Stage-1 cross-sectional pretext weight).

EVENT-TIMED: the policy is NOT on a fixed decision clock. The encoder turns each full session into a context at
EVERY `block_seconds` block (78 blocks/day at 300s); the policy chooses WHEN to trade (a per-block act-gate) under
a SOFT per-day budget of ~`max_actions_per_day` trades, and trades execute T+1. So the candidate grid is the
encoder's blocks -- there is no separate per-candidate storage; one full-session encode yields every context.

The series no longer varies "lookback": the two-tier hierarchy reaches the WHOLE session by design, so every real
design encodes the full RTH session and instead varies (context arch x policy arch x block cadence x training
strategy x trade budget). `large` is the smaller MINIMUM/floor. Full-session SSL is dominated by the tier-1
activations; `grad_checkpoint` (recompute tier-1 in backward) + `amp` (bf16) keep ONE day/micro-batch within an
80 GB H100 up to ~d512, and `ssl_accum` builds the effective target-batch at fixed peak VRAM. Verify with
nvidia-smi and tune ssl_accum / ssl_batch_size. The 2xH100 sweep runs two designs at once (one per GPU).
"""
from __future__ import annotations

from dataclasses import dataclass


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
    ssl_accum: int = 8               # grad-accum: effective SSL target-batch = ssl_batch_size * ssl_accum days
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
    max_actions_per_day: float = 5.0  # SOFT per-day trade budget (the policy gates WHEN to act)
    budget_lambda: float = 0.1        # penalty on the per-day act-gate RATE exceeding max_actions_per_day/nB
    gate_init_bias: float = 2.0       # initial act-gate logit (sigmoid(2)=0.88): start TRADING, not in CASH
    gate_entropy_coef: float = 1e-3   # Bernoulli gate-entropy bonus -> exploration on WHEN to trade
    friction_warmup_frac: float = 0.3 # ramp turnover cost + budget penalty 0->full over this frac of policy_steps
    ssl_perstock_coef: float = 1.0    # weight of the per-stock cross-sectional SSL pretext (relative-value signal)
    amp: bool = False                 # bf16 autocast (frees ~44% activation -> bigger batch at same VRAM)
    grad_checkpoint: bool = False     # recompute tier-1 in backward (needed for full-session SSL at d>=384)

    def __post_init__(self) -> None:
        if self.d_model % self.enc_heads:
            raise ValueError(f"{self.name}: enc_heads {self.enc_heads} must divide d_model {self.d_model}")
        if self.policy_token_dim % self.policy_heads:
            raise ValueError(f"{self.name}: policy_heads {self.policy_heads} must divide "
                             f"policy_token_dim {self.policy_token_dim}")
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"{self.name}: schedule must be 'cosine' or 'constant'")
        if self.temperature <= 0:
            raise ValueError(f"{self.name}: temperature must be > 0")
        if self.session_seconds % self.block_seconds:
            raise ValueError(f"{self.name}: block_seconds {self.block_seconds} must divide "
                             f"session_seconds {self.session_seconds}")
        if self.max_actions_per_day <= 0 or self.budget_lambda < 0:
            raise ValueError(f"{self.name}: need max_actions_per_day>0 and budget_lambda>=0")
        for f in ("session_seconds", "block_seconds", "ssl_steps", "policy_steps", "ssl_batch_size",
                  "ssl_accum", "batch_days"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{self.name}: {f} must be positive")


FULL = 23400  # full RTH session (09:30->16:00) in seconds; 78 blocks at 300s
_SERIES = [
    # tiny: CPU smoke / CI only (short session, 4 blocks of 30s).
    Phase1Design("tiny", "smoke/CI only", session_seconds=120, block_seconds=30, d_model=24, enc_layers=1,
                 enc_heads=2, policy_token_dim=24, policy_layers=1, policy_heads=2, ssl_steps=40, policy_steps=60,
                 ssl_batch_size=2, ssl_accum=1, batch_days=4, dropout=0.0, max_actions_per_day=2.0),

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

    Phase1Design("balanced", "BALANCED d512/10L, full session, budget 5; entropy 0.01, temp 1.5, cost 1e-3",
                 session_seconds=FULL, block_seconds=300, d_model=512, enc_layers=10, enc_heads=8,
                 policy_token_dim=512, policy_layers=6, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=16, batch_days=64, entropy_coef=0.01, temperature=1.5, cost=1e-3,
                 amp=True, grad_checkpoint=True),

    Phase1Design("coarse_blocks", "COARSE 600s blocks (39/day), d384/8L, full session; constant lr, risk 0.2",
                 session_seconds=FULL, block_seconds=600, d_model=384, enc_layers=8, enc_heads=8,
                 policy_token_dim=512, policy_layers=4, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=8, batch_days=48, schedule="constant", pol_weight_decay=5e-2,
                 risk_lambda=0.2, amp=True, grad_checkpoint=True),

    Phase1Design("active", "ACTIVE budget 8 (looser budget_lambda 0.05), d512/10L, full session; bf16, entropy",
                 session_seconds=FULL, block_seconds=300, d_model=512, enc_layers=10, enc_heads=8,
                 policy_token_dim=640, policy_layers=6, policy_heads=8, ssl_steps=3000, policy_steps=8000,
                 ssl_batch_size=1, ssl_accum=16, batch_days=64, amp=True, grad_checkpoint=True, ssl_lr=2.5e-4,
                 entropy_coef=0.02, max_actions_per_day=8.0, budget_lambda=0.05),
]
DESIGNS: dict[str, Phase1Design] = {d.name: d for d in _SERIES}

DEFAULT_DESIGN = "wide"
# Variety run on the 2xH100 box, with `large` as the smaller minimum. ('tiny' = CPU smoke only.)
SWEEP = ["large", "wide", "deep", "balanced", "coarse_blocks", "active"]
