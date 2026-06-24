"""A series of model + training-strategy DESIGNS for the Phase-1 two-stage framework.

Each design fully specifies BOTH transformers' architecture AND the training strategy/setup for BOTH stages:

  ARCHITECTURE
    context (two-tier causal transformer over RAW seconds): max_seconds (causal lookback, rolls from open),
      d_model / enc_layers / enc_heads (block_seconds is the tier-1 block length, set on the encoder config).
    policy (set-transformer): policy_token_dim / policy_layers / policy_heads.
  TRAINING STRATEGY / SETUP (per stage where it differs)
    budget: ssl_steps, policy_steps, ssl_batch_size (effective SSL batch = ssl_batch_size x 51 actions),
      batch_windows (windows per policy step).
    optimization: ssl_lr / pol_lr, ssl_weight_decay / pol_weight_decay, ssl_warmup_frac / pol_warmup_frac,
      schedule ('cosine' warmup->decay, or 'constant'), grad_clip, amp (bf16 autocast).
    policy objective: cost (turnover), risk_lambda (downside penalty), entropy_coef (exploration),
      temperature (allocation sharpness).

The series is an ISO-VRAM SWEEP: every design targets ~75 GB on one 80 GB H100, but explores a DIFFERENT point
in (context arch x policy arch x training strategy) space. `large` (~22 GB) is the smaller MINIMUM/floor. VRAM
was sized from a calibrated measurement (method reproduces the real `large` ~= 22 GB); ssl_batch_size is the
per-design knob that lands ~75 GB. The 2xH100 sweep runs two designs at once (one per GPU) -> ~150 GB in use.
Verify with nvidia-smi and nudge ssl_batch_size +/-1 if a card's headroom differs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Design:
    name: str
    note: str
    # --- architecture ---
    max_seconds: int                 # raw-second causal lookback (one token per second)
    d_model: int
    enc_layers: int
    enc_heads: int
    policy_token_dim: int
    policy_layers: int
    policy_heads: int
    # --- training budget ---
    ssl_steps: int                   # SSL OPTIMIZER steps (each = ssl_accum micro-batches)
    policy_steps: int
    ssl_batch_size: int              # decision-rows per micro-batch; peak VRAM = ssl_batch_size * 51 stocks
    batch_windows: int
    ssl_accum: int = 8               # grad-accum: effective SSL target-batch = ssl_batch_size * ssl_accum
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
    amp: bool = False                 # bf16 autocast (frees ~44% activation -> bigger batch at same VRAM)

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
        for f in ("max_seconds", "ssl_steps", "policy_steps", "ssl_batch_size", "ssl_accum", "batch_windows"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{self.name}: {f} must be positive")


# The context encoder reads RAW 1-second bars DIRECTLY (one token/second) + covariates; `max_seconds` is the
# causal lookback in real seconds. Sizing is GPU-MEASURED (scratchpad raw-second probe): peak VRAM ~ ssl_batch_size
# * 51 stocks * ~0.16 MB * max_seconds. On one 80 GB H100 that caps ssl_batch_size at ~2 (S=3600) / 1 (S>=5400);
# 51 stocks * 10800s already ~87 GB so a 3h lookback is INFEASIBLE and was dropped. The tiny per-step batch is
# compensated by GRADIENT ACCUMULATION (ssl_accum) -> a healthy SSL target-batch at fixed peak VRAM. SSL steps are
# OPTIMIZER steps; each does ssl_accum micro-batches of (ssl_batch_size * 51) raw-second sequences, so they are far
# costlier than the old chunk-token steps -> step counts are right-sized down accordingly. Verify with nvidia-smi
# and tune ssl_batch_size / ssl_accum / max_seconds. The variety spans lookback x context arch x policy arch x
# training strategy; `large` is the smaller minimum.
_SERIES = [
    # tiny: CPU smoke / CI only.
    Phase1Design("tiny", "smoke/CI only", max_seconds=120, d_model=24, enc_layers=1, enc_heads=2,
                 policy_token_dim=24, policy_layers=1, policy_heads=2, ssl_steps=40, policy_steps=60,
                 ssl_batch_size=4, ssl_accum=1, batch_windows=4, dropout=0.0),

    # large: the MINIMUM -- short lookback, modest model (S=1800 allows a bigger ssl_batch).
    Phase1Design("large", "MINIMUM: 1800s lookback, d256/4L", max_seconds=1800, d_model=256, enc_layers=4,
                 enc_heads=8, policy_token_dim=256, policy_layers=4, policy_heads=8, ssl_steps=3000,
                 policy_steps=8000, ssl_batch_size=4, ssl_accum=8, batch_windows=32),

    # ===== variety: vary lookback (max_seconds) x context arch x policy arch x training strategy =====
    Phase1Design("wide", "WIDE d512/8L, 3600s lookback; standard cosine", max_seconds=3600, d_model=512,
                 enc_layers=8, enc_heads=8, policy_token_dim=512, policy_layers=4, policy_heads=8,
                 ssl_steps=3000, policy_steps=8000, ssl_batch_size=2, ssl_accum=8, batch_windows=48),

    Phase1Design("deep", "DEEP-NARROW d384/16L, 3600s; warmup-heavy, clip 0.5, lr 1.5e-4", max_seconds=3600,
                 d_model=384, enc_layers=16, enc_heads=8, policy_token_dim=384, policy_layers=6, policy_heads=8,
                 ssl_steps=3500, policy_steps=8000, ssl_batch_size=1, ssl_accum=16, batch_windows=48,
                 ssl_lr=1.5e-4, ssl_warmup_frac=0.10, pol_warmup_frac=0.10, grad_clip=0.5),

    Phase1Design("balanced", "BALANCED d512/10L, 3600s; entropy 0.01, temp 1.5, cost 1e-3", max_seconds=3600,
                 d_model=512, enc_layers=10, enc_heads=8, policy_token_dim=512, policy_layers=6, policy_heads=8,
                 ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=16, batch_windows=64,
                 entropy_coef=0.01, temperature=1.5, cost=1e-3),

    Phase1Design("long_ctx", "LONG 7200s lookback (24 blocks), d384/8L; constant lr, wd 5e-2, risk 0.2",
                 max_seconds=7200, d_model=384, enc_layers=8, enc_heads=8, policy_token_dim=512, policy_layers=4,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=8, batch_windows=48,
                 schedule="constant", pol_weight_decay=5e-2, risk_lambda=0.2),

    # FULL SESSION: the two-tier hierarchy makes the whole ~6.5h RTH session tractable (78 blocks x 300s). bf16.
    # NOTE: raw bars for the full session are ~134 GB across 374 windows -> needs a big-RAM box (or lazy loading).
    Phase1Design("full_session", "FULL ~23400s session (78 blocks), d512/10L, bf16 AMP, entropy",
                 max_seconds=23400, d_model=512, enc_layers=10, enc_heads=8, policy_token_dim=640, policy_layers=6,
                 policy_heads=8, ssl_steps=3000, policy_steps=8000, ssl_batch_size=1, ssl_accum=16, batch_windows=64,
                 amp=True, ssl_lr=2.5e-4, entropy_coef=0.02),
]
DESIGNS: dict[str, Phase1Design] = {d.name: d for d in _SERIES}

DEFAULT_DESIGN = "wide"
# Variety run on the 2xH100 box, with `large` as the smaller minimum. ('tiny' = CPU smoke only.)
SWEEP = ["large", "wide", "deep", "balanced", "long_ctx", "full_session"]
