"""A series of model-setting DESIGNS for the Phase-1 two-stage framework.

Each design parameterizes BOTH learning stages and their training budget:
  * Stage 1 (context): the causal-attention transformer -- chunk granularity (chunk_sec x max_chunks ~ how much
    of the session it rolls over) and capacity (d_model / enc_layers / enc_heads).
  * Stage 2 (policy): the permutation-equivariant set-transformer -- token_dim / policy_layers / policy_heads.
  * Training: SSL + policy steps, the SSL minibatch (effective batch = ssl_batch_size x n_actions), the number
    of windows per policy step, lr, and the portfolio cost/risk knobs.

The series spans capacity AND context granularity so the sweep explores both axes. Sizes are chosen so each
design fits comfortably on ONE 80 GB H100 (rough peak VRAM in `note`); the 2xH100 sweep runs two designs/seeds
at once (see scripts/sweep_phase1.py) -> up to 160 GB of aggregate VRAM in use. Heads always divide their dim.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Design:
    name: str
    note: str
    # Stage 1 -- context (causal transformer)
    chunk_sec: int
    max_chunks: int
    d_model: int
    enc_layers: int
    enc_heads: int
    # Stage 2 -- policy (set-transformer)
    policy_token_dim: int
    policy_layers: int
    policy_heads: int
    dropout: float
    # training budget
    ssl_steps: int
    policy_steps: int
    ssl_batch_size: int
    batch_windows: int
    lr: float
    cost: float
    risk_lambda: float

    def __post_init__(self) -> None:
        if self.d_model % self.enc_heads:
            raise ValueError(f"{self.name}: enc_heads {self.enc_heads} must divide d_model {self.d_model}")
        if self.policy_token_dim % self.policy_heads:
            raise ValueError(f"{self.name}: policy_heads must divide policy_token_dim")
        for f in ("chunk_sec", "max_chunks", "ssl_steps", "policy_steps", "ssl_batch_size", "batch_windows"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{self.name}: {f} must be positive")


_SERIES = [
    # name     note                                          ch_s ch_n  d   eL eH   pT pL pH  drop  sslS  polS  sslB bw   lr    cost  risk
    Phase1Design("tiny",  "smoke/CI only (<1 GB)",            60,  12,  24, 1, 2,   24, 1, 2, 0.0,    40,    60,   8,  4, 3e-4, 5e-4, 0.10),
    Phase1Design("small", "fast baseline (~4-6 GB)",         300,  80, 128, 2, 4,  128, 2, 4, 0.0,  8000,  8000,  32, 16, 3e-4, 5e-4, 0.10),
    Phase1Design("base",  "default serious run (~12-20 GB)", 300,  80, 256, 4, 8,  256, 3, 8, 0.0, 15000, 20000,  24, 24, 2e-4, 5e-4, 0.10),
    Phase1Design("large", "finer context (~30-45 GB)",       180, 130, 512, 6, 8,  384, 4, 8, 0.1, 25000, 30000,  12, 32, 2e-4, 5e-4, 0.10),
    Phase1Design("xlarge","session-fine, high cap (~55-75 GB)",120,200, 768, 8,12,  512, 6, 8, 0.1, 40000, 40000,   8, 32, 1.5e-4,5e-4,0.10),
]
DESIGNS: dict[str, Phase1Design] = {d.name: d for d in _SERIES}

DEFAULT_DESIGN = "base"
# The series to run on a 2xH100 box (small..xlarge). 'tiny' is excluded (it is for the CPU smoke / CI only).
SWEEP = ["small", "base", "large", "xlarge"]
