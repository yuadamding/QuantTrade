"""Learning-framework models, split into the two decoupled stages of the design.

  * context_encoder -- Stage 1 CONTEXT LEARNING: a causal-attention transformer over raw-second chunk tokens,
    trained self-supervised then FROZEN. Pure market state (no policy concept).
  * decision_policy -- Stage 2 POLICY LEARNING: a permutation-equivariant set-transformer over action tokens,
    trained on the FROZEN context. All policy machinery (previous position, constraints, allocation) lives here.

The split is structural: decision_policy holds no encoder reference, so policy gradients cannot reach the
context encoder. Training the policy on cached frozen embeddings (see rl_quant.training) makes that literal.
"""
from __future__ import annotations

from rl_quant.models.context_encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    ContextForwardHead,
    PerStockForwardHead,
)
from rl_quant.models.decision_policy import DecisionPolicyConfig, DecisionPolicyHead

__all__ = [
    "ContextEncoder",
    "ContextEncoderConfig",
    "ContextForwardHead",
    "DecisionPolicyConfig",
    "DecisionPolicyHead",
    "PerStockForwardHead",
]
