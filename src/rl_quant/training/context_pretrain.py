"""Stage 1 — self-supervised CONTEXT learning for the second->hour market-context encoder.

Trains ``SecondToHourContextEncoder`` (policy-free) with ``SecondContextForwardHead`` to predict, from each
decision's hourly market embedding, the **next-period whole-market move + dispersion** — a label derived from the
data itself (no trading reward, no policy). The encoder is then frozen and its embeddings precomputed
(``encode_split``) as the compact input to Stage-2 decision-policy learning (``training.decision_policy``).

Decoupling rationale: market context is the same regardless of what you hold, so it is learned independently of
the policy; this also amortizes the heavy per-second data — you pay it once here, then iterate policy cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rl_quant.datasets.hour_from_second import HourFromSecondDataSplit
from rl_quant.models.second_to_hour import SecondContextForwardHead, SecondToHourContextEncoder


@dataclass(frozen=True)
class ContextPretrainConfig:
    # Settings tuned for the small per-window data regime (~150-200 train rows): a compact context encoder
    # learned on the DENSE self-supervised market-move target, regularized to generalize OOS. The SSL target
    # is dense (every row), so more epochs help WITHOUT the joint trainer's reward-overfit.
    d_model: int = 128
    n_heads: int = 4
    second_layers: int = 1
    hour_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.15
    max_second_tokens: int | None = 128
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    huber_beta: float = 1.0


def forward_market_targets(
    action_returns: torch.Tensor, valid_mask: torch.Tensor, *, cash_index: int = 0
) -> torch.Tensor:
    """Self-supervised target ``[B, 2]`` per decision: (equal-weighted mean, dispersion) of the realized
    next-period returns across the **non-CASH** universe. A market-move + volatility proxy derived purely
    from labels already in the dataset — no new inputs. NaN (invalid) cells are masked out. ``cash_index``
    is the column to exclude (derived from ``action_names.index('CASH')``, not assumed to be 0)."""
    non_cash = torch.ones(action_returns.shape[1], dtype=torch.bool, device=action_returns.device)
    non_cash[cash_index] = False
    r = action_returns[:, non_cash]
    m = valid_mask[:, non_cash].to(r.dtype)
    count = m.sum(dim=1).clamp_min(1.0)
    r0 = torch.nan_to_num(r, nan=0.0) * m
    mean = r0.sum(dim=1) / count
    var = (((torch.nan_to_num(r, nan=0.0) - mean[:, None]) ** 2) * m).sum(dim=1) / count
    return torch.stack([mean, var.clamp_min(0.0).sqrt()], dim=1)


def train_second_context_encoder(
    train_data: HourFromSecondDataSplit,
    config: ContextPretrainConfig = ContextPretrainConfig(),
    *,
    device: torch.device | str = "cpu",
) -> tuple[SecondToHourContextEncoder, SecondContextForwardHead, dict]:
    """Fit the policy-free context encoder + SSL head on the per-second context split. Returns the trained
    encoder (to be frozen), the head, and a small metrics dict."""
    data = train_data.to(device)
    encoder = SecondToHourContextEncoder(
        second_feature_dim=int(data.second_features.shape[-1]),
        hour_feature_dim=int(data.hour_features.shape[-1]),
        hours_lookback=data.hours_lookback, seconds_per_hour=data.seconds_per_hour,
        d_model=config.d_model, n_heads=config.n_heads, second_layers=config.second_layers,
        hour_layers=config.hour_layers, feedforward_dim=config.feedforward_dim, dropout=config.dropout,
        max_second_tokens=config.max_second_tokens,
    ).to(device)
    head = SecondContextForwardHead(d_model=config.d_model).to(device)
    opt = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    cash_index = data.action_names.index("CASH")
    idx_all = data.valid_start_indices.to(device).long()
    first, last = 0.0, 0.0
    for epoch in range(config.epochs):
        perm = idx_all[torch.randperm(idx_all.shape[0], device=device)]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, perm.shape[0], config.batch_size):
            idx = perm[start:start + config.batch_size]
            second, mask, hour = data.state(idx)
            target = forward_market_targets(
                data.action_returns[idx], data.label_valid_actions(idx), cash_index=cash_index
            )
            pred = head(encoder(second, mask, hour))
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        if epoch == 0:
            first = mean_loss
        last = mean_loss
    return encoder, head, {"first_epoch_loss": first, "last_epoch_loss": last, "epochs": config.epochs}


@torch.no_grad()
def encode_split(
    encoder: SecondToHourContextEncoder, data: HourFromSecondDataSplit, *, batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Precompute the frozen hourly context embedding ``[D, d_model]`` for every decision row -- the compact
    Stage-2 (decision-policy) input. Indexed by absolute decision row so the policy can gather per episode."""
    encoder = encoder.to(device).eval()
    data = data.to(device)
    rows = int(data.second_features.shape[0])
    out = torch.empty((rows, encoder.d_model), dtype=torch.float32, device=device)
    for start in range(0, rows, batch_size):
        idx = torch.arange(start, min(start + batch_size, rows), device=device)
        second, mask, hour = data.state(idx)
        out[idx] = encoder(second, mask, hour).float()
    return out
