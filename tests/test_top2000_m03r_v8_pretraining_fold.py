from __future__ import annotations

import datetime as dt
import hashlib
import json

import torch

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_SETTINGS,
)
from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03RV8DevelopmentTrainingPlan,
)
from rl_quant.training.top2000_m03r_v8_policy import (
    Top2000M03RV8DevelopmentPolicy,
)
from rl_quant.training.top2000_m03r_v8_pretraining_fold import (
    evaluate_m03r_v8_pretraining_fold,
)
from rl_quant.training.top2000_m03r_v8_pretraining_step import (
    model_state_sha256,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _cache() -> Top2000VerifiedDevelopmentCache:
    rows = 1001
    assets = 4
    dates = tuple(
        (dt.date(2022, 1, 1) + dt.timedelta(days=index)).isoformat()
        for index in range(rows)
    )
    actions = ("CASH", "A", "B", "C")
    daily = torch.ones(rows, assets, 5)
    trend = torch.arange(rows, dtype=torch.float32).view(-1, 1) * 0.0001
    daily[:, 1, :4] += trend
    daily[:, 2, :4] += trend * 0.5
    daily[:, 3, :4] -= trend * 0.25
    daily[..., 4] = 1000.0
    daily[:, 0, :4] = 1.0
    daily[:, 0, 4] = 0.0
    return Top2000VerifiedDevelopmentCache(
        daily_ohlcv=daily,
        availability=torch.ones(rows, assets, dtype=torch.bool),
        exchange_dates=dates,
        action_ids=actions,
        cache_sha256="a" * 64,
        cache_identity="b" * 64,
        search_identity="c" * 64,
        action_hash=_digest(list(actions)),
        bar_seconds=300,
        acknowledgement=DEVELOPMENT_ACK,
        development_only=True,
        bars_only=True,
    )


def test_inner_validation_uses_training_tail_without_model_mutation() -> None:
    plan = M03RV8DevelopmentTrainingPlan(
        setting_index=0,
        setting_id=M03R_V8_TOP2000_DEV_SETTINGS[0].setting_id,
        cache_path="/immutable/cache.pt",
        cache_sha256="a" * 64,
        output_root="/immutable/output",
        source_manifest_sha256="d" * 64,
    )
    policy = Top2000M03RV8DevelopmentPolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    before = model_state_sha256(policy)
    evidence = evaluate_m03r_v8_pretraining_fold(
        _cache(),
        plan,
        render_top2000_m03r_v7_development_folds(1001)[0],
        policy,
        device=torch.device("cpu"),
    )

    assert evidence.fold_index == 0
    assert evidence.valid_date_counts == (63, 63, 63, 63)
    assert evidence.receipt_sha256
    assert model_state_sha256(policy) == before
