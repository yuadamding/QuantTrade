from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch

RISK_AWARE_POLICY_MODEL_VERSION = 3

EXPOSURE_FEATURE_NAMES = [
    "steps_today_over_episode_length",
    "leveraged_bars_today_over_cap",
    "consecutive_leveraged_bars_over_cap",
    "current_effective_leverage",
    "current_action_is_leveraged",
    "current_group_share_today",
]
EXPOSURE_FEATURE_DIM = len(EXPOSURE_FEATURE_NAMES)


@dataclass(frozen=True)
class ActionMeta:
    name: str
    asset_class: str
    group: str
    underlying: str | None
    leverage: float
    inverse: bool
    max_weight: float
    max_consecutive_bars: int | None = None
    allowed_in_high_vol: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureConstraintConfig:
    max_effective_leverage: float | None = 1.0
    max_leveraged_bars_per_day: int | None = 60
    max_same_group_share_per_day: float | None = None
    max_consecutive_leveraged_bars: int | None = 30
    min_group_share_observations: int = 20


KNOWN_ACTION_META: dict[str, dict[str, Any]] = {
    "CASH": {
        "asset_class": "cash",
        "group": "cash",
        "underlying": None,
        "leverage": 0.0,
        "inverse": False,
    },
    "SPY": {
        "asset_class": "etf",
        "group": "broad_market",
        "underlying": "SPY",
        "leverage": 1.0,
        "inverse": False,
    },
    "QQQ": {
        "asset_class": "etf",
        "group": "broad_tech",
        "underlying": "QQQ",
        "leverage": 1.0,
        "inverse": False,
    },
    "IWM": {
        "asset_class": "etf",
        "group": "small_cap",
        "underlying": "IWM",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLF": {
        "asset_class": "etf",
        "group": "financials",
        "underlying": "XLF",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLK": {
        "asset_class": "etf",
        "group": "technology",
        "underlying": "XLK",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLE": {
        "asset_class": "etf",
        "group": "energy",
        "underlying": "XLE",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLI": {
        "asset_class": "etf",
        "group": "industrials",
        "underlying": "XLI",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLU": {
        "asset_class": "etf",
        "group": "utilities",
        "underlying": "XLU",
        "leverage": 1.0,
        "inverse": False,
    },
    "XLV": {
        "asset_class": "etf",
        "group": "healthcare",
        "underlying": "XLV",
        "leverage": 1.0,
        "inverse": False,
    },
    "TLT": {
        "asset_class": "etf",
        "group": "treasury_duration",
        "underlying": "TLT",
        "leverage": 1.0,
        "inverse": False,
    },
    "GLD": {
        "asset_class": "etf",
        "group": "gold",
        "underlying": "GLD",
        "leverage": 1.0,
        "inverse": False,
    },
    "SOXL": {
        "asset_class": "leveraged_etf",
        "group": "semiconductor",
        "underlying": "SOXX",
        "leverage": 3.0,
        "inverse": False,
    },
    "SOXS": {
        "asset_class": "leveraged_etf",
        "group": "semiconductor",
        "underlying": "SOXX",
        "leverage": 3.0,
        "inverse": True,
    },
    "TQQQ": {
        "asset_class": "leveraged_etf",
        "group": "broad_tech",
        "underlying": "QQQ",
        "leverage": 3.0,
        "inverse": False,
    },
    "SQQQ": {
        "asset_class": "leveraged_etf",
        "group": "broad_tech",
        "underlying": "QQQ",
        "leverage": 3.0,
        "inverse": True,
    },
    "TZA": {
        "asset_class": "leveraged_etf",
        "group": "small_cap",
        "underlying": "IWM",
        "leverage": 3.0,
        "inverse": True,
    },
    "TSLL": {
        "asset_class": "leveraged_etf",
        "group": "tesla",
        "underlying": "TSLA",
        "leverage": 2.0,
        "inverse": False,
    },
    "TSLG": {
        "asset_class": "leveraged_etf",
        "group": "tesla",
        "underlying": "TSLA",
        "leverage": 2.0,
        "inverse": False,
    },
    "NVD": {
        "asset_class": "leveraged_etf",
        "group": "nvidia",
        "underlying": "NVDA",
        "leverage": 2.0,
        "inverse": True,
    },
    "BITO": {
        "asset_class": "etf",
        "group": "crypto",
        "underlying": "BTC",
        "leverage": 1.0,
        "inverse": False,
    },
    "DRIP": {
        "asset_class": "leveraged_etf",
        "group": "energy",
        "underlying": "XOP",
        "leverage": 2.0,
        "inverse": True,
    },
    "SPDN": {
        "asset_class": "inverse_etf",
        "group": "broad_market",
        "underlying": "SPY",
        "leverage": 1.0,
        "inverse": True,
    },
    "SNDQ": {
        "asset_class": "inverse_etf",
        "group": "broad_tech",
        "underlying": "QQQ",
        "leverage": 1.0,
        "inverse": True,
    },
}


def _default_max_weight(leverage: float) -> float:
    if leverage <= 0:
        return 1.0
    return min(1.0, 1.0 / max(float(leverage), 1.0))


def infer_action_meta(name: str) -> ActionMeta:
    symbol = name.upper()
    values = KNOWN_ACTION_META.get(
        symbol,
        {"asset_class": "etf", "group": symbol.lower(), "underlying": symbol, "leverage": 1.0, "inverse": False},
    )
    leverage = float(values["leverage"])
    return ActionMeta(
        name=name,
        asset_class=str(values["asset_class"]),
        group=str(values["group"]),
        underlying=values["underlying"],
        leverage=leverage,
        inverse=bool(values["inverse"]),
        max_weight=float(values.get("max_weight", _default_max_weight(leverage))),
        max_consecutive_bars=values.get("max_consecutive_bars"),
        allowed_in_high_vol=bool(values.get("allowed_in_high_vol", True)),
    )


def build_action_metadata(action_names: list[str]) -> list[ActionMeta]:
    return [infer_action_meta(name) for name in action_names]


def action_metadata_to_dicts(action_meta: list[ActionMeta]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in action_meta]


def action_weight_tensor(
    action_meta: list[ActionMeta],
    *,
    device: torch.device | str,
    max_effective_leverage: float | None = None,
) -> torch.Tensor:
    target = None if max_effective_leverage is None else float(max_effective_leverage)
    weights = []
    for item in action_meta:
        if target is None or item.leverage <= 0:
            weights.append(item.max_weight)
        else:
            weights.append(min(item.max_weight, target / max(item.leverage, 1.0)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def action_leverage_tensor(action_meta: list[ActionMeta], *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor([item.leverage for item in action_meta], dtype=torch.float32, device=device)


def action_is_leveraged_tensor(action_meta: list[ActionMeta], *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor([item.leverage > 1.0 for item in action_meta], dtype=torch.bool, device=device)


def action_is_inverse_tensor(action_meta: list[ActionMeta], *, device: torch.device | str) -> torch.Tensor:
    return torch.tensor([item.inverse for item in action_meta], dtype=torch.bool, device=device)


def group_ids_for_actions(action_meta: list[ActionMeta], *, device: torch.device | str) -> tuple[torch.Tensor, list[str]]:
    groups = sorted({item.group for item in action_meta})
    group_index = {group: index for index, group in enumerate(groups)}
    return torch.tensor([group_index[item.group] for item in action_meta], dtype=torch.long, device=device), groups


def trade_notional(
    previous_action: torch.Tensor,
    action: torch.Tensor,
    action_weights: torch.Tensor,
    *,
    cash_index: int = 0,
) -> torch.Tensor:
    changed = action != previous_action
    previous_weights = action_weights[previous_action.long()]
    next_weights = action_weights[action.long()]
    previous_weights = torch.where(
        previous_action.long() == int(cash_index),
        torch.zeros_like(previous_weights),
        previous_weights,
    )
    next_weights = torch.where(action.long() == int(cash_index), torch.zeros_like(next_weights), next_weights)
    return torch.where(changed, previous_weights + next_weights, torch.zeros_like(next_weights))


def make_exposure_features(
    *,
    current_action: torch.Tensor,
    action_leverage: torch.Tensor,
    action_weights: torch.Tensor,
    action_is_leveraged: torch.Tensor,
    action_group_ids: torch.Tensor,
    group_counts_today: torch.Tensor,
    steps_today: torch.Tensor,
    leveraged_bars_today: torch.Tensor,
    consecutive_leveraged_bars: torch.Tensor,
    constraints: ExposureConstraintConfig,
    episode_length: int,
) -> torch.Tensor:
    current_action = current_action.long()
    current_group = action_group_ids[current_action]
    current_group_count = group_counts_today.gather(1, current_group.unsqueeze(1)).squeeze(1)
    group_share = current_group_count.float() / steps_today.float().clamp_min(1.0)
    leverage_cap = constraints.max_effective_leverage if constraints.max_effective_leverage is not None else 1.0
    leveraged_cap = (
        constraints.max_leveraged_bars_per_day
        if constraints.max_leveraged_bars_per_day is not None
        else max(int(episode_length), 1)
    )
    consecutive_cap = (
        constraints.max_consecutive_leveraged_bars
        if constraints.max_consecutive_leveraged_bars is not None
        else max(int(episode_length), 1)
    )
    current_effective_leverage = action_leverage[current_action] * action_weights[current_action]
    return torch.stack(
        [
            steps_today.float() / max(float(episode_length), 1.0),
            leveraged_bars_today.float() / max(float(leveraged_cap), 1.0),
            consecutive_leveraged_bars.float() / max(float(consecutive_cap), 1.0),
            current_effective_leverage / max(float(leverage_cap), 1.0),
            action_is_leveraged[current_action].float(),
            group_share,
        ],
        dim=1,
    ).clamp(0.0, 8.0)


def apply_exposure_masks(
    mask: torch.Tensor,
    *,
    current_action: torch.Tensor,
    action_leverage: torch.Tensor,
    action_weights: torch.Tensor,
    action_is_leveraged: torch.Tensor,
    action_group_ids: torch.Tensor,
    group_counts_today: torch.Tensor,
    steps_today: torch.Tensor,
    leveraged_bars_today: torch.Tensor,
    consecutive_leveraged_bars: torch.Tensor,
    constraints: ExposureConstraintConfig,
    cash_index: int = 0,
) -> torch.Tensor:
    out = mask.clone()
    action_count = out.shape[1]
    candidates = torch.arange(action_count, dtype=torch.long, device=out.device).unsqueeze(0).expand_as(out)
    candidate_leverage = action_leverage[candidates]
    candidate_weights = action_weights[candidates]
    candidate_is_leveraged = action_is_leveraged[candidates]
    cash_column = int(cash_index)

    if constraints.max_effective_leverage is not None:
        effective_leverage = candidate_leverage * candidate_weights
        out = out & (effective_leverage <= float(constraints.max_effective_leverage) + 1e-12)

    if constraints.max_leveraged_bars_per_day is not None:
        exhausted = leveraged_bars_today >= int(constraints.max_leveraged_bars_per_day)
        out = out & ~(exhausted.unsqueeze(1) & candidate_is_leveraged)

    if constraints.max_consecutive_leveraged_bars is not None:
        exhausted = consecutive_leveraged_bars >= int(constraints.max_consecutive_leveraged_bars)
        out = out & ~(exhausted.unsqueeze(1) & candidate_is_leveraged)

    if constraints.max_same_group_share_per_day is not None:
        min_obs = max(int(constraints.min_group_share_observations), 1)
        enough_history = steps_today >= min_obs
        candidate_groups = action_group_ids[candidates]
        candidate_group_counts = torch.gather(group_counts_today, 1, candidate_groups)
        candidate_group_share = candidate_group_counts.float() / steps_today.float().clamp_min(1.0).unsqueeze(1)
        group_exhausted = enough_history.unsqueeze(1) & (
            candidate_group_share >= float(constraints.max_same_group_share_per_day)
        )
        out = out & ~group_exhausted

    out[:, cash_column] = mask[:, cash_column]
    empty_rows = ~out.any(dim=1)
    if bool(empty_rows.any().item()):
        out[empty_rows, cash_column] = True
    return out


def action_concentration(
    rollout_records: list[dict[str, Any]],
    *,
    action_meta: list[ActionMeta],
) -> dict[str, Any]:
    if not rollout_records:
        return {
            "rows": 0,
            "max_action": None,
            "max_action_share": 0.0,
            "max_group": None,
            "max_group_share": 0.0,
            "max_risky_group": None,
            "max_risky_group_share": 0.0,
            "leveraged_action_share": 0.0,
            "inverse_action_share": 0.0,
        }
    by_name = {item.name: item for item in action_meta}
    action_counts = Counter(str(row.get("asset", row.get("action", ""))) for row in rollout_records)
    group_counts: Counter[str] = Counter()
    risky_group_counts: Counter[str] = Counter()
    leveraged = 0
    inverse = 0
    for row in rollout_records:
        name = str(row.get("asset", row.get("action", "")))
        meta = by_name.get(name, infer_action_meta(name))
        group_counts[meta.group] += 1
        if meta.asset_class != "cash" and meta.leverage > 0:
            risky_group_counts[meta.group] += 1
        leveraged += int(meta.leverage > 1.0)
        inverse += int(meta.inverse)
    rows = len(rollout_records)
    max_action, max_action_count = action_counts.most_common(1)[0]
    max_group, max_group_count = group_counts.most_common(1)[0]
    if risky_group_counts:
        max_risky_group, max_risky_group_count = risky_group_counts.most_common(1)[0]
    else:
        max_risky_group, max_risky_group_count = None, 0
    return {
        "rows": rows,
        "max_action": max_action,
        "max_action_share": max_action_count / rows,
        "max_group": max_group,
        "max_group_share": max_group_count / rows,
        "max_risky_group": max_risky_group,
        "max_risky_group_share": max_risky_group_count / rows,
        "leveraged_action_share": leveraged / rows,
        "inverse_action_share": inverse / rows,
        "action_counts": dict(action_counts),
        "group_counts": dict(group_counts),
        "risky_group_counts": dict(risky_group_counts),
    }


def rollout_return_diagnostics(rollout_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"rows": 0, "gross_sum": 0.0, "net_sum": 0.0, "min_net": float("inf"), "max_net": -float("inf")}
    )
    by_day: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"rows": 0, "gross_sum": 0.0, "net_sum": 0.0, "min_equity": float("inf"), "max_equity": -float("inf")}
    )
    segments: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None
    previous_asset: str | None = None
    for row in rollout_records:
        asset = str(row.get("asset", row.get("action", "")))
        gross = float(row.get("gross_return", 0.0))
        net = float(row.get("bar_return", row.get("net_return", 0.0)))
        equity = float(row.get("equity", 1.0))
        timestamp = str(row.get("timestamp", row.get("decision_timestamp", "")))
        day = timestamp[:10]

        action_bucket = by_action[asset]
        action_bucket["rows"] = int(action_bucket["rows"]) + 1
        action_bucket["gross_sum"] = float(action_bucket["gross_sum"]) + gross
        action_bucket["net_sum"] = float(action_bucket["net_sum"]) + net
        action_bucket["min_net"] = min(float(action_bucket["min_net"]), net)
        action_bucket["max_net"] = max(float(action_bucket["max_net"]), net)

        day_bucket = by_day[day]
        day_bucket["rows"] = int(day_bucket["rows"]) + 1
        day_bucket["gross_sum"] = float(day_bucket["gross_sum"]) + gross
        day_bucket["net_sum"] = float(day_bucket["net_sum"]) + net
        day_bucket["min_equity"] = min(float(day_bucket["min_equity"]), equity)
        day_bucket["max_equity"] = max(float(day_bucket["max_equity"]), equity)
        day_bucket["last_equity"] = equity

        if current_segment is None or asset != previous_asset:
            if current_segment is not None:
                segments.append(current_segment)
            current_segment = {
                "start": timestamp,
                "end": timestamp,
                "asset": asset,
                "rows": 0,
                "gross_sum": 0.0,
                "net_sum": 0.0,
                "start_equity": equity,
                "end_equity": equity,
            }
        current_segment["rows"] += 1
        current_segment["gross_sum"] += gross
        current_segment["net_sum"] += net
        current_segment["end"] = timestamp
        current_segment["end_equity"] = equity
        previous_asset = asset
    if current_segment is not None:
        segments.append(current_segment)

    rows = max(len(rollout_records), 1)
    for bucket in by_action.values():
        bucket["row_share"] = int(bucket["rows"]) / rows
        bucket["mean_net"] = float(bucket["net_sum"]) / max(int(bucket["rows"]), 1)
    return {
        "by_action": dict(by_action),
        "by_day": dict(by_day),
        "holding_segments": segments,
    }


def reportability_flags(
    *,
    test_metrics: dict[str, Any],
    baselines: dict[str, Any],
    concentration: dict[str, Any],
    max_group_share: float = 0.75,
    max_leveraged_share: float = 0.50,
) -> dict[str, Any]:
    reasons: list[str] = []
    test_return = float(test_metrics.get("total_return", 0.0))
    cash_return = float(baselines.get("CASH", {}).get("test", {}).get("total_return", 0.0))
    qqq_return = baselines.get("BuyAndHold_QQQ", {}).get("test", {}).get("total_return")
    if test_return < cash_return:
        reasons.append("test_return_below_cash")
    if qqq_return is not None and test_return < float(qqq_return):
        reasons.append("test_return_below_buy_and_hold_qqq")
    concentration_group_share = float(concentration.get("max_risky_group_share", concentration.get("max_group_share", 0.0)))
    if concentration_group_share > max_group_share:
        reasons.append("max_group_share_exceeds_limit")
    if float(concentration.get("leveraged_action_share", 0.0)) > max_leveraged_share:
        reasons.append("leveraged_action_share_exceeds_limit")
    return {"reportable": not reasons, "reasons": reasons}
