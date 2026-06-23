from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import torch

_WARNED_UNKNOWN_ACTION_SYMBOLS: set[str] = set()


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
    allow_leveraged_actions: bool = True
    allow_inverse_actions: bool = True
    max_leveraged_bars_per_day: int | None = 30
    max_same_group_share_per_day: float | None = 0.50
    max_consecutive_leveraged_bars: int | None = 15
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


def unknown_action_metadata_symbols(action_names: list[str]) -> list[str]:
    """Action names with no explicit risk metadata (leverage/inverse are inferred, not known).

    Callers building reportable or leverage/inverse-constrained runs should gate on this:
    an unknown leveraged/inverse instrument would otherwise be treated as 1x long and slip
    past the exposure caps.
    """
    return [name for name in action_names if name.upper() not in KNOWN_ACTION_META]


def infer_action_meta(name: str, *, strict: bool = False) -> ActionMeta:
    symbol = name.upper()
    known = symbol in KNOWN_ACTION_META
    if not known:
        if strict:
            raise ValueError(
                f"Unknown action symbol {name!r} has no entry in KNOWN_ACTION_META; refusing to "
                "infer leverage/inverse for a risk-constrained run. Add explicit metadata or call "
                "with strict=False to accept the 1x-long, non-inverse fallback."
            )
        if symbol not in _WARNED_UNKNOWN_ACTION_SYMBOLS:
            _WARNED_UNKNOWN_ACTION_SYMBOLS.add(symbol)
            warnings.warn(
                f"No risk metadata for action {name!r}; assuming 1x long, non-inverse. A leveraged "
                "or inverse instrument without metadata would bypass the leverage/inverse exposure "
                "caps. Add it to KNOWN_ACTION_META or gate with unknown_action_metadata_symbols().",
                stacklevel=2,
            )
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


def build_action_metadata(action_names: list[str], *, strict: bool = False) -> list[ActionMeta]:
    return [infer_action_meta(name, strict=strict) for name in action_names]


def validate_action_index_for_actions(action_names: list[str], index: object, *, name: str) -> int:
    """Reject a non-integer (bool/float/string) or out-of-range action index and return it as an int.

    Cross-cutting contract shared by the env (initial_action, cash_index) and the evaluator: an index is
    gathered against / compared to the action space, so a bool/float/string would silently coerce
    (``int(True) == 1``, ``int(2.9) == 2``) and select the WRONG action. ``numbers.Integral`` accepts numpy
    ints while still rejecting ``bool`` (a bool IS an Integral, so it is excluded explicitly)."""
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise ValueError(f"{name} must be an integer action index, got {index!r}.")
    out = int(index)
    if not (0 <= out < len(action_names)):
        raise ValueError(f"{name}={out} is outside the action space (0..{len(action_names) - 1}).")
    return out


def validate_cash_index_for_actions(action_names: list[str], cash_index: object) -> int:
    """Validate that ``cash_index`` is a real CASH action for ``action_names`` and return it as an int.

    Shared by the env and the evaluator so both fail closed identically: ``cash_index`` is special everywhere
    (cash-idle penalty, zero shadow exposure, label/force-restore fallback), so an out-of-range, wrong-type, or
    non-cash index would silently mis-charge the idle penalty / zero the wrong action's exposure / restore the
    wrong action. We inspect ONLY the cash symbol via ``infer_action_meta(strict=True)`` -- building metadata
    for every action would emit spurious warnings for unknown non-cash symbols (e.g. a new ETF ticker)."""
    index = validate_action_index_for_actions(action_names, cash_index, name="constraints.cash_index")
    symbol = str(action_names[index])
    if infer_action_meta(symbol, strict=True).asset_class != "cash":
        raise ValueError(
            f"constraints.cash_index={index} points to {symbol!r}, which is "
            "not a cash action (action-metadata asset_class != 'cash')."
        )
    return index


def action_metadata_to_dicts(action_meta: list[ActionMeta]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in action_meta]


def stable_action_metadata_hash(action_meta: list[ActionMeta]) -> str:
    payload = json.dumps(action_metadata_to_dicts(action_meta), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_action_risk_config_hash(config: ExposureConstraintConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
