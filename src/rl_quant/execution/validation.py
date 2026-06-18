"""Numeric / integer coercion + validation helpers for the execution layer.

These are the SINGLE source of the framework's numeric-field contracts (reject bool where a
float is meant, reject NaN/inf, reject fractional ints instead of silently truncating). The
execution engine and the intraday env both enforce the SAME rules through these helpers -- the
public aliases at the bottom exist so callers outside this package can do likewise.
"""

from __future__ import annotations

import math


def _coerce_float(name: str, value: object) -> float:
    # Numeric fields must end up as real floats: reject bool (True would silently become 1.0) and any
    # non-numeric type, and RETURN the coerced float so the caller can store it -- a value that only
    # *validated* but stayed a string would later break arithmetic (e.g. "0.01" + 0.05).
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool; got {value!r}.")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc


def _coerce_finite(name: str, value: object) -> float:
    coerced = _coerce_float(name, value)
    if not math.isfinite(coerced):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return coerced


def _coerce_finite_nonnegative(name: str, value: object) -> float:
    # NOTE: a bare ``value < 0`` does NOT reject NaN (every NaN comparison is False), so check finiteness.
    coerced = _coerce_float(name, value)
    if not math.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return coerced


def _coerce_positive_price(name: str, value: object) -> float:
    # Equity/ETF prices must be strictly positive: a non-positive mid/quote/entry would produce
    # meaningless P&L and costs. (A bare ``value <= 0`` would pass NaN, so check finiteness too.)
    coerced = _coerce_float(name, value)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return coerced


def _require_nonnegative_int(name: str, value: object) -> int:
    # Bars/lots must be integer-like: reject bool, and reject a float that is non-finite or has a
    # fractional part instead of silently truncating it (int(1.9) == 1 would rescale every dollar P&L).
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be integer-like; got {value!r}.")
    try:
        coerced = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be integer-like; got {value!r}.") from exc
    if coerced < 0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")
    return coerced


def _require_positive_int(name: str, value: object) -> int:
    coerced = _require_nonnegative_int(name, value)
    if coerced <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")
    return coerced


def _require_int_allow_negative(name: str, value: object) -> int:
    # Like _require_nonnegative_int but permits negatives (latency_steps <= 0 collapses to "now").
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool; got {value!r}.")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be integer-like; got {value!r}.")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be integer-like; got {value!r}.") from exc


def _require_bool(name: str, value: object) -> bool:
    # Governed flags must be REAL bools: bool("false") is True and bool(0) is False, so coercing a config
    # flag with bool(...) would silently flip behaviour (and, for a result-moving flag, the reported numbers).
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}.")
    return value


def _coerce_finite_positive(name: str, value: object) -> float:
    # Strictly-positive finite scalar (e.g. reward_scale): a zero/negative/NaN/inf would zero, flip, or
    # blow up every reward and any bps figure normalised by it.
    coerced = _coerce_finite(name, value)
    if coerced <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return coerced


# Public aliases so other modules (e.g. the intraday env) can enforce the SAME numeric/integer validation
# as ExecutionConfig instead of int()-truncating a fractional config value or float()-coercing a bool.
require_positive_int = _require_positive_int
require_nonnegative_int = _require_nonnegative_int
require_bool = _require_bool
coerce_finite_nonnegative = _coerce_finite_nonnegative
coerce_finite_positive = _coerce_finite_positive
