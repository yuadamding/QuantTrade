"""Ranker-quality metrics for the action SCORER (separate from sequential-policy P&L).

The second-context path is a contextual action scorer with sequential diagnostics, not yet a full sequential
RL policy (see README). It should therefore be judged as a RANKER too: does the predicted score rank the
candidate actions by their realized outcome? A model can be a good ranker but a bad policy after turnover/
costs, or a bad ranker that got lucky in one sequential regime -- these metrics separate the two.

Each metric is CROSS-SECTIONAL: per decision row, it compares the predicted ``scores`` against the realized
``realized`` returns ACROSS the candidate actions (restricted to ``valid_mask``), then averages over rows.
Inputs are [rows][actions] -- tensors (``.tolist()``) or nested sequences. Signed-return-safe (no NDCG-style
non-negativity assumption). Pure functions, stdlib only; change no backtest number.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _rows(value: object) -> list[list[float]]:
    listed = value.tolist() if hasattr(value, "tolist") else value
    return [list(row) for row in listed]  # type: ignore[union-attr]


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_pairs(
    score_row: Sequence[float], return_row: Sequence[float], mask_row: Sequence[object] | None
) -> tuple[list[float], list[float]]:
    """Score/return pairs for actions that are valid (mask True or no mask) AND finite in both arrays."""
    scores: list[float] = []
    rets: list[float] = []
    for j, (s, r) in enumerate(zip(score_row, return_row)):
        if (mask_row is None or bool(mask_row[j])) and _finite(s) and _finite(r):
            scores.append(float(s))
            rets.append(float(r))
    return scores, rets


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None when undefined (< 2 points or zero variance in either series)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties given their average rank (for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def _mean_over_rows(scores: object, realized: object, valid_mask: object | None, per_row) -> float:
    """Apply ``per_row(scores, rets)`` to each row's valid pairs, averaging the defined results; NaN if none."""
    s_rows = _rows(scores)
    r_rows = _rows(realized)
    m_rows = _rows(valid_mask) if valid_mask is not None else [None] * len(s_rows)
    if not (len(s_rows) == len(r_rows) == len(m_rows)):
        raise ValueError("scores, realized, and valid_mask must have the same number of rows.")
    total = 0.0
    count = 0
    for i, (s_row, r_row, m_row) in enumerate(zip(s_rows, r_rows, m_rows)):
        # Per-row action widths must agree. _valid_pairs zips scores/returns and indexes the mask positionally,
        # so a width mismatch would SILENTLY truncate (or misalign the mask) and report a wrong metric instead
        # of failing. Fail closed on mismatch rather than scoring a partial row.
        if len(s_row) != len(r_row):
            raise ValueError(f"row {i}: scores width {len(s_row)} != realized width {len(r_row)}.")
        if m_row is not None and len(m_row) != len(s_row):
            raise ValueError(f"row {i}: valid_mask width {len(m_row)} != scores width {len(s_row)}.")
        value = per_row(*_valid_pairs(s_row, r_row, m_row))
        if value is not None:
            total += value
            count += 1
    return total / count if count else float("nan")


def information_coefficient(scores: object, realized: object, valid_mask: object | None = None) -> float:
    """Mean cross-sectional Pearson correlation between predicted scores and realized returns across the valid
    actions of each row (the classic IC). +1 = perfect ranking, 0 = none, -1 = inverted. NaN if no row has >=2
    valid finite actions with dispersion."""
    return _mean_over_rows(scores, realized, valid_mask, lambda s, r: _pearson(s, r))


def rank_information_coefficient(scores: object, realized: object, valid_mask: object | None = None) -> float:
    """Mean cross-sectional SPEARMAN (rank) correlation -- the IC robust to outliers/monotone-but-nonlinear
    scores. NaN if no row is defined."""
    return _mean_over_rows(scores, realized, valid_mask,
                           lambda s, r: _pearson(_ranks(s), _ranks(r)) if len(s) >= 2 else None)


def top_k_mean_return(scores: object, realized: object, k: int = 1, valid_mask: object | None = None) -> float:
    """Mean realized return of the top-``k`` actions BY SCORE within each row (the realized payoff of acting on
    the scorer's top picks). k is clamped to the number of valid actions in a row. NaN if no row has a valid
    action. Raises on k < 1."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}.")

    def per_row(s: list[float], r: list[float]) -> float | None:
        if not s:
            return None
        top = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]
        return sum(r[i] for i in top) / len(top)

    return _mean_over_rows(scores, realized, valid_mask, per_row)


def selection_regret(scores: object, realized: object, valid_mask: object | None = None) -> float:
    """Mean opportunity cost of trusting the scorer: per row, the BEST valid realized return minus the realized
    return of the action the scorer ranks first (argmax score). 0 = the scorer always picks the best action;
    larger = more return left on the table. NaN if no row has a valid action."""
    def per_row(s: list[float], r: list[float]) -> float | None:
        if not s:
            return None
        chosen = max(range(len(s)), key=lambda i: s[i])
        return max(r) - r[chosen]

    return _mean_over_rows(scores, realized, valid_mask, per_row)
