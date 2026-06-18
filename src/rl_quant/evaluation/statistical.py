"""Statistical-credibility metrics -- the axis SEPARATE from mechanical reportability.

A mechanically-reportable backtest (causal, complete, cost-aware) can still be OVERFIT. This module computes
the standard selection-bias / multiple-testing controls (Bailey & Lopez de Prado): the Probabilistic Sharpe
Ratio (PSR), the expected maximum Sharpe under the null over N trials, and the Deflated Sharpe Ratio (DSR).
These change NO backtest number; they judge whether an observed Sharpe is CREDIBLE given how many
strategies / configs / seeds were tried.

Intended consumer (the named, non-dead-API justification): the PR-D D4 A/B and the report layer. A
flag-on-vs-off Sharpe delta must not be taken at face value -- it should be deflated for the number of
configs/seeds tried (DSR) before concluding "the dynamic feature helped." Pure functions, stdlib only
(``statistics.NormalDist`` for the normal CDF/inverse -- no torch/scipy), so it is independent of the model
and trainer. All Sharpe inputs are PER-OBSERVATION (non-annualized) Sharpe ratios.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist

_NORMAL = NormalDist()
# Euler-Mascheroni constant, used in the expected-maximum-of-N-Gaussians approximation.
_EULER_MASCHERONI = 0.5772156649015329


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _non_normality_denominator(sharpe: float, skewness: float, kurtosis: float) -> float:
    # sqrt(1 - skew*SR + (kurtosis-1)/4 * SR^2) -- the standard-error adjustment for non-normal returns;
    # kurtosis is NON-excess (3 for a normal). The (kurtosis-1)/4 coefficient is NOT a typo for
    # (kurtosis-3)/4: it is Lo (2002)'s i.i.d.-normal SR-estimator variance term SR^2/2 COMBINED with
    # Mertens (2002)'s excess-kurtosis term (kurtosis-3)/4 * SR^2 -- i.e.
    # SR^2/2 + (kurtosis-3)/4 * SR^2 = (kurtosis-1)/4 * SR^2. So for a normal return (kurtosis=3) the
    # denominator is sqrt(1 + SR^2/2) (the classic Sharpe-estimator variance), NOT 1. Clamped > 0 to stay
    # defined. (A test pins this against the closed form; Monte Carlo confirms the SR^2/2 term is real.)
    value = 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    return math.sqrt(max(value, 1e-12))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > ``benchmark_sharpe``) given the observed per-period Sharpe over ``n_observations``,
    adjusted for return skew/kurtosis (Bailey & Lopez de Prado, 2012). Monotone increasing in
    observed_sharpe and in n_observations; equals 0.5 when observed_sharpe == benchmark_sharpe."""
    if not isinstance(n_observations, int) or isinstance(n_observations, bool) or n_observations < 2:
        raise ValueError(f"n_observations must be an integer >= 2; got {n_observations!r}.")
    for name, value in (("observed_sharpe", observed_sharpe), ("benchmark_sharpe", benchmark_sharpe),
                        ("skewness", skewness), ("kurtosis", kurtosis)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number; got {value!r}.")
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1)
    z /= _non_normality_denominator(observed_sharpe, skewness, kurtosis)
    return _NORMAL.cdf(z)


# A PSR computed from very few net returns is fragile no matter how high it reads: the skew/kurtosis
# standard-error correction is itself estimated from that same handful of points, and the normal-CDF
# mapping leans on the per-period Sharpe estimator being roughly Gaussian -- only true once n is large
# enough. This is the observation count below which a PSR should be reported as "present but not yet
# credible". It is a REPORTABILITY heuristic (the common CLT rule of thumb), NOT a hard statistical law:
# it gates NO model selection and changes NO PSR value -- it only annotates how much to trust one.
PSR_MIN_CREDIBLE_OBSERVATIONS = 30


def psr_is_credible(probabilistic_sharpe: float | None, n_observations: int) -> bool:
    """True iff a PSR was estimable (not ``None``) AND rests on at least ``PSR_MIN_CREDIBLE_OBSERVATIONS``
    net returns. A reportability annotation only -- it never changes the PSR value or any selection."""
    return probabilistic_sharpe is not None and n_observations >= PSR_MIN_CREDIBLE_OBSERVATIONS


def effective_sample_size(returns: Sequence[float]) -> float:
    """Autocorrelation-deflated effective sample size of a return series: ``n / (1 + 2 * sum_k rho_k)``,
    summing the lag-k autocorrelations over the INITIAL POSITIVE SEQUENCE (stop at the first ``rho_k <= 0``),
    clamped to ``[1, n]``.

    PSR / Sharpe assume i.i.d. returns; with POSITIVE serial correlation the independent information is LESS
    than ``n``, so a PSR computed from the raw count is over-confident -- this is the honest ``n`` to judge it
    against (``effective_observations < observations`` flags that inflation). Uses population (1/n) moments to
    match the PSR moment convention. Returns ``float(n)`` when ``n < 2`` or the series has zero variance (no
    autocorrelation to estimate), and NEVER inflates above the raw count: negative autocorrelation truncates
    the sum to empty rather than claiming MORE independent observations than were collected."""
    n = len(returns)
    if n < 2:
        return float(n)
    mean = sum(returns) / n
    centered = [r - mean for r in returns]
    c0 = sum(c * c for c in centered) / n
    if c0 <= 0.0:
        return float(n)
    total = 0.0
    for k in range(1, n):
        ck = sum(centered[t] * centered[t + k] for t in range(n - k)) / n
        rho = ck / c0
        if rho <= 0.0:
            break
        total += rho
    return max(1.0, min(float(n), n / (1.0 + 2.0 * total)))


def expected_maximum_sharpe(n_trials: int, *, trials_sharpe_std: float = 1.0) -> float:
    """Expected MAXIMUM of ``n_trials`` i.i.d. ``N(0, trials_sharpe_std^2)`` Sharpe estimates under the null
    (no skill) -- the benchmark the Deflated Sharpe deflates against. More trials => higher expected max by
    chance alone. Returns 0.0 for a single trial (no selection)."""
    _require_positive_int("n_trials", n_trials)
    if isinstance(trials_sharpe_std, bool) or not isinstance(trials_sharpe_std, (int, float)) or not (
        math.isfinite(float(trials_sharpe_std)) and trials_sharpe_std >= 0.0
    ):
        raise ValueError(f"trials_sharpe_std must be a finite non-negative number; got {trials_sharpe_std!r}.")
    if n_trials == 1:
        return 0.0
    upper = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    lower = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(trials_sharpe_std) * ((1.0 - _EULER_MASCHERONI) * upper + _EULER_MASCHERONI * lower)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trials_sharpe_std: float = 1.0,
) -> float:
    """Probability the observed Sharpe is real AFTER accounting for selection over ``n_trials``
    configs/seeds -- the PSR with the benchmark set to the expected maximum Sharpe under the null
    (Bailey & Lopez de Prado, 2014). For a single trial it reduces to PSR vs 0; more trials lower it."""
    benchmark = expected_maximum_sharpe(n_trials, trials_sharpe_std=trials_sharpe_std)
    return probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark_sharpe=benchmark,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )


# Confidence the Deflated Sharpe must clear to PROMOTE (one-sided). 0.95 is the conventional 5%
# selection-risk bar; like the PSR observation floor it is a REPORTABILITY/decision threshold, not a hard
# statistical law, and it gates nothing on its own -- the A/B / report layer reads the verdict and decides.
DSR_PROMOTION_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PromotionVerdict:
    """A reportable promotion decision for a candidate Sharpe AFTER deflating for the number of configs/seeds
    tried. ``promote`` is True ONLY when the Deflated Sharpe clears the confidence bar AND the estimate rests
    on enough net returns to be credible. ``reasons`` is empty when promoted and otherwise names every failed
    gate, so a report can show WHY a candidate was held -- not merely that it was."""

    promote: bool
    deflated_sharpe_ratio: float
    n_trials: int
    n_observations: int
    confidence: float
    is_significant: bool
    is_credible: bool
    reasons: tuple[str, ...]


def deflated_sharpe_promotion_verdict(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    trials_sharpe_std: float = 1.0,
    confidence: float = DSR_PROMOTION_CONFIDENCE,
) -> PromotionVerdict:
    """Synthesize the two controls a promotion gate needs into ONE verdict -- the review's point that PSR
    alone (n_trials = 1) is NOT a promotion gate. (1) The Deflated Sharpe Ratio (probability the observed
    per-period Sharpe is real AFTER deflating for selection over ``n_trials`` configs/seeds) must reach
    ``confidence``; AND (2) the estimate must rest on at least ``PSR_MIN_CREDIBLE_OBSERVATIONS`` net returns
    (a high DSR off a handful of points is not promotable). A pure decision helper: it changes no backtest
    number and gates nothing on its own. Requires ``n_observations >= 2`` (inherited from the DSR/PSR
    contract -- caller pre-checks estimability); raises on a confidence outside (0, 1)."""
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 < float(confidence) < 1.0):
        raise ValueError(f"confidence must be a number in (0, 1); got {confidence!r}.")
    dsr = deflated_sharpe_ratio(
        observed_sharpe,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        trials_sharpe_std=trials_sharpe_std,
    )
    is_significant = dsr >= float(confidence)
    is_credible = n_observations >= PSR_MIN_CREDIBLE_OBSERVATIONS
    reasons: list[str] = []
    if not is_significant:
        reasons.append(f"deflated_sharpe_ratio {dsr:.4f} < confidence {float(confidence):.4f}")
    if not is_credible:
        reasons.append(
            f"n_observations {n_observations} < PSR_MIN_CREDIBLE_OBSERVATIONS {PSR_MIN_CREDIBLE_OBSERVATIONS}"
        )
    return PromotionVerdict(
        promote=is_significant and is_credible,
        deflated_sharpe_ratio=dsr,
        n_trials=n_trials,
        n_observations=n_observations,
        confidence=float(confidence),
        is_significant=is_significant,
        is_credible=is_credible,
        reasons=tuple(reasons),
    )


def _series_sharpe(values: Sequence[float]) -> float:
    """Per-observation Sharpe (mean / population std) of a return series; 0.0 when fewer than 2 points or
    zero dispersion (a flat series carries no rank information)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    if var <= 0.0:
        return 0.0
    return mean / math.sqrt(var)


def probability_of_backtest_overfitting(
    performance: Sequence[Sequence[float]], *, n_splits: int = 16
) -> float:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation (Bailey, Borwein,
    Lopez de Prado & Zhu, 2017). ``performance`` is a matrix with one ROW per observation (time) and one
    COLUMN per candidate config/strategy (``performance[t][n]`` = config n's return at observation t).

    The T observations are split into ``n_splits`` contiguous blocks; for every way to choose n_splits/2
    blocks as IN-SAMPLE (the rest OUT-OF-SAMPLE), the config with the best IS Sharpe is found and its OOS
    RANK among all configs is taken. PBO is the fraction of splits in which that IS-best config lands in the
    OOS BOTTOM half (logit <= 0) -- i.e. the probability that an in-sample-selected config is, out of sample,
    no better than the median. ~0.5 means selection is indistinguishable from luck; near 1.0 means the
    selection procedure overfits; near 0.0 means a genuinely dominant config. The companion to PSR/DSR for a
    project that searches over many configs/seeds. Pure / stdlib-only; changes no backtest number.

    Raises on a non-rectangular matrix, < 2 configs, an odd or < 2 ``n_splits``, T < n_splits, or non-finite
    values."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2; got {n_splits!r}.")
    rows = [list(row) for row in performance]
    n_obs = len(rows)
    if n_obs < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} observations; got {n_obs}.")
    n_configs = len(rows[0]) if rows else 0
    if n_configs < 2:
        raise ValueError(f"need at least 2 configs (matrix columns); got {n_configs}.")
    for r in rows:
        if len(r) != n_configs:
            raise ValueError("performance matrix must be rectangular (every row the same number of configs).")
        for value in r:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"performance entries must be finite numbers; got {value!r}.")

    # Contiguous, as-even-as-possible blocks of observation indices.
    bounds = [round(k * n_obs / n_splits) for k in range(n_splits + 1)]
    blocks = [list(range(bounds[k], bounds[k + 1])) for k in range(n_splits)]

    def sharpe_over(indices: list[int], config: int) -> float:
        return _series_sharpe([rows[t][config] for t in indices])

    overfit = 0
    total = 0
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_set = set(is_blocks)
        is_rows = [t for b in is_blocks for t in blocks[b]]
        oos_rows = [t for b in range(n_splits) if b not in is_set for t in blocks[b]]
        is_sharpe = [sharpe_over(is_rows, n) for n in range(n_configs)]
        oos_sharpe = [sharpe_over(oos_rows, n) for n in range(n_configs)]
        best = max(range(n_configs), key=lambda n: is_sharpe[n])  # IS-selected config (first on ties)
        # OOS rank of the IS-best (1 = worst): relative rank in (0, 1); logit <= 0  <=>  bottom half.
        rank = sum(1 for s in oos_sharpe if s < oos_sharpe[best]) + 1
        omega = rank / (n_configs + 1)
        total += 1
        if omega <= 0.5:
            overfit += 1
    return overfit / total
