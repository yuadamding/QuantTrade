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
import random
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
        # OOS MIDRANK of the IS-best (ties averaged): relative rank omega in (0, 1). Midranks matter when OOS
        # scores tie -- a strict "< best" count would push every tie down to rank 1 and spuriously flag
        # overfitting. The IS-best is "overfit" only when it is STRICTLY below the OOS median (omega < 0.5);
        # exactly at the median (e.g. all-tied/degenerate configs -> omega == 0.5) is uninformative, not overfit.
        below = sum(1 for s in oos_sharpe if s < oos_sharpe[best])
        ties = sum(1 for s in oos_sharpe if s == oos_sharpe[best])  # includes the best itself
        midrank = below + (ties + 1) / 2.0
        omega = midrank / (n_configs + 1)
        total += 1
        if omega < 0.5:
            overfit += 1
    return overfit / total


def _stationary_bootstrap_indices(n_obs: int, mean_block: float, rng: random.Random) -> list[int]:
    """Politis-Romano stationary-bootstrap index sequence: geometrically-distributed block lengths with mean
    ``mean_block`` (so serial dependence survives resampling), wrapping at the series end."""
    restart_prob = 1.0 / mean_block
    indices: list[int] = []
    cur = rng.randrange(n_obs)
    for i in range(n_obs):
        if i == 0 or rng.random() < restart_prob:
            cur = rng.randrange(n_obs)
        else:
            cur = (cur + 1) % n_obs
        indices.append(cur)
    return indices


def _validate_differentials(
    performance_differentials: Sequence[Sequence[float]], n_bootstrap: int, block_size: float | None
) -> tuple[list[list[float]], int, int, float]:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be a positive integer; got {n_bootstrap!r}.")
    rows = [list(r) for r in performance_differentials]
    n_obs = len(rows)
    if n_obs < 2:
        raise ValueError(f"need at least 2 observations; got {n_obs}.")
    n_models = len(rows[0]) if rows else 0
    if n_models < 1:
        raise ValueError(f"need at least 1 model column; got {n_models}.")
    for r in rows:
        if len(r) != n_models:
            raise ValueError("performance_differentials must be rectangular (every row the same width).")
        for value in r:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"performance_differentials entries must be finite numbers; got {value!r}.")
    if block_size is None:
        mean_block = max(1.0, float(round(math.sqrt(n_obs))))
    elif isinstance(block_size, bool) or not isinstance(block_size, (int, float)) or block_size < 1:
        raise ValueError(f"block_size must be a number >= 1; got {block_size!r}.")
    else:
        mean_block = float(block_size)
    return rows, n_obs, n_models, mean_block


def _bootstrap_column_means(
    rows: list[list[float]], n_obs: int, n_models: int, n_bootstrap: int, mean_block: float, rng: random.Random
) -> list[list[float]]:
    out: list[list[float]] = []
    for _ in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n_obs, mean_block, rng)
        out.append([sum(rows[t][k] for t in idx) / n_obs for k in range(n_models)])
    return out


def white_reality_check(
    performance_differentials: Sequence[Sequence[float]], *, n_bootstrap: int = 1000,
    block_size: float | None = None, seed: int = 0,
) -> float:
    """White's Reality Check (2000) bootstrap p-value for the null that NO model truly outperforms the
    benchmark, correcting for data-snooping across all ``M`` models. ``performance_differentials[t][k]`` is
    model k's per-period OUTPERFORMANCE vs the benchmark at observation t (higher = better). A LOW p-value
    rejects the null -- the best model's edge survives the multiple-comparison correction. Uses the stationary
    bootstrap (mean block ~ sqrt(T) by default) for serial dependence; deterministic given ``seed``.
    Pure/stdlib; changes no backtest number."""
    rows, n_obs, n_models, mean_block = _validate_differentials(performance_differentials, n_bootstrap, block_size)
    root_n = math.sqrt(n_obs)
    dbar = [sum(rows[t][k] for t in range(n_obs)) / n_obs for k in range(n_models)]
    observed = max(root_n * dbar[k] for k in range(n_models))
    rng = random.Random(seed)
    exceed = 0
    for bmeans in _bootstrap_column_means(rows, n_obs, n_models, n_bootstrap, mean_block, rng):
        stat = max(root_n * (bmeans[k] - dbar[k]) for k in range(n_models))
        if stat >= observed:
            exceed += 1
    return (1 + exceed) / (n_bootstrap + 1)


def hansens_spa(
    performance_differentials: Sequence[Sequence[float]], *, n_bootstrap: int = 1000,
    block_size: float | None = None, seed: int = 0,
) -> float:
    """Hansen's Superior Predictive Ability test (2005): the studentized, less-conservative refinement of
    White's Reality Check. Same input / convention / null as ``white_reality_check`` (low p-value -> the best
    model genuinely outperforms the benchmark), but more powerful because it (1) studentizes each differential
    by its standard deviation and (2) recenters only models that are NOT hopelessly bad -- those above the
    ``-sqrt(2 log log T)`` studentized threshold -- so poor models do not inflate the null distribution. It is
    asymptotically MORE POWERFUL than RC (whose inclusion of inferior models makes it conservative); being a
    different, studentized statistic, its finite-sample p-value is typically -- but not strictly -- below RC's.
    Deterministic given ``seed``. Pure/stdlib; changes no backtest number."""
    rows, n_obs, n_models, mean_block = _validate_differentials(performance_differentials, n_bootstrap, block_size)
    root_n = math.sqrt(n_obs)
    dbar = [sum(rows[t][k] for t in range(n_obs)) / n_obs for k in range(n_models)]
    omega = []
    for k in range(n_models):
        var = sum((rows[t][k] - dbar[k]) ** 2 for t in range(n_obs)) / n_obs
        omega.append(math.sqrt(var) if var > 0.0 else 0.0)
    # A zero-variance POSITIVE differential is DETERMINISTIC out-performance (the model beats the benchmark
    # every single period): the studentized statistic is +inf (divide-by-zero). Reject at the minimum p-value
    # rather than studentizing to 0 and dropping the column -- which would falsely report "no evidence" for a
    # model that dominates the benchmark with certainty. (Float roundoff hides this for non-exact constants
    # like 0.001, where var is tiny-positive and the column already rejects; this catches the exact-zero case.)
    if any(omega[k] == 0.0 and dbar[k] > 0.0 for k in range(n_models)):
        return 1.0 / (n_bootstrap + 1)
    studentized = [root_n * dbar[k] / omega[k] if omega[k] > 0.0 else 0.0 for k in range(n_models)]
    observed = max(0.0, max(studentized))
    # Hansen "consistent" recentering: keep d̄_k only for models above -sqrt(2 log log T) (log log T needs
    # T >= 3); hopelessly-bad models get g_k = 0 so their (very negative) studentized bootstrap term never
    # enters the max -- this is the SPA improvement over RC, which recenters (and thus retains) every model.
    threshold = math.sqrt(2.0 * math.log(math.log(n_obs))) if n_obs >= 3 else 0.0
    g = [dbar[k] if studentized[k] >= -threshold else 0.0 for k in range(n_models)]
    rng = random.Random(seed)
    exceed = 0
    for bmeans in _bootstrap_column_means(rows, n_obs, n_models, n_bootstrap, mean_block, rng):
        stat = 0.0
        for k in range(n_models):
            if omega[k] > 0.0:
                val = root_n * (bmeans[k] - g[k]) / omega[k]
                if val > stat:
                    stat = val
        if stat >= observed:
            exceed += 1
    return (1 + exceed) / (n_bootstrap + 1)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile (q in [0, 1]) of an already-sorted list (numpy 'linear' convention)."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(lo)]
    frac = pos - lo
    return sorted_values[int(lo)] * (1.0 - frac) + sorted_values[int(hi)] * frac


def _finite_floats(values: Sequence[float], name: str) -> list[float]:
    """Validate that every entry is a finite, non-bool real number and return them as floats (ValueError
    otherwise -- so a misuse fails closed with a clear message instead of a bare TypeError)."""
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"{name} entries must be finite numbers; got {v!r}.")
        out.append(float(v))
    return out


def walk_forward_degradation_ratio(
    in_sample_scores: Sequence[float], out_of_sample_scores: Sequence[float]
) -> float:
    """Walk-forward efficiency (Pardo): mean OUT-OF-SAMPLE score / mean IN-SAMPLE score across walk-forward
    folds. 1.0 = out-of-sample matches in-sample (no degradation); 0 < r < 1 = the edge degrades out of sample
    (lower = more overfit); r <= 0 = the OOS edge vanished or reversed. Inputs are equal-length per-fold
    performance scores (e.g. Sharpe or return per fold). Requires a POSITIVE in-sample mean -- there must be
    an in-sample edge to degrade FROM. Pure; changes no backtest number."""
    is_scores = _finite_floats(in_sample_scores, "in_sample_scores")
    oos_scores = _finite_floats(out_of_sample_scores, "out_of_sample_scores")
    if len(is_scores) != len(oos_scores):
        raise ValueError("in_sample_scores and out_of_sample_scores must have equal length (one pair per fold).")
    if len(is_scores) < 1:
        raise ValueError("need at least one walk-forward fold.")
    is_mean = sum(is_scores) / len(is_scores)
    if is_mean <= 0.0:
        raise ValueError(f"in-sample mean score must be positive to measure degradation; got {is_mean!r}.")
    return (sum(oos_scores) / len(oos_scores)) / is_mean


def block_bootstrap_confidence_interval(
    returns: Sequence[float], *, statistic: str = "mean", confidence: float = 0.95,
    n_bootstrap: int = 1000, block_size: float | None = None, seed: int = 0,
) -> tuple[float, float]:
    """Percentile confidence interval ``(low, high)`` for a return series' ``statistic`` -- "mean" or "sharpe"
    (per-observation mean/std) -- via the Politis-Romano STATIONARY block bootstrap (mean block ~ sqrt(n) by
    default), which preserves serial dependence rather than assuming i.i.d. returns. ``confidence`` is the
    two-sided coverage (0.95 -> the 2.5th/97.5th bootstrap percentiles). Deterministic given ``seed``.
    Pure/stdlib; changes no backtest number."""
    if statistic not in ("mean", "sharpe"):
        raise ValueError(f"statistic must be 'mean' or 'sharpe'; got {statistic!r}.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (0.0 < float(confidence) < 1.0):
        raise ValueError(f"confidence must be a number in (0, 1); got {confidence!r}.")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be a positive integer; got {n_bootstrap!r}.")
    values = _finite_floats(returns, "returns")
    n_obs = len(values)
    if n_obs < 2:
        raise ValueError(f"need at least 2 observations; got {n_obs}.")
    if block_size is None:
        mean_block = max(1.0, float(round(math.sqrt(n_obs))))
    elif isinstance(block_size, bool) or not isinstance(block_size, (int, float)) or block_size < 1:
        raise ValueError(f"block_size must be a number >= 1; got {block_size!r}.")
    else:
        mean_block = float(block_size)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n_obs, mean_block, rng)
        sample = [values[t] for t in idx]
        stats.append(sum(sample) / n_obs if statistic == "mean" else _series_sharpe(sample))
    stats.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    return (_percentile(stats, alpha), _percentile(stats, 1.0 - alpha))


def statistical_credibility_report(
    returns: Sequence[float], *, n_trials: int,
    candidate_performance: Sequence[Sequence[float]] | None = None,
    benchmark_differentials: Sequence[Sequence[float]] | None = None,
    confidence: float = DSR_PROMOTION_CONFIDENCE, trials_sharpe_std: float = 1.0, seed: int = 0,
) -> dict[str, object]:
    """Assemble the full statistical-credibility report for ONE candidate's per-period net ``returns`` that
    was selected from ``n_trials`` configs/seeds/universes -- the single artifact (the review's
    ``statistical_credibility.json``) that combines every control in this module so a result is never judged
    by a raw Sharpe alone:

      * the per-period Sharpe + its Probabilistic Sharpe Ratio (skew/kurtosis-adjusted) and credibility flag;
      * the autocorrelation-deflated effective sample size (PSR/Sharpe assume i.i.d.);
      * the expected-maximum-Sharpe / Deflated Sharpe Ratio / promotion verdict for the DECLARED trial count
        (so the edge is deflated for selection, not taken at face value);
      * and -- only when supplied -- the Probability of Backtest Overfitting (from a full
        ``candidate_performance`` matrix) and White's Reality Check / Hansen's SPA p-values (from
        ``benchmark_differentials``), the data-snooping controls across the whole candidate family.

    Pure / stdlib, deterministic given ``seed``. Fields are ``None`` where a metric is not estimable (e.g.
    fewer than 2 returns, or zero dispersion). ``n_trials`` is caller-supplied (a declared or registry-inferred
    count); raises (via the underlying primitives) on an invalid ``n_trials`` or malformed inputs."""
    n_obs = len(returns)
    report: dict[str, object] = {
        "n_observations": n_obs,
        "n_trials": n_trials,
        "effective_observations": effective_sample_size(returns),
        "expected_maximum_sharpe": expected_maximum_sharpe(n_trials, trials_sharpe_std=trials_sharpe_std),
    }
    per_period_sharpe: float | None = None
    psr: float | None = None
    dsr: float | None = None
    promotion: dict[str, object] | None = None
    if n_obs >= 2:
        avg = sum(returns) / n_obs
        m2 = sum((r - avg) ** 2 for r in returns) / n_obs
        if m2 > 0.0:
            per_period_sharpe = avg / math.sqrt(m2)
            skewness = (sum((r - avg) ** 3 for r in returns) / n_obs) / (m2 ** 1.5)
            kurtosis = (sum((r - avg) ** 4 for r in returns) / n_obs) / (m2 ** 2)
            psr = probabilistic_sharpe_ratio(per_period_sharpe, benchmark_sharpe=0.0, n_observations=n_obs,
                                             skewness=skewness, kurtosis=kurtosis)
            dsr = deflated_sharpe_ratio(per_period_sharpe, n_trials=n_trials, n_observations=n_obs,
                                        skewness=skewness, kurtosis=kurtosis, trials_sharpe_std=trials_sharpe_std)
            verdict = deflated_sharpe_promotion_verdict(
                per_period_sharpe, n_trials=n_trials, n_observations=n_obs, skewness=skewness, kurtosis=kurtosis,
                trials_sharpe_std=trials_sharpe_std, confidence=confidence)
            promotion = {
                "promote": verdict.promote, "is_significant": verdict.is_significant,
                "is_credible": verdict.is_credible, "confidence": verdict.confidence,
                "reasons": list(verdict.reasons),
            }
    report.update({
        "per_period_sharpe": per_period_sharpe,
        "probabilistic_sharpe_ratio": psr,
        "psr_is_credible": psr_is_credible(psr, n_obs),
        "deflated_sharpe_ratio": dsr,
        "deflated_sharpe_promotion": promotion,
        # Stable schema: the data-snooping fields are always present (None when their optional input was not
        # supplied), so a consumer never has to distinguish "absent" from "not computed".
        "probability_of_backtest_overfitting": None,
        "white_reality_check_p_value": None,
        "hansen_spa_p_value": None,
    })
    if candidate_performance is not None:
        report["probability_of_backtest_overfitting"] = probability_of_backtest_overfitting(candidate_performance)
    if benchmark_differentials is not None:
        report["white_reality_check_p_value"] = white_reality_check(benchmark_differentials, seed=seed)
        report["hansen_spa_p_value"] = hansens_spa(benchmark_differentials, seed=seed)
    return report
