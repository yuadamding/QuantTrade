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
from statistics import NormalDist

_NORMAL = NormalDist()
# Euler-Mascheroni constant, used in the expected-maximum-of-N-Gaussians approximation.
_EULER_MASCHERONI = 0.5772156649015329


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return value


def _non_normality_denominator(sharpe: float, skewness: float, kurtosis: float) -> float:
    # sqrt(1 - skew*SR + (kurtosis-1)/4 * SR^2) -- the standard-error adjustment for non-normal returns
    # (kurtosis is NON-excess: 3 for a normal distribution). Clamped > 0 to stay defined.
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
