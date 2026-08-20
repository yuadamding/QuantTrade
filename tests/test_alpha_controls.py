from __future__ import annotations

import numpy as np
import pytest

from rl_quant.evaluation.alpha_controls import (
    fit_ridge_baseline,
    fixed_random_score,
    permute_targets_within_date,
    reversal_score,
    shift_targets_without_wrap,
    trailing_return_score,
)


def test_trailing_momentum_uses_only_complete_declared_history() -> None:
    returns = np.arange(2 * 3 * 6, dtype=np.float64).reshape(2, 3, 6) / 1_000.0
    valid = np.ones_like(returns, dtype=np.bool_)
    valid[0, 1, 2] = False

    score, score_valid = trailing_return_score(
        returns,
        valid,
        lookback_sessions=3,
        skip_recent_sessions=1,
    )

    np.testing.assert_allclose(score[0, 0], returns[0, 0, 2:5].sum())
    assert not score_valid[0, 1]
    assert score[0, 1] == 0.0


def test_reversal_is_exact_negative_momentum() -> None:
    returns = np.arange(12, dtype=np.float64).reshape(1, 2, 6) / 100.0
    valid = np.ones_like(returns, dtype=np.bool_)
    momentum, momentum_valid = trailing_return_score(
        returns, valid, lookback_sessions=3
    )
    reversal, reversal_valid = reversal_score(
        returns, valid, lookback_sessions=3
    )

    np.testing.assert_array_equal(momentum_valid, reversal_valid)
    np.testing.assert_allclose(reversal, -momentum)


def test_ridge_control_recovers_planted_linear_signal() -> None:
    feature = np.linspace(-2.0, 2.0, 20)[:, None]
    target = 0.3 + 2.0 * feature[:, 0]
    prediction = fit_ridge_baseline(
        feature,
        target,
        np.asarray(((-3.0,), (3.0,))),
        ridge_penalty=1e-8,
    )

    np.testing.assert_allclose(prediction, (-5.7, 6.3), atol=1e-7, rtol=0.0)


def test_fixed_random_control_is_reproducible_and_asset_bound() -> None:
    first = fixed_random_score(("A", "B", "C"), seed=17)
    second = fixed_random_score(("A", "B", "C"), seed=17)
    changed = fixed_random_score(("A", "B", "C"), seed=18)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, changed)


def test_target_permutation_preserves_each_date_multiset_and_invalid_cells() -> None:
    targets = np.asarray(((1.0, 2.0, 99.0, 3.0), (4.0, 5.0, 6.0, 7.0)))
    valid = np.asarray(((True, True, False, True), (True, True, True, True)))
    permuted = permute_targets_within_date(targets, valid, seed=3)

    assert sorted(permuted[0, valid[0]]) == [1.0, 2.0, 3.0]
    assert sorted(permuted[1, valid[1]]) == [4.0, 5.0, 6.0, 7.0]
    assert permuted[0, 2] == 99.0


@pytest.mark.parametrize("periods", (1, -1))
def test_target_shift_never_wraps(periods: int) -> None:
    targets = np.arange(12, dtype=np.float64).reshape(4, 3)
    valid = np.ones_like(targets, dtype=np.bool_)
    shifted, shifted_valid = shift_targets_without_wrap(
        targets, valid, periods=periods
    )

    if periods > 0:
        assert not shifted_valid[0].any()
        np.testing.assert_array_equal(shifted[1], targets[0])
    else:
        assert not shifted_valid[-1].any()
        np.testing.assert_array_equal(shifted[-2], targets[-1])
