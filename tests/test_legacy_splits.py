from __future__ import annotations

import math

import pytest

from rl_quant.datasets.splits import split_days, time_split


def _window(*dates: str) -> dict[str, object]:
    return {"dates": list(dates), "n_days": len(dates)}


@pytest.mark.parametrize(
    ("train_frac", "val_frac"),
    [
        (-0.1, 0.2),
        (0.0, 0.2),
        (0.5, -0.1),
        (1.0, 0.0),
        (1.1, 0.0),
        (0.5, 1.1),
        (0.8, 0.3),
        (0.8, 0.2),
        (math.nan, 0.1),
        (0.5, math.inf),
        (True, 0.1),
    ],
)
def test_time_split_rejects_invalid_fractions(train_frac: float, val_frac: float) -> None:
    with pytest.raises(ValueError):
        time_split(list(range(10)), train_frac, val_frac)


def test_time_split_produces_three_non_empty_chronological_splits() -> None:
    train, validation, test = time_split(list(range(10)), 0.6, 0.2)
    assert train == list(range(6))
    assert validation == [6, 7]
    assert test == [8, 9]


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_time_split_rejects_geometry_that_rounds_any_split_to_empty(size: int) -> None:
    with pytest.raises(ValueError, match="non-empty train, validation, and test"):
        time_split(list(range(size)), 0.5, 0.1)


def test_time_split_rejects_zero_validation_fraction_as_an_empty_split() -> None:
    with pytest.raises(ValueError, match="validation=0"):
        time_split(list(range(10)), 0.5, 0.0)


def test_intraday_split_rejects_a_date_repeated_across_window_boundaries() -> None:
    windows = [
        _window("2024-01-01", "2024-01-02"),
        _window("2024-01-02", "2024-01-03"),
        _window("2024-01-04"),
    ]
    with pytest.raises(ValueError, match="duplicate decision dates.*train/validation"):
        split_days(windows, "intraday", 1 / 3, 1 / 3)  # type: ignore[arg-type]


def test_split_days_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        split_days([], "typo", 0.5, 0.25)
