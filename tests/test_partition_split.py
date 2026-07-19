from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from rl_quant.protocol.partition import PartitionWindow, derive_reportable_partition_split


def test_derived_partition_split_is_deeply_immutable() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = [
        PartitionWindow(
            label=f"window-{index}",
            start=start + timedelta(days=index),
            end_exclusive=start + timedelta(days=index + 1),
        )
        for index in range(5)
    ]

    split = derive_reportable_partition_split(windows, val_count=1, test_count=1)

    assert isinstance(split.train, tuple)
    assert tuple(window.label for window in split.train) == ("window-0", "window-1", "window-2")
    assert tuple(window.label for window in split.val) == ("window-3",)
    assert tuple(window.label for window in split.test) == ("window-4",)
    with pytest.raises(FrozenInstanceError):
        split.train = ()  # type: ignore[misc]
