"""Shared partition-window protocol.

Parses ``YYYY-MM-DD`` / ``YYYY-MM-DD_to_YYYY-MM-DD`` partition labels into sortable half-open spans
and enforces the strict latest-period selection invariants. Imported by BOTH the protocol-partition
and calendar-holdout training scripts so the two latest-period gates cannot drift apart -- a fix here
stays fixed in both paths.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

_LABEL_RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:_to_(\d{4}-\d{2}-\d{2}))?")


def label_span(label: str) -> tuple[datetime, datetime] | None:
    """Parse a partition label into a half-open ``[start, end)`` calendar span.

    The real label format is ``<start>_to_<end>`` with ``end`` exclusive (the builder sets
    ``end = last_trading_day + 1``); a bare ``<date>`` label is the one-day window
    ``[date, date + 1 day)`` (a partition holds at least a day of data, so it is never empty).
    The pattern is FULLY anchored (``fullmatch``), so any trailing garbage or non-range suffix
    (``2026-01-01abc``, ``..._to_..._v2``) is rejected rather than silently truncated. Returns
    ``None`` when a date is not a real calendar date or when an explicit range has ``end <= start``
    (an empty/inverted range is malformed, not a sortable window)."""
    match = _LABEL_RANGE.fullmatch(label)
    if not match:
        return None
    try:
        start = datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return None
    if match.group(2) is None:
        return (start, start + timedelta(days=1))
    try:
        end = datetime.strptime(match.group(2), "%Y-%m-%d")
    except ValueError:
        return None
    return (start, end) if end > start else None


def chronological_latest_label(labels: list[str]) -> str | None:
    """Latest label by its WINDOW-END date (most recent data), independent of input/lexicographic order.

    Labels are ``<start>_to_<end>`` ranges (or bare dates). Ranking by the start prefix alone would
    crown a window whose data ENDS earlier than an overlapping sibling (a wide backfill vs a short
    tail), so rank by parsed end, then start, then position. Directory (lexicographic) order is also
    unreliable for unsortable suffixes (``2026-06-15_v10`` sorts before ``_v2``). Unparseable labels
    are ignored; if none parse we fall back to the last given label."""
    ranked: list[tuple[datetime, datetime, int, str]] = []
    for index, label in enumerate(labels):
        span = label_span(label)
        if span is not None:
            ranked.append((span[1], span[0], index, label))
    if not ranked:
        return labels[-1] if labels else None
    return max(ranked, key=lambda item: (item[0], item[1], item[2]))[3]


def strict_latest_partition_violations(
    *,
    selected_labels: list[str],
    all_available_labels: list[str],
    allow_truncated_training_history: bool,
) -> list[str]:
    """Strict latest-period reporting violations; empty list means the selection is admissible.

    Enforces the stated protocol -- latest periods for test, ALL earlier periods for train/validation:
      1. Every available AND selected label must be a real ISO calendar window -- a ``<date>`` (a
         one-day window) or ``<start>_to_<end>`` with end > start -- so spans are sortable (rejects
         ``partition_9``, ``2026-99-99``, ``2026-01-01_to_garbage``, suffixes, empty/inverted ranges);
         same-start-date distinct labels are ambiguous. Duplicate and unknown selected labels are
         always reported (independent of ``allow_truncated_training_history``).
      2. Available windows must be a strictly non-overlapping walk-forward (rejects a window contained
         in or overlapping another, which would make the latest period ambiguous and leak train/test).
      3. The final selected partition must be the latest available one, ranked by WINDOW END (most
         recent data) -- so no OLD period is reported as the headline latest-period test.
      4. No earlier available partition may be silently excluded from the train/validation history,
         unless ``allow_truncated_training_history`` explicitly permits it.
    """
    violations: list[str] = []
    # Selection-level checks that do NOT depend on chronological ordering are collected FIRST, so they
    # still surface even when label parsing/ordering below fails closed (avoids fix-one / rerun churn).
    duplicates = sorted(label for label, count in Counter(selected_labels).items() if count > 1)
    if duplicates:
        violations.append(f"selected partitions contain duplicate labels: {duplicates[:5]}")
    available_set = set(all_available_labels)
    unknown_selected = sorted({label for label in selected_labels if label not in available_set})
    if unknown_selected:
        # Always an error, in every mode: a selected label not present on disk is never admissible.
        violations.append(f"selected partitions contain unknown labels not present on disk: {unknown_selected[:5]}")
    # A label (available OR selected) must be a real ISO calendar window -- a <date> (one-day window)
    # or <start>_to_<end> with end > start -- so spans are sortable real windows (rejects partition_9,
    # 2026-99-99, 2026-01-01_to_garbage, trailing-suffix labels, and empty/inverted ranges).
    invalid_labels = sorted({label for label in (*all_available_labels, *selected_labels) if label_span(label) is None})
    if invalid_labels:
        violations.append(
            "partition labels must be real ISO calendar windows -- <date> or <start>_to_<end> with "
            f"end > start; got invalid labels: {invalid_labels[:5]}"
        )
        return violations
    distinct_labels = list(dict.fromkeys(all_available_labels))
    # Chronological order must be UNAMBIGUOUS from the label. The start prefix carries only day
    # granularity, so two distinct labels sharing a START date (e.g. a rebuild leaving two windows that
    # begin on the same day) cannot be ordered from the label. Fail closed: the latest-period test is
    # then undefined.
    by_start: dict[str, list[str]] = {}
    for label in distinct_labels:
        by_start.setdefault(label[:10], []).append(label)
    ambiguous_starts = sorted(start for start, group in by_start.items() if len(group) > 1)
    if ambiguous_starts:
        violations.append(
            "partition labels are not chronologically unambiguous; multiple distinct labels share a "
            f"start date, so the latest-period test cannot be determined from the label: "
            f"{[label for start in ambiguous_starts[:3] for label in by_start[start]][:6]} "
            "(use non-overlapping <start>_to_<end> windows, or record a revision in the partition "
            "manifest -- directory-label suffixes are rejected in strict mode)"
        )
        return violations
    # Windows must be a strictly NON-overlapping walk-forward. Labels are half-open [start, end) ranges;
    # an overlapping/contained window (e.g. a short window inside a wide backfill from mixed
    # --skip-existing builds) makes "the latest period" ambiguous by date order alone and risks
    # train/test leakage. Fail closed. (Adjacent windows share at most a boundary, end_i <= start_{i+1},
    # so they pass.) The diagnostic names both the overlapping window and the container it overlaps.
    spans = sorted(((label_span(label), label) for label in distinct_labels), key=lambda item: item[0])
    running_max_end: datetime | None = None
    active_label: str | None = None
    overlapping: list[str] = []
    for (start, end), label in spans:
        if running_max_end is not None and start < running_max_end:
            overlapping.append(f"{label} overlaps {active_label}")
        if running_max_end is None or end > running_max_end:
            running_max_end = end
            active_label = label
    if overlapping:
        violations.append(
            "partition windows overlap (a proper walk-forward must be strictly non-overlapping); "
            f"overlapping/contained windows: {overlapping[:5]}"
        )
        return violations
    # With a clean, ordered universe, collect the remaining independent violations. Latest available is
    # by PARSED window end, not the caller's positional ordering (see chronological_latest_label).
    latest_available = chronological_latest_label(all_available_labels)
    if selected_labels and latest_available is not None and selected_labels[-1] != latest_available:
        violations.append(
            f"final selected partition ({selected_labels[-1]}) is not the latest available partition "
            f"({latest_available})"
        )
    if not allow_truncated_training_history:
        # Every available partition before the test must be present (rejects a skipped earliest OR
        # middle partition); unknown selected labels were already reported above, in every mode.
        selected_set = set(selected_labels)
        missing = [label for label in distinct_labels if label not in selected_set]
        if missing:
            violations.append(
                f"selected partitions do not cover all available history; {len(missing)} earlier/middle "
                f"partition(s) are silently excluded from train/validation: {missing[:5]} "
                "(pass --allow-truncated-training-history to override)"
            )
    return violations


@dataclass(frozen=True)
class PartitionWindow:
    """A partition's half-open ``[start, end_exclusive)`` calendar window plus its directory label."""

    label: str
    start: datetime
    end_exclusive: datetime
    complete: bool = True


@dataclass(frozen=True)
class PartitionSplit:
    """A reportable train/validation/test split expressed as contiguous partition blocks."""

    train: list[PartitionWindow]
    val: list[PartitionWindow]
    test: list[PartitionWindow]


def partition_windows_from_labels(labels: list[str], *, complete: bool = True) -> list[PartitionWindow]:
    """Parse directory labels into ``PartitionWindow`` objects via :func:`label_span`.

    Raises on ANY malformed label: a reportable split must not silently drop unparseable partitions
    (that would let a stale window masquerade as the latest by simply being unreadable)."""
    windows: list[PartitionWindow] = []
    for label in labels:
        span = label_span(label)
        if span is None:
            raise ValueError(f"unparseable partition label {label!r}; cannot derive a reportable split.")
        windows.append(PartitionWindow(label=label, start=span[0], end_exclusive=span[1], complete=complete))
    return windows


def derive_reportable_partition_split(
    available_windows: list[PartitionWindow],
    *,
    val_count: int,
    test_count: int,
    allow_truncated_training_history: bool = False,
    train_window_count: int | None = None,
) -> PartitionSplit:
    """Deterministically derive the reportable train/val/test partition split (the hard rule).

    TEST is the latest complete partition suffix, VALIDATION is the immediately preceding block, and
    TRAIN is everything earlier (optionally truncated to the most recent ``train_window_count`` blocks,
    which must be explicitly allowed). Incomplete windows are dropped; the remaining windows are ordered
    by ``(end_exclusive, start, label)`` and MUST be strictly non-overlapping (a clean walk-forward) --
    otherwise "latest" is ill-defined. This is the single source of truth: the split is computed from
    ALL available complete partitions, never from a caller-restricted subset."""
    if val_count <= 0 or test_count <= 0:
        raise ValueError(f"val_count and test_count must be positive; got val={val_count}, test={test_count}.")
    if train_window_count is not None:
        if not allow_truncated_training_history:
            raise ValueError("training-history truncation (train_window_count) must be explicitly allowed.")
        if train_window_count <= 0:
            raise ValueError(f"train_window_count must be positive; got {train_window_count}.")
    ordered = sorted((w for w in available_windows if w.complete), key=lambda w: (w.end_exclusive, w.start, w.label))
    running_max_end: datetime | None = None
    for window in ordered:
        if running_max_end is not None and window.start < running_max_end:
            raise ValueError(f"overlapping partition windows are not a valid walk-forward: {window.label!r}")
        running_max_end = window.end_exclusive if running_max_end is None else max(running_max_end, window.end_exclusive)
    if len(ordered) < val_count + test_count + 1:
        raise ValueError(
            f"need at least {val_count + test_count + 1} complete non-overlapping partitions for a "
            f"train(>=1)/val({val_count})/test({test_count}) split; got {len(ordered)}."
        )
    test = ordered[-test_count:]
    val = ordered[-(test_count + val_count) : -test_count]
    train = ordered[: -(test_count + val_count)]
    if train_window_count is not None:
        train = train[-train_window_count:]
    return PartitionSplit(train=train, val=val, test=test)
