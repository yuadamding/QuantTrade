"""Source-bound price labels for the separate, finalized-bars research track.

These are gross price-prediction labels, NOT native V5 total-return/factor
targets, fills, or profitability. Candidate actions and known issue conflicts
mask the affected intervals until event/identity accounting is qualified.
Labels are stored separately from causal features and carry per-bucket maturity.
"""

from __future__ import annotations

import itertools
import math
import os
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_features_v1 import (
    FEATURE_NAMES, SCHEMA as FEATURE_SCHEMA, _bar, _parquet_rows, feature_specification,
)
from rl_quant.data_sources.massive.qt200_aggregate_panel_v1 import SCHEMA as PANEL_SCHEMA, _object
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.execution.qt200_aggregate_execution_v1 import validate_window

SCHEMA = "rl-quant.qt200-bars-price-targets-v1"
BOUNDARIES = (0, 1, 5, 10, 21, 42, 63, 126)
BUCKET_NAMES = tuple(f"price_{a}_{b}" for a, b in zip(BOUNDARIES, BOUNDARIES[1:]))
REASONS = dict(future_calendar_incomplete=1, daily_path_missing=2,
               start_window_missing=4, end_window_missing=8,
               candidate_action_unresolved=16, known_issue_conflict=32,
               maturity_outside_calendar=64)


def target_specification() -> dict:
    return dict(schema=SCHEMA, research_track="QT200-AGG-DEV-01",
        bucket_names=list(BUCKET_NAMES), boundaries=list(BOUNDARIES), entry_lag_sessions=1,
        label="gross-simple-price-return-between-nonoverlapping-boundaries",
        price_proxy="09:35-minute-open-with-complete-09:35-to-09:45-observed-window",
        price_proxy_is_vwap=False, price_proxy_is_a_fill=False, transaction_costs_in_label=False,
        support="all-daily-marks-in-bucket-and-both-boundary-windows",
        candidate_actions="mask-any-candidate-including-boundary-dates",
        known_issue_conflicts="mask-inclusive-bucket-interval",
        missing_encoding="nullable-value-false-mask-and-reason-bitset",
        label_maturity="after-close-next-exchange-session-after-bucket-end",
        historical_availability="provider-finalized-with-declared-lag-assumption",
        actual_historical_revision_availability_proven=False,
        costs_dividends_splits_terminal_outcomes="not-inferred-from-price-labels",
        normalization_fitted=False, native_v5_target_substitution=False,
        dataset_training_authorization=False)


def target_rows(*, ticker: str, sessions: Sequence[str], daily_sessions: set[str],
                window_open: Mapping[str, float], event_dates: Sequence[str] = (),
                conflict_dates: Sequence[str] = ()) -> list[dict]:
    """Keep every calendar origin and every bucket; never compress missing time."""
    days = list(sessions)
    if (not ticker or not days or days != sorted(set(days))
            or any(date.fromisoformat(d).isoformat() != d for d in days)
            or not (set(daily_sessions) | set(window_open)).issubset(days)):
        raise ValueError("Target calendar/instrument differs")
    events, conflicts = sorted(set(event_dates)), sorted(set(conflict_dates))
    if any(date.fromisoformat(d).isoformat() != d for d in events + conflicts):
        raise ValueError("Target event/identity date differs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in window_open.values()):
        raise ValueError("Invalid observed target window price")
    # Prefix sums avoid repeatedly scanning all 126 sessions per origin.
    absent = [0]
    for day in days:
        absent.append(absent[-1] + (day not in daily_sessions))
    result = []
    for i, day in enumerate(days):
        values, reasons, starts, ends, maturity = [], [], [], [], []
        for lo, hi in zip(BOUNDARIES, BOUNDARIES[1:]):
            a, b = i + 1 + lo, i + 1 + hi
            first = days[a] if a < len(days) else None
            last = days[b] if b < len(days) else None
            available = days[b + 1] if b + 1 < len(days) else None
            reason = 0
            value = None
            if last is None:
                reason |= REASONS["future_calendar_incomplete"]
            else:
                if absent[b + 1] != absent[a]:
                    reason |= REASONS["daily_path_missing"]
                if first not in window_open:
                    reason |= REASONS["start_window_missing"]
                if last not in window_open:
                    reason |= REASONS["end_window_missing"]
                # Include the first boundary too: no assumed ex-date timing.
                for dates, name in ((events, "candidate_action_unresolved"), (conflicts, "known_issue_conflict")):
                    if bisect_right(dates, last) != bisect_left(dates, first):
                        reason |= REASONS[name]
                if available is None:
                    reason |= REASONS["maturity_outside_calendar"]
                if not reason:
                    value = math.expm1(math.log(window_open[last]) - math.log(window_open[first]))
                    if not math.isfinite(value) or value <= -1:
                        raise ValueError("Target numeric range differs")
            values.append(value)
            reasons.append(reason)
            starts.append(first)
            ends.append(last)
            maturity.append(available)
        result.append(dict(ticker=ticker, decision_session=day,
            entry_session=days[i + 1] if i + 1 < len(days) else None,
            bucket_start_sessions=starts, bucket_end_sessions=ends,
            label_maturity_sessions=maturity, values=values,
            observed_mask=[v is not None for v in values], reason_bits=reasons,
            training_eligible=False))
    return result


def read_completion(root: Path, digest: str, schema: str) -> tuple[bytes, dict]:
    if root.resolve() != root:
        raise ValueError("Canonical evidence path required")
    body = transport.read_regular(root / "COMPLETE.json", 4 * 1024 * 1024)
    result = transport.parse_json(body)
    if (transport.digest(body) != digest or result.get("schema") != schema
            or result.get("training_ready") is not False):
        raise ValueError("Preparation parent differs")
    return body, result


def verify_artifact(root: Path, parent: dict, name: str) -> bytes:
    proof = parent["files"][name]
    if proof["path"] != name or Path(name).name != name:
        raise ValueError("Evidence file path differs")
    body = transport.read_regular(root / name, 512 * 1024 * 1024)
    if len(body) != proof["bytes"] or transport.digest(body) != proof["sha256"]:
        raise ValueError("Consumed evidence file differs")
    return body


def materialize_targets(*, panel_root: Path, panel_sha256: str, feature_root: Path,
                        feature_sha256: str, output: Path) -> dict:
    """Derive labels and feature row links from accepted, read-only sources.

    No forecast model is fitted. A completed preparation remains nonauthorizing
    even for an interval without a *known* action or dated identity conflict.
    """
    roots = (panel_root, feature_root, output)
    if any(p.resolve() != p for p in roots) or any(a == b or a in b.parents or b in a.parents
            for a, b in itertools.combinations(roots, 2)):
        raise ValueError("Separate canonical generations required")
    pbody, panel = read_completion(panel_root, panel_sha256, PANEL_SCHEMA)
    fbody, feature = read_completion(feature_root, feature_sha256, FEATURE_SCHEMA)
    if (panel.get("ordered_tickers") != list(SYMBOLS)
            or panel.get("source_observation_integration_complete") is not True
            or feature.get("panel_sha256") != panel_sha256
            or feature.get("numeric_feature_preparation_complete") is not True
            or feature.get("feature_names") != list(FEATURE_NAMES)
            or feature.get("normalization_fitted") is not False
            or feature.get("future_execution_inputs_read") is not False):
        raise ValueError("Panel/feature identity or preparation contract differs")
    pnames = ("daily-bars.parquet", "minute-window-bars.parquet", "session-index.parquet",
              "candidate-economic-observations.json.gz", "decision-clock-plan.json.gz")
    for name in pnames:
        verify_artifact(panel_root, panel, name)
    for name in ("features.parquet", "specification.json"):
        verify_artifact(feature_root, feature, name)
    if transport.parse_json(verify_artifact(feature_root, feature, "specification.json")) != feature_specification():
        raise ValueError("Feature specification differs")
    clocks = _object(verify_artifact(panel_root, panel, "decision-clock-plan.json.gz"))["clocks"]
    days = [row["decision_session"] for row in clocks]
    if len(days) != panel["sessions"] or days != sorted(set(days)):
        raise ValueError("Target calendar differs")
    calendar_days = set(days)
    for i, row in enumerate(clocks):
        if (row["latest_feature_session"] != (days[i - 1] if i else None)
                or row["execution_session"] != (days[i + 1] if i + 1 < len(days) else None)):
            raise ValueError("Target/feature clock mismatch")
    events = _object(verify_artifact(panel_root, panel, "candidate-economic-observations.json.gz"))["observations"]
    # Keep at most compact window summaries in memory, not millions of row dicts.
    windows = defaultdict(dict)
    minute_count = 0
    minute_groups = itertools.groupby(_parquet_rows(panel_root / "minute-window-bars.parquet"),
                                     lambda r: (r["ticker"], r["session_date_et"]))
    for (ticker, day), group in minute_groups:
        rows = list(group)
        if day not in calendar_days or day in windows[ticker] or any(r["adjusted"] is not False for r in rows):
            raise ValueError("Duplicate/off-calendar/adjusted minute source")
        validated = validate_window(rows, day)
        windows[ticker][day] = dict(start=minute_count, count=len(rows),
            price=float(validated[0][1]) if len(rows) == 10 else None)
        minute_count += len(rows)
    if minute_count != panel["counts"]["minute_rows"]:
        raise ValueError("Minute population differs")
    daily = defaultdict(dict)
    daily_count = 0
    for row in _parquet_rows(panel_root / "daily-bars.parquet"):
        ticker, day = row["ticker"], row["session_date_et"]
        if ticker not in SYMBOLS or day not in calendar_days or day in daily[ticker] or row["adjusted"] is not False:
            raise ValueError("Duplicate/off-panel/adjusted daily source")
        _bar(row)
        daily[ticker][day] = daily_count
        daily_count += 1
    if daily_count != panel["counts"]["daily_rows"]:
        raise ValueError("Daily population differs")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(mode=0o700)
    fields = [("ticker", pa.string()), ("decision_session", pa.string()), ("entry_session", pa.string()),
              ("feature_row_index", pa.int64())]
    fields += [(name, pa.list_(pa.string(), 7)) for name in
               ("bucket_start_sessions", "bucket_end_sessions", "label_maturity_sessions")]
    fields += [("values", pa.list_(pa.float64(), 7)), ("observed_mask", pa.list_(pa.bool_(), 7)),
               ("reason_bits", pa.list_(pa.uint16(), 7)), ("training_eligible", pa.bool_())]
    schema = pa.schema(fields)
    counts, observed, reason_counts, by_symbol = Counter(), Counter(), Counter(), []
    groups = itertools.groupby(_parquet_rows(panel_root / "session-index.parquet"), lambda r: r["ticker"])
    fgroups = itertools.groupby(_parquet_rows(feature_root / "features.parquet"), lambda r: r["ticker"])
    seen = []
    with pq.ParquetWriter(output / "targets.parquet", schema, compression="zstd", compression_level=9) as writer:
        for (ticker, group), (fticker, fgroup) in zip(groups, fgroups, strict=True):
            index, features = list(group), list(fgroup)
            seen.append(ticker)
            if (ticker != fticker or [r["session_date"] for r in index] != days
                    or [r["decision_session"] for r in features] != days):
                raise ValueError("Feature/target rectangle differs")
            action_days, conflict_days = [], []
            for i, (row, frow) in enumerate(zip(index, features, strict=True)):
                day = days[i]
                indices = row["candidate_economic_observation_indices"]
                if any(type(n) is not int or not 0 <= n < len(events) for n in indices):
                    raise ValueError("Candidate event index differs")
                if indices:
                    action_days.append(day)
                if row["identity_evidence"] == "dated_issue_conflict":
                    conflict_days.append(day)
                window = windows[ticker].get(day, {})
                if (row["daily_row"] != daily[ticker].get(day)
                        or row["window_row_start"] != window.get("start")
                        or row["window_row_count"] != window.get("count", 0)
                        or frow["latest_feature_session"] != clocks[i]["latest_feature_session"]
                        or len(frow["values"]) != len(FEATURE_NAMES)
                        or frow["observed_mask"] != [v is not None for v in frow["values"]]
                        or frow["training_eligible"] is not False):
                    raise ValueError("Source row link/mask/feature clock differs")
            rows = target_rows(ticker=ticker, sessions=days, daily_sessions=set(daily[ticker]),
                window_open={d: r["price"] for d, r in windows[ticker].items() if r["price"] is not None},
                event_dates=action_days, conflict_dates=conflict_days)
            symbol_count = Counter()
            for i, row in enumerate(rows):
                row["feature_row_index"] = counts["rows"]
                counts["rows"] += 1
                symbol_count["fully_observed_target_rows"] += all(row["observed_mask"])
                symbol_count["rows_with_any_observed_target"] += any(row["observed_mask"])
                for name, valid, reason in zip(BUCKET_NAMES, row["observed_mask"], row["reason_bits"], strict=True):
                    observed[name] += valid
                    for why, bit in REASONS.items():
                        reason_counts[why] += bool(reason & bit)
                counts["fully_observed_features_and_any_target"] += all(features[i]["observed_mask"]) and any(row["observed_mask"])
            counts.update(symbol_count)
            by_symbol.append(dict(ticker=ticker, **symbol_count))
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    if seen != list(SYMBOLS) or counts["rows"] != len(SYMBOLS) * len(days) or counts["rows"] != feature["counts"]["rows"]:
        raise ValueError("Target population differs")
    transport.write_once(output / "specification.json", transport.canonical(target_specification()))
    for name in pnames:
        verify_artifact(panel_root, panel, name)
    for name in ("features.parquet", "specification.json"):
        verify_artifact(feature_root, feature, name)
    if (transport.read_regular(panel_root / "COMPLETE.json", 4 * 1024 * 1024) != pbody
            or transport.read_regular(feature_root / "COMPLETE.json", 4 * 1024 * 1024) != fbody):
        raise ValueError("Preparation parent changed during target generation")
    files = {}
    for name in ("targets.parquet", "specification.json"):
        path = output / name
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        body = transport.read_regular(path, 256 * 1024 * 1024)
        files[name] = dict(path=name, bytes=len(body), sha256=transport.digest(body))
    result = dict(schema=SCHEMA, panel_sha256=panel_sha256, feature_sha256=feature_sha256,
        ordered_tickers=list(SYMBOLS), sessions=days, files=files, counts=dict(counts),
        observed_by_bucket=dict(observed), unresolved_reason_counts=dict(reason_counts),
        by_symbol=by_symbol, targets_generated=True, numeric_target_preparation_complete=True,
        feature_inputs_unchanged=True, source_reverified_before_and_after=True,
        label_maturity_is_assumed=True, normalization_fitted=False, forecast_fitted=False,
        historical_identity_qualified=False, corporate_action_accounting_qualified=False,
        joint_h100_runtime_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
