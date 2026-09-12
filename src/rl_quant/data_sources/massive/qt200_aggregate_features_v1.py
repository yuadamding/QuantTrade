"""Causal bars-only numeric features for the separately identified QT200 study.

All features at decision t end at exchange session t-1. No future execution
volume, target validity, current-day bar or fit-wide statistic enters them.
Cross-session features cannot bridge missing bars, known identity conflicts,
or candidate corporate actions in the unadjusted series. This is not a claim
that unresolved historical identities/actions have been qualified.
"""

from __future__ import annotations

import itertools
import math
import os
from bisect import bisect_right
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_panel_v1 import SCHEMA as PANEL_SCHEMA, _object
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.execution.qt200_aggregate_execution_v1 import number

SCHEMA = "rl-quant.qt200-bars-only-features-v1"
FEATURE_NAMES = (
    "log_close_open", "log_high_low", "close_location", "log1p_volume",
    "log1p_transactions", "log_vwap_close", "close_log_return_1",
    "close_log_return_5", "close_log_return_20", "close_log_return_63",
    "volume_log_change_1", "close_log_volatility_20",
)


def feature_specification() -> dict:
    return dict(schema=SCHEMA, research_track="QT200-AGG-DEV-01", feature_names=list(FEATURE_NAMES),
        feature_lag_sessions=1, longest_return_sessions=63, volatility_sessions=20,
        volatility_ddof=0, prices="unadjusted", volume_transform="natural-log-one-plus",
        missing_encoding="nullable-value-and-explicit-false-mask",
        zero_range_close_location="missing", cross_session_action_handling="mask-bridge",
        normalization="fit-interval-only-per-feature-population-mean-and-scale",
        target_or_execution_inputs_read=False, native_90_dimension_substitution=False,
        historical_availability="provider-finalized-with-declared-lag-assumption",
        dataset_training_authorization=False)


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Feature overflow/nonfinite value")
    return value


def _log_ratio(numerator, denominator) -> float:
    # Taking logs separately avoids overflow in a binary floating ratio.
    return _finite(math.log(number(numerator, positive=True)) - math.log(number(denominator, positive=True)))


def _bar(row: dict) -> None:
    o, h, low, c = [number(row[key], positive=True) for key in "ohlc"]
    if not low <= min(o, c) <= max(o, c) <= h:
        raise ValueError("Invalid OHLC feature input")
    number(row["v"])
    if row.get("n") is not None:
        n = row["n"]
        if type(n) is not int or n < 0:
            raise ValueError("Invalid observed transaction count")
    if row.get("vw") is not None:
        number(row["vw"], positive=True)


def feature_rows(*, ticker: str, sessions: Sequence[str], bars: Mapping[str, dict],
                 event_dates: Iterable[str] = (), conflict_dates: Iterable[str] = ()) -> list[dict]:
    """Numerical preparation only; eligibility is deliberately not inferred.

    Missingness is a full-calendar property, never a compressed observed-row
    offset. Candidate events mask bridges even before accounting is resolved.
    Optional VWAP and count stay missing when not observed.
    """
    days = list(sessions)
    if not days or days != sorted(set(days)) or any(date.fromisoformat(d).isoformat() != d for d in days):
        raise ValueError("A canonical complete session sequence is required")
    if not set(bars).issubset(days):
        raise ValueError("Off-calendar bar cannot silently enter model features")
    events = sorted(set(event_dates))
    conflicts = set(conflict_dates)
    if any(date.fromisoformat(d).isoformat() != d for d in [*events, *conflicts]):
        raise ValueError("Noncanonical event/identity date")
    for d, row in bars.items():
        if row.get("ticker") != ticker or row.get("session_date_et") != d or row.get("adjusted") is not False:
            raise ValueError("Feature instrument/date/adjustment mismatch")
        _bar(row)
    available = [bars.get(d) if d not in conflicts else None for d in days]
    result = []
    for decision_index, decision_day in enumerate(days):
        end = decision_index - 1
        values: list[float | None] = [None] * len(FEATURE_NAMES)
        current = available[end] if end >= 0 else None

        def window(length):
            start = end - length
            if start < 0 or current is None:
                return None
            if bisect_right(events, days[end]) != bisect_right(events, days[start]):
                return None
            rows = available[start:end + 1]
            return rows if all(row is not None for row in rows) else None

        if current is not None:
            values[0] = _log_ratio(current["c"], current["o"])
            values[1] = _log_ratio(current["h"], current["l"])
            spread = number(current["h"]) - number(current["l"])
            values[2] = _finite(float((number(current["c"]) - number(current["l"])) / spread)) if spread else None
            values[3] = _finite(math.log1p(number(current["v"])))
            if current.get("n") is not None:
                values[4] = _finite(math.log1p(current["n"]))
            if current.get("vw") is not None:
                values[5] = _log_ratio(current["vw"], current["c"])
            for index, length in enumerate((1, 5, 20, 63), 6):
                rows = window(length)
                if rows is not None:
                    values[index] = _log_ratio(rows[-1]["c"], rows[0]["c"])
            rows = window(1)
            if rows is not None:
                values[10] = _finite(math.log1p(number(rows[-1]["v"])) - math.log1p(number(rows[0]["v"])))
            rows = window(20)
            if rows is not None:
                values[11] = _finite(pstdev(_log_ratio(b["c"], a["c"]) for a, b in zip(rows, rows[1:])))
        result.append(dict(ticker=ticker, decision_session=decision_day,
            latest_feature_session=days[end] if end >= 0 else None, values=values,
            observed_mask=[v is not None for v in values], training_eligible=False))
    return result


def fit_normalizer(rows: Sequence[dict], *, fit_sessions: Sequence[str], heldout_start: str) -> dict:
    """Fit exclusively on specified prior sessions; never on a joined full panel."""
    days = list(fit_sessions)
    if (not days or days != sorted(set(days)) or days[-1] >= heldout_start
            or any(date.fromisoformat(d).isoformat() != d for d in [*days, heldout_start])):
        raise ValueError("Normalization fit crosses its held-out boundary")
    columns: list[list[float]] = [[] for _ in FEATURE_NAMES]
    identities = set()
    selected = set(days)
    seen_days = set()
    for row in rows:
        if row["decision_session"] not in selected:
            continue
        key = row["ticker"], row["decision_session"]
        if key in identities:
            raise ValueError("Duplicate fitting observation")
        identities.add(key)
        seen_days.add(key[1])
        feature_day = row.get("latest_feature_session")
        if feature_day is not None and not feature_day < key[1]:
            raise ValueError("Normalization input contains a contemporaneous/future feature")
        values, mask = row["values"], row["observed_mask"]
        if len(values) != len(FEATURE_NAMES) or mask != [v is not None for v in values]:
            raise ValueError("Fit feature/mask contract differs")
        for column, value in zip(columns, values, strict=True):
            if value is not None:
                column.append(_finite(float(value)))
    if seen_days != selected or any(not column for column in columns):
        raise ValueError("Fit interval or feature support is incomplete")
    return dict(feature_names=list(FEATURE_NAMES), fit_sessions=days, heldout_start=heldout_start,
                mean=[fmean(column) for column in columns],
                scale=[pstdev(column) or 1.0 for column in columns],
                counts=[len(column) for column in columns], training_authorized=False)


def _parquet_rows(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def materialize_features(*, panel_root: Path, panel_sha256: str, output: Path) -> dict:
    """Consume only one committed observed panel and publish actual numeric rows.

    Does not read future execution bars or generate outcome labels. Source
    issue continuity and corporate-action accounting remain external gates.
    """
    if (output.resolve() != output or panel_root.resolve() != panel_root or output == panel_root
            or output in panel_root.parents or panel_root in output.parents):
        raise ValueError("Separate canonical output generation required")
    complete_path = panel_root / "COMPLETE.json"
    before = transport.read_regular(complete_path, 4 * 1024 * 1024)
    parent = transport.parse_json(before)
    if (transport.digest(before) != panel_sha256 or parent.get("schema") != PANEL_SCHEMA
            or parent.get("training_ready") is not False or parent.get("ordered_tickers") != list(SYMBOLS)
            or parent.get("source_observation_integration_complete") is not True):
        raise ValueError("Observed panel parent differs")
    names = ("daily-bars.parquet", "session-index.parquet", "candidate-economic-observations.json.gz",
             "decision-clock-plan.json.gz")

    def verify(name):
        proof = parent["files"][name]
        body = transport.read_regular(panel_root / name, 512 * 1024 * 1024)
        if proof["path"] != name or len(body) != proof["bytes"] or transport.digest(body) != proof["sha256"]:
            raise ValueError("Consumed panel file differs")
        return body

    clocks = _object(verify("decision-clock-plan.json.gz"))["clocks"]
    sessions = [row["decision_session"] for row in clocks]
    if len(sessions) != parent["sessions"] or sessions != sorted(set(sessions)):
        raise ValueError("Feature calendar differs")
    for index, row in enumerate(clocks):
        if row["latest_feature_session"] != (sessions[index - 1] if index else None):
            raise ValueError("Feature clock does not lag one complete session")
    verify("daily-bars.parquet")
    verify("session-index.parquet")
    # Event indices are already derived by the parent from both event sources.
    events = _object(verify("candidate-economic-observations.json.gz"))["observations"]
    support = {}
    groups = itertools.groupby(_parquet_rows(panel_root / "session-index.parquet"), lambda r: r["ticker"])
    seen = []
    for ticker, rows in groups:
        seen.append(ticker)
        dates, action_dates, conflicts = [], set(), set()
        for row in rows:
            dates.append(row["session_date"])
            indices = row["candidate_economic_observation_indices"]
            if any(type(i) is not int or not 0 <= i < len(events) for i in indices):
                raise ValueError("Corporate-action index outside parent inventory")
            if indices:
                action_dates.add(row["session_date"])
            if row["identity_evidence"] == "dated_issue_conflict":
                conflicts.add(row["session_date"])
        if dates != sessions:
            raise ValueError("Incomplete/duplicated session rectangle")
        support[ticker] = (action_dates, conflicts)
    if seen != list(SYMBOLS):
        raise ValueError("Feature universe differs")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(mode=0o700)
    schema = pa.schema([("ticker", pa.string()), ("decision_session", pa.string()),
        ("latest_feature_session", pa.string()), ("values", pa.list_(pa.float64(), len(FEATURE_NAMES))),
        ("observed_mask", pa.list_(pa.bool_(), len(FEATURE_NAMES))), ("training_eligible", pa.bool_())])
    counts, seen = Counter(), []
    groups = itertools.groupby(_parquet_rows(panel_root / "daily-bars.parquet"), lambda r: r["ticker"])
    with pq.ParquetWriter(output / "features.parquet", schema, compression="zstd", compression_level=9) as writer:
        for ticker, group in groups:
            seen.append(ticker)
            bars = {}
            for row in group:
                d = row["session_date_et"]
                if d in bars:
                    raise ValueError("Duplicate daily source row")
                bars[d] = row
            action_dates, conflicts = support[ticker]
            rows = feature_rows(ticker=ticker, sessions=sessions, bars=bars,
                                event_dates=action_dates, conflict_dates=conflicts)
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            counts["rows"] += len(rows)
            for row in rows:
                counts["fully_observed_rows"] += all(row["observed_mask"])
                for name, observed in zip(FEATURE_NAMES, row["observed_mask"], strict=True):
                    counts[name] += observed
    if seen != list(SYMBOLS) or counts["rows"] != len(SYMBOLS) * len(sessions):
        raise ValueError("Feature source population incomplete")
    transport.write_once(output / "specification.json", transport.canonical(feature_specification()))
    for name in names:
        verify(name)
    if transport.read_regular(complete_path, 4 * 1024 * 1024) != before:
        raise ValueError("Panel completion changed during feature preparation")
    files = {}
    for name in ("features.parquet", "specification.json"):
        path = output / name
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        body = transport.read_regular(path, 256 * 1024 * 1024)
        files[name] = dict(path=name, bytes=len(body), sha256=transport.digest(body))
    result = dict(schema=SCHEMA, panel_sha256=panel_sha256, files=files,
        feature_names=list(FEATURE_NAMES), counts=dict(counts),
        numeric_feature_preparation_complete=True, source_reverified_before_and_after=True,
        future_execution_inputs_read=False, normalization_fitted=False, targets_generated=False,
        historical_identity_qualified=False, corporate_action_accounting_qualified=False,
        execution_model_qualified=False, forecast_pipeline_qualified=False,
        joint_h100_runtime_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
