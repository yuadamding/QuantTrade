"""Compact, calendar-keyed observed inputs for QT200-AGG-DEV-01.

This consumes accepted normalization receipts, NOT a native/PIT authority.
Daily and minute observations remain separate tables. The session index never
stitches aliases, imputes bars, computes returns across missing days, or turns
aggregate volume into executable liquidity. Decimal strings are lossless.
"""

from __future__ import annotations

import gzip
import io
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_calendar_v1 import PINNED, clock_rows
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import EPOCH, _bound, normalize_bar
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import END, START, SYMBOLS, DailyQuery
from rl_quant.data_sources.massive.qt200_daily_reference_v1 import _brackets, _calendar, _identity_class
from rl_quant.data_sources.massive.qt200_minute_capture_v1 import minute_queries
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import ALIASES, SupplementQuery

SCHEMA = "rl-quant.qt200-aggregate-observed-panel-v1"
_BAR_FIELDS = ("ticker", "product", "bar_start_ms", "session_date_et", "o", "h", "l", "c", "v",
               "vw", "n", "vwap_observed", "transaction_count_observed", "adjusted", "observed_at_ns",
               "historical_available_at_ns", "raw_row_sha256")


def _rows(body: bytes):
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
        total = 0
        for line in stream:
            total += len(line)
            if len(line) > 1_048_576 or total > 2_000_000_000:
                raise transport.ResearchCaptureError("Normalized input inflation exceeds bound")
            yield transport.parse_json(line)


def _object(body: bytes) -> dict:
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise transport.ResearchCaptureError("Input object exceeds bound")
    return transport.parse_json(raw)


def _complete(root: Path, expected: str, schema: str) -> tuple[dict, bytes]:
    raw = transport.read_regular(root / "COMPLETE.json", 2 * 1024 * 1024)
    obj = transport.parse_json(raw)
    if transport.digest(raw) != expected or obj.get("schema") != schema:
        raise transport.ResearchCaptureError("Parent completion/schema differs")
    if obj.get("training_ready") is not False:
        raise transport.ResearchCaptureError("Research parent cannot assert training readiness")
    return obj, raw


def _observed_bar(row: dict, query) -> dict:
    """Recheck schema without rewriting the retained raw-source identity."""
    source = row["source"]
    if (row.get("adjusted") is not False or row.get("historical_available_at_ns") is not None
            or type(source.get("received_at_ns")) is not int
            or row.get("observed_at_ns") != source["received_at_ns"]):
        raise transport.ResearchCaptureError("Observation/availability or adjustment differs")
    raw = {k: row[k] for k in "ohlcv"}
    raw["t"] = row["bar_start_ms"]
    for key, mask in (("vw", "vwap_observed"), ("n", "transaction_count_observed")):
        if type(row.get(mask)) is not bool or (row.get(key) is not None) != row[mask]:
            raise transport.ResearchCaptureError("Optional observation mask differs")
        if row[mask]:
            raw[key] = row[key]
    checked = normalize_bar(raw, query, received_at_ns=source["received_at_ns"])
    for key in _BAR_FIELDS:
        if key != "raw_row_sha256" and checked[key] != row[key]:
            raise transport.ResearchCaptureError("Normalized bar fields differ")
    for digest in (row["raw_row_sha256"], source.get("page_sha256")):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise transport.ResearchCaptureError("Source-row/page identity missing")
    if any(type(source.get(k)) is not int or source[k] < 0 for k in ("page", "row")):
        raise transport.ResearchCaptureError("Source position invalid")
    return {**{k: row[k] for k in _BAR_FIELDS},
            "source_json": transport.canonical(source).decode("ascii").rstrip("\n")}


def _window_facts(rows: list[dict], day: str) -> dict:
    minutes = []
    for row in rows:
        stamp = (EPOCH + timedelta(milliseconds=row["bar_start_ms"])).astimezone(transport.ET)
        if stamp.date().isoformat() != day or stamp.hour != 9 or not 35 <= stamp.minute < 45:
            raise transport.ResearchCaptureError("Minute row outside session execution window")
        minutes.append(stamp.minute)
    if minutes != sorted(set(minutes)):
        raise transport.ResearchCaptureError("Duplicate/unordered window slots")
    missing = [m for m in range(35, 45) if m not in minutes]
    return dict(observed_slots=len(minutes), missing_slot_mask=sum(1 << (m - 35) for m in missing),
        missing_slots=missing, complete=not missing,
        observed_volume_sum=str(sum((Decimal(r["v"]) for r in rows), Decimal(0))) if rows else None,
        observed_vwap_slots=sum(r["vwap_observed"] for r in rows))


def _identity_evidence(resolution: dict, ticker: str, day: str, extra: list[dict]) -> str:
    base = _identity_class(resolution, ticker, day, _brackets(resolution))
    # Issuer query results of OTHER tickers/classes must not veto or qualify
    # this ticker/date. A matching CIK never establishes issue continuity.
    exact = [r for r in extra if r["query_date"] == day and r["provider_row"].get("ticker") == ticker]
    if base == "dated_issue_conflict" or any(r["classification"] == "conflicting_issue_identifiers" for r in exact):
        return "dated_issue_conflict"
    if exact and all(r["classification"] == "exact_date_issue_identifiers_match_only" for r in exact):
        return "exact_date_issue_identifier_match_only"
    return base if not exact else "exact_date_issue_unresolved"


def _bar_schema():
    import pyarrow as pa

    types = {"bar_start_ms": pa.int64(), "observed_at_ns": pa.int64(), "historical_available_at_ns": pa.int64(),
             "n": pa.int64(), **{k: pa.bool_() for k in ("adjusted", "vwap_observed", "transaction_count_observed")}}
    return pa.schema([(k, types.get(k, pa.string())) for k in (*_BAR_FIELDS, "source_json")])


def _panel_schema():
    import pyarrow as pa

    return pa.schema([
        ("ticker", pa.string()), ("session_date", pa.string()), ("daily_row", pa.int64()),
        ("window_row_start", pa.int64()), ("window_row_count", pa.int16()),
        ("window_missing_slot_mask", pa.int16()), ("any_minute_observed", pa.bool_()),
        ("window_observed_volume_sum", pa.string()), ("window_vwap_observed_slots", pa.int16()),
        ("identity_evidence", pa.string()), ("candidate_economic_observation_indices", pa.list_(pa.int32())),
        ("feature_session", pa.string()), ("feature_daily_row", pa.int64()),
        ("execution_session", pa.string()), ("execution_window_complete", pa.bool_()),
        ("observation_support_complete", pa.bool_()), ("training_eligible", pa.bool_()),
    ])


def build_panel(*, daily_root: Path, daily_sha256: str, reference_root: Path, reference_sha256: str,
                supplement_root: Path, supplement_sha256: str, calendar_root: Path, calendar_sha256: str,
                minute_root: Path, minute_sha256: str, output: Path) -> dict:
    """Materialize compact observed inputs; verify every consumed parent twice.

    Roots refer to accepted normalized products. This does not re-run HTTP/raw
    normalization, authenticate third-party history or promote research sources.
    The outer operator also binds the exact successful LSF producer receipts.
    """
    roots = [daily_root, reference_root, supplement_root, calendar_root, minute_root]
    if output.resolve() != output or any(output == r or r in output.parents or output in r.parents for r in roots):
        raise transport.ResearchCaptureError("Output must be a separate canonical generation")
    bindings = []

    def complete(root, digest, schema):
        obj, raw = _complete(root, digest, schema)
        bindings.append((root / "COMPLETE.json", transport.digest(raw), len(raw)))
        return obj

    def artifact(root, proof, name):
        raw = _bound(root, proof, name, 128 * 1024 * 1024)
        bindings.append((root / name, transport.digest(raw), len(raw)))
        return raw

    d = complete(daily_root, daily_sha256, "rl-quant.qt200-daily-research-capture-v1-normalized")
    r = complete(reference_root, reference_sha256, "rl-quant.qt200-daily-reference-reconciliation-v1")
    s = complete(supplement_root, supplement_sha256, "rl-quant.qt200-supplement-reconciliation-v1")
    c = complete(calendar_root, calendar_sha256, "rl-quant.qt200-aggregate-research-calendar-v1")
    m = complete(minute_root / "analysis", minute_sha256, "qt200-minute-execution-input-preparation-v1")
    if (d["invalid_rows"] != 0 or not d["observed_bar_schema_accepted"] or s["invalid_alias_rows"] != 0
            or m["invalid_rows"] != 0 or r["daily_capture_sha256"] != d["capture_sha256"]
            or s["identity_evidence_sha256"] != r["files"]["identity-and-actions.json.gz"]["sha256"]
            or m["ordered_tickers"] != [q.ticker for q in minute_queries()] or len(m["series"]) != 203):
        raise transport.ResearchCaptureError("Parent populations or source links differ")
    sessions = _object(artifact(calendar_root, c["files"]["sessions.json.gz"], "sessions.json.gz"))["sessions"]
    check = dict(schema="quanttrade-xnys-calendar-source-v1", calendar="XNYS", coverage_start_date=START,
                 coverage_end_date=END, exchange_calendars_version=PINNED["exchange-calendars"], sessions=sessions)
    check["receipt_sha256"] = transport.digest(transport.canonical(check))
    _calendar(transport.canonical(check))
    days = [row["session_date"] for row in sessions]
    if len(days) != 2428 or c["sessions"] != len(days) or c["dependency_versions"] != PINNED:
        raise transport.ResearchCaptureError("Full aggregate calendar differs")
    clocks = _object(artifact(calendar_root, c["files"]["decision-clock-plan.json.gz"], "decision-clock-plan.json.gz"))["clocks"]
    if clocks != clock_rows(sessions):
        raise transport.ResearchCaptureError("Prior-feature/next-execution clock differs")
    reference_body = artifact(reference_root, r["files"]["identity-and-actions.json.gz"], "identity-and-actions.json.gz")
    reference = _object(reference_body)
    if reference["ordered_tickers"] != list(SYMBOLS):
        raise transport.ResearchCaptureError("Ordered issue panel differs")
    entities = {row["qt200_ticker"]: row for row in reference["security_resolutions"]}
    supplementary = {name: artifact(supplement_root, proof, name) for name, proof in s["files"].items()}
    extra = defaultdict(list)
    for row in _rows(supplementary["dated-reference-observations.jsonl.gz"]):
        extra[row["qt200_candidate"]].append(row)
    events = [dict(origin="retained_reference", observation=row) for row in reference["economic_observations"]]
    events.extend(dict(origin="supplement_tail", observation=row) for row in _rows(supplementary["economic-tail-observations.jsonl.gz"]))
    event_index = defaultdict(list)
    for index, item in enumerate(events):
        row = item["observation"]
        if item["origin"] == "retained_reference":
            day = row["effective_date"]
            candidates = {q["qt200_ticker"] for q in row["candidate_joins"]}
        else:
            raw = row["provider_row"]
            day = raw.get("execution_date") if row["product"] == "splits" else raw.get("ex_dividend_date")
            candidates = {raw.get("ticker")}.intersection(SYMBOLS)
        for ticker in candidates:
            event_index[ticker, day].append(index)

    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(mode=0o700)
    counts, daily_index, minute_index, observed_dates = Counter(), {}, {}, {}
    off_calendar = defaultdict(Counter)
    files = {}
    calendar_days = set(days)
    daily_body = artifact(daily_root, d["bars"], "observed-daily-bars.jsonl.gz")
    sidecar = iter(_rows(artifact(reference_root, r["files"]["bar-reference-sidecar.jsonl.gz"], "bar-reference-sidecar.jsonl.gz")))
    with pq.ParquetWriter(output / "daily-bars.parquet", _bar_schema(), compression="zstd", compression_level=9) as writer:
        batch, prior = [], {}
        for row in _rows(daily_body):
            ticker, day = row["ticker"], row["session_date_et"]
            normalized = _observed_bar(row, DailyQuery("day", ticker, START, END))
            linked = next(sidecar, None)
            if (linked is None or (linked["ticker"], linked["session_date_et"], linked["source"]) != (ticker, day, row["source"])
                    or day <= prior.get(ticker, "")):
                raise transport.ResearchCaptureError("Daily/reference row linkage differs")
            prior[ticker] = day
            daily_index[ticker, day] = counts["daily_rows"]
            counts["daily_rows"] += 1
            if day not in calendar_days:
                off_calendar["daily"][day] += 1
            batch.append(normalized)
            if len(batch) == 4096:
                writer.write_table(pa.Table.from_pylist(batch, schema=_bar_schema()))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=_bar_schema()))
    if next(sidecar, None) is not None or counts["daily_rows"] != d["observed_rows"] or r["daily_rows"] != d["observed_rows"]:
        raise transport.ResearchCaptureError("Daily source population differs")

    with pq.ParquetWriter(output / "minute-window-bars.parquet", _bar_schema(), compression="zstd", compression_level=9) as writer:
        for query, parent in zip(minute_queries(), m["series"], strict=True):
            if parent["ticker"] != query.ticker:
                raise transport.ResearchCaptureError("Minute series order differs")
            root = minute_root / "windows" / query.ticker
            proof = complete(root, parent["completion_sha256"], "rl-quant.qt200-minute-research-capture-v1-execution-inputs")
            if proof["ticker"] != query.ticker or proof["invalid_rows"] != 0:
                raise transport.ResearchCaptureError("Minute series invalid")
            coverage = _object(artifact(root, proof["files"]["coverage.json.gz"], "coverage.json.gz"))
            body = artifact(root, proof["files"]["execution-window-bars.jsonl.gz"], "execution-window-bars.jsonl.gz")
            by_day, batch, prior = defaultdict(list), [], -1
            for row in _rows(body):
                normalized = _observed_bar(row, query)
                if row["bar_start_ms"] <= prior or row["source_capture_sha256"] != proof["capture_sha256"]:
                    raise transport.ResearchCaptureError("Minute order/capture differs")
                prior = row["bar_start_ms"]
                day = row["session_date_et"]
                by_day[day].append(normalized)
                batch.append(normalized)
            for day, rows in by_day.items():
                facts = _window_facts(rows, day)
                minute_index[query.ticker, day] = dict(start=counts["minute_rows"], **facts)
                counts["minute_rows"] += len(rows)
                if day not in calendar_days:
                    off_calendar["minute"][day] += len(rows)
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=_bar_schema()))
            dates = coverage["observed_dates"]
            observed_dates[query.ticker] = set(dates)
            missing = {day: minute_index.get((query.ticker, day), _window_facts([], day))["missing_slots"]
                       for day in dates if minute_index.get((query.ticker, day), {}).get("observed_slots", 0) != 10}
            if (dates != sorted(set(dates)) or not set(by_day).issubset(dates) or coverage["invalid_rows"]
                    or missing != coverage["missing_window_slots"] or len(batch) != proof["selected_window_rows"]
                    or len(dates) != proof["observed_dates"]):
                raise transport.ResearchCaptureError("Minute coverage/population reconciliation differs")
    if counts["minute_rows"] != m["selected_window_rows"]:
        raise transport.ResearchCaptureError("Global minute population differs")

    summaries = []
    with pq.ParquetWriter(output / "session-index.parquet", _panel_schema(), compression="zstd", compression_level=9) as writer:
        for ticker in SYMBOLS:
            rows, summary, gaps = [], Counter(), []
            for clock in clocks:
                day = clock["decision_session"]
                window = minute_index.get((ticker, day))
                facts = window or _window_facts([], day)
                identity = _identity_evidence(entities[ticker], ticker, day, extra[ticker])
                feature = daily_index.get((ticker, clock["latest_feature_session"]))
                execution = minute_index.get((ticker, clock["execution_session"]))
                execution_complete = execution is not None and execution["complete"]
                summary["missing_daily_sessions"] += (ticker, day) not in daily_index
                summary["incomplete_window_sessions"] += not facts["complete"]
                summary["absent_minute_sessions"] += day not in observed_dates[ticker]
                summary[identity] += 1
                if not facts["complete"]:
                    gaps.append(dict(session=day, missing_slots=facts["missing_slots"], any_minute_observed=day in observed_dates[ticker]))
                rows.append(dict(ticker=ticker, session_date=day, daily_row=daily_index.get((ticker, day)),
                    window_row_start=window["start"] if window else None, window_row_count=facts["observed_slots"],
                    window_missing_slot_mask=facts["missing_slot_mask"], any_minute_observed=day in observed_dates[ticker],
                    window_observed_volume_sum=facts["observed_volume_sum"], window_vwap_observed_slots=facts["observed_vwap_slots"],
                    identity_evidence=identity, candidate_economic_observation_indices=event_index[ticker, day],
                    feature_session=clock["latest_feature_session"], feature_daily_row=feature,
                    execution_session=clock["execution_session"], execution_window_complete=execution_complete,
                    observation_support_complete=feature is not None and execution_complete,
                    training_eligible=False))
            writer.write_table(pa.Table.from_pylist(rows, schema=_panel_schema()))
            summaries.append(dict(ticker=ticker, counts=dict(summary), window_gaps=gaps))
            counts["session_rows"] += len(rows)

    # Candidate aliases are retained separately, never spliced into equity rows.
    alias_rows = []
    for row in _rows(supplementary["alias-daily-bars.jsonl.gz"]):
        matches = [q for q in ALIASES if q[1] == row["ticker"] and q[0] == row["qt200_candidate"]]
        if len(matches) != 1:
            raise transport.ResearchCaptureError("Alias daily role differs")
        target, ticker, start, end = matches[0]
        query = SupplementQuery("day", ticker, start, end, target)
        alias_rows.append(_observed_bar(row, query))
    if len(alias_rows) != s["alias_bar_rows"]:
        raise transport.ResearchCaptureError("Alias daily population differs")
    pq.write_table(pa.Table.from_pylist(alias_rows, schema=_bar_schema()), output / "alias-daily-bars.parquet", compression="zstd", compression_level=9)
    for name, body in {
        "identity-and-actions.json.gz": reference_body,
        "dated-reference-observations.jsonl.gz": supplementary["dated-reference-observations.jsonl.gz"],
        "candidate-economic-observations.json.gz": gzip.compress(transport.canonical(dict(observations=events, accounting_authorized=False)), mtime=0),
        "decision-clock-plan.json.gz": gzip.compress(transport.canonical(dict(clocks=clocks)), mtime=0),
        "coverage.json.gz": gzip.compress(transport.canonical(dict(symbols=summaries, off_calendar={k: dict(v) for k, v in off_calendar.items()},
            aliases=[dict(ticker=q[1], qt200_candidate=q[0], observed_minute_dates=len(observed_dates[q[1]]), automatic_stitching=False) for q in ALIASES])), mtime=0),
    }.items():
        transport.write_once(output / name, body)
    for path, digest, size in bindings:
        body = transport.read_regular(path, 128 * 1024 * 1024)
        if len(body) != size or transport.digest(body) != digest:
            raise transport.ResearchCaptureError("Consumed parent changed during panel construction")
    for path in sorted(output.iterdir()):
        if path.suffix == ".parquet":
            expected = {"daily-bars.parquet": counts["daily_rows"],
                        "minute-window-bars.parquet": counts["minute_rows"],
                        "session-index.parquet": counts["session_rows"],
                        "alias-daily-bars.parquet": len(alias_rows)}[path.name]
            if pq.read_metadata(path).num_rows != expected:
                raise transport.ResearchCaptureError("Parquet persisted row count differs")
        # Final completion must not get ahead of a buffered columnar file.
        import os

        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        body = transport.read_regular(path, 512 * 1024 * 1024)
        files[path.name] = dict(path=path.name, bytes=len(body), sha256=transport.digest(body))
    result = dict(schema=SCHEMA, research_track="QT200-AGG-DEV-01", files=files, counts=dict(counts),
        alias_daily_rows=len(alias_rows), ordered_tickers=list(SYMBOLS), sessions=len(days),
        parents=dict(daily=daily_sha256, reference=reference_sha256, supplement=supplement_sha256,
                     calendar=calendar_sha256, minute=minute_sha256),
        consumed_files=[dict(path=str(p), bytes=n, sha256=h) for p, h, n in bindings],
        source_bytes_reverified_before_and_after=True, original_http_bodies_replayed=False,
        source_observation_integration_complete=True, decimal_values_preserved_as_strings=True,
        missing_daily_sessions=sum(r["counts"]["missing_daily_sessions"] for r in summaries),
        incomplete_window_sessions=sum(r["counts"]["incomplete_window_sessions"] for r in summaries),
        raw_trade_semantics_qualified=False, historical_identity_qualified=False,
        corporate_action_accounting_qualified=False, execution_model_qualified=False,
        bars_only_features_and_targets_qualified=False, joint_h100_runtime_qualified=False,
        point_in_time_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
