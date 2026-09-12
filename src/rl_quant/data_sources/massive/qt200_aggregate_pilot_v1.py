"""Receipt-bound bars ingestion and REST/flat-file comparison, not V5 promotion.

Retain provider-finalized observations, including missing optional fields. Raw
trade sequence equality is a diagnostic lookup only, never a correction link.
No network, credential, price imputation, trade replay or model code is used.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rl_quant.data_sources.massive.qt200_research_capture_v1 import (
    ET,
    MAX_PAGE_BYTES,
    PilotQuery,
    ResearchCaptureError,
    canonical,
    digest,
    parse_json,
    pilot_queries,
    read_regular,
    replay_pilot,
    write_once,
)

SCHEMA = "rl-quant.qt200-aggregate-pilot-v1"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _decimal(value: object, *, positive: bool) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ResearchCaptureError("Missing or nonnumeric aggregate value")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ResearchCaptureError("Invalid aggregate decimal") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ResearchCaptureError("Invalid aggregate sign or finite-value contract")
    return str(number)


def normalize_bar(row: dict, query: PilotQuery, *, received_at_ns: int) -> dict:
    """Parse observed OHLCV; optional absence stays null, not an observed zero.

    Numeric strings preserve the parsed decimal value without forcing a binary
    float tensor or an arbitrary decimal scale. The original JSON stays bound.
    This validates interval structure, not exchange-calendar/issue eligibility.
    """
    query.validate()
    if query.product not in {"day", "minute"}:
        raise ResearchCaptureError("Expected an aggregate query")
    if type(received_at_ns) is not int or received_at_ns <= 0:
        raise ResearchCaptureError("Missing actual observation time")
    timestamp = row.get("t")
    if type(timestamp) is not int or not 0 <= timestamp <= 253402300799999:
        raise ResearchCaptureError("Invalid millisecond timestamp")
    stamp = (EPOCH + timedelta(milliseconds=timestamp)).astimezone(ET)
    if not query.start <= stamp.date().isoformat() <= query.end:
        raise ResearchCaptureError("Aggregate escaped query dates")
    if stamp.second or stamp.microsecond or (
        query.product == "day" and (stamp.hour or stamp.minute)
    ):
        raise ResearchCaptureError("Aggregate start is not interval aligned")
    if "otc" in row and row["otc"] is not False:
        raise ResearchCaptureError("OTC observation outside equity pilot")
    values = {key: _decimal(row.get(key), positive=key != "v") for key in "ohlcv"}
    low, high = Decimal(values["l"]), Decimal(values["h"])
    if low > high or any(not low <= Decimal(values[key]) <= high for key in "oc"):
        raise ResearchCaptureError("OHLC range inconsistent")
    # VWAP may have a different eligible population from extrema. Do not force
    # it into the OHLC range or replace a missing VWAP with average closes.
    vwap = _decimal(row["vw"], positive=True) if "vw" in row else None
    count = row.get("n")
    if "n" in row and (type(count) is not int or count < 0):
        raise ResearchCaptureError("Invalid transaction count")
    return {
        "ticker": query.ticker, "product": query.product,
        "bar_start_ms": timestamp, "session_date_et": stamp.date().isoformat(),
        **values, "vw": vwap, "n": count, "vwap_observed": "vw" in row,
        "transaction_count_observed": "n" in row, "adjusted": False,
        "observed_at_ns": received_at_ns,
        "historical_available_at_ns": None, "raw_row_sha256": digest(canonical(row)),
    }


def _bound(path: Path, proof: dict, expected_name: str, cap: int) -> bytes:
    if proof.get("path") != expected_name:
        raise ResearchCaptureError("Unexpected bound evidence path")
    body = read_regular(path / expected_name, cap)
    if len(body) != proof.get("bytes") or digest(body) != proof.get("sha256"):
        raise ResearchCaptureError("Bound evidence changed")
    return body


def _query_rows(root: Path, query: PilotQuery, entry: dict):
    directory = root / query.name
    complete = parse_json(_bound(directory, entry["completion"], "COMPLETE.json", 1048576))
    for index, proof in enumerate(complete["pages"]):
        page = parse_json(_bound(directory, proof, f"page-{index:04d}.receipt.json", 1048576))
        packed = _bound(directory, page["body"], f"page-{index:04d}.json.gz", MAX_PAGE_BYTES + 1048576)
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            raw = stream.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES or stream.read(1):
                raise ResearchCaptureError("Unbounded response inflation")
        if digest(raw) != page["raw_body_sha256"] or len(raw) != page["raw_body_bytes"]:
            raise ResearchCaptureError("Raw response changed after replay")
        for ordinal, row in enumerate(parse_json(raw).get("results", [])):
            yield row, {"page": index, "row": ordinal, "page_sha256": proof["sha256"],
                        "received_at_ns": page["received_at_ns"]}


def compare_original(original: dict, candidates: list[tuple[dict, dict]]) -> dict:
    """Compare every sequence-key candidate; never choose one as a target.

    Missing REST correction/TRF fields are reported as missing. In particular,
    an absent correction indicator is NOT silently equated to flat-file 0.
    """
    matches = []
    for row, position in candidates:
        absent, equal, different = [], [], []
        for key, value in original.items():
            if key == "ticker":
                continue  # REST ticker is query-scoped, not a row field.
            if key not in row:
                absent.append(key)
                continue
            observed = row[key]
            if key == "conditions":
                expected = [int(code) for code in value.split(",") if code]
                same = observed == expected
            elif key == "id":
                same = observed == value
            else:
                try:
                    same = not isinstance(observed, bool) and Decimal(str(observed)) == Decimal(value)
                except InvalidOperation:
                    same = False
            (equal if same else different).append(key)
        matches.append({"position": position, "rest_row": row,
                        "equal_fields": sorted(equal), "different_fields": sorted(different),
                        "absent_rest_fields": sorted(absent),
                        "correction_link_established": False})
    return {"original_fields": original, "candidate_count": len(matches), "candidates": matches,
            "lookup": "ticker/query-date/sequence-only-not-an-economic-identity",
            "correction_target_reference": None}


def analyze_pilot(*, capture_root: Path, plan_sha256: str, completion_sha256: str,
                  samples_csv: Path, samples_sha256: str,
                  pairs_json: Path, pairs_sha256: str, output_root: Path) -> dict:
    """Build a compact, nonauthorizing dataset and report from exact captures.

    Publish only into a new output root. Source integrity and actual capture
    chronology are replayed before AND after analysis. Unsupported economics
    stay unresolved, regardless of observed product agreement.
    """
    replay = replay_pilot(root=capture_root, plan_sha256=plan_sha256,
                          completion_sha256=completion_sha256)
    complete = parse_json(read_regular(capture_root / "COMPLETE.json", 1048576))
    if digest(canonical(complete)) != completion_sha256:
        raise ResearchCaptureError("Completion changed after replay")
    sample_bytes = read_regular(samples_csv, 4 * 1048576)
    pair_bytes = read_regular(pairs_json, 16 * 1048576)
    if digest(sample_bytes) != samples_sha256 or digest(pair_bytes) != pairs_sha256:
        raise ResearchCaptureError("Original comparison evidence changed")
    pairs = parse_json(pair_bytes)
    plan = parse_json(read_regular(capture_root / "plan.json", 1048576))
    source_hash = plan.get("original_trade_object_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in "0123456789abcdef" for c in source_hash)
            or pairs.get("original_source_compressed_sha256") != source_hash):
        raise ResearchCaptureError("Original trade source differs")
    samples = list(csv.DictReader(io.StringIO(sample_bytes.decode("utf-8"))))
    selected = {q.ticker for q in pilot_queries()}
    examples = [{"origin": "original-samples", "original_fields": row} for row in samples
                if row["ticker"] in selected]
    for group in pairs["groups"]:
        if group["ticker_raw"] in selected:
            examples.extend({"origin": "pair-probe", **row} for row in group["rows"])
    keys = {(e["original_fields"]["ticker"], int(e["original_fields"]["sequence_number"]))
            for e in examples}
    candidates = defaultdict(list)
    summaries, bars, diagnostics = [], [], []
    for query, entry in zip(pilot_queries(), complete["queries"], strict=True):
        counts, fields, previous, query_bars = Counter(), Counter(), None, []
        for row, position in _query_rows(capture_root, query, entry):
            counts["rows"] += 1
            fields.update(row.keys())
            if query.product == "trades":
                counts["blank_or_absent_ids"] += row.get("id") in ("", None)
                counts["zero_size_records"] += row.get("size") == 0
                counts["correction:" + str(row.get("correction", "ABSENT"))] += 1
                key = (query.ticker, row.get("sequence_number"))
                if key in keys:
                    candidates[key].append((row, position))
                continue
            try:
                normalized = normalize_bar(row, query, received_at_ns=position["received_at_ns"])
                timestamp = normalized["bar_start_ms"]
                if previous is not None and timestamp <= previous:
                    raise ResearchCaptureError("Duplicate or out-of-order aggregate interval")
                previous = timestamp
                normalized.update(source_query=query.name, source_position=position)
                query_bars.append(normalized)
            except ResearchCaptureError as exc:
                diagnostics.append({"query": query.name, "position": position,
                                    "raw_row_sha256": digest(canonical(row)), "reason": str(exc)})
        if counts["rows"] != entry["result_count"]:
            raise ResearchCaptureError("Consumed row population differs")
        summary = {"query": query.name, "counts": dict(sorted(counts.items())),
                   "field_presence_counts": dict(sorted(fields.items()))}
        if query.product != "trades":
            summary.update(valid_bar_rows=len(query_bars),
                           invalid_bar_rows=counts["rows"] - len(query_bars),
                           observed_dates=sorted({b["session_date_et"] for b in query_bars}),
                           expected_exchange_calendar_coverage_assessed=False)
        if query.product == "minute":
            # Only enumerate observed input bars in the proposed proxy window.
            # No price proxy, volume-to-fill rule or realized fill is invented.
            window = [b for b in query_bars if "09:35" <=
                      (EPOCH + timedelta(milliseconds=b["bar_start_ms"])).astimezone(ET).strftime("%H:%M") < "09:45"]
            summary["proposed_window"] = {"interval": "[09:35,09:45) America/New_York",
                "observed_minutes": len(window), "expected_minute_slots": 10,
                "all_minute_slots_observed": len(window) == 10,
                "observed_volume_sum": str(sum((Decimal(b["v"]) for b in window), Decimal(0))),
                "vwap_observed_minutes": sum(b["vwap_observed"] for b in window),
                "fills_derived": False}
        summaries.append(summary)
        bars.extend(query_bars)
    comparisons = [{"example_origin": example["origin"],
                    "source_row_number": example.get("source_row_number"),
                    **compare_original(example["original_fields"], candidates[
                        (example["original_fields"]["ticker"], int(example["original_fields"]["sequence_number"]))])}
                   for example in examples]
    if replay_pilot(root=capture_root, plan_sha256=plan_sha256,
                    completion_sha256=completion_sha256) != replay:
        raise ResearchCaptureError("Capture replay changed during analysis")
    if read_regular(samples_csv, 4 * 1048576) != sample_bytes or read_regular(pairs_json, 16 * 1048576) != pair_bytes:
        raise ResearchCaptureError("Comparison evidence changed during analysis")
    if output_root.resolve() != output_root or output_root == capture_root or capture_root in output_root.parents:
        raise ResearchCaptureError("Output must be a separate canonical directory")
    output_root.mkdir(mode=0o700)  # Existing/partial output never overwritten.
    dataset = write_once(output_root / "observed-bars.jsonl.gz",
                         gzip.compress(b"".join(canonical(row) for row in bars), compresslevel=9, mtime=0))
    comparison = write_once(output_root / "source-comparisons.json.gz",
                            gzip.compress(canonical({"comparisons": comparisons}), compresslevel=9, mtime=0))
    result = {"schema": SCHEMA, "capture_replay": replay, "plan_sha256": plan_sha256,
              "samples_sha256": samples_sha256, "pairs_sha256": pairs_sha256,
              "queries": summaries, "bars": dataset, "bar_rows": len(bars),
              "comparisons": comparison, "original_examples_compared": len(comparisons),
              "invalid_bars": diagnostics, "observed_bar_schema_accepted": bool(bars) and not diagnostics,
              "full_history_dataset_complete": False, "source_writes": False,
              "correction_links_established": 0, "historical_availability_recovered": False,
              "point_in_time_qualified": False, "native_v5_qualified": False,
              "training_ready": False, "profitability_authorized": False}
    write_once(output_root / "COMPLETE.json", canonical(result))
    return result
