"""Source-bound QT200 identity/economic diagnostics, never an identity authority.

This reader consumes the immutable *support extraction* bundle, not provider
credentials or network endpoints. A matching FIGI is useful evidence, but FIGI
is not required by the native security-ID contract. CIK, ticker, exchange and
company name alone do not establish an issuance. No rows from this diagnostic
are promoted to SourcedTickerHistoryRecord or an economic accounting authority.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import stat
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

SCHEMA = "rl-quant.qt200-identity-evidence-v1"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
# Dedicated original-dividend limits; these never widen support-file limits.
MAX_DIVIDEND_SOURCE_BYTES = 256 * 1024 * 1024
MAX_DIVIDEND_SOURCE_PAGES = 128
MAX_DIVIDEND_SOURCE_RESULTS = 500_000
_INPUTS = (
    "qt200-universe-spec.json", "reference-records.jsonl",
    "ticker-event-captures.jsonl", "dividends.jsonl", "splits.jsonl",
)
_GAPS = (
    "source_bound_issue_identity_and_listing_delisting_intervals_required",
    "historical_identity_availability_and_ticker_reuse_coverage_required",
    "explicit_successor_spinoff_merger_and_terminal_consideration_required",
    "complete_economic_surface_and_event_interaction_coverage_required",
    "native_identity_and_economic_authority_promotion_not_performed",
)


class QT200IdentityEvidenceError(ValueError):
    """The bounded input bundle or its source links are inconsistent."""


def _fail(message: str) -> None:
    raise QT200IdentityEvidenceError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        _fail("Expected a lowercase SHA-256")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(data: bytes) -> Any:
    def invalid(value: str) -> None:
        _fail("Nonfinite JSON constant: " + value)
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QT200IdentityEvidenceError("Malformed JSON") from exc


def _day(value: Any) -> str:
    if not isinstance(value, str):
        _fail("Expected an ISO event date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise QT200IdentityEvidenceError("Invalid event date") from exc
    if parsed.isoformat() != value:
        _fail("Noncanonical event date")
    return value


def _identity(s: os.stat_result) -> tuple[int, ...]:
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_uid, s.st_mode)


def _read(root: Path, relative: str) -> tuple[bytes, tuple[int, ...]]:
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or ".." in path.parts:
        _fail("Unsafe inventory path")
    target = root / relative
    if target.resolve(strict=True) != target:
        _fail("Symlink in source path")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_size > MAX_FILE_BYTES):
            _fail("Source is not an owned bounded regular file")
        data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) != before.st_size or _identity(before) != _identity(os.fstat(stream.fileno())):
            _fail("Source changed during read")
    if _identity(before) != _identity(target.lstat()):
        _fail("Source pathname changed during read")
    return data, _identity(before)


def _load_bundle(root: Path, completion_sha256: str, universe_sha256: str):
    root = root.absolute()
    before_root = root.lstat()
    if (root.resolve(strict=True) != root or not stat.S_ISDIR(before_root.st_mode)
            or before_root.st_uid != os.getuid()):
        _fail("Support root must be an owned non-symlink directory")
    complete_raw, state = _read(root, "COMPLETE.json")
    if _sha(complete_raw) != _digest(completion_sha256):
        _fail("Support completion hash mismatch")
    complete = _json(complete_raw)
    if (complete.get("schema") != "quanttrade-qt200-support-extraction-complete-v1"
            or complete.get("status") != "COMPLETE_SUPPORT_DATA_EXTRACTION_ONLY"):
        _fail("Not a completed support-extraction bundle")
    inventory = complete.get("inventory", [])
    if not isinstance(inventory, list) or not 1 <= len(inventory) <= 64:
        _fail("Unbounded or absent support inventory")
    if _sha(_canonical(inventory)) != complete.get("inventory_sha256"):
        _fail("Inventory semantic hash mismatch")
    states, contents, bindings = {"COMPLETE.json": state}, {}, []
    total = 0
    for row in inventory:
        relative = row["relative_path"]
        if relative in states:
            _fail("Duplicate inventory path")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("Invalid inventory byte count")
        total += size
        if total > MAX_BUNDLE_BYTES:
            _fail("Support bundle exceeds diagnostic byte bound")
        data, states[relative] = _read(root, relative)
        if len(data) != size or _sha(data) != _digest(row["sha256"]):
            _fail("Inventory physical hash/size mismatch: " + relative)
        bindings.append(dict(row))
        if relative in _INPUTS or relative == "manifest.json":
            contents[relative] = data
    if (complete.get("inventoried_file_count_excluding_complete") != len(inventory)
            or complete.get("inventoried_bytes_excluding_complete") != total):
        _fail("Inventory totals mismatch")
    if not set(_INPUTS).issubset(contents) or "manifest.json" not in contents:
        _fail("Missing diagnostic input")
    if _sha(contents["manifest.json"]) != complete.get("manifest_sha256"):
        _fail("Manifest completion cross-link mismatch")
    manifest = _json(contents["manifest.json"])
    if manifest.get("files") != [row for row in inventory if row["relative_path"] != "manifest.json"]:
        # The publisher preserves insertion order in manifest.files and sorts
        # COMPLETE.inventory; compare exact unique entries, not their order.
        expected = sorted((row for row in inventory if row["relative_path"] != "manifest.json"),
                          key=lambda row: row["relative_path"])
        observed = sorted(manifest.get("files", []), key=lambda row: row["relative_path"])
        if observed != expected:
            _fail("Manifest inventory cross-link mismatch")
    if (_sha(contents["qt200-universe-spec.json"]) != _digest(universe_sha256)
            or manifest.get("universe_sha256") != universe_sha256):
        _fail("Universe physical hash mismatch")
    return root, states, contents, manifest, bindings


def _records(data: bytes) -> list[dict[str, Any]]:
    lines = data.splitlines()
    if len(lines) > 50_000 or any(not line or len(line) > 256 * 1024 for line in lines):
        _fail("Invalid or unbounded JSONL records")
    rows = [_json(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        _fail("JSONL record is not an object")
    return rows


def _location(row: dict[str, Any], file: str, index: int, sources: dict[str, Any]):
    p = row["provenance"]
    source = sources.get(p["source_file"])
    if source is None or any(p.get(key) != source.get(key) for key in
                             ("source_sha256", "upstream_receipt_sha256")):
        _fail("Record provenance does not match source manifest")
    for key in ("source_sha256", "upstream_receipt_sha256", "raw_response_body_sha256"):
        _digest(p[key])
    captured = p["captured_at_ms"]
    if isinstance(captured, bool) or not isinstance(captured, int) or captured < 0:
        _fail("Invalid source capture time")
    return dict(file=file, record_index=index, captured_at_ms=captured,
                source_file=p["source_file"], source_sha256=p["source_sha256"],
                upstream_receipt_sha256=p["upstream_receipt_sha256"],
                raw_response_body_sha256=p["raw_response_body_sha256"],
                provider_result_index=p.get("result_index"), provider_page_index=p.get("page_index"))


def _summarize(tickers, references, captures, actions, sources):
    current = defaultdict(list)
    for index, row in enumerate(references):
        location = _location(row, "reference-records.jsonl", index, sources)
        provider = row["provider_record"]
        if provider.get("active") is True and provider.get("ticker") in tickers:
            current[provider["ticker"]].append((provider, location))
    entities, timelines = {}, defaultdict(list)
    for ticker in tickers:
        rows = current[ticker]
        record = rows[0][0] if len(rows) == 1 else {}
        figi = record.get("composite_figi")
        entities[ticker] = dict(
            ticker=ticker, active_exact_reference_count=len(rows),
            current_composite_figi=figi, current_share_class_figi=record.get("share_class_figi"),
            current_issuer_cik=record.get("cik"), current_primary_exchange=record.get("primary_exchange"),
            current_security_type=record.get("type"),
            current_reference_sources=[location for _, location in rows],
            current_issue_identifier_observed=bool(figi or record.get("share_class_figi")),
            native_security_id=None, historical_identity_qualified=False,
            ticker_history_qualified=False, successor_or_terminal_joins_qualified=False,
            absence_of_corporate_action_records_proves_no_events=False,
            gaps=list(_GAPS), ticker_event_observations=[], economic_candidate_counts={},
        )
        if len(rows) != 1:
            entities[ticker]["gaps"].append("unique_current_exact_reference_missing")
        if not figi and not record.get("share_class_figi"):
            entities[ticker]["gaps"].append("issue_level_identifier_missing_cik_is_not_substitute")
    for index, row in enumerate(captures):
        location = _location(row, "ticker-event-captures.jsonl", index, sources)
        result = row.get("provider_response", {}).get("results", {})
        events = result.get("events", [])
        metadata = row["response_metadata"]
        if metadata.get("completed_at_ms") != location["captured_at_ms"]:
            _fail("Ticker-event capture timestamp cross-link mismatch")
        for ticker in row["qt200_tickers"]:
            if ticker not in entities:
                _fail("Ticker-event assignment outside frozen universe")
            entity = entities[ticker]
            figi = entity["current_composite_figi"]
            matched = bool(figi and result.get("composite_figi") == figi
                           and row["request_identifier"] in (figi, ticker)
                           and metadata.get("response_status") == 200
                           and row.get("provider_response", {}).get("status") == "OK")
            ordered = sorted((_day(event["date"]), event["ticker_change"]["ticker"], position)
                             for position, event in enumerate(events)
                             if event.get("type") == "ticker_change")
            conflicts = len({day for day, _, _ in ordered}) != len(ordered)
            if conflicts:
                entity["gaps"].append("duplicate_or_conflicting_provider_event_dates")
            if ordered and ordered[-1][1] != ticker:
                entity["gaps"].append("current_ticker_differs_from_last_provider_event")
            if not matched:
                entity["gaps"].append("provider_events_not_issue_identifier_linked")
            for event_index, (day, symbol, provider_event_index) in enumerate(ordered):
                end = ordered[event_index + 1][0] if event_index + 1 < len(ordered) else None
                observation = dict(provider_ticker=symbol, event_date=day,
                                   provider_event_index=provider_event_index,
                                   next_provider_event_date=end, same_current_composite_figi=matched,
                                   interval_is_listing_or_tradability_authority=False,
                                   historical_available_at_ms=None, source=location)
                entity["ticker_event_observations"].append(observation)
                if matched and not conflicts:
                    timelines[symbol].append(dict(qt200_ticker=ticker, start=day, end=end))
    economic = []
    duplicates, by_event_id = [], {}
    counts = {ticker: Counter() for ticker in tickers}
    for surface, rows in actions.items():
        for index, row in enumerate(rows):
            location = _location(row, surface + ".jsonl", index, sources)
            record = row["provider_record"]
            ticker = record.get("ticker")
            day_field = "ex_dividend_date" if surface == "dividends" else "execution_date"
            problems = []
            try:
                event_date = _day(record.get(day_field))
            except QT200IdentityEvidenceError:
                event_date = None
                problems.append("invalid_effective_date")
            candidates = row.get("candidate_qt200_tickers", [])
            if len(set(candidates)) != len(candidates) or any(x not in entities for x in candidates):
                _fail("Invalid economic candidate assignment")
            key = (surface, record.get("id"))
            if not isinstance(key[1], str) or not key[1]:
                problems.append("provider_event_id_missing")
            elif key in by_event_id:
                previous, previous_index = by_event_id[key]
                kind = "duplicate_provider_event_id" if previous == record else "conflicting_provider_event_id"
                duplicates.append(dict(surface=surface, provider_event_id=key[1],
                                       first_record_index=previous_index, record_index=index, kind=kind))
                problems.append(kind)
            else:
                by_event_id[key] = (record, index)
            fields = ("cash_amount",) if surface == "dividends" else ("split_from", "split_to")
            for field in fields:
                value = record.get(field)
                minimum = 0 if surface == "dividends" else 0.0
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < minimum
                        or (surface == "splits" and value == 0)):
                    problems.append("invalid_" + field)
            if surface == "dividends":
                if record.get("currency") != "USD":
                    problems.append("usd_cash_currency_not_established")
                for field in ("pay_date", "record_date"):
                    try:
                        _day(record.get(field))
                    except QT200IdentityEvidenceError:
                        problems.append("invalid_or_missing_" + field)
            joins = []
            for candidate in candidates:
                brackets = [item for item in timelines[ticker] if item["qt200_ticker"] == candidate]
                compatible = bool(event_date and any(item["start"] <= event_date
                                  and (item["end"] is None or event_date < item["end"])
                                  for item in brackets))
                if candidate != ticker and not compatible:
                    join_state = "alias_outside_provider_event_bracket_or_unlinked"
                elif candidate != ticker:
                    join_state = "same_figi_event_bracket_candidate_only"
                else:
                    join_state = "exact_current_ticker_not_historical_identity"
                joins.append(dict(qt200_ticker=candidate, candidate_state=join_state,
                                  event_bracket_compatible=compatible, identity_qualified=False))
                counts[candidate][surface] += 1
                counts[candidate][join_state] += 1
            economic.append(dict(surface=surface, record_index=index, provider_record=record,
                                 effective_date=event_date, candidate_joins=joins, problems=problems,
                                 source=location, historical_available_at_ms=None,
                                 accounting_authorized=False))
    for ticker, entity in entities.items():
        entity["economic_candidate_counts"] = dict(counts[ticker])
        entity["gaps"] = sorted(set(entity["gaps"]))
        if not entity["ticker_event_observations"]:
            entity["gaps"].append("provider_ticker_events_missing")
    return list(entities.values()), economic, duplicates


def build_qt200_identity_evidence_v1(
    *, support_root: Path, expected_completion_sha256: str, expected_universe_sha256: str,
) -> dict[str, Any]:
    """Read bounded immutable metadata and return a nonauthorizing diagnostic.

    Physical hashes bind every support file. Original provider raw response
    bodies omitted by that bundle are *not* reauthenticated by this reader.
    Source locations and upstream receipt hashes are preserved, not upgraded.
    """
    root, states, contents, manifest, bindings = _load_bundle(
        Path(support_root), expected_completion_sha256, expected_universe_sha256)
    spec = _json(contents["qt200-universe-spec.json"])
    tickers = spec["tickers"]
    if (len(tickers) != 200 or len(set(tickers)) != 200 or spec.get("ticker_count") != 200
            or any(not isinstance(ticker, str) or not ticker for ticker in tickers)):
        _fail("Expected the exact ordered 200-ticker specification")
    sources = {}
    for source in manifest["sources"]:
        name = source["source_file"]
        if name in sources:
            _fail("Duplicate upstream source name")
        sources[name] = source
    entities, economic, duplicates = _summarize(
        tickers, _records(contents["reference-records.jsonl"]),
        _records(contents["ticker-event-captures.jsonl"]),
        {surface: _records(contents[surface + ".jsonl"]) for surface in ("dividends", "splits")},
        sources)
    for relative, expected in states.items():
        if (root / relative).resolve(strict=True) != root / relative:
            _fail("Input path changed to a symlink")
        if _identity((root / relative).lstat()) != expected:
            _fail("Input identity changed after diagnostic")
    result = dict(
        schema=SCHEMA, operation="identity_and_economic_evidence_diagnostic_only",
        support_completion_sha256=expected_completion_sha256,
        universe_sha256=expected_universe_sha256, input_bindings=bindings,
        source_identity_rechecked=True, support_files_physically_rehashed=True,
        original_provider_raw_bodies_reauthenticated=False,
        upstream_receipt_semantics_revalidated=False, ordered_tickers=tickers,
        security_candidates=entities, economic_observations=economic,
        duplicate_economic_event_observations=duplicates,
        issue_identifier_missing_tickers=[row["ticker"] for row in entities
                                          if not row["current_issue_identifier_observed"]],
        native_security_ids_emitted=0, historical_identity_qualified=False,
        economic_accounting_qualified=False, strict_pit_qualified=False,
        finalized_accounting_research_authorized=False,
        training_ready_for_adaptive_v5=False, provider_calls=0, model_tests_run=False,
        blockers=list(_GAPS),
    )
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Diagnostic output exceeds byte bound")
    return result


def load_qt200_dated_identity_diagnostic_v1(
    *, capture_root: Path, expected_completion_sha256: str, expected_universe_sha256: str,
) -> dict[str, Any]:
    """Verify a completed dated-overview capture and describe sparse observations.

    This is deliberately separate from the older support-bundle reader. Query
    dates, provider list dates, and local capture times retain distinct meanings.
    HTTP 404 is only a query result, never a delisting or nonexistence authority.
    """
    root = Path(capture_root).absolute()
    if (root.resolve(strict=True) != root or not root.is_dir()
            or root.stat().st_uid != os.getuid()):
        _fail("Dated capture root must be an owned nonsymlink directory")
    states = {}
    total = 0

    def read_bound(relative: str, digest: str, size: int | None = None):
        nonlocal total
        if relative in states:
            _fail("Duplicate dated inventory path")
        data, states[relative] = _read(root, relative)
        total += len(data)
        if total > 1_000_000_000:
            _fail("Dated capture exceeds one-gigabyte diagnostic bound")
        if _sha(data) != _digest(digest) or (size is not None and len(data) != size):
            _fail("Dated capture physical hash/size mismatch: " + relative)
        return data

    completion = _json(read_bound("COMPLETE.json", expected_completion_sha256))
    if completion.get("schema") != "qt200-dated-identity-capture-completion-v1":
        _fail("Not a completed dated capture")
    for flag in ("identity_qualified", "point_in_time_qualified", "training_ready"):
        if completion.get(flag) is not False:
            _fail("Dated capture contains unexpected authority claims")
    proof = completion["plan"]
    if proof.get("path") != "plan.json":
        _fail("Unexpected dated plan path")
    plan = _json(read_bound("plan.json", proof["sha256"], proof["bytes"]))
    if (plan.get("schema") != "qt200-dated-identity-capture-plan-v1"
            or plan.get("universe_sha256") != _digest(expected_universe_sha256)
            or plan.get("production_qualification") is not False):
        _fail("Dated plan schema/universe/nonqualification binding mismatch")
    read_bound("capture.py", plan["source_sha256"])
    requests = plan["requests"]
    if not isinstance(requests, list) or not 1 <= len(requests) <= 660:
        _fail("Invalid dated request inventory")
    for pair in requests:
        if (not isinstance(pair, list) or len(pair) != 2
                or not isinstance(pair[0], str) or not pair[0]
                or pair[0] != pair[0].strip()):
            _fail("Invalid dated request pair")
        _day(pair[1])
    if len({tuple(pair) for pair in requests}) != len(requests):
        _fail("Duplicate dated request pair")
    n = len(requests)
    entries = completion["files"]
    if (not isinstance(entries, list) or len(entries) != n
            or plan.get("requested_count") != n or completion.get("requested_count") != n
            or completion.get("completed_count") != n):
        _fail("Incomplete dated request coverage")
    indices = [entry["index"] for entry in entries]
    if (any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or sorted(indices) != list(range(n))):
        _fail("Dated response indices are not an exact one-to-one inventory")
    observations, statuses, failures = [], Counter(), []
    fields = ("ticker", "name", "cik", "composite_figi", "share_class_figi",
              "primary_exchange", "type", "active", "list_date", "delisted_utc")
    for entry in sorted(entries, key=lambda item: item["index"]):
        index = entry["index"]
        ticker, query_date = requests[index]
        if (entry.get("path") != f"response-{index:04d}.json"
                or [entry.get("ticker"), entry.get("date")] != requests[index]):
            _fail("Dated response inventory/request cross-link mismatch")
        row = _json(read_bound(entry["path"], entry["sha256"], entry["bytes"]))
        expected_url = "https://api.massive.com/v3/reference/tickers/" + quote(ticker, safe="")
        expected_url += "?date=" + query_date
        if [row.get("ticker"), row.get("date")] != requests[index] or row.get("request_url") != expected_url:
            _fail("Dated response query binding mismatch")
        requested, captured = row.get("requested_at_ms"), row.get("completed_at_ms")
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in (requested, captured)):
            _fail("Invalid dated acquisition timestamp")
        issues = []
        if captured < requested:
            issues.append("capture_completed_before_request")
        capture_date = datetime.fromtimestamp(captured / 1000, tz=timezone.utc).date().isoformat()
        if query_date > capture_date:
            issues.append("query_date_after_capture_date")
        status = row.get("http_status")
        statuses[str(status)] += 1
        if row.get("transport_error"):
            failures.append(index)
        provider_fields, raw_sha = None, None
        if status in (200, 404):
            if (row.get("historical_known_at") is not None
                    or row.get("capture_time_is_historical_availability") is not False
                    or row.get("absent_response_proves_delisting") is not False):
                _fail("Dated response contains unexpected historical authority claims")
            try:
                body = base64.b64decode(row["raw_response_body_base64"], validate=True)
            except (ValueError, binascii.Error, TypeError) as exc:
                raise QT200IdentityEvidenceError("Invalid dated raw-body base64") from exc
            if (len(body) > 1048576 or len(body) != row.get("raw_response_content_length")
                    or _sha(body) != _digest(row["raw_response_body_sha256"])):
                _fail("Dated raw-body physical hash/length mismatch")
            payload = _json(body)
            if (not isinstance(payload, dict) or payload.get("results") != row.get("results")
                    or payload.get("request_id") != row.get("provider_request_id")):
                _fail("Dated parsed response/raw-body cross-link mismatch")
            raw_sha = row["raw_response_body_sha256"]
            if status == 200:
                result = payload.get("results")
                if not isinstance(result, dict) or result.get("ticker") != ticker:
                    _fail("Dated successful response ticker mismatch")
                provider_fields = {field: result.get(field) for field in fields}
                if payload.get("status") != "OK":
                    issues.append("provider_payload_status_not_OK")
                if result.get("list_date") is not None:
                    try:
                        if _day(result["list_date"]) > query_date:
                            issues.append("provider_list_date_after_query_date")
                    except QT200IdentityEvidenceError:
                        issues.append("invalid_provider_list_date")
                else:
                    issues.append("provider_list_date_missing_no_listing_inference")
                if not result.get("composite_figi") and not result.get("share_class_figi"):
                    issues.append("issue_level_identifier_not_observed_cik_not_substitute")
            elif payload.get("results") is not None:
                issues.append("http_404_with_results_not_interpreted")
        else:
            if not row.get("transport_error"):
                _fail("Non-200/404 response lacks explicit capture failure")
            issues.append("request_failed_no_identity_interpretation")
        observations.append(dict(
            index=index, ticker=ticker, query_date=query_date, http_status=status,
            requested_at_ms=requested, captured_at_ms=captured, historical_known_at_ms=None,
            provider_fields=provider_fields, issues=issues,
            response_file_sha256=entry["sha256"], raw_response_body_sha256=raw_sha,
            delisting_proven=False, historical_identity_qualified=False))
    if (dict(statuses) != completion.get("http_status_counts")
            or sorted(failures) != completion.get("failure_indices")):
        _fail("Dated completion status/failure reconciliation mismatch")
    grouped = defaultdict(list)
    for observation in observations:
        grouped[observation["ticker"]].append(observation)
    changes = []
    compare = ("cik", "composite_figi", "share_class_figi", "primary_exchange", "type", "name", "list_date")
    for ticker, rows in sorted(grouped.items()):
        previous = None
        for observation in sorted(rows, key=lambda item: item["query_date"]):
            value = observation["provider_fields"]
            if value is None:
                continue
            if previous is not None:
                old = previous["provider_fields"]
                differing = {field: dict(before=old[field], after=value[field])
                             for field in compare if old[field] != value[field]}
                if differing:
                    changes.append(dict(ticker=ticker, earlier_query_date=previous["query_date"],
                                        later_query_date=observation["query_date"], fields=differing,
                                        interval_between_snapshots_qualified=False,
                                        successor_or_corporate_action_inferred=False))
            previous = observation
    for relative, expected in states.items():
        target = root / relative
        if target.resolve(strict=True) != target or _identity(target.lstat()) != expected:
            _fail("Dated source identity changed after diagnostic")
    result = dict(
        schema="rl-quant.qt200-dated-identity-diagnostic-v1",
        capture_completion_sha256=expected_completion_sha256, plan_file_sha256=proof["sha256"],
        capture_source_sha256=plan["source_sha256"], universe_sha256=expected_universe_sha256,
        response_count=n, observed_http_status_counts=dict(statuses), failure_indices=failures,
        observations=observations, observed_field_changes=changes,
        physical_file_hashes_and_raw_body_crosslinks_verified=True,
        http_status_is_capture_attestation_not_independent_provider_signature=True,
        absence_proves_delisting=False, sparse_snapshots_prove_continuous_listing=False,
        active_flag_does_not_prove_listing_or_tradability=True,
        same_ticker_does_not_prove_same_issue_across_dates=True,
        historical_identity_qualified=False, point_in_time_qualified=False,
        native_security_ids_emitted=0, successor_joins_emitted=0,
        training_ready_for_adaptive_v5=False, provider_calls=0,
    )
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Dated diagnostic exceeds output byte bound")
    return result


def _dated_issue_assertion(entity: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    """Compare observed identifiers, never infer an issue interval or join."""
    fields = ("composite_figi", "share_class_figi")
    current = {field: entity["current_" + field] for field in fields}
    provider = observation["provider_fields"] or {}
    observed = {field: provider.get(field) for field in fields}

    def usable(value: Any) -> bool:
        return isinstance(value, str) and bool(value) and value == value.strip()

    matches, conflicts, incomparable = [], [], []
    for field in fields:
        if not usable(current[field]) or not usable(observed[field]):
            incomparable.append(field)
        elif current[field] == observed[field]:
            matches.append(field)
        else:
            conflicts.append(field)
    if conflicts:
        classification = "conflicts_with_current_observed_issue_identifiers"
    elif matches:
        classification = "supports_current_observed_issue_identifiers"
    else:
        classification = "unknown_issue_identity"
    reasons = []
    if observation["http_status"] != 200:
        reasons.append("unsuccessful_query_is_not_issue_or_delisting_evidence")
    if entity["active_exact_reference_count"] != 1:
        reasons.append("unique_current_reference_not_established")
    if not matches and not conflicts:
        reasons.append("no_comparable_issue_identifier_cik_and_name_are_insufficient")
    if any(value is not None and not usable(value) for value in (*current.values(), *observed.values())):
        reasons.append("malformed_or_empty_issue_identifier_not_compared")
    return dict(
        observation_index=observation["index"], query_ticker=observation["ticker"],
        query_date=observation["query_date"], captured_at_ms=observation["captured_at_ms"],
        response_file_sha256=observation["response_file_sha256"],
        raw_response_body_sha256=observation["raw_response_body_sha256"],
        current_issue_identifiers=current, observed_issue_identifiers=observed,
        classification=classification, matching_identifier_fields=matches,
        conflicting_identifier_fields=conflicts, incomparable_identifier_fields=incomparable,
        reasons=reasons, alias_query=observation["ticker"] != entity["ticker"],
        historical_known_at_ms=None, identity_qualified=False,
        continuous_interval_inferred=False, successor_join_inferred=False,
    )


def build_qt200_issue_resolution_evidence_v1(
    *, support_root: Path, expected_support_completion_sha256: str,
    capture_root: Path, expected_capture_completion_sha256: str,
    expected_universe_sha256: str,
) -> dict[str, Any]:
    """Cross-check source-bound observations without promoting any identity.

    Both readers reopen their original input chains; callers cannot supply
    precomputed reports or qualification flags. A matching FIGI supports only
    the observed identifier assertion. It does not turn sparse query dates into
    validity intervals, capture times into historical knowledge, or current
    same-ticker economic candidates into an issue-level accounting join.
    """
    support = build_qt200_identity_evidence_v1(
        support_root=Path(support_root),
        expected_completion_sha256=expected_support_completion_sha256,
        expected_universe_sha256=expected_universe_sha256,
    )
    dated = load_qt200_dated_identity_diagnostic_v1(
        capture_root=Path(capture_root),
        expected_completion_sha256=expected_capture_completion_sha256,
        expected_universe_sha256=expected_universe_sha256,
    )
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in dated["observations"]:
        by_ticker[observation["ticker"]].append(observation)
    resolutions, by_target, assigned = [], {}, set()
    totals: Counter[str] = Counter()
    for entity in support["security_candidates"]:
        ticker = entity["ticker"]
        symbols = sorted({ticker} | {
            event["provider_ticker"] for event in entity["ticker_event_observations"]
        })
        observations = sorted(
            (row for symbol in symbols for row in by_ticker[symbol]),
            key=lambda row: (row["ticker"], row["query_date"], row["index"]),
        )
        assertions = [_dated_issue_assertion(entity, row) for row in observations]
        counts = Counter(row["classification"] for row in assertions)
        totals.update(counts)
        assigned.update(row["index"] for row in observations)
        flags = []
        if not entity["current_issue_identifier_observed"]:
            flags.append("current_issue_identifier_not_observed")
        if entity["active_exact_reference_count"] != 1:
            flags.append("unique_current_reference_not_established")
        if counts["conflicts_with_current_observed_issue_identifiers"]:
            flags.append("conflicting_dated_issue_identifiers_require_source_resolution")
        if any(row["alias_query"] and row["conflicting_identifier_fields"] for row in assertions):
            flags.append("alias_query_conflicts_with_current_issue_identifiers")
        if counts["unknown_issue_identity"]:
            flags.append("some_dated_issue_assertions_remain_unknown")
        missing_queries = [symbol for symbol in symbols if not by_ticker[symbol]]
        if missing_queries:
            flags.append("candidate_query_symbols_not_observed_in_dated_capture")
        for gap in ("current_ticker_differs_from_last_provider_event",
                    "duplicate_or_conflicting_provider_event_dates",
                    "provider_events_not_issue_identifier_linked"):
            if gap in entity["gaps"]:
                flags.append(gap)
        result = dict(
            qt200_ticker=ticker, current_reference_evidence=entity,
            candidate_query_symbols=symbols, query_symbols_without_observations=missing_queries,
            dated_assertions=assertions, assertion_counts=dict(sorted(counts.items())),
            review_flags=sorted(flags),
            event_brackets_remain_unqualified=True,
            support_does_not_establish_continuous_identity=True,
            native_security_id=None, historical_identity_qualified=False,
            historical_known_at_ms=None, accounting_authorized=False,
        )
        resolutions.append(result)
        by_target[ticker] = result
    economic = []
    for event in support["economic_observations"]:
        symbol = event["provider_record"].get("ticker")
        joins = []
        for candidate in event["candidate_joins"]:
            resolution = by_target[candidate["qt200_ticker"]]
            exact_date = [row for row in resolution["dated_assertions"]
                          if row["query_ticker"] == symbol
                          and row["query_date"] == event["effective_date"]]
            joins.append(dict(
                candidate,
                cross_capture_review_flags=resolution["review_flags"],
                exact_event_date_dated_assertions=exact_date,
                review_flags_are_not_event_date_validity_masks=True,
                dated_assertions_do_not_authorize_economic_join=True,
            ))
        economic.append(dict(event, candidate_joins=joins))
    result = dict(
        schema="rl-quant.qt200-issue-resolution-evidence-v1",
        operation="cross_capture_issue_assertion_review_only",
        support_completion_sha256=expected_support_completion_sha256,
        dated_capture_completion_sha256=expected_capture_completion_sha256,
        universe_sha256=expected_universe_sha256,
        support_diagnostic_receipt_sha256=support["receipt_sha256"],
        dated_diagnostic_receipt_sha256=dated["receipt_sha256"],
        support_input_bindings=support["input_bindings"],
        ordered_tickers=support["ordered_tickers"], security_resolutions=resolutions,
        dated_observations=dated["observations"], observed_field_changes=dated["observed_field_changes"],
        unassigned_dated_observation_indices=sorted(
            row["index"] for row in dated["observations"] if row["index"] not in assigned),
        assertion_counts=dict(sorted(totals.items())),
        economic_observations=economic,
        duplicate_economic_event_observations=support["duplicate_economic_event_observations"],
        support_original_provider_raw_bodies_reauthenticated=False,
        support_upstream_receipt_semantics_revalidated=False,
        dated_physical_files_and_raw_body_crosslinks_verified=True,
        name_or_cik_used_as_issue_identifier=False, conflicting_identifier_takes_precedence=True,
        continuous_identity_intervals_emitted=0, native_security_ids_emitted=0,
        successor_joins_emitted=0, historical_identity_qualified=False,
        point_in_time_qualified=False, economic_accounting_qualified=False,
        training_ready_for_adaptive_v5=False, provider_calls=0, model_tests_run=False,
    )
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Issue-resolution evidence exceeds diagnostic byte bound")
    return result


def _corporate_action_candidate_targets(contents, tickers, sources):
    """Reproduce support extraction candidates, never historical identity."""
    current = defaultdict(list)
    for i, row in enumerate(_records(contents["reference-records.jsonl"])):
        _location(row, "reference-records.jsonl", i, sources)
        provider = row["provider_record"]
        if provider.get("active") is True and provider.get("ticker") in tickers:
            current[provider["ticker"]].append(provider)
    current_figi = {t: rows[0]["composite_figi"] for t, rows in current.items()
                    if len(rows) == 1 and rows[0].get("composite_figi")}
    figi_targets = defaultdict(set)
    for ticker, figi in current_figi.items():
        figi_targets[figi].add(ticker)
    targets = {ticker: {ticker} for ticker in tickers}
    for i, row in enumerate(_records(contents["ticker-event-captures.jsonl"])):
        _location(row, "ticker-event-captures.jsonl", i, sources)
        result = row.get("provider_response", {}).get("results") or {}
        figi = result.get("composite_figi")
        if row["response_metadata"].get("response_status") != 200 or not figi:
            continue
        for ticker in row["qt200_tickers"]:
            if (ticker not in targets or current_figi.get(ticker) != figi
                    or len(figi_targets[figi]) != 1):
                continue
            for event in result.get("events", []):
                if event.get("type") != "ticker_change":
                    continue
                symbol = (event.get("ticker_change") or {}).get("ticker")
                if symbol and event.get("date"):
                    _day(event["date"])
                    if not isinstance(symbol, str) or symbol != symbol.strip():
                        _fail("Malformed candidate ticker event")
                    targets.setdefault(symbol, set()).add(ticker)
    return targets


def reconcile_qt200_split_source_v1(
    *, source_root: Path, relative_payload_path: str,
    expected_payload_sha256: str, expected_receipt_file_sha256: str,
    expected_commit_file_sha256: str, support_root: Path,
    expected_support_completion_sha256: str, expected_universe_sha256: str,
) -> dict[str, Any]:
    """Reauthenticate only the retained split candidates against original pages.

    The original native transaction and complete raw page chain are reopened.
    Selection is reconstructed from the support bundle's exact current symbols
    and explicitly observed same-FIGI ticker-event candidates. That is the
    existing extraction rule, NOT a historical identity/eligibility rule.
    Duplicate provider rows at different page/result positions are preserved;
    duplicate selected provenance positions, missing rows and added rows fail.
    No files are written, no provider is contacted, and no accounting is run.
    Each original file is bounded by MAX_FILE_BYTES before native loading.
    """
    from time import time_ns

    from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
        parse_massive_economic_raw_rest_capture_v8,
    )
    from rl_quant.data_sources.massive.source_receipts import load_massive_source_bundle

    root = Path(source_root).absolute()
    if (root.resolve(strict=True) != root or not root.is_dir()
            or root.stat().st_uid != os.getuid()):
        _fail("Split source root must be an owned nonsymlink directory")
    transaction = {}
    states = {}
    for suffix, expected in (
        ("", expected_payload_sha256), (".receipt.json", expected_receipt_file_sha256),
        (".commit.json", expected_commit_file_sha256),
    ):
        relative = relative_payload_path + suffix
        body, states[relative] = _read(root, relative)
        if (root / relative).lstat().st_nlink != 1:
            _fail("Split transaction must have a single link")
        if _sha(body) != _digest(expected):
            _fail("Split transaction expected physical hash mismatch")
        transaction[suffix] = body
    try:
        loaded = load_massive_source_bundle(
            root=root, relative_payload_path=relative_payload_path,
            verified_at_ms=time_ns() // 1_000_000,
        )
        capture = parse_massive_economic_raw_rest_capture_v8(root=root, loaded_source=loaded)
    except (ValueError, OSError) as exc:
        raise QT200IdentityEvidenceError("Native split transaction/page replay failed") from exc
    if (capture.surface_id != "massive-splits-v1" or capture.fixed_runtime_captured is not False
            or loaded.receipt.physical_sha256 != expected_payload_sha256
            or loaded.receipt.source_object_key != relative_payload_path
            or capture.page_count > 128):
        _fail("Expected a bounded nonauthorizing native split transaction")

    support, support_states, contents, manifest, bindings = _load_bundle(
        Path(support_root), expected_support_completion_sha256, expected_universe_sha256)
    spec = _json(contents["qt200-universe-spec.json"])
    tickers = spec["tickers"]
    if (spec.get("ticker_count") != 200 or len(tickers) != 200
            or any(not isinstance(t, str) or not t or t != t.strip() for t in tickers)
            or len(set(tickers)) != 200):
        _fail("Expected the exact ordered 200-ticker specification")
    if (manifest.get("requested_economic_coverage_start") != capture.coverage_start_date
            or manifest.get("requested_economic_coverage_end") != capture.coverage_end_date):
        _fail("Split capture/support coverage interval differs")
    sources = {}
    for row in manifest["sources"]:
        if row["source_file"] in sources:
            _fail("Duplicate upstream source name")
        sources[row["source_file"]] = row
    source_name = PurePosixPath(relative_payload_path).name
    source = sources.get(source_name)
    if (source is None or source.get("source_sha256") != expected_payload_sha256
            or source.get("upstream_receipt_sha256") != loaded.receipt.receipt_sha256
            or source.get("source_bytes") != loaded.receipt.content_length):
        _fail("Split support/original source binding differs")

    targets = _corporate_action_candidate_targets(contents, tickers, sources)

    selected = _records(contents["splits.jsonl"])
    observed, observed_positions = [], set()
    for i, row in enumerate(selected):
        _location(row, "splits.jsonl", i, sources)
        p = row["provenance"]
        position = (p.get("page_index"), p.get("result_index"))
        if any(type(n) is not int or n < 0 for n in position):
            _fail("Split provenance page/result index is invalid")
        if position in observed_positions:
            _fail("Duplicate selected split provenance position")
        observed_positions.add(position)
        if (p.get("source_file") != source_name or p.get("capture_index") is not None
                or p.get("historical_available_at") is not None
                or any(row.get(flag) is not False for flag in (
                    "corporate_action_permanent_identity_verified", "point_in_time_complete",
                    "automatic_adjustment_or_stitching_authorized"))
                or row.get("historical_available_at") is not None):
            _fail("Split provenance scope or nonqualification flags differ")
        observed.append((position, row))

    reconciled = []
    all_count, next_selected = 0, 0
    for page in capture.pages:
        records = page.parsed_body()["results"]
        for position, provider in enumerate(records):
            all_count += 1
            if all_count > 50_000:
                _fail("Split source result inventory exceeds diagnostic bound")
            event_day = _day(provider.get("execution_date"))
            if not capture.coverage_start_date <= event_day <= capture.coverage_end_date:
                _fail("Split source result outside captured coverage interval")
            symbol = provider.get("ticker")
            if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
                _fail("Split source ticker is malformed")
            if symbol not in targets:
                continue
            if next_selected >= len(observed):
                _fail("Missing selected split source row")
            original_position = (page.page_index, position)
            stored_position, row = observed[next_selected]
            p = row["provenance"]
            expected_provenance = dict(
                source_file=source_name, source_sha256=expected_payload_sha256,
                upstream_receipt_sha256=loaded.receipt.receipt_sha256, capture_index=None,
                page_index=page.page_index, result_index=position,
                provider_request_id=page.provider_request_id,
                raw_response_body_sha256=page.raw_response_body_sha256,
                captured_at_ms=page.completed_at_ms,
                captured_at_utc=datetime.fromtimestamp(page.completed_at_ms / 1000, timezone.utc).isoformat(),
                historical_available_at=None,
            )
            if (stored_position != original_position or p != expected_provenance
                    or _canonical(row["provider_record"]) != _canonical(provider)
                    or row.get("candidate_qt200_tickers") != sorted(targets[symbol])):
                _fail("Selected split page/result/provenance inventory mismatch")
            reconciled.append(dict(support_record_index=next_selected,
                page_index=page.page_index, result_index=position,
                provider_record=provider, provenance=expected_provenance,
                candidate_qt200_tickers=sorted(targets[symbol]),
                identity_qualified=False, historical_available_at_ms=None,
                accounting_authorized=False))
            next_selected += 1
    if next_selected != len(observed):
        _fail("Additional selected split source rows")
    for directory, expected_states in ((root, states), (support, support_states)):
        for relative, expected in expected_states.items():
            target = directory / relative
            if target.resolve(strict=True) != target or _identity(target.lstat()) != expected:
                _fail("Split reconciliation input changed during replay")
    result = dict(
        schema=SCHEMA, operation="split_original_source_provenance_reconciliation_only",
        source_payload_relative_path=relative_payload_path,
        source_payload_sha256=expected_payload_sha256,
        source_receipt_file_sha256=expected_receipt_file_sha256,
        source_commit_file_sha256=expected_commit_file_sha256,
        source_receipt_sha256=loaded.receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded.commit.receipt_sha256,
        support_completion_sha256=expected_support_completion_sha256,
        support_input_bindings=bindings, universe_sha256=expected_universe_sha256,
        ordered_tickers=tickers, candidate_symbols=sorted(targets),
        capture_kind=capture.capture_kind, fixed_runtime_captured=False,
        coverage_start_date=capture.coverage_start_date, coverage_end_date=capture.coverage_end_date,
        page_count=capture.page_count, raw_page_inventory_sha256=capture.raw_page_inventory_sha256,
        original_source_result_count=all_count, selected_split_count=next_selected,
        reconciled_splits=reconciled, split_original_provider_raw_bodies_reauthenticated=True,
        exact_selected_position_inventory_verified=True, original_transaction_replayed=True,
        all_support_original_provider_raw_bodies_reauthenticated=False,
        historical_identity_qualified=False, historical_availability_qualified=False,
        economic_accounting_qualified=False, point_in_time_qualified=False,
        native_security_ids_emitted=0, successor_joins_emitted=0,
        training_ready_for_adaptive_v5=False, provider_calls=0, source_writes=False,
    )
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Split reconciliation output exceeds diagnostic byte bound")
    return result


def _hash_original_dividend(root: Path, relative: str) -> tuple[str, tuple[int, ...]]:
    """Bound and stream one original payload; do not retain a second raw copy."""
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or ".." in path.parts:
        _fail("Unsafe inventory path")
    target = root / relative
    if target.resolve(strict=True) != target:
        _fail("Symlink in source path")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or before.st_size > MAX_DIVIDEND_SOURCE_BYTES):
            _fail("Dividend source must be an owned single-link bounded regular file")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                _fail("Dividend source changed during read")
            digest.update(chunk)
            remaining -= len(chunk)
        if (stream.read(1) or _identity(before) != _identity(os.fstat(stream.fileno()))
                or os.fstat(stream.fileno()).st_nlink != 1):
            _fail("Dividend source changed during read")
    after = target.lstat()
    if _identity(before) != _identity(after) or after.st_nlink != 1:
        _fail("Dividend source pathname changed during read")
    return digest.hexdigest(), _identity(before)


def reconcile_qt200_dividend_source_v1(
    *, source_root: Path, relative_payload_path: str,
    expected_payload_sha256: str, expected_receipt_file_sha256: str,
    expected_commit_file_sha256: str, support_root: Path,
    expected_support_completion_sha256: str, expected_universe_sha256: str,
) -> dict[str, Any]:
    """Reopen original dividend pages and reconcile exact support candidates.

    This is the dividend counterpart of the split diagnostic, with a dedicated
    256 MiB original-payload bound for the retained 186,640,862-byte capture.
    Sidecars and every support file retain the 16 MiB bound. Native validation
    checks the complete transaction and raw page chain; replay is capped at
    128 pages and 500,000 results (retained capture: 81 / 401,325).
    A verified original row is not historical identity, availability, dividend
    cash accounting, or training authorization. Capture times remain capture
    times. Identical provider rows at distinct positions remain distinct.
    No provider requests, writes, source modifications, or accounting occur.
    """
    from time import time_ns

    from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
        parse_massive_economic_raw_rest_capture_v8,
    )
    from rl_quant.data_sources.massive.source_receipts import load_massive_source_bundle

    root = Path(source_root).absolute()
    if (root.resolve(strict=True) != root or not root.is_dir()
            or root.stat().st_uid != os.getuid()):
        _fail("Dividend source root must be an owned nonsymlink directory")
    physical, payload_state = _hash_original_dividend(root, relative_payload_path)
    if physical != _digest(expected_payload_sha256):
        _fail("Dividend transaction expected physical hash mismatch")
    states = {relative_payload_path: payload_state}
    for suffix, expected in (
        (".receipt.json", expected_receipt_file_sha256),
        (".commit.json", expected_commit_file_sha256),
    ):
        relative = relative_payload_path + suffix
        body, states[relative] = _read(root, relative)
        if (root / relative).lstat().st_nlink != 1:
            _fail("Dividend transaction must have a single link")
        if _sha(body) != _digest(expected):
            _fail("Dividend transaction expected physical hash mismatch")
    try:
        loaded = load_massive_source_bundle(
            root=root, relative_payload_path=relative_payload_path,
            verified_at_ms=time_ns() // 1_000_000,
        )
        capture = parse_massive_economic_raw_rest_capture_v8(root=root, loaded_source=loaded)
    except (ValueError, OSError) as exc:
        raise QT200IdentityEvidenceError("Native dividend transaction/page replay failed") from exc
    if (capture.surface_id != "massive-dividends-v1" or capture.fixed_runtime_captured is not False
            or loaded.receipt.physical_sha256 != expected_payload_sha256
            or loaded.receipt.source_object_key != relative_payload_path
            or capture.page_count > MAX_DIVIDEND_SOURCE_PAGES
            or sum(page.result_count for page in capture.pages) > MAX_DIVIDEND_SOURCE_RESULTS):
        _fail("Expected a bounded nonauthorizing native dividend transaction")

    support, support_states, contents, manifest, bindings = _load_bundle(
        Path(support_root), expected_support_completion_sha256, expected_universe_sha256)
    spec = _json(contents["qt200-universe-spec.json"])
    tickers = spec["tickers"]
    if (spec.get("ticker_count") != 200 or len(tickers) != 200
            or any(not isinstance(t, str) or not t or t != t.strip() for t in tickers)
            or len(set(tickers)) != 200):
        _fail("Expected the exact ordered 200-ticker specification")
    if (manifest.get("requested_economic_coverage_start") != capture.coverage_start_date
            or manifest.get("requested_economic_coverage_end") != capture.coverage_end_date):
        _fail("Dividend capture/support coverage interval differs")
    sources = {}
    for row in manifest["sources"]:
        if row["source_file"] in sources:
            _fail("Duplicate upstream source name")
        sources[row["source_file"]] = row
    source_name = PurePosixPath(relative_payload_path).name
    source = sources.get(source_name)
    if (source is None or source.get("source_sha256") != expected_payload_sha256
            or source.get("upstream_receipt_sha256") != loaded.receipt.receipt_sha256
            or source.get("source_bytes") != loaded.receipt.content_length):
        _fail("Dividend support/original source binding differs")
    targets = _corporate_action_candidate_targets(contents, tickers, sources)

    observed, observed_positions = [], set()
    for i, row in enumerate(_records(contents["dividends.jsonl"])):
        _location(row, "dividends.jsonl", i, sources)
        p = row["provenance"]
        position = (p.get("page_index"), p.get("result_index"))
        if any(type(n) is not int or n < 0 for n in position):
            _fail("Dividend provenance page/result index is invalid")
        if position in observed_positions:
            _fail("Duplicate selected dividend provenance position")
        observed_positions.add(position)
        if (p.get("source_file") != source_name or p.get("capture_index") is not None
                or p.get("historical_available_at") is not None
                or any(row.get(flag) is not False for flag in (
                    "corporate_action_permanent_identity_verified", "point_in_time_complete",
                    "automatic_adjustment_or_stitching_authorized"))
                or row.get("historical_available_at") is not None):
            _fail("Dividend provenance scope or nonqualification flags differ")
        observed.append((position, row))

    reconciled = []
    all_count, next_selected = 0, 0
    for page in capture.pages:
        for position, provider in enumerate(page.parsed_body()["results"]):
            all_count += 1
            if all_count > MAX_DIVIDEND_SOURCE_RESULTS:
                _fail("Dividend source result inventory exceeds diagnostic bound")
            event_day = _day(provider.get("ex_dividend_date"))
            if not capture.coverage_start_date <= event_day <= capture.coverage_end_date:
                _fail("Dividend source result outside captured coverage interval")
            symbol = provider.get("ticker")
            if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
                _fail("Dividend source ticker is malformed")
            if symbol not in targets:
                continue
            if next_selected >= len(observed):
                _fail("Missing selected dividend source row")
            original_position = (page.page_index, position)
            stored_position, row = observed[next_selected]
            expected_provenance = dict(
                source_file=source_name, source_sha256=expected_payload_sha256,
                upstream_receipt_sha256=loaded.receipt.receipt_sha256, capture_index=None,
                page_index=page.page_index, result_index=position,
                provider_request_id=page.provider_request_id,
                raw_response_body_sha256=page.raw_response_body_sha256,
                captured_at_ms=page.completed_at_ms,
                captured_at_utc=datetime.fromtimestamp(page.completed_at_ms / 1000, timezone.utc).isoformat(),
                historical_available_at=None,
            )
            if (stored_position != original_position or row["provenance"] != expected_provenance
                    or _canonical(row["provider_record"]) != _canonical(provider)
                    or row.get("candidate_qt200_tickers") != sorted(targets[symbol])):
                _fail("Selected dividend page/result/provenance inventory mismatch")
            reconciled.append(dict(support_record_index=next_selected,
                page_index=page.page_index, result_index=position,
                provider_record=provider, provenance=expected_provenance,
                candidate_qt200_tickers=sorted(targets[symbol]),
                identity_qualified=False, historical_available_at_ms=None,
                accounting_authorized=False))
            next_selected += 1
    if next_selected != len(observed):
        _fail("Additional selected dividend source rows")
    for directory, expected_states in ((root, states), (support, support_states)):
        for relative, expected in expected_states.items():
            target = directory / relative
            if target.resolve(strict=True) != target or _identity(target.lstat()) != expected:
                _fail("Dividend reconciliation input changed during replay")
    result = dict(
        schema=SCHEMA, operation="dividend_original_source_provenance_reconciliation_only",
        source_payload_relative_path=relative_payload_path,
        source_payload_sha256=expected_payload_sha256,
        source_receipt_file_sha256=expected_receipt_file_sha256,
        source_commit_file_sha256=expected_commit_file_sha256,
        source_receipt_sha256=loaded.receipt.receipt_sha256,
        source_commit_receipt_sha256=loaded.commit.receipt_sha256,
        support_completion_sha256=expected_support_completion_sha256,
        support_input_bindings=bindings, universe_sha256=expected_universe_sha256,
        ordered_tickers=tickers, candidate_symbols=sorted(targets),
        capture_kind=capture.capture_kind, fixed_runtime_captured=False,
        coverage_start_date=capture.coverage_start_date, coverage_end_date=capture.coverage_end_date,
        page_count=capture.page_count, raw_page_inventory_sha256=capture.raw_page_inventory_sha256,
        original_source_result_count=all_count, selected_dividend_count=next_selected,
        reconciled_dividends=reconciled, dividend_original_provider_raw_bodies_reauthenticated=True,
        exact_selected_position_inventory_verified=True, original_transaction_replayed=True,
        all_support_original_provider_raw_bodies_reauthenticated=False,
        historical_identity_qualified=False, historical_availability_qualified=False,
        economic_accounting_qualified=False, point_in_time_qualified=False,
        native_security_ids_emitted=0, successor_joins_emitted=0,
        training_ready_for_adaptive_v5=False, provider_calls=0, source_writes=False,
    )
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Dividend reconciliation output exceeds diagnostic byte bound")
    return result


_IDENTITY_CAPTURE_NAMES = (
    "reference-tickers-current-v1.json", "ticker-events-v1.json",
    "missing-figi-ticker-events-v1.json",
)
_IDENTITY_CAPTURE_SCHEMAS = (
    "quanttrade-massive-reference-tickers-capture-v1",
    "quanttrade-massive-ticker-events-capture-v1",
    "quanttrade-massive-missing-figi-ticker-events-capture-v1",
)


def _identity_capture_bytes(value: Any) -> bytes:
    """The original producer hashes ASCII canonical JSON WITH a newline."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _identity_fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail("Original identity " + name + " field inventory differs")


def _identity_url(value, path):
    if not isinstance(value, str):
        _fail("Original identity URL is missing")
    parts = urlsplit(value)
    denied = {"apikey", "api_key", "authorization", "token", "access_token",
              "password", "secret", "client_secret", "key"}
    if (parts.scheme != "https" or parts.netloc != "api.massive.com"
            or parts.path != path or parts.fragment or parts.username or parts.password
            or any(k.lower() in denied for k, _ in parse_qsl(parts.query))):
        _fail("Original identity URL leaves the secret-free fixed surface")
    return parts


def _identity_response(response):
    requested, completed = response.get("requested_at_ms"), response.get("completed_at_ms")
    if (any(type(t) is not int or t < 0 for t in (requested, completed))
            or completed < requested):
        _fail("Original identity response chronology differs")
    request_id = response.get("provider_request_id")
    if (not isinstance(request_id, str) or not request_id
            or "json" not in str(response.get("response_content_type", "")).lower()):
        _fail("Original identity response metadata differs")
    encoded = response.get("raw_response_body_base64")
    if not isinstance(encoded, str) or len(encoded) > MAX_FILE_BYTES:
        _fail("Original identity raw response exceeds diagnostic bound")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise QT200IdentityEvidenceError("Original identity response base64 differs") from exc
    if (base64.b64encode(raw).decode("ascii") != encoded
            or type(response.get("raw_response_content_length")) is not int
            or len(raw) != response["raw_response_content_length"]
            or _sha(raw) != _digest(response.get("raw_response_body_sha256"))):
        _fail("Original identity raw response hash/length differs")
    body = _json(raw)
    if not isinstance(body, dict) or body.get("request_id") != request_id:
        _fail("Original identity response request ID differs")
    return body


def _identity_provenance(name, source, response, index, capture_index=None):
    return dict(source_file=name, source_sha256=source["source_sha256"],
        upstream_receipt_sha256=source["upstream_receipt_sha256"], capture_index=capture_index,
        page_index=response.get("page_index"), provider_request_id=response["provider_request_id"],
        raw_response_body_sha256=response["raw_response_body_sha256"], result_index=index,
        captured_at_ms=response["completed_at_ms"],
        captured_at_utc=datetime.fromtimestamp(response["completed_at_ms"] / 1000, timezone.utc).isoformat(),
        historical_available_at=None)


def _identity_event_position_inventory(rows):
    """Preserve response/event order, ignoring only old set-order target blocks."""
    groups, seen_responses = [], set()
    previous_response, previous_ticker = None, None
    for row in rows:
        provenance = row.get("provenance", {})
        response = (provenance.get("source_file"), provenance.get("result_index"))
        ticker = row.get("qt200_ticker")
        if (not isinstance(response[0], str) or type(response[1]) is not int or response[1] < 0
                or not isinstance(ticker, str) or not ticker):
            _fail("Malformed selected ticker-change source position")
        if response != previous_response:
            if response in seen_responses:
                _fail("Repeated noncontiguous ticker-change response position")
            seen_responses.add(response)
            groups.append((response, {}))
            previous_response, previous_ticker = response, None
        blocks = groups[-1][1]
        if ticker != previous_ticker:
            if ticker in blocks:
                _fail("Repeated noncontiguous ticker-change target block")
            blocks[ticker] = []
            previous_ticker = ticker
        blocks[ticker].append(row)
    return groups


def reconcile_qt200_identity_source_v1(
    *, source_root: Path, expected_capture_sha256: dict[str, str],
    expected_implementation_sha256: dict[str, str], expected_entitlement_receipt_sha256: str,
    support_root: Path, expected_support_completion_sha256: str, expected_universe_sha256: str,
) -> dict[str, Any]:
    """Reauthenticate three self-receipted identity captures and their selection.

    These legacy producer files are NOT native source-object transactions or
    provider signatures. Producer hashes establish declared implementation
    identity, not a fresh execution attestation. All original responses,
    including unselected and 404 rows, are checked; source observations at
    distinct positions are never deduplicated. Only the original extraction's
    derived alias table uses its documented last-observation-per-event rule.
    This returns an ordinary diagnostic, with no writes or provider calls and
    no historical identity, availability, PIT, accounting or training authority.
    """
    if (set(expected_capture_sha256) != set(_IDENTITY_CAPTURE_NAMES)
            or set(expected_implementation_sha256) != set(_IDENTITY_CAPTURE_NAMES)):
        _fail("Expected exactly three original identity capture bindings")
    entitlement = _digest(expected_entitlement_receipt_sha256)
    root = Path(source_root).absolute()
    if (root.resolve(strict=True) != root or not root.is_dir()
            or root.stat().st_uid != os.getuid()):
        _fail("Original identity root must be an owned nonsymlink directory")
    support, support_states, contents, manifest, bindings = _load_bundle(
        Path(support_root), expected_support_completion_sha256, expected_universe_sha256)
    spec = _json(contents["qt200-universe-spec.json"])
    tickers = spec.get("tickers")
    if (spec.get("ticker_count") != 200 or not isinstance(tickers, list) or len(tickers) != 200
            or any(not isinstance(t, str) or not t or t != t.strip() for t in tickers)
            or len(set(tickers)) != 200):
        _fail("Expected the exact ordered 200-ticker specification")
    start = _day(manifest.get("requested_economic_coverage_start"))
    end = _day(manifest.get("requested_economic_coverage_end"))
    if start > end:
        _fail("Identity support coverage interval differs")
    end_exclusive = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    sources = {}
    for row in manifest["sources"]:
        if row["source_file"] in sources:
            _fail("Duplicate upstream source name")
        sources[row["source_file"]] = row
    documents, states, source_bindings = {}, {}, []
    for index, name in enumerate(_IDENTITY_CAPTURE_NAMES):
        raw, states[name] = _read(root, name)
        if (root / name).lstat().st_nlink != 1:
            _fail("Original identity capture must have a single link")
        physical = _digest(expected_capture_sha256[name])
        if _sha(raw) != physical:
            _fail("Original identity capture physical hash differs")
        document = _json(raw)
        fields = {"schema", "capture_id", "endpoint", "query", "entitlement_receipt_sha256",
                  "implementation_source_sha256", "credential_values_recorded", "receipt_sha256"}
        fields |= {"captures"} if index == 0 else {
            "reference_capture_file_sha256", "identifier_inventory_sha256", "rows"}
        if index == 2:
            fields.add("request_implementation_source_sha256")
        _identity_fields(document, fields, "capture")
        if (raw != _identity_capture_bytes(document)
                or document["receipt_sha256"] != _sha(_identity_capture_bytes(
                    {k: v for k, v in document.items() if k != "receipt_sha256"}))):
            _fail("Original identity canonical file/self-receipt differs")
        if (document["schema"] != _IDENTITY_CAPTURE_SCHEMAS[index]
                or document["credential_values_recorded"] is not False
                or document["entitlement_receipt_sha256"] != entitlement
                or not isinstance(document["capture_id"], str) or not document["capture_id"]
                or document["implementation_source_sha256"] != _digest(expected_implementation_sha256[name])):
            _fail("Original identity capture schema/producer binding differs")
        endpoint = ("https://api.massive.com/v3/reference/tickers" if index == 0
                    else "https://api.massive.com/vX/reference/tickers")
        queries = ("us-stocks-current-active-and-inactive-complete-pagination",
                   "ticker-change-events-by-distinct-common-stock-composite-figi",
                   "ticker-change-events-by-missing-figi-common-stock-ticker")
        if document["endpoint"] != endpoint or document["query"] != queries[index]:
            _fail("Original identity endpoint/query differs")
        if index and document["reference_capture_file_sha256"] != expected_capture_sha256[_IDENTITY_CAPTURE_NAMES[0]]:
            _fail("Original identity event/reference file binding differs")
        if index == 2 and document["request_implementation_source_sha256"] != expected_implementation_sha256[_IDENTITY_CAPTURE_NAMES[1]]:
            _fail("Original missing-FIGI request producer binding differs")
        source = sources.get(name)
        if (source is None or source.get("source_sha256") != physical
                or source.get("source_bytes") != len(raw)
                or source.get("upstream_receipt_sha256") != document["receipt_sha256"]):
            _fail("Original identity capture/support binding differs")
        documents[name] = document
        source_bindings.append(dict(source_file=name, source_sha256=physical, bytes=len(raw),
            receipt_sha256=document["receipt_sha256"],
            declared_implementation_sha256=document["implementation_source_sha256"]))

    reference = documents[_IDENTITY_CAPTURE_NAMES[0]]
    captures = reference["captures"]
    if not isinstance(captures, list) or len(captures) != 2:
        _fail("Original reference must contain both active-status captures")
    all_reference, page_count, prior_capture_end = [], 0, None
    reference_page_fields = {"page_index", "request_url", "final_response_url", "requested_at_ms",
        "completed_at_ms", "provider_request_id", "response_content_type", "raw_response_body_base64",
        "raw_response_body_sha256", "raw_response_content_length", "result_count", "result_inventory_sha256", "next_url"}
    for capture_index, capture in enumerate(captures):
        _identity_fields(capture, {"active", "page_count", "result_count", "pages"}, "status capture")
        pages = capture["pages"]
        if (capture["active"] is not bool(capture_index) or not isinstance(pages, list)
                or not 1 <= len(pages) <= 128 or type(capture["page_count"]) is not int
                or capture["page_count"] != len(pages)):
            _fail("Original reference active-status/page inventory differs")
        prior_end, prior_next, subtotal = None, None, 0
        for page_index, page in enumerate(pages):
            _identity_fields(page, reference_page_fields, "reference page")
            if type(page["page_index"]) is not int or page["page_index"] != page_index:
                _fail("Original reference page order differs")
            requested_url = _identity_url(page["request_url"], "/v3/reference/tickers")
            _identity_url(page["final_response_url"], "/v3/reference/tickers")
            if page["request_url"] != page["final_response_url"]:
                _fail("Original reference final request URL differs")
            if page_index == 0:
                expected_query = dict(market="stocks", locale="us", active=str(capture["active"]).lower(),
                                      limit="1000", sort="ticker", order="asc")
                if sorted(parse_qsl(requested_url.query)) != sorted(expected_query.items()):
                    _fail("Original reference initial query differs")
            elif page["request_url"] != prior_next or page["requested_at_ms"] < prior_end:
                _fail("Original reference pagination/chronology chain differs")
            body = _identity_response(page)
            if page_index == 0 and prior_capture_end is not None and page["requested_at_ms"] < prior_capture_end:
                _fail("Original reference cross-capture chronology differs")
            rows = body.get("results")
            if (body.get("status") != "OK" or not isinstance(rows, list)
                    or any(not isinstance(row, dict) for row in rows)
                    or type(page["result_count"]) is not int or page["result_count"] != len(rows)
                    or page["result_inventory_sha256"] != _sha(_identity_capture_bytes(rows))
                    or body.get("next_url") != page["next_url"]):
                _fail("Original reference response/count inventory differs")
            if (page["next_url"] is None) != (page_index == len(pages) - 1):
                _fail("Original reference pagination does not close")
            if page["next_url"] is not None:
                _identity_url(page["next_url"], "/v3/reference/tickers")
            subtotal += len(rows)
            if len(all_reference) + len(rows) > 100_000:
                _fail("Original reference result bound exceeded")
            all_reference.extend((row, _identity_provenance(_IDENTITY_CAPTURE_NAMES[0],
                sources[_IDENTITY_CAPTURE_NAMES[0]], page, i, capture_index)) for i, row in enumerate(rows))
            prior_end, prior_next = page["completed_at_ms"], page["next_url"]
            page_count += 1
        if type(capture["result_count"]) is not int or capture["result_count"] != subtotal:
            _fail("Original reference total count differs")
        prior_capture_end = prior_end

    current = defaultdict(list)
    for row, _ in all_reference:
        if row.get("active") is True and row.get("ticker") in tickers:
            current[row["ticker"]].append(row)
    current_figi = {t: rows[0]["composite_figi"] for t, rows in current.items()
                    if len(rows) == 1 and rows[0].get("composite_figi")}
    figi_targets = defaultdict(list)
    for ticker in tickers:
        if ticker in current_figi:
            figi_targets[current_figi[ticker]].append(ticker)
    event_rows, selected_captures, event_counts = [], [], {}
    alias_events = defaultdict(list)
    event_fields = {"identifier", "requested_at_ms", "completed_at_ms", "provider_request_id",
        "response_content_type", "raw_response_body_base64", "raw_response_body_sha256",
        "raw_response_content_length", "event_count", "response_status"}
    for index, name in enumerate(_IDENTITY_CAPTURE_NAMES[1:], 1):
        expected_identifiers = sorted({row.get("composite_figi") if index == 1 else row.get("ticker")
            for row, _ in all_reference if row.get("type") == "CS" and (
                bool(row.get("composite_figi")) if index == 1 else
                not row.get("composite_figi") and not row.get("share_class_figi") and bool(row.get("ticker")))})
        document = documents[name]
        rows = document["rows"]
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 50_000
                or document["identifier_inventory_sha256"] != _sha(_identity_capture_bytes(expected_identifiers))):
            _fail("Original event identifier inventory differs")
        observed_identifiers = []
        for response_index, response in enumerate(rows):
            _identity_fields(response, event_fields, "event response")
            observed_identifiers.append(response["identifier"])
            body = _identity_response(response)
            result = body.get("results") or {}
            if not isinstance(result, dict):
                _fail("Original event result shape differs")
            events = result.get("events") or []
            if (not isinstance(events, list) or type(response["event_count"]) is not int
                    or len(events) != response["event_count"] or len(events) > 10_000
                    or any(not isinstance(event, dict) or event.get("type") != "ticker_change"
                           or not isinstance(event.get("ticker_change"), dict) for event in events)):
                _fail("Original ticker-change event inventory differs")
            if type(response["response_status"]) is not int:
                _fail("Original event response status differs")
            if response["response_status"] == 200:
                if not isinstance(body.get("results"), dict):
                    _fail("Original successful event response differs")
            elif response["response_status"] == 404:
                if body.get("status") != "NOT_FOUND" or body.get("message") != "No events found for given ID" or events:
                    _fail("Original not-found event response differs")
            else:
                _fail("Original event response status differs")
            figi = result.get("composite_figi")
            matched = set(figi_targets.get(response["identifier"], [])) | set(figi_targets.get(figi, []))
            if response["identifier"] in tickers:
                matched.add(response["identifier"])
            if not matched:
                continue
            provenance = _identity_provenance(name, sources[name], response, response_index)
            metadata = {k: v for k, v in response.items() if k != "raw_response_body_base64"}
            metadata.update(captured_at_utc=provenance["captured_at_utc"],
                            capture_time_is_historical_availability_time=False)
            selected_captures.append(dict(qt200_tickers=sorted(matched), request_identifier=response["identifier"],
                response_metadata=metadata, provider_response=body, provenance=provenance, point_in_time_complete=False))
            for ticker in sorted(matched):
                match = bool(response["response_status"] == 200 and current_figi.get(ticker) is not None
                             and figi == current_figi[ticker] and len(figi_targets[figi]) == 1)
                for event_index, event in enumerate(events):
                    item = dict(qt200_ticker=ticker, provider_event=event, composite_figi=figi,
                        permanent_identity_match_verified=match, event_index=event_index, provenance=provenance)
                    event_rows.append(item)
                    if match and event["ticker_change"].get("ticker") and event.get("date"):
                        _day(event["date"])
                        alias_events[ticker].append(item)
        if observed_identifiers != expected_identifiers:
            _fail("Original event identifiers missing, additional, duplicated or reordered")
        event_counts[name] = len(rows)

    aliases, alias_targets = [], defaultdict(set)
    for ticker in tickers:
        unique = {(item["provider_event"]["date"], item["provider_event"]["ticker_change"]["ticker"]): item
                  for item in alias_events[ticker]}
        ordered = [unique[key] for key in sorted(unique)]
        conflicting = len({item["provider_event"]["date"] for item in ordered}) != len(ordered)
        for i, item in enumerate(ordered):
            event = item["provider_event"]
            alias = event["ticker_change"]["ticker"]
            alias_targets[alias].add(ticker)
            next_date = ordered[i+1]["provider_event"]["date"] if i+1 < len(ordered) else None
            lower, upper = max(start, event["date"]), min(end_exclusive, next_date or end_exclusive)
            aliases.append(dict(qt200_ticker=ticker, provider_ticker=alias, composite_figi=current_figi[ticker],
                provider_event_date=event["date"], next_provider_event_date=next_date,
                candidate_interval_start_inclusive=lower, candidate_interval_end_exclusive=upper,
                candidate_interval_overlaps_requested_dates=lower < upper, current_ticker=alias == ticker,
                evidence="explicit_provider_ticker_change_same_composite_figi", provenance=item["provenance"],
                permanent_identity_match_verified=True, interval_continuity_verified=False,
                historical_tradability_verified=False, automatic_price_history_stitching_authorized=False,
                conflicting_event_dates=conflicting))
    selected_reference = []
    for row, provenance in all_reference:
        symbol, figi = row.get("ticker"), row.get("composite_figi")
        matched = set(figi_targets.get(figi, [])) | alias_targets.get(symbol, set())
        if symbol in tickers:
            matched.add(symbol)
        if matched:
            selected_reference.append(dict(qt200_tickers=sorted(matched), provider_record=row,
                provenance=provenance, current_snapshot_only=True,
                same_composite_figi_qt200_tickers=figi_targets.get(figi, []),
                historical_identity_from_symbol_match_verified=False))
    expected_alias_payload = dict(records=aliases,
        rule="Only explicit provider ticker_change events joined by exact current composite FIGI.",
        interval_semantics="Candidate date bounds, not verified continuous listing/tradability.",
        merger_spinoff_share_class_continuity_inferred=False,
        automatic_price_history_stitching_authorized=False, training_ready=False, PIT=False)
    checks = {"reference-records.jsonl": selected_reference, "ticker-event-captures.jsonl": selected_captures,
              "ticker-change-events.jsonl": event_rows, "historical-aliases.json": expected_alias_payload}
    for name, expected in checks.items():
        if name not in support_states:
            _fail("Missing identity selected support inventory: " + name)
        raw, state = _read(support, name)
        if state != support_states[name]:
            _fail("Identity selected support changed during replay")
        observed = _records(raw) if name.endswith(".jsonl") else _json(raw)
        if name == "ticker-change-events.jsonl":
            # The historical extractor iterated `for ticker in matched` where
            # matched was a set. Only contiguous target-block order within the
            # same original response is unspecified; response order, per-target
            # original event order, values and provenance remain exact.
            observed = _identity_event_position_inventory(observed)
            expected = _identity_event_position_inventory(expected)
        if _canonical(observed) != _canonical(expected):
            _fail("Identity selected support inventory/provenance differs: " + name)
    for directory, expected_states in ((root, states), (support, support_states)):
        for relative, expected in expected_states.items():
            target = directory / relative
            if target.resolve(strict=True) != target or _identity(target.lstat()) != expected:
                _fail("Identity reconciliation input changed during replay")
    result = dict(schema=SCHEMA, operation="identity_original_capture_provenance_reconciliation_only",
        source_bindings=source_bindings, support_input_bindings=bindings,
        support_completion_sha256=expected_support_completion_sha256, universe_sha256=expected_universe_sha256,
        entitlement_receipt_sha256=entitlement, ordered_tickers=tickers,
        original_reference_result_count=len(all_reference), original_reference_page_count=page_count,
        original_event_response_counts=event_counts, selected_reference_count=len(selected_reference),
        selected_event_capture_count=len(selected_captures), selected_ticker_change_count=len(event_rows),
        selected_alias_count=len(aliases), candidate_symbols=sorted(set(tickers) | set(alias_targets)),
        identity_original_provider_raw_bodies_reauthenticated=True, original_capture_self_receipts_reauthenticated=True,
        exact_selected_position_inventory_verified=True, original_distinct_positions_preserved=True,
        native_source_transactions_present=False, third_party_signature_verified=False,
        producer_execution_attested=False, all_support_original_provider_raw_bodies_reauthenticated=False,
        historical_identity_qualified=False, historical_availability_qualified=False,
        economic_accounting_qualified=False, point_in_time_qualified=False,
        native_security_ids_emitted=0, successor_joins_emitted=0,
        training_ready_for_adaptive_v5=False, provider_calls=0, source_writes=False)
    result["receipt_sha256"] = _sha(_canonical(result))
    if len(_canonical(result)) > MAX_REPORT_BYTES:
        _fail("Identity reconciliation output exceeds diagnostic byte bound")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-universe-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.absolute()
    parent = target.parent
    if (parent.resolve(strict=True) != parent or parent.stat().st_uid != os.getuid()
            or not parent.is_dir()):
        _fail("Output parent must be an owned existing nonsymlink directory")
    source = args.support_root.absolute()
    if target == source or source in target.parents:
        _fail("Diagnostic output must not modify the source bundle")
    result = build_qt200_identity_evidence_v1(
        support_root=source, expected_completion_sha256=args.expected_completion_sha256,
        expected_universe_sha256=args.expected_universe_sha256)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_canonical(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps(dict(schema=SCHEMA, receipt_sha256=result["receipt_sha256"],
                          training_ready_for_adaptive_v5=False), sort_keys=True))


if __name__ == "__main__":
    main()
