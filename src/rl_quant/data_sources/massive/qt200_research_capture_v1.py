"""Bounded, immutable REST observations for a separate QT200 research pilot.

This module neither replays corrections nor publishes native V5 authorities.
Retrieval times are observations made now, not historical availability claims.
The transport is fixed HTTPS with header authentication and no redirects.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as day_time, timedelta
from io import BytesIO
from pathlib import Path
from urllib import error, parse, request
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive.qt200_capture_pipeline_v1 import RequestPacer

SCHEMA = "rl-quant.qt200-research-rest-capture-v1"
MAX_PAGE_BYTES = 32 * 1024 * 1024
MAX_CAPTURE_BYTES = 1_000_000_000
MAX_PAGES_PER_QUERY = 128
MAX_CAPTURE_SECONDS = 1800
HOST = "api.massive.com"
ET = ZoneInfo("America/New_York")


class ResearchCaptureError(ValueError):
    """A bounded capture cannot be accepted; the original evidence is retained."""


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ResearchCaptureError("Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ResearchCaptureError("Nonfinite JSON number")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ResearchCaptureError("Invalid JSON response") from exc
    if not isinstance(value, dict):
        raise ResearchCaptureError("Expected a JSON object")
    return value


def read_regular(path: Path, cap: int) -> bytes:
    if not path.is_absolute() or path.resolve() != path:
        raise ResearchCaptureError("Noncanonical input path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or before.st_size > cap):
            raise ResearchCaptureError("Unsafe input metadata")
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_ctime_ns", "st_size", "st_mtime_ns")
        if (len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k)
                                            for k in fields)):
            raise ResearchCaptureError("Input changed during read")
        return raw


def write_once(path: Path, raw: bytes) -> dict:
    """Reserve a new file; interruption leaves evidence, never overwrite it."""
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise ResearchCaptureError("Noncanonical publication path")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return {"path": path.name, "bytes": len(raw), "sha256": digest(raw)}


@dataclass(frozen=True)
class PilotQuery:
    product: str
    ticker: str
    start: str
    end: str

    def validate(self) -> None:
        if (self.product not in {"trades", "day", "minute"}
                or self.ticker not in {"ABBV", "WMT", "AAPL"}):
            raise ResearchCaptureError("Query outside frozen pilot population")
        if self.start != "2017-01-03" or self.end != (
            "2017-12-29" if self.product == "day" else "2017-01-03"
        ):
            raise ResearchCaptureError("Query outside frozen pilot interval")

    def validate_url(self, url: str) -> str:
        return validate_url(url, self)

    def inspect_page(self, raw: bytes) -> dict:
        return inspect_page(raw, self)

    @property
    def name(self) -> str:
        self.validate()
        return f"{self.product}-{self.ticker}"

    @property
    def url(self) -> str:
        self.validate()
        if self.product == "trades":
            path = f"/v3/trades/{self.ticker}"
            query = {"timestamp": self.start, "sort": "timestamp",
                     "order": "asc", "limit": "50000"}
        else:
            path = (f"/v2/aggs/ticker/{self.ticker}/range/1/"
                    f"{self.product}/{self.start}/{self.end}")
            query = {"adjusted": "false", "sort": "asc", "limit": "50000"}
        return f"https://{HOST}{path}?{parse.urlencode(sorted(query.items()))}"


def pilot_queries() -> tuple[PilotQuery, ...]:
    # Small aggregate captures first; full trade pagination remains mandatory.
    return tuple(PilotQuery(product, ticker, "2017-01-03",
                            "2017-12-29" if product == "day" else "2017-01-03")
                 for product in ("day", "minute", "trades")
                 for ticker in ("ABBV", "WMT", "AAPL"))


def validate_url(url: str, query: PilotQuery) -> str:
    query.validate()
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ResearchCaptureError("Noncanonical request URL")
    parts = parse.urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != HOST or parts.fragment
            or parts.username is not None or parts.password is not None):
        raise ResearchCaptureError("Request outside fixed HTTPS origin")
    pairs = parse.parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
    fields = dict(pairs)
    if len(fields) != len(pairs) or any(not v for _, v in pairs):
        raise ResearchCaptureError("Ambiguous request parameters")
    if query.product == "trades":
        if parts.path != f"/v3/trades/{query.ticker}":
            raise ResearchCaptureError("Trade endpoint changed")
        allowed = {"cursor", "timestamp", "sort", "order", "limit"}
        expected = {"timestamp": query.start, "sort": "timestamp", "order": "asc"}
    else:
        prefix = f"/v2/aggs/ticker/{query.ticker}/range/1/{query.product}/"
        if not parts.path.startswith(prefix):
            raise ResearchCaptureError("Aggregate endpoint changed")
        bounds = parts.path[len(prefix):].split("/")
        if len(bounds) != 2 or bounds[1] != query.end:
            raise ResearchCaptureError("Aggregate end date changed")
        # The documented aggregate cursor may advance the path's start to ms.
        if bounds[0] != query.start:
            if not bounds[0].isascii() or not bounds[0].isdigit() or len(bounds[0]) != 13:
                raise ResearchCaptureError("Aggregate cursor start is invalid")
            start_ms = int(datetime.combine(date.fromisoformat(query.start), day_time(), ET).timestamp() * 1000)
            end_ms = int(datetime.combine(date.fromisoformat(query.end) + timedelta(days=1), day_time(), ET).timestamp() * 1000)
            if not start_ms <= int(bounds[0]) < end_ms:
                raise ResearchCaptureError("Aggregate cursor escaped requested dates")
        allowed = {"cursor", "adjusted", "sort", "limit"}
        expected = {"adjusted": "false", "sort": "asc"}
    if not set(fields) <= allowed:
        raise ResearchCaptureError("Unapproved or secret-bearing URL parameter")
    if any(key in fields and fields[key] != value for key, value in expected.items()):
        raise ResearchCaptureError("Provider cursor changed query semantics")
    if "limit" in fields and fields["limit"] != "50000":
        raise ResearchCaptureError("Provider cursor changed page bound")
    if "cursor" not in fields and url != query.url:
        raise ResearchCaptureError("Noninitial URL requires a provider cursor")
    return url


def inspect_page(raw: bytes, query: PilotQuery) -> dict:
    """Structural census only: never correct, deduplicate or promote records."""
    body = parse_json(raw)
    if body.get("status") != "OK" or not isinstance(body.get("request_id"), str) or not body["request_id"]:
        raise ResearchCaptureError("Provider response is not a successful identified page")
    rows = body.get("results", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ResearchCaptureError("Provider results shape changed")
    if len(rows) > 50000:
        raise ResearchCaptureError("Provider result count exceeds request bound")
    if query.product != "trades":
        if body.get("ticker") != query.ticker or body.get("adjusted") is not False:
            raise ResearchCaptureError("Ticker or unadjusted aggregate contract differs")
        if type(body.get("resultsCount")) is not int or body["resultsCount"] != len(rows):
            raise ResearchCaptureError("Aggregate row count differs")
    next_url = body.get("next_url")
    if next_url is not None:
        validate_url(next_url, query)
    summary = {"result_count": len(rows), "provider_request_id": body["request_id"],
               "next_url": next_url, "field_names": sorted({k for row in rows for k in row}),
               "records_deduplicated": 0, "correction_links_established": 0,
               "point_in_time_qualified": False, "training_ready": False}
    if query.product == "trades":
        codes = {}
        for row in rows:
            key = json.dumps(row.get("correction"), ensure_ascii=True, sort_keys=True)
            codes[key] = codes.get(key, 0) + 1
        summary.update(blank_or_absent_ids=sum(row.get("id") in (None, "") for row in rows),
                       zero_size_records=sum(row.get("size") == 0 for row in rows),
                       raw_correction_counts=codes)
    else:
        summary.update(vwap_present_records=sum("vw" in row for row in rows),
                       transaction_count_present_records=sum("n" in row for row in rows))
    return summary


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ResearchCaptureError("Provider redirect rejected")


def _fetch(url: str, query: PilotQuery, api_key: str) -> tuple[bytes, dict]:
    query.validate_url(url)
    started = time.time_ns()
    req = request.Request(url, method="GET", headers={
        "Authorization": "Bearer " + api_key, "Accept": "application/json",
        "User-Agent": "QuantTrade-QT200-Research-Capture-V1"})
    try:
        response = request.build_opener(_NoRedirect()).open(req, timeout=45)
    except error.HTTPError as exc:
        response = exc
    with response:
        if response.geturl() != url:
            raise ResearchCaptureError("Provider response origin changed")
        raw = response.read(MAX_PAGE_BYTES + 1)
        metadata = {"request_url": url, "http_status": response.code,
                    "requested_at_ns": started, "received_at_ns": time.time_ns(),
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified")}
    if len(raw) > MAX_PAGE_BYTES:
        raise ResearchCaptureError("Provider response exceeds byte bound")
    if api_key.encode() in raw or api_key.encode() in canonical(metadata):
        raise ResearchCaptureError("Provider response contains sensitive material")
    return raw, metadata


def capture_pilot(*, root: Path, api_key: str, plan_sha256: str) -> dict:
    """Acquire the frozen nine-query pilot; failures retain a BLOCKED receipt.

    The caller must have published plan.json before loading its credential.
    There is no injected transport or qualification/authorization override.
    """
    return _capture_queries(root=root, api_key=api_key, plan_sha256=plan_sha256,
                            queries=pilot_queries(), schema=SCHEMA,
                            maximum_bytes=MAX_CAPTURE_BYTES,
                            maximum_pages=MAX_PAGES_PER_QUERY)


def _capture_queries(*, root: Path, api_key: str, plan_sha256: str,
                     queries: tuple[PilotQuery, ...], schema: str,
                     maximum_bytes: int, maximum_pages: int,
                     request_pacer: RequestPacer | None = None) -> dict:
    """Shared transport implementation; public entry points freeze their scope."""
    if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
        raise ResearchCaptureError("Malformed acquisition credential")
    if request_pacer is not None and type(request_pacer) is not RequestPacer:
        raise ResearchCaptureError("Request pacer must be the package-owned implementation")
    plan_raw = read_regular(root / "plan.json", 1024 * 1024)
    plan = parse_json(plan_raw)
    if (digest(plan_raw) != plan_sha256 or plan.get("schema") != schema + "-plan"
            or plan.get("queries") != [asdict(q) for q in queries]
            or plan.get("maximum_raw_response_bytes") != maximum_bytes
            or plan.get("maximum_page_bytes") != MAX_PAGE_BYTES
            or plan.get("maximum_pages_per_query") != maximum_pages
            or plan.get("maximum_elapsed_seconds") != MAX_CAPTURE_SECONDS
            or plan.get("capture_implementation_sha256") != digest(Path(__file__).read_bytes())
            or plan.get("retry_count") != 0 or plan.get("training_ready") is not False):
        raise ResearchCaptureError("Frozen acquisition plan differs")
    if set(p.name for p in root.iterdir()) != {"plan.json"}:
        raise ResearchCaptureError("Capture root already consumed; do not overwrite or retry")
    write_once(root / "STARTED.json", canonical({"schema": schema + "-started",
               "plan_sha256": plan_sha256, "started_at_ns": time.time_ns()}))
    totals, results, bytes_used = [], [], 0
    deadline = time.monotonic() + MAX_CAPTURE_SECONDS
    try:
        for query in queries:
            query_root = root / query.name
            query_root.mkdir(mode=0o700)
            url, visited, pages = query.url, set(), []
            while url is not None:
                if time.monotonic() >= deadline:
                    raise ResearchCaptureError("Acquisition deadline exhausted")
                query.validate_url(url)
                if url in visited or len(pages) >= maximum_pages:
                    raise ResearchCaptureError("Pagination loop or page budget exhausted")
                if shutil.disk_usage(root).free < 2 * MAX_PAGE_BYTES + 1024 * 1024:
                    raise ResearchCaptureError("Insufficient capture filesystem space")
                visited.add(url)
                if request_pacer is not None:
                    request_pacer.acquire(deadline=deadline)
                raw, metadata = _fetch(url, query, api_key)
                bytes_used += len(raw)
                if bytes_used > maximum_bytes:
                    raise ResearchCaptureError("Capture byte budget exhausted")
                # Preserve even a provider error response, but never mark it complete.
                body_proof = write_once(query_root / f"page-{len(pages):04d}.json.gz",
                                        gzip.compress(raw, compresslevel=9, mtime=0))
                page = {"schema": schema + "-page", **metadata,
                        "page_index": len(pages), "raw_body_sha256": digest(raw),
                        "raw_body_bytes": len(raw), "body": body_proof,
                        "predecessor_page_sha256": pages[-1]["sha256"] if pages else None,
                        "plan_sha256": plan_sha256, "query": asdict(query),
                        "capture_time_is_historical_availability": False}
                proof = write_once(query_root / f"page-{len(pages):04d}.receipt.json", canonical(page))
                if metadata["http_status"] != 200:
                    raise ResearchCaptureError("Provider request failed; no retry was issued")
                summary = query.inspect_page(raw)
                pages.append(proof)
                totals.append({"query": query.name, "page": page["page_index"], **summary})
                url = summary["next_url"]
                time.sleep(0.3)
            complete = {"schema": schema + "-query-complete", "query": asdict(query),
                        "plan_sha256": plan_sha256, "pages": pages,
                        "page_count": len(pages), "pagination_complete": True,
                        "result_count": sum(row["result_count"] for row in totals if row["query"] == query.name),
                        "empty_provider_response_proves_no_trading": False,
                        "point_in_time_qualified": False, "training_ready": False}
            proof = write_once(query_root / "COMPLETE.json", canonical(complete))
            results.append({"query": query.name, "completion": proof,
                            "page_count": len(pages), "result_count": complete["result_count"]})
            print(json.dumps({"query_complete": query.name, "pages": len(pages),
                              "records": complete["result_count"]}), flush=True)
        census = write_once(root / "page-census.json", canonical(totals))
        result = {"schema": schema + "-complete", "plan_sha256": plan_sha256,
                  "queries": results, "query_count": len(results), "page_census": census,
                  "raw_response_bytes": bytes_used, "completed_at_ns": time.time_ns(),
                  "capture_complete": True, "credential_values_recorded": False,
                  "point_in_time_qualified": False, "native_v5_qualified": False,
                  "training_ready": False, "profitability_authorized": False}
        write_once(root / "COMPLETE.json", canonical(result))
        return result
    except Exception as exc:
        # Never serialize exception text or request objects: they may hold auth.
        write_once(root / "BLOCKED.json", canonical({"schema": schema + "-blocked",
                   "plan_sha256": plan_sha256, "completed_queries": results,
                   "error_type": type(exc).__name__, "raw_response_bytes": bytes_used,
                   "capture_complete": False, "training_ready": False,
                   "failed_at_ns": time.time_ns(), "retry_issued": False}))
        raise ResearchCaptureError("Capture incomplete; inspect secret-free retained receipts") from None


def replay_pilot(*, root: Path, plan_sha256: str, completion_sha256: str) -> dict:
    """Reopen every committed raw page without writes or a network connection.

    This verifies retained acquisition content, not historical availability or
    the provider's economic interpretation. It cannot authorize native V5.
    """
    return _replay_queries(root=root, plan_sha256=plan_sha256,
                           completion_sha256=completion_sha256,
                           queries=pilot_queries(), schema=SCHEMA,
                           maximum_bytes=MAX_CAPTURE_BYTES,
                           maximum_pages=MAX_PAGES_PER_QUERY)


def _replay_queries(*, root: Path, plan_sha256: str, completion_sha256: str,
                    queries: tuple[PilotQuery, ...], schema: str,
                    maximum_bytes: int, maximum_pages: int) -> dict:
    plan_raw = read_regular(root / "plan.json", 1024 * 1024)
    if digest(plan_raw) != plan_sha256:
        raise ResearchCaptureError("Plan content changed")
    plan = parse_json(plan_raw)
    if (plan.get("schema") != schema + "-plan"
            or plan.get("queries") != [asdict(q) for q in queries]):
        raise ResearchCaptureError("Pilot plan population changed")
    root_complete_raw = read_regular(root / "COMPLETE.json", 1024 * 1024)
    if digest(root_complete_raw) != completion_sha256:
        raise ResearchCaptureError("Externally bound completion content changed")
    root_complete = parse_json(root_complete_raw)
    if (root_complete.get("schema") != schema + "-complete"
            or root_complete.get("plan_sha256") != plan_sha256
            or root_complete.get("query_count") != len(queries)
            or root_complete.get("capture_complete") is not True
            or any(root_complete.get(key) is not False for key in
                   ("point_in_time_qualified", "native_v5_qualified", "training_ready",
                    "profitability_authorized", "credential_values_recorded"))):
        raise ResearchCaptureError("Capture completion claims differ")
    expected_root = {"plan.json", "STARTED.json", "COMPLETE.json", "page-census.json"}
    expected_root.update(q.name for q in queries)
    if set(p.name for p in root.iterdir()) != expected_root:
        raise ResearchCaptureError("Capture inventory contains missing or unexpected entries")
    started = parse_json(read_regular(root / "STARTED.json", 1024 * 1024))
    if started.get("plan_sha256") != plan_sha256:
        raise ResearchCaptureError("Start receipt plan differs")

    def reopen(directory: Path, proof: dict, name: str, cap: int) -> bytes:
        if not isinstance(proof, dict) or proof.get("path") != name:
            raise ResearchCaptureError("Artifact path escaped exact inventory")
        value = read_regular(directory / name, cap)
        if proof.get("sha256") != digest(value) or proof.get("bytes") != len(value):
            raise ResearchCaptureError("Artifact content differs")
        return value

    entries = root_complete.get("queries")
    if not isinstance(entries, list) or len(entries) != len(queries):
        raise ResearchCaptureError("Completed query inventory differs")
    totals, bytes_used, last_received = [], 0, started.get("started_at_ns", 0)
    for query, entry in zip(queries, entries, strict=True):
        if entry.get("query") != query.name:
            raise ResearchCaptureError("Query ordering differs")
        query_root = root / query.name
        complete = parse_json(reopen(query_root, entry["completion"], "COMPLETE.json", 1024 * 1024))
        if (complete.get("query") != asdict(query) or complete.get("plan_sha256") != plan_sha256
                or complete.get("pagination_complete") is not True
                or complete.get("point_in_time_qualified") is not False
                or complete.get("training_ready") is not False):
            raise ResearchCaptureError("Query completion differs")
        pages = complete.get("pages")
        if not isinstance(pages, list) or not 1 <= len(pages) <= maximum_pages:
            raise ResearchCaptureError("Invalid page inventory")
        expected_url, seen, count = query.url, set(), 0
        expected_names = {"COMPLETE.json"}
        for index, proof in enumerate(pages):
            receipt_name = f"page-{index:04d}.receipt.json"
            body_name = f"page-{index:04d}.json.gz"
            expected_names.update((receipt_name, body_name))
            page = parse_json(reopen(query_root, proof, receipt_name, 1024 * 1024))
            if (expected_url is None or expected_url in seen
                    or page.get("request_url") != expected_url
                    or page.get("http_status") != 200 or page.get("query") != asdict(query)
                    or page.get("page_index") != index or page.get("plan_sha256") != plan_sha256
                    or page.get("predecessor_page_sha256") != (pages[index - 1]["sha256"] if index else None)
                    or page.get("capture_time_is_historical_availability") is not False):
                raise ResearchCaptureError("Page chain changed")
            query.validate_url(expected_url)
            seen.add(expected_url)
            requested, received = page.get("requested_at_ns"), page.get("received_at_ns")
            if (type(requested) is not int or type(received) is not int
                    or not last_received <= requested <= received):
                raise ResearchCaptureError("Capture chronology differs")
            last_received = received
            packed = reopen(query_root, page["body"], body_name, MAX_PAGE_BYTES + 1024 * 1024)
            with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
                raw = stream.read(MAX_PAGE_BYTES + 1)
                if len(raw) > MAX_PAGE_BYTES or stream.read(1):
                    raise ResearchCaptureError("Decompressed page exceeds bound")
            if digest(raw) != page.get("raw_body_sha256") or len(raw) != page.get("raw_body_bytes"):
                raise ResearchCaptureError("Original page content differs")
            bytes_used += len(raw)
            if bytes_used > maximum_bytes:
                raise ResearchCaptureError("Capture exceeds byte budget")
            summary = query.inspect_page(raw)
            totals.append({"query": query.name, "page": index, **summary})
            count += summary["result_count"]
            expected_url = summary["next_url"]
        if (expected_url is not None or complete.get("result_count") != count
                or entry.get("result_count") != count
                or complete.get("page_count") != len(pages) or entry.get("page_count") != len(pages)
                or set(p.name for p in query_root.iterdir()) != expected_names):
            raise ResearchCaptureError("Pagination or result inventory incomplete")
    census_raw = reopen(root, root_complete["page_census"], "page-census.json", 16 * 1024 * 1024)
    if (census_raw != canonical(totals) or root_complete.get("raw_response_bytes") != bytes_used
            or type(root_complete.get("completed_at_ns")) is not int
            or root_complete["completed_at_ns"] < last_received):
        raise ResearchCaptureError("Census or terminal chronology changed")
    return {"capture_sha256": digest(root_complete_raw), "query_count": len(entries),
            "page_count": len(totals), "raw_response_bytes": bytes_used,
            "read_only_replay_verified": True, "point_in_time_qualified": False,
            "native_v5_qualified": False, "training_ready": False}
