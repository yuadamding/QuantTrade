"""One-pass packed second capture through the unchanged header-only HTTP owner.

Direct mode has a distinct plan/receipt schema. Original response bytes are
framed before interpretation; neither loose originals nor expanded native JSON
are written. A consumed or partially committed root is never retried or repaired.
Filesystem checks supplement, not replace, the caller's project/fileset quota
admission. Completion remains nonauthorizing source-preservation evidence.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import gzip
import os
from pathlib import Path
import struct

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_alias_v1 as aliases
from rl_quant.data_sources.massive import raw_second_capture_v1 as capture
from rl_quant.data_sources.massive import raw_second_evidence_v1 as evidence
from rl_quant.datasets import massive_raw_second_packed_v1 as packed
from rl_quant.datasets import massive_raw_seconds_v1 as raw_seconds

SCHEMA = "rl-quant.raw-second-rest-packed-capture-v1"
MAX_URL_BYTES = 4096
MAX_QUERY_METADATA_BYTES = 2048
MAX_CENSUS_BYTES = 1024**2
MAX_ROOT_METADATA_BYTES = 2 * 1024**2
ALLOCATION_RESERVE_BYTES = 1024**2
STORAGE_RESERVE_BYTES = (packed.MAX_PACK_BYTES + packed.MAX_INDEX_BYTES
                         + MAX_ROOT_METADATA_BYTES + ALLOCATION_RESERVE_BYTES)
_FALSE_FLAGS = ("credential_values_recorded", "point_in_time_qualified", "native_v5_qualified",
                "training_ready", "profitability_authorized", "continuous_identity_qualified",
                "provider_tickers_rewritten", "raw_values_transformed")


def _implementations(alias_routes=()) -> dict:
    modules = {"capture": transport, "second": capture, "alias": aliases,
               "packed": packed, "raw_loader": raw_seconds, "evidence": evidence}
    result = {name + "_implementation_sha256": transport.digest(
        transport.read_regular(Path(module.__file__).resolve(), 16 * 1024**2))
        for name, module in modules.items()}
    result["direct_implementation_sha256"] = transport.digest(
        transport.read_regular(Path(__file__).resolve(), 16 * 1024**2))
    result.update(capture._documented_alias_implementation(alias_routes))
    return result


def _fields(queries, alias_routes=(), *, evidence_relocation=None) -> dict:
    queries = capture._queries(queries, alias_routes, evidence_relocation=evidence_relocation)
    if any(len(q.url.encode()) > MAX_URL_BYTES
           or len(transport.canonical(asdict(q))) > MAX_QUERY_METADATA_BYTES for q in queries):
        raise ValueError("Direct query metadata exceeds the declared bounded index format")
    # Every URL occurs in both native manifest and physical page descriptor.
    # Even a maximum population must fit the index before the first GET.
    index_reserve = (capture.MAX_QUERIES * (MAX_QUERY_METADATA_BYTES + 1024)
        + capture.MAX_QUERIES * capture.MAX_PAGES * (2 * MAX_URL_BYTES + 2048)
        + packed.MAX_RECEIPT_BYTES)
    if index_reserve > packed.MAX_INDEX_BYTES:
        raise ValueError("Direct index reservation cannot contain its bounded population")
    return dict(schema=SCHEMA + "-plan", queries=[asdict(q) for q in queries],
        alias_routes=[capture._alias_route_dict(route) for route in alias_routes],
        maximum_raw_response_bytes=capture.MAX_BYTES, maximum_page_bytes=transport.MAX_PAGE_BYTES,
        maximum_pages_per_query=capture.MAX_PAGES, maximum_elapsed_seconds=transport.MAX_CAPTURE_SECONDS,
        maximum_queries=capture.MAX_QUERIES, maximum_query_seconds=capture.MAX_QUERY_SECONDS,
        maximum_packed_bytes=packed.MAX_PACK_BYTES, maximum_index_bytes=packed.MAX_INDEX_BYTES,
        maximum_frame_header_bytes=packed.MAX_HEADER_BYTES, maximum_receipt_bytes=packed.MAX_RECEIPT_BYTES,
        maximum_request_url_bytes=MAX_URL_BYTES, maximum_query_metadata_bytes=MAX_QUERY_METADATA_BYTES,
        maximum_census_bytes=MAX_CENSUS_BYTES, maximum_root_metadata_bytes=MAX_ROOT_METADATA_BYTES,
        storage_reserve_bytes=STORAGE_RESERVE_BYTES, allocation_reserve_bytes=ALLOCATION_RESERVE_BYTES,
        maximum_page_working_memory_bytes=3 * transport.MAX_PAGE_BYTES + 1024**2,
        pre_request_reservation="full-permitted-page-gzip-frame-and-index-before-every-GET",
        retry_count=0, concurrent_requests=1, minimum_request_gap_seconds=0.3,
        market_fields=["open", "high", "low", "close", "volume"],
        continuous_identity_qualified=False, provider_tickers_rewritten=False,
        training_ready=False, point_in_time_qualified=False, native_v5_qualified=False,
        **capture._documented_alias_fields(alias_routes))


def publish_direct_second_capture_plan(*, root: Path, queries: tuple[raw_seconds.SecondQuery, ...],
                                       alias_routes: tuple[capture.AliasRoute, ...] = ()) -> str:
    """Publish all bounded query/implementation scope before credential loading."""
    root = packed._path(root)
    fields = _fields(queries, alias_routes)
    implementation = (_implementations(alias_routes) if capture._documented_aliases(alias_routes)
                      else _implementations())
    body = transport.canonical({**fields, **implementation})
    if len(body) > 1024**2:
        raise ValueError("Direct capture plan exceeds bound")
    root.mkdir(parents=True, exist_ok=False)
    return transport.write_once(root / "plan.json", body)["sha256"]


def _plan(root: Path, expected: str, *, current: bool, evidence_relocation=None) -> tuple[dict, tuple]:
    if evidence_relocation is not None and (current or type(evidence_relocation) is not evidence.EvidenceRelocation):
        raise ValueError("Evidence relocation is an explicit replay-only operation")
    root = packed._path(root)
    packed._digest(expected)
    body = transport.read_regular(root / "plan.json", 1024**2)
    if transport.digest(body) != expected:
        raise ValueError("Direct capture plan content changed")
    plan = transport.parse_json(body)
    routes = tuple(capture._alias_route_from_dict(row) for row in plan["alias_routes"])
    queries = capture._queries(tuple(raw_seconds.SecondQuery(**row) for row in plan["queries"]), routes,
                               evidence_relocation=evidence_relocation)
    fields = _fields(queries, routes, evidence_relocation=evidence_relocation)
    implementation = (_implementations(routes) if capture._documented_aliases(routes)
                      else _implementations())
    # Original plans predate this resolver. Read-only replay preserves all their
    # recorded hashes; new acquisition must bind the complete current surface.
    if not current and "evidence_implementation_sha256" not in plan:
        implementation.pop("evidence_implementation_sha256", None)
    if (set(plan) != set(fields) | set(implementation)
            or any(type(plan[k]) is not type(v) or plan[k] != v for k, v in fields.items())):
        raise ValueError("Direct plan contract differs")
    for key, expected_value in implementation.items():
        packed._digest(plan[key])
        if current and plan[key] != expected_value:
            raise ValueError("Direct implementation changed after planning")
    return plan, queries


def _provenance(plan: dict, plan_sha256: str, started_sha256: str) -> dict:
    return dict(capture_schema=SCHEMA, plan_sha256=plan_sha256, started_sha256=started_sha256,
        implementations={k: v for k, v in plan.items() if k.endswith("_implementation_sha256")},
        alias_routes_sha256=transport.digest(transport.canonical(plan["alias_routes"])),
        continuous_identity_qualified=False, provider_tickers_rewritten=False)


class _DirectSecondSink:
    """Private exact-type sink; only the existing transport owns HTTP and retry."""

    def __init__(self, root: Path, plan_sha256: str):
        self.root = packed._path(root)
        self.plan_sha256 = plan_sha256
        self.plan, self.queries = _plan(root, plan_sha256, current=True)
        self.writer = None
        self.http_attempts = 0
        self.query = None

    def validate_scope(self, root, plan_sha256, queries, schema, maximum_bytes, maximum_pages):
        if (root != self.root or plan_sha256 != self.plan_sha256 or queries != self.queries
                or any(type(q) is not capture.SecondHTTPQuery for q in queries)
                or schema != SCHEMA or maximum_bytes != capture.MAX_BYTES or maximum_pages != capture.MAX_PAGES
                or _plan(root, plan_sha256, current=True) != (self.plan, self.queries)):
            raise ValueError("Packed sink is not bound to this explicit second capture")

    def start(self):
        started = transport.read_regular(self.root / "STARTED.json", 1024**2)
        self.writer = packed.SecondPackWriter(self.root / "packed",
            provenance=_provenance(self.plan, self.plan_sha256, transport.digest(started)))

    def close(self):
        if self.writer is not None:
            self.writer.close()

    @property
    def completed_frames(self):
        if self.writer is None:
            return 0
        return sum(len(row["pages"]) for row in self.writer._captures) + len(self.writer._pages)

    def begin_query(self, query):
        self.query = query
        self.writer.begin_query(raw_seconds.SecondQuery(**asdict(query)))

    def before_request(self, url):
        """Reserve a maximum allowed body before GET, not after receiving it."""
        if len(url.encode()) > MAX_URL_BYTES:
            raise ValueError("Direct request URL exceeds reserved index metadata")
        worst_frame = 4 + packed.MAX_HEADER_BYTES + transport.MAX_PAGE_BYTES + 1024**2
        if (self.writer._raw_total + transport.MAX_PAGE_BYTES > capture.MAX_BYTES
                or self.writer._offset + worst_frame > packed.MAX_PACK_BYTES):
            raise ValueError("Full-page pre-GET raw/packed reservation exhausted")
        remaining = (packed.MAX_PACK_BYTES - self.writer._offset + packed.MAX_INDEX_BYTES
                     + MAX_ROOT_METADATA_BYTES + ALLOCATION_RESERVE_BYTES)
        space = os.statvfs(self.root)
        if space.f_bavail * space.f_frsize < remaining or space.f_favail < 8:
            raise ValueError("Insufficient direct packed file/index/failure reservation")

    def request_started(self):
        self.http_attempts += 1
        if self.http_attempts > capture.MAX_QUERIES * capture.MAX_PAGES:
            raise ValueError("Direct HTTP attempt budget exceeded")

    def preserve_page(self, query, raw, metadata, page_index, predecessor):
        zipped = gzip.compress(raw, compresslevel=9, mtime=0)
        page = dict(schema=SCHEMA + "-page", **metadata, query=asdict(query),
            page_index=page_index, request_index=self.http_attempts - 1,
            raw_body_sha256=transport.digest(raw), raw_body_bytes=len(raw),
            compressed_body=dict(bytes=len(zipped), sha256=transport.digest(zipped)),
            predecessor_page_sha256=predecessor, plan_sha256=self.plan_sha256,
            capture_time_is_historical_availability=False)
        receipt = transport.canonical(page)
        oversized_metadata = len(receipt) > packed.MAX_RECEIPT_BYTES
        if oversized_metadata:
            # An oversized provider header cannot prevent preserving its bounded
            # original body. This failure-only receipt is not replay-eligible.
            receipt = transport.canonical(dict(schema=SCHEMA + "-oversized-metadata",
                metadata_sha256=transport.digest(transport.canonical(metadata)),
                plan_sha256=self.plan_sha256, request_url=metadata["request_url"],
                requested_at_ns=metadata["requested_at_ns"], received_at_ns=metadata["received_at_ns"],
                http_status=metadata["http_status"], metadata_complete=False))
        captured = raw_seconds.CapturedSecondPage(metadata["request_url"],
            (metadata["received_at_ns"] + 999_999) // 1_000_000, raw)
        self.writer.append_page(captured, compressed_body=zipped, source_receipt=receipt)
        if oversized_metadata:
            raise ValueError("Original body retained; direct response metadata exceeds bound")
        return page, dict(bytes=len(receipt), sha256=transport.digest(receipt))

    def check_census(self, totals):
        if len(transport.canonical(totals)) > MAX_CENSUS_BYTES:
            raise ValueError("Original body retained; direct page census exceeds bound")

    def finish_query(self, complete):
        manifest = self.writer.finish_query()
        body = transport.canonical(complete)
        return dict(bytes=len(body), sha256=transport.digest(body), manifest_sha256=manifest)

    def complete(self, results, totals, bytes_used, deadline):
        if transport.time.monotonic() >= deadline:
            raise ValueError("Direct acquisition deadline exhausted before commit")
        if _plan(self.root, self.plan_sha256, current=True) != (self.plan, self.queries):
            raise ValueError("Direct plan/alias source changed during capture")
        verified = self.writer.finalize()
        started_raw = transport.read_regular(self.root / "STARTED.json", 1024**2)
        result = dict(schema=SCHEMA + "-complete", plan_sha256=self.plan_sha256,
            started_sha256=transport.digest(started_raw), queries=results, page_census=totals,
            query_count=len(results), page_count=len(totals), raw_response_bytes=bytes_used,
            http_attempts=self.http_attempts, packed_index_sha256=verified["index_sha256"],
            packed_sha256=verified["pack_sha256"], packed_bytes=verified["pack_bytes"],
            index_bytes=verified["index_bytes"], capture_complete=True,
            completed_at_ns=transport.time.time_ns(), **{key: False for key in _FALSE_FLAGS})
        # Full source replay must pass before the final acquisition authority is
        # created. An independently valid index alone is not capture completion.
        _verify(self.root, self.plan_sha256, result, committed=False)
        if transport.time.monotonic() >= deadline:
            raise ValueError("Direct acquisition deadline exhausted during final replay")
        if _plan(self.root, self.plan_sha256, current=True) != (self.plan, self.queries):
            raise ValueError("Direct implementation or alias source changed during final replay")
        completed = transport.time.time_ns()
        if completed < result["completed_at_ns"]:
            raise ValueError("Capture completion clock moved backwards during replay")
        result["completed_at_ns"] = completed
        body = transport.canonical(result)
        if len(body) > MAX_ROOT_METADATA_BYTES:
            raise ValueError("Direct acquisition terminal exceeds reserved metadata")
        transport.write_once(self.root / "COMPLETE.json", body)
        return result


def capture_direct_seconds(*, root: Path, plan_sha256: str, api_key: str) -> dict:
    """Acquire one new explicit batch, with no credential reader or loose copies."""
    sink = _DirectSecondSink(root, plan_sha256)
    return transport._capture_queries(root=root, plan_sha256=plan_sha256, api_key=api_key,
        queries=sink.queries, schema=SCHEMA, maximum_bytes=capture.MAX_BYTES,
        maximum_pages=capture.MAX_PAGES, _packed_sink=sink)


def _frame_receipts(root: Path, index: dict) -> list[list[tuple[bytes, dict]]]:
    result = []
    with os.fdopen(os.open(root / "packed" / packed.PACK_NAME, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = packed._snapshot(stream, packed.MAX_PACK_BYTES)
        for member in index["captures"]:
            receipts = []
            for page in member["pages"]:
                stream.seek(page["frame_offset"])
                if stream.read(4) != struct.pack(">I", page["header_bytes"]):
                    raise ValueError("Direct frame length differs")
                header_raw = stream.read(page["header_bytes"])
                if transport.digest(header_raw) != page["header_sha256"]:
                    raise ValueError("Direct frame header differs")
                header = transport.parse_json(header_raw)
                value = header["source_receipt_base64"]
                receipt = base64.b64decode(value, validate=True)
                if len(receipt) > packed.MAX_RECEIPT_BYTES:
                    raise ValueError("Direct framed receipt exceeds bound")
                parsed = transport.parse_json(receipt)
                if transport.canonical(parsed) != receipt:
                    raise ValueError("Direct receipt is not its canonical original")
                receipts.append((receipt, parsed))
            result.append(receipts)
        packed._unchanged(stream, before)
    return result


def _verify(root: Path, plan_sha256: str, complete: dict, *, committed: bool, evidence_relocation=None) -> dict:
    plan, queries = _plan(root, plan_sha256, current=False, evidence_relocation=evidence_relocation)
    expected_names = {"plan.json", "STARTED.json", "packed"} | ({"COMPLETE.json"} if committed else set())
    if {p.name for p in root.iterdir()} != expected_names:
        raise ValueError("Direct source contains missing/unexpected artifacts")
    started_raw = transport.read_regular(root / "STARTED.json", 1024**2)
    started = transport.parse_json(started_raw)
    if (set(started) != {"schema", "plan_sha256", "started_at_ns"}
            or started["schema"] != SCHEMA + "-started" or started["plan_sha256"] != plan_sha256
            or type(started["started_at_ns"]) is not int or started["started_at_ns"] <= 0):
        raise ValueError("Direct capture start identity differs")
    keys = {"schema", "plan_sha256", "started_sha256", "queries", "page_census", "query_count",
            "page_count", "raw_response_bytes", "http_attempts", "packed_index_sha256", "packed_sha256",
            "packed_bytes", "index_bytes", "capture_complete", "completed_at_ns", *_FALSE_FLAGS}
    if (set(complete) != keys or complete["schema"] != SCHEMA + "-complete"
            or complete["plan_sha256"] != plan_sha256
            or complete["started_sha256"] != transport.digest(started_raw)
            or complete["capture_complete"] is not True
            or any(complete[key] is not False for key in _FALSE_FLAGS)):
        raise ValueError("Direct terminal claims differ")
    for key in ("query_count", "page_count", "raw_response_bytes", "http_attempts", "packed_bytes", "index_bytes"):
        if type(complete[key]) is not int or complete[key] < 0:
            raise ValueError("Direct terminal count/byte field is not an integer")
    index_sha = complete["packed_index_sha256"]
    verified = packed.verify_packed_second_store(root=root / "packed", index_sha256=index_sha)
    index = packed._index(root / "packed", index_sha)
    if transport.canonical(index["provenance"]) != transport.canonical(
            _provenance(plan, plan_sha256, transport.digest(started_raw))):
        raise ValueError("Direct packed index provenance differs")
    if len(index["captures"]) != len(queries):
        raise ValueError("Direct packed query population differs")
    receipts = _frame_receipts(root, index)
    totals, entries, used, request_index = [], [], 0, 0
    last_received = started["started_at_ns"]
    for query, member, receipt_rows, ref in zip(queries, index["captures"], receipts, verified["captures"], strict=True):
        actual_query, pages = packed.PackedSecondCaptureRef.from_dict(ref).load()
        if asdict(actual_query) != asdict(query):
            raise ValueError("Direct source literal ticker/query changed")
        expected_url, seen, proofs, count = query.url, set(), [], 0
        for number, (page, descriptor, (receipt_raw, receipt)) in enumerate(zip(pages, member["pages"], receipt_rows, strict=True)):
            receipt_keys = {"schema", "request_url", "http_status", "requested_at_ns", "received_at_ns",
                "content_type", "etag", "last_modified", "page_index", "request_index", "raw_body_sha256",
                "raw_body_bytes", "compressed_body", "predecessor_page_sha256", "plan_sha256", "query",
                "capture_time_is_historical_availability"}
            if (set(receipt) != receipt_keys or receipt["schema"] != SCHEMA + "-page"
                    or expected_url is None or expected_url in seen or receipt["request_url"] != expected_url
                    or len(expected_url.encode()) > MAX_URL_BYTES or receipt["http_status"] != 200
                    or type(receipt["http_status"]) is not int
                    or transport.canonical(receipt["query"]) != transport.canonical(asdict(query))
                    or receipt["page_index"] != number
                    or receipt["request_index"] != request_index or receipt["plan_sha256"] != plan_sha256
                    or receipt["predecessor_page_sha256"] != (proofs[-1]["sha256"] if proofs else None)
                    or receipt["capture_time_is_historical_availability"] is not False):
                raise ValueError("Direct receipt URL/cursor/predecessor or query identity differs")
            if any(type(receipt[key]) is not int or receipt[key] < 0
                   for key in ("page_index", "request_index", "raw_body_bytes")):
                raise ValueError("Direct receipt count/index field is not an integer")
            query.validate_url(expected_url)
            seen.add(expected_url)
            requested, received = receipt["requested_at_ns"], receipt["received_at_ns"]
            if (type(requested) is not int or type(received) is not int
                    or not last_received <= requested <= received
                    or page.received_at_ms != (received + 999_999) // 1_000_000):
                raise ValueError("Direct capture receipt chronology differs")
            last_received = received
            if (receipt["raw_body_sha256"] != transport.digest(page.body)
                    or receipt["raw_body_bytes"] != len(page.body)
                    or transport.canonical(receipt["compressed_body"]) != transport.canonical(
                        dict(bytes=descriptor["gzip_bytes"], sha256=descriptor["gzip_sha256"]))):
                raise ValueError("Direct original body/frame identity differs")
            if used + transport.MAX_PAGE_BYTES > capture.MAX_BYTES:
                raise ValueError("Direct history violates pre-GET full-page reservation")
            worst_frame = 4 + packed.MAX_HEADER_BYTES + transport.MAX_PAGE_BYTES + 1024**2
            if descriptor["frame_offset"] + worst_frame > packed.MAX_PACK_BYTES:
                raise ValueError("Direct history violates pre-GET packed reservation")
            used += len(page.body)
            request_index += 1
            summary = query.inspect_page(page.body)
            totals.append(dict(query=query.name, page=number, **summary))
            count += summary["result_count"]
            expected_url = summary["next_url"]
            proofs.append(dict(bytes=len(receipt_raw), sha256=transport.digest(receipt_raw)))
        if expected_url is not None:
            raise ValueError("Direct pagination did not terminate")
        query_complete = dict(schema=SCHEMA + "-query-complete", query=asdict(query),
            plan_sha256=plan_sha256, pages=proofs, page_count=len(pages), pagination_complete=True,
            result_count=count, empty_provider_response_proves_no_trading=False,
            point_in_time_qualified=False, training_ready=False)
        body = transport.canonical(query_complete)
        entries.append(dict(query=query.name, completion=dict(bytes=len(body), sha256=transport.digest(body),
            manifest_sha256=member["manifest_sha256"]), page_count=len(pages), result_count=count))
    if (len(transport.canonical(totals)) > MAX_CENSUS_BYTES
            or transport.canonical(complete["queries"]) != transport.canonical(entries)
            or transport.canonical(complete["page_census"]) != transport.canonical(totals)
            or complete["query_count"] != len(queries)
            or complete["page_count"] != request_index or complete["http_attempts"] != request_index
            or complete["raw_response_bytes"] != used or complete["packed_bytes"] != verified["pack_bytes"]
            or complete["index_bytes"] != verified["index_bytes"] or complete["packed_sha256"] != verified["pack_sha256"]
            or type(complete["completed_at_ns"]) is not int or complete["completed_at_ns"] < last_received):
        raise ValueError("Direct census/count/terminal chronology differs")
    if (_plan(root, plan_sha256, current=False, evidence_relocation=evidence_relocation) != (plan, queries)
            or packed._index(root / "packed", index_sha) != index):
        raise ValueError("Direct plan/index changed while replaying")
    return dict(query_count=len(queries), page_count=request_index, raw_response_bytes=used,
        captures=verified["captures"], packed_index_sha256=index_sha, packed_sha256=verified["pack_sha256"],
        packed_bytes=verified["pack_bytes"], index_bytes=verified["index_bytes"],
        source_provenance=index["provenance"], read_only_replay_verified=True,
        point_in_time_qualified=False, training_ready=False, native_v5_qualified=False,
        continuous_identity_qualified=False, provider_tickers_rewritten=False)


def replay_direct_second_capture(*, root: Path, plan_sha256: str, completion_sha256: str,
                                  evidence_relocation: evidence.EvidenceRelocation | None = None) -> dict:
    """Reconstruct every original response/receipt; an index alone is insufficient."""
    root = packed._path(root)
    relocated = None
    if evidence_relocation is not None:
        if type(evidence_relocation) is not evidence.EvidenceRelocation:
            raise ValueError("Expected an explicit typed evidence relocation")
        relocated = evidence_relocation.verify(capture_root=root, plan_sha256=plan_sha256,
                                               completion_sha256=completion_sha256)
    body = transport.read_regular(root / "COMPLETE.json", MAX_ROOT_METADATA_BYTES)
    if transport.digest(body) != completion_sha256:
        raise ValueError("Direct capture completion content changed")
    complete = transport.parse_json(body)
    result = _verify(root, plan_sha256, complete, committed=True, evidence_relocation=evidence_relocation)
    if transport.read_regular(root / "COMPLETE.json", MAX_ROOT_METADATA_BYTES) != body:
        raise ValueError("Direct capture completion changed during replay")
    if evidence_relocation is not None:
        if evidence_relocation.verify(capture_root=root, plan_sha256=plan_sha256,
                                      completion_sha256=completion_sha256) != relocated:
            raise ValueError("Relocated evidence changed while replaying")
        result["evidence_relocation"] = relocated
    return dict(capture_sha256=completion_sha256, plan_sha256=plan_sha256, **result)
