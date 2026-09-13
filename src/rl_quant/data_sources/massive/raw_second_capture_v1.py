"""Bounded REST-second acquisition using the existing header-only transport.

Plans precede credential loading. Raw HTTP evidence is never training readiness.
The second-input handoff is byte-preserving, not a resampler or feature builder.
No provider credential loader or remote credential transport is introduced.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_documented_alias_v1 as documented
from rl_quant.data_sources.massive import raw_second_evidence_v1 as evidence
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.data_sources.massive.raw_second_alias_v1 import SecondAliasRoute
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, SecondCaptureRef, SecondQuery, _safe_url, publish_second_capture,
)

SCHEMA = "rl-quant.raw-second-rest-capture-v1"
ALIAS_SCHEMA = "rl-quant.raw-second-rest-exact-day-alias-capture-v1"
MAX_QUERIES = 24
MAX_QUERY_SECONDS = 3600
MAX_BYTES = 128_000_000
MAX_PAGES = 8
AliasRoute = SecondAliasRoute | documented.SecondDocumentedAliasRoute


def _alias_route_dict(route: AliasRoute) -> dict:
    """Keep legacy payloads exact; only the opt-in route carries a discriminator."""
    if type(route) is SecondAliasRoute:
        return asdict(route)
    if type(route) is documented.SecondDocumentedAliasRoute:
        return route.to_dict()
    raise ValueError("Unsupported typed alias route")


def _alias_route_from_dict(value: dict) -> AliasRoute:
    if type(value) is not dict:
        raise ValueError("Alias route must be a typed serialized object")
    if "schema" in value:
        if value["schema"] != documented.SCHEMA:
            raise ValueError("Unsupported alias route schema")
        return documented.SecondDocumentedAliasRoute.from_dict(value)
    if set(value) != {"fixed_slot_ticker", "provider_ticker", "session_date", "identity", "event"}:
        raise ValueError("Legacy alias serialization fields differ")
    return SecondAliasRoute.from_dict(value)


def _documented_aliases(routes: tuple[AliasRoute, ...]) -> bool:
    """Validate exact types and identify the explicit documented-route variant."""
    if (not isinstance(routes, tuple)
            or any(type(route) not in (SecondAliasRoute, documented.SecondDocumentedAliasRoute)
                   for route in routes)):
        raise ValueError("Alias routes require exact supported types")
    return any(type(route) is documented.SecondDocumentedAliasRoute for route in routes)


def _documented_alias_fields(routes: tuple[AliasRoute, ...]) -> dict:
    return {"documented_alias_schema": documented.SCHEMA} if _documented_aliases(routes) else {}


def _documented_alias_implementation(routes: tuple[AliasRoute, ...]) -> dict:
    if not _documented_aliases(routes):
        return {}
    return {"documented_alias_implementation_sha256": transport.digest(
        transport.read_regular(Path(documented.__file__).resolve(), 16 * 1024**2))}


class SecondHTTPQuery(SecondQuery):
    @property
    def name(self) -> str:
        return f"second-{self.ticker}-{self.start_ms}-{self.end_ms}"

    def validate_url(self, url: str) -> str:
        _safe_url(url, self)
        return url

    def inspect_page(self, raw: bytes) -> dict:
        body = transport.parse_json(raw)
        rows = body.get("results", [])
        if (body.get("status") != "OK" or body.get("ticker") != self.ticker
                or body.get("adjusted") is not False or not isinstance(rows, list)
                or any(not isinstance(r, dict) for r in rows)
                or type(body.get("resultsCount")) is not int or body["resultsCount"] != len(rows)
                or not isinstance(body.get("request_id"), str) or not body["request_id"]
                or len(rows) > 50_000):
            raise transport.ResearchCaptureError("Wrong second response identity/population")
        next_url = body.get("next_url")
        if next_url is not None:
            self.validate_url(next_url)
        if next_url is None and (len(rows) >= 50_000 or body.get("queryCount", 0) >= 50_000):
            raise transport.ResearchCaptureError("Unresolved query-limit coverage")
        return dict(result_count=len(rows), provider_request_id=body["request_id"], next_url=next_url,
                    field_names=sorted({k for row in rows for k in row}),
                    point_in_time_qualified=False, training_ready=False)


def _queries(queries: tuple[SecondQuery, ...],
             alias_routes: tuple[AliasRoute, ...] = (), *, evidence_relocation=None) -> tuple[SecondHTTPQuery, ...]:
    if (not isinstance(queries, tuple) or not 1 <= len(queries) <= MAX_QUERIES
            or any(not isinstance(q, SecondQuery) for q in queries)):
        raise ValueError("Expected a bounded explicit second-query population")
    result = tuple(SecondHTTPQuery(**asdict(q)) for q in queries)
    _documented_aliases(alias_routes)
    if (not isinstance(alias_routes, tuple) or len(alias_routes) > MAX_QUERIES
            or len({(r.provider_ticker, r.session_date) for r in alias_routes}) != len(alias_routes)
            or len({(r.fixed_slot_ticker, r.session_date) for r in alias_routes}) != len(alias_routes)):
        raise ValueError("Alias routes must be unique typed exact-date evidence")
    used = set()
    if len({q.name for q in result}) != len(result):
        raise ValueError("Duplicate second query")
    for index, query in enumerate(result):
        day = datetime.fromtimestamp(query.start_ms / 1000, timezone.utc).astimezone(
            ZoneInfo("America/New_York")).date().isoformat()
        if any(query.ticker == route.fixed_slot_ticker and route.session_date == day
               for route in alias_routes):
            raise ValueError("Fixed-slot and alias captures cannot overlap on one routed date")
        if query.ticker not in SYMBOLS:
            matches = []
            for route_index, route in enumerate(alias_routes):
                if route.provider_ticker == query.ticker:
                    # A route for another day must not authorize this query.
                    if route.session_date == day:
                        route.validate(SecondQuery(**asdict(query)), evidence_relocation=evidence_relocation)
                        matches.append(route_index)
            if len(matches) != 1:
                raise ValueError("Second pilot requires fixed QT200 symbols or exact-date alias evidence")
            used.update(matches)
        if query.end_ms - query.start_ms >= MAX_QUERY_SECONDS * 1000:
            raise ValueError("Pilot queries are limited to one hour each")
        for earlier in result[:index]:
            if (query.ticker == earlier.ticker
                    and max(query.start_ms, earlier.start_ms) <= min(query.end_ms, earlier.end_ms)):
                raise ValueError("Overlapping source queries")
    if used != set(range(len(alias_routes))):
        raise ValueError("Unused or unsupported alias route")
    return result


def _fields(queries, alias_routes: tuple[AliasRoute, ...] = (), *, evidence_relocation=None) -> dict:
    body = dict(schema=(ALIAS_SCHEMA if alias_routes else SCHEMA) + "-plan",
        queries=[asdict(q) for q in _queries(queries, alias_routes, evidence_relocation=evidence_relocation)],
        maximum_raw_response_bytes=MAX_BYTES, maximum_page_bytes=transport.MAX_PAGE_BYTES,
        maximum_pages_per_query=MAX_PAGES, maximum_elapsed_seconds=transport.MAX_CAPTURE_SECONDS,
        retry_count=0, concurrent_requests=1, minimum_request_gap_seconds=0.3,
        market_fields=["open", "high", "low", "close", "volume"],
        training_ready=False, point_in_time_qualified=False, native_v5_qualified=False)
    if alias_routes:
        body.update(alias_routes=[_alias_route_dict(r) for r in alias_routes],
                    alias_route_schema="rl-quant.raw-second-exact-day-alias-v1",
                    continuous_identity_qualified=False, provider_tickers_rewritten=False)
    body.update(_documented_alias_fields(alias_routes))
    return body


def publish_second_capture_plan(*, root: Path, queries: tuple[SecondQuery, ...],
                                alias_routes: tuple[AliasRoute, ...] = ()) -> str:
    """Publish the bounded scope before an acquisition owner opens its key."""
    body = {**_fields(queries, alias_routes),
        "capture_implementation_sha256": transport.digest(Path(transport.__file__).read_bytes()),
        "second_implementation_sha256": transport.digest(Path(__file__).read_bytes())}
    if alias_routes:
        from rl_quant.data_sources.massive import raw_second_alias_v1 as aliases
        body["alias_implementation_sha256"] = transport.digest(Path(aliases.__file__).read_bytes())
    body.update(_documented_alias_implementation(alias_routes))
    root.mkdir(parents=True, exist_ok=False)
    return transport.write_once(root / "plan.json", transport.canonical(body))["sha256"]


def _plan(root: Path, expected_sha256: str, *, current: bool,
          evidence_relocation=None) -> tuple[SecondHTTPQuery, ...]:
    if evidence_relocation is not None and (current or type(evidence_relocation) is not evidence.EvidenceRelocation):
        raise ValueError("Evidence relocation is an explicit replay-only operation")
    raw = transport.read_regular(root / "plan.json", 1_048_576)
    body = transport.parse_json(raw)
    if transport.digest(raw) != expected_sha256:
        raise ValueError("Second capture plan changed")
    routes = tuple(_alias_route_from_dict(r) for r in body.get("alias_routes", []))
    queries = _queries(tuple(SecondQuery(**q) for q in body["queries"]), routes,
                       evidence_relocation=evidence_relocation)
    fields = _fields(queries, routes, evidence_relocation=evidence_relocation)
    implementation_keys = {"capture_implementation_sha256", "second_implementation_sha256"}
    if routes:
        implementation_keys.add("alias_implementation_sha256")
    documented_implementation = _documented_alias_implementation(routes)
    implementation_keys.update(documented_implementation)
    if (set(body) != set(fields) | implementation_keys
            or any(body[k] != v or type(body[k]) is not type(v) for k, v in fields.items())):
        raise ValueError("Second capture plan contract differs")
    if current and (body["second_implementation_sha256"] != transport.digest(Path(__file__).read_bytes())
                    or body["capture_implementation_sha256"] != transport.digest(Path(transport.__file__).read_bytes())):
        raise ValueError("Capture implementation changed after planning")
    if current and routes:
        from rl_quant.data_sources.massive import raw_second_alias_v1 as aliases
        if body["alias_implementation_sha256"] != transport.digest(Path(aliases.__file__).read_bytes()):
            raise ValueError("Alias implementation changed after planning")
    for key, expected in documented_implementation.items():
        value = body[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                or (current and value != expected)):
            raise ValueError("Documented alias implementation binding differs")
    return queries


def _capture_schema(root: Path) -> str:
    body = transport.parse_json(transport.read_regular(root / "plan.json", 1_048_576))
    return ALIAS_SCHEMA if body["schema"] == ALIAS_SCHEMA + "-plan" else SCHEMA


def _routing_metadata(root: Path) -> dict:
    body = transport.parse_json(transport.read_regular(root / "plan.json", 1_048_576))
    if body["schema"] == ALIAS_SCHEMA + "-plan":
        result = dict(alias_routes=body["alias_routes"], continuous_identity_qualified=False,
                      provider_tickers_rewritten=False)
        if "documented_alias_schema" in body:
            result["documented_alias_schema"] = body["documented_alias_schema"]
        return result
    return {}


def capture_seconds(*, root: Path, plan_sha256: str, api_key: str) -> dict:
    queries = _plan(root, plan_sha256, current=True)
    schema = _capture_schema(root)
    result = transport._capture_queries(root=root, plan_sha256=plan_sha256, api_key=api_key,
        queries=queries, schema=schema, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES)
    if schema == ALIAS_SCHEMA and _plan(root, plan_sha256, current=True) != queries:
        raise ValueError("Alias source route changed during acquisition")
    return result


def replay_second_capture(*, root: Path, plan_sha256: str, completion_sha256: str,
                          evidence_relocation: evidence.EvidenceRelocation | None = None) -> dict:
    relocated = None
    if evidence_relocation is not None:
        if type(evidence_relocation) is not evidence.EvidenceRelocation:
            raise ValueError("Expected an explicit typed evidence relocation")
        relocated = evidence_relocation.verify(capture_root=root, plan_sha256=plan_sha256,
                                               completion_sha256=completion_sha256)
    queries = _plan(root, plan_sha256, current=False, evidence_relocation=evidence_relocation)
    schema = _capture_schema(root)
    result = transport._replay_queries(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256,
        queries=queries, schema=schema, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES)
    if schema == ALIAS_SCHEMA and _plan(root, plan_sha256, current=False,
                                      evidence_relocation=evidence_relocation) != queries:
        raise ValueError("Alias source route changed during replay")
    if evidence_relocation is not None:
        if evidence_relocation.verify(capture_root=root, plan_sha256=plan_sha256,
                                      completion_sha256=completion_sha256) != relocated:
            raise ValueError("Relocated evidence changed while replaying")
        result["evidence_relocation"] = relocated
    return result


def _pages(root: Path, query: SecondHTTPQuery, entry: dict) -> tuple[CapturedSecondPage, ...]:
    """Call only after the complete transport chain has been hash-replayed."""
    pages = []
    base = root / query.name
    complete_raw = transport.read_regular(base / "COMPLETE.json", 1_048_576)
    if transport.digest(complete_raw) != entry["completion"]["sha256"]:
        raise ValueError("Query completion changed during handoff")
    complete = transport.parse_json(complete_raw)
    for index in range(entry["page_count"]):
        receipt_raw = transport.read_regular(base / f"page-{index:04d}.receipt.json", 1_048_576)
        if transport.digest(receipt_raw) != complete["pages"][index]["sha256"]:
            raise ValueError("Page receipt changed during handoff")
        receipt = transport.parse_json(receipt_raw)
        packed = transport.read_regular(base / f"page-{index:04d}.json.gz", transport.MAX_PAGE_BYTES + 1_048_576)
        if transport.digest(packed) != receipt["body"]["sha256"]:
            raise ValueError("Capture body changed during handoff")
        with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
            body = stream.read(transport.MAX_PAGE_BYTES + 1)
            if len(body) > transport.MAX_PAGE_BYTES or stream.read(1):
                raise ValueError("Expanded second page exceeds bound")
        if transport.digest(body) != receipt["raw_body_sha256"]:
            raise ValueError("Original second response hash differs")
        # Round up, never make receipt-time observations available early.
        pages.append(CapturedSecondPage(receipt["request_url"], (receipt["received_at_ns"] + 999_999) // 1_000_000, body))
    return tuple(pages)


def materialize_second_sources(*, root: Path, plan_sha256: str, completion_sha256: str, output: Path) -> dict:
    before = replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256)
    queries = _plan(root, plan_sha256, current=False)
    complete = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1_048_576))
    output.mkdir(parents=True, exist_ok=False)
    refs = []
    for query, entry in zip(queries, complete["queries"], strict=True):
        ref = publish_second_capture(output / query.name, SecondQuery(**asdict(query)), _pages(root, query, entry))
        ref.load()
        refs.append(asdict(ref))
    if replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256) != before:
        raise ValueError("Source capture changed during materialization")
    result = dict(schema=_capture_schema(root) + "-raw-sources", capture_sha256=completion_sha256,
        plan_sha256=plan_sha256, captures=refs, raw_values_transformed=False,
        historical_issue_identity_qualified=False, corporate_actions_qualified=False,
        point_in_time_qualified=False, training_ready=False, **_routing_metadata(root))
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result


def verify_second_sources(*, root: Path, plan_sha256: str, completion_sha256: str,
                          output: Path, output_sha256: str) -> dict:
    before = replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256)
    queries = _plan(root, plan_sha256, current=False)
    capture = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1_048_576))
    raw = transport.read_regular(output / "COMPLETE.json", 1_048_576)
    if transport.digest(raw) != output_sha256:
        raise ValueError("Raw-source completion changed")
    body = transport.parse_json(raw)
    expected = dict(schema=_capture_schema(root) + "-raw-sources", capture_sha256=completion_sha256,
        plan_sha256=plan_sha256, captures=body["captures"], raw_values_transformed=False,
        historical_issue_identity_qualified=False, corporate_actions_qualified=False,
        point_in_time_qualified=False, training_ready=False, **_routing_metadata(root))
    if transport.canonical(body) != transport.canonical(expected) or len(body["captures"]) != len(queries):
        raise ValueError("Raw-source completion claims differ")
    if {p.name for p in output.iterdir()} != {"COMPLETE.json", *(q.name for q in queries)}:
        raise ValueError("Raw-source file population differs")
    for query, entry, ref in zip(queries, capture["queries"], body["captures"], strict=True):
        if ref["path"] != str(output / query.name):
            raise ValueError("Raw-source reference escaped its population")
        actual_query, actual_pages = SecondCaptureRef(**ref).load()
        if asdict(actual_query) != asdict(query) or actual_pages != _pages(root, query, entry):
            raise ValueError("Raw inputs differ from captured response bytes")
    if replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256) != before:
        raise ValueError("Capture changed during verification")
    return dict(output_sha256=output_sha256, nonmaterializing=True, raw_response_identity_verified=True,
                training_ready=False)
