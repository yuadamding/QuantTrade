"""Current Massive stock-action REST capture with fixed V8 coverage queries."""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from io import BytesIO
from pathlib import Path
from urllib import error, parse, request

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)

MASSIVE_ECONOMIC_REST_BASE_V8 = "https://api.massive.com"
MASSIVE_ECONOMIC_REST_SURFACES_V8 = {
    "massive-dividends-v1": "/stocks/v1/dividends",
    "massive-splits-v1": "/stocks/v1/splits",
}
MASSIVE_ECONOMIC_REST_DATE_FIELDS_V8 = {
    "massive-dividends-v1": "ex_dividend_date",
    "massive-splits-v1": "execution_date",
}
MASSIVE_ECONOMIC_REST_SORTS_V8 = {
    "massive-dividends-v1": "ex_dividend_date.asc,ticker.asc",
    "massive-splits-v1": "execution_date.asc,ticker.asc",
}
MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA = "rl-quant.massive-economic-raw-rest-capture-v8"
MASSIVE_ECONOMIC_RAW_CAPTURE_V8_DATASET = "massive-economic-raw-rest-capture-v8"
MASSIVE_ECONOMIC_RAW_CAPTURE_V8_OBJECT_PREFIX = (
    "massive-profitability-p0/raw-economic-rest-v8/"
)
MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA,
        "base_url": MASSIVE_ECONOMIC_REST_BASE_V8,
        "surfaces": MASSIVE_ECONOMIC_REST_SURFACES_V8,
        "query": "all-market-inclusive-date-range-limit-5000-fixed-sort",
        "initial_cursor": "prohibited",
        "transport": "fixed-HTTPS-GET-bearer-header-no-redirect",
        "raw_body": "exact-base64-and-physical-sha256",
        "pagination": "complete-provider-next-url-chain",
        "generic_parse": "always-fixed-runtime-captured-false",
    }
)


class MassiveEconomicProviderCaptureV8Error(ValueError):
    """Current Massive response capture or its immutable bytes differ."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicProviderCaptureV8Error(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicProviderCaptureV8Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicProviderCaptureV8Error(f"{name} must be nonnegative")
    return value


def _canonical_date(name: str, value: object) -> str:
    raw = _text(name, value)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise MassiveEconomicProviderCaptureV8Error(
            f"{name} must be an ISO calendar date"
        ) from exc
    if parsed.isoformat() != raw:
        raise MassiveEconomicProviderCaptureV8Error(f"{name} is not canonical")
    return raw


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicProviderCaptureV8Error(f"{name} fields differ")


def _request_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise MassiveEconomicProviderCaptureV8Error(
            "provider request ID must be text or integer"
        )
    return _text("provider request ID", str(value))


def _safe_massive_url(value: object, *, expected_path: str) -> str:
    url = _text("Massive request URL", value)
    parsed = parse.urlsplit(url)
    query_keys = tuple(key.lower() for key, _ in parse.parse_qsl(parsed.query))
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.massive.com"
        or parsed.path != expected_path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "apikey" in query_keys
    ):
        raise MassiveEconomicProviderCaptureV8Error(
            "Massive request URL is outside the fixed secret-free surface"
        )
    return url


def _json_without_duplicate_keys(raw: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MassiveEconomicProviderCaptureV8Error(
                    "raw provider response contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicProviderCaptureV8Error(
            "raw provider response is not JSON"
        ) from exc


def build_massive_economic_query_parameters_v8(
    *, surface_id: str, coverage_start_date: str, coverage_end_date: str
) -> tuple[tuple[str, str], ...]:
    """Return the only initial query allowed for one V8 all-market capture."""

    if surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V8:
        raise MassiveEconomicProviderCaptureV8Error("V8 REST surface is unsupported")
    start = _canonical_date("economic coverage start", coverage_start_date)
    end = _canonical_date("economic coverage end", coverage_end_date)
    if end < start:
        raise MassiveEconomicProviderCaptureV8Error(
            "economic coverage date interval is inverted"
        )
    date_field = MASSIVE_ECONOMIC_REST_DATE_FIELDS_V8[surface_id]
    return tuple(
        sorted(
            (
                (f"{date_field}.gte", start),
                (f"{date_field}.lte", end),
                ("limit", "5000"),
                ("sort", MASSIVE_ECONOMIC_REST_SORTS_V8[surface_id]),
            )
        )
    )


class _RejectAllRedirectsV8(request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        raise MassiveEconomicProviderCaptureV8Error(
            "Massive economic capture rejects every HTTP redirect"
        )


@dataclass(frozen=True, slots=True)
class MassiveEconomicRawRestPageV8:
    page_index: int
    request_url: str
    final_response_url: str
    http_method: str
    http_status: int
    requested_at_ms: int
    completed_at_ms: int
    provider_request_id: str
    response_content_type: str
    provider_etag: str | None
    provider_last_modified: str | None
    raw_response_body_base64: str
    raw_response_body_sha256: str
    raw_response_content_length: int
    next_url: str | None
    result_count: int
    result_inventory_sha256: str
    receipt_sha256: str

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def raw_body(self) -> bytes:
        try:
            return b64decode(self.raw_response_body_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise MassiveEconomicProviderCaptureV8Error(
                "raw provider response is not canonical base64"
            ) from exc

    def parsed_body(self) -> Mapping[str, object]:
        value = _json_without_duplicate_keys(self.raw_body())
        if not isinstance(value, dict):
            raise MassiveEconomicProviderCaptureV8Error(
                "raw provider response root is not an object"
            )
        return value

    def validate(self, *, page_count: int, expected_path: str) -> None:
        if (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or not 0 <= self.page_index < page_count
        ):
            raise MassiveEconomicProviderCaptureV8Error("page index is invalid")
        requested_url = _safe_massive_url(self.request_url, expected_path=expected_path)
        final_url = _safe_massive_url(
            self.final_response_url, expected_path=expected_path
        )
        if final_url != requested_url:
            raise MassiveEconomicProviderCaptureV8Error(
                "Massive response URL differs after redirect handling"
            )
        if self.http_method != "GET" or self.http_status != 200:
            raise MassiveEconomicProviderCaptureV8Error(
                "raw provider response did not come from a successful fixed GET"
            )
        requested = _nonnegative_int("page request time", self.requested_at_ms)
        completed = _nonnegative_int("page completion time", self.completed_at_ms)
        if completed < requested:
            raise MassiveEconomicProviderCaptureV8Error(
                "provider page completion predates request"
            )
        _request_id(self.provider_request_id)
        _text("provider response content type", self.response_content_type)
        if "json" not in self.response_content_type.lower():
            raise MassiveEconomicProviderCaptureV8Error(
                "provider response content type is not JSON"
            )
        for name in ("provider_etag", "provider_last_modified"):
            value = getattr(self, name)
            if value is not None:
                _text(name, value)
        raw = self.raw_body()
        if (
            hashlib.sha256(raw).hexdigest()
            != _digest("raw response body", self.raw_response_body_sha256)
            or len(raw) != self.raw_response_content_length
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "raw provider response bytes differ"
            )
        body = self.parsed_body()
        if body.get("status") != "OK":
            raise MassiveEconomicProviderCaptureV8Error(
                "provider response status is not OK"
            )
        results = body.get("results")
        if not isinstance(results, list) or any(
            not isinstance(row, dict) for row in results
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "provider result inventory is malformed"
            )
        if self.result_count != len(
            results
        ) or self.result_inventory_sha256 != semantic_sha256(results):
            raise MassiveEconomicProviderCaptureV8Error(
                "provider result inventory differs"
            )
        if body.get("count") is not None and body.get("count") != len(results):
            raise MassiveEconomicProviderCaptureV8Error(
                "provider response count differs"
            )
        body_request_id = body.get("request_id")
        if (
            body_request_id is not None
            and _request_id(body_request_id) != self.provider_request_id
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "provider body/header request IDs differ"
            )
        body_next = body.get("next_url")
        if body_next is not None:
            body_next = _safe_massive_url(body_next, expected_path=expected_path)
        if body_next != self.next_url:
            raise MassiveEconomicProviderCaptureV8Error(
                "provider next URL differs from raw body"
            )
        if (self.next_url is None) is not (self.page_index == page_count - 1):
            raise MassiveEconomicProviderCaptureV8Error(
                "provider pagination did not close exactly"
            )
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicProviderCaptureV8Error("raw page receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveEconomicRawRestCaptureV8:
    surface_id: str
    provider_id: str
    provider_dataset: str
    endpoint_base: str
    coverage_start_date: str
    coverage_end_date: str
    initial_request_url: str
    query_parameters: tuple[tuple[str, str], ...]
    requested_at_ms: int
    completed_at_ms: int
    pages: tuple[MassiveEconomicRawRestPageV8, ...]
    page_count: int
    provider_request_ids: tuple[str, ...]
    raw_page_inventory_sha256: str
    pagination_complete: bool
    capture_kind: str
    adapter_source_sha256: str
    adapter_spec_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    fixed_runtime_captured: bool = False
    schema: str = MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"receipt_sha256", "fixed_runtime_captured"}
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA:
            raise MassiveEconomicProviderCaptureV8Error("capture schema differs")
        if self.surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V8:
            raise MassiveEconomicProviderCaptureV8Error(
                "current Massive REST surface is unsupported"
            )
        if (
            self.provider_id != "massive"
            or self.provider_dataset != self.surface_id
            or self.endpoint_base != MASSIVE_ECONOMIC_REST_BASE_V8
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "Massive provider identity differs"
            )
        start = _canonical_date("coverage start", self.coverage_start_date)
        end = _canonical_date("coverage end", self.coverage_end_date)
        if end < start:
            raise MassiveEconomicProviderCaptureV8Error(
                "capture coverage interval is inverted"
            )
        expected_parameters = build_massive_economic_query_parameters_v8(
            surface_id=self.surface_id,
            coverage_start_date=start,
            coverage_end_date=end,
        )
        if self.query_parameters != expected_parameters:
            raise MassiveEconomicProviderCaptureV8Error(
                "capture does not use the frozen all-market query"
            )
        expected_path = MASSIVE_ECONOMIC_REST_SURFACES_V8[self.surface_id]
        initial = _safe_massive_url(
            self.initial_request_url, expected_path=expected_path
        )
        if tuple(sorted(parse.parse_qsl(parse.urlsplit(initial).query))) != (
            expected_parameters
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "initial request query differs from frozen parameters"
            )
        initial_keys = {
            key.lower() for key, _ in parse.parse_qsl(parse.urlsplit(initial).query)
        }
        if "cursor" in initial_keys:
            raise MassiveEconomicProviderCaptureV8Error(
                "initial economic request cannot begin from a cursor"
            )
        requested = _nonnegative_int("capture request time", self.requested_at_ms)
        completed = _nonnegative_int("capture completion time", self.completed_at_ms)
        if completed < requested:
            raise MassiveEconomicProviderCaptureV8Error(
                "capture completion predates request"
            )
        if (
            not self.pagination_complete
            or self.page_count <= 0
            or len(self.pages) != self.page_count
            or len(self.provider_request_ids) != self.page_count
            or len(set(self.provider_request_ids)) != self.page_count
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "capture pagination inventory differs"
            )
        previous_completed = requested
        expected_url = initial
        for index, page in enumerate(self.pages):
            page.validate(page_count=self.page_count, expected_path=expected_path)
            if (
                page.page_index != index
                or page.provider_request_id != self.provider_request_ids[index]
                or page.request_url != expected_url
                or page.requested_at_ms < previous_completed
            ):
                raise MassiveEconomicProviderCaptureV8Error(
                    "capture page chronology or URL chain differs"
                )
            previous_completed = page.completed_at_ms
            if page.next_url is not None:
                expected_url = page.next_url
        if (
            self.requested_at_ms != self.pages[0].requested_at_ms
            or self.completed_at_ms != self.pages[-1].completed_at_ms
            or self.raw_page_inventory_sha256
            != semantic_sha256(tuple(page.receipt_sha256 for page in self.pages))
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "capture page inventory differs"
            )
        if self.capture_kind not in {
            "fixed-massive-rest-production-v8",
            "synthetic-test-response-v8",
        }:
            raise MassiveEconomicProviderCaptureV8Error("capture kind is invalid")
        if not isinstance(self.fixed_runtime_captured, bool):
            raise MassiveEconomicProviderCaptureV8Error(
                "fixed-runtime capture flag is not boolean"
            )
        if self.fixed_runtime_captured and self.capture_kind != (
            "fixed-massive-rest-production-v8"
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "synthetic capture cannot be fixed-runtime captured"
            )
        if (
            self.adapter_source_sha256 != MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256
            or self.adapter_spec_sha256 != MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "capture adapter implementation differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_RAW_CAPTURE_V8_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_ECONOMIC_RAW_CAPTURE_V8_OBJECT_PREFIX
            )
            or self.loaded_source.receipt.request_id != self.provider_request_ids[-1]
            or self.loaded_source.receipt.downloaded_at_ms != self.completed_at_ms
        ):
            raise MassiveEconomicProviderCaptureV8Error(
                "capture immutable source transaction differs"
            )
        if self.receipt_sha256 != semantic_sha256(self.semantic_unsigned()):
            raise MassiveEconomicProviderCaptureV8Error("raw capture receipt differs")


def _page_payload(page: MassiveEconomicRawRestPageV8) -> dict[str, object]:
    return asdict(page)


def _capture_payload_from_parts(
    *,
    surface_id: str,
    coverage_start_date: str,
    coverage_end_date: str,
    query_parameters: tuple[tuple[str, str], ...],
    pages: tuple[MassiveEconomicRawRestPageV8, ...],
    capture_kind: str,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA,
        "surface_id": surface_id,
        "provider_id": "massive",
        "provider_dataset": surface_id,
        "endpoint_base": MASSIVE_ECONOMIC_REST_BASE_V8,
        "coverage_start_date": coverage_start_date,
        "coverage_end_date": coverage_end_date,
        "initial_request_url": pages[0].request_url,
        "query_parameters": query_parameters,
        "requested_at_ms": pages[0].requested_at_ms,
        "completed_at_ms": pages[-1].completed_at_ms,
        "pages": tuple(_page_payload(page) for page in pages),
        "page_count": len(pages),
        "provider_request_ids": tuple(page.provider_request_id for page in pages),
        "raw_page_inventory_sha256": semantic_sha256(
            tuple(page.receipt_sha256 for page in pages)
        ),
        "pagination_complete": pages[-1].next_url is None,
        "capture_kind": capture_kind,
        "adapter_source_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256,
        "adapter_spec_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256,
    }


def parse_massive_economic_raw_rest_capture_v8(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicRawRestCaptureV8:
    """Reopen committed V8 bytes; generic parsing is always nonauthorizing."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicProviderCaptureV8Error(
            "raw capture source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicProviderCaptureV8Error(
            "raw capture source is not canonical JSON"
        )
    expected_fields = {
        "schema",
        "surface_id",
        "provider_id",
        "provider_dataset",
        "endpoint_base",
        "coverage_start_date",
        "coverage_end_date",
        "initial_request_url",
        "query_parameters",
        "requested_at_ms",
        "completed_at_ms",
        "pages",
        "page_count",
        "provider_request_ids",
        "raw_page_inventory_sha256",
        "pagination_complete",
        "capture_kind",
        "adapter_source_sha256",
        "adapter_spec_sha256",
    }
    _exact_keys(payload, expected_fields, name="raw capture source")
    raw_pages = payload["pages"]
    if not isinstance(raw_pages, list) or any(
        not isinstance(page, dict) for page in raw_pages
    ):
        raise MassiveEconomicProviderCaptureV8Error("capture pages are malformed")
    pages: list[MassiveEconomicRawRestPageV8] = []
    for page in raw_pages:
        _exact_keys(
            page,
            set(MassiveEconomicRawRestPageV8.__dataclass_fields__),
            name="raw capture page",
        )
        pages.append(MassiveEconomicRawRestPageV8(**page))
    raw_parameters = payload["query_parameters"]
    if not isinstance(raw_parameters, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in raw_parameters
    ):
        raise MassiveEconomicProviderCaptureV8Error(
            "capture query parameters are malformed"
        )
    raw_request_ids = payload["provider_request_ids"]
    if not isinstance(raw_request_ids, list):
        raise MassiveEconomicProviderCaptureV8Error("capture request IDs are malformed")
    provisional = MassiveEconomicRawRestCaptureV8(
        schema=payload["schema"],
        surface_id=payload["surface_id"],
        provider_id=payload["provider_id"],
        provider_dataset=payload["provider_dataset"],
        endpoint_base=payload["endpoint_base"],
        coverage_start_date=payload["coverage_start_date"],
        coverage_end_date=payload["coverage_end_date"],
        initial_request_url=payload["initial_request_url"],
        query_parameters=tuple((row[0], row[1]) for row in raw_parameters),
        requested_at_ms=payload["requested_at_ms"],
        completed_at_ms=payload["completed_at_ms"],
        pages=tuple(pages),
        page_count=payload["page_count"],
        provider_request_ids=tuple(raw_request_ids),
        raw_page_inventory_sha256=payload["raw_page_inventory_sha256"],
        pagination_complete=payload["pagination_complete"],
        capture_kind=payload["capture_kind"],
        adapter_source_sha256=payload["adapter_source_sha256"],
        adapter_spec_sha256=payload["adapter_spec_sha256"],
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
        fixed_runtime_captured=False,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _build_page(
    *,
    page_index: int,
    request_url: str,
    final_response_url: str,
    expected_path: str,
    requested_at_ms: int,
    completed_at_ms: int,
    provider_request_id: str,
    content_type: str,
    provider_etag: str | None,
    provider_last_modified: str | None,
    body: bytes,
) -> MassiveEconomicRawRestPageV8:
    parsed = _json_without_duplicate_keys(body)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise MassiveEconomicProviderCaptureV8Error(
            "Massive response result inventory is malformed"
        )
    results = parsed["results"]
    next_url = parsed.get("next_url")
    if next_url is not None:
        next_url = _safe_massive_url(next_url, expected_path=expected_path)
    provisional = MassiveEconomicRawRestPageV8(
        page_index=page_index,
        request_url=_safe_massive_url(request_url, expected_path=expected_path),
        final_response_url=_safe_massive_url(
            final_response_url, expected_path=expected_path
        ),
        http_method="GET",
        http_status=200,
        requested_at_ms=requested_at_ms,
        completed_at_ms=completed_at_ms,
        provider_request_id=_request_id(provider_request_id),
        response_content_type=_text("response content type", content_type),
        provider_etag=provider_etag,
        provider_last_modified=provider_last_modified,
        raw_response_body_base64=b64encode(body).decode("ascii"),
        raw_response_body_sha256=hashlib.sha256(body).hexdigest(),
        raw_response_content_length=len(body),
        next_url=next_url,
        result_count=len(results),
        result_inventory_sha256=semantic_sha256(results),
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    return result


def _publish_capture(
    *,
    root: str | Path,
    surface_id: str,
    coverage_start_date: str,
    coverage_end_date: str,
    query_parameters: tuple[tuple[str, str], ...],
    pages: tuple[MassiveEconomicRawRestPageV8, ...],
    capture_kind: str,
    entitlement_receipt_sha256: str,
    capture_id: str,
) -> MassiveEconomicRawRestCaptureV8:
    if not pages:
        raise MassiveEconomicProviderCaptureV8Error("capture contains no pages")
    payload = _capture_payload_from_parts(
        surface_id=surface_id,
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
        query_parameters=query_parameters,
        pages=pages,
        capture_kind=capture_kind,
    )
    relative = (
        f"{MASSIVE_ECONOMIC_RAW_CAPTURE_V8_OBJECT_PREFIX}"
        f"{surface_id}-{_text('capture ID', capture_id)}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_RAW_CAPTURE_V8_DATASET,
        source_object_key=relative,
        requested_at_ms=pages[0].requested_at_ms,
        downloaded_at_ms=pages[-1].completed_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=_digest(
            "entitlement receipt", entitlement_receipt_sha256
        ),
        committed_at_ms=pages[-1].completed_at_ms,
        request_id=pages[-1].provider_request_id,
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=pages[-1].completed_at_ms,
    )
    return parse_massive_economic_raw_rest_capture_v8(root=root, loaded_source=loaded)


def capture_massive_economic_rest_surface_v8(
    *,
    root: str | Path,
    surface_id: str,
    coverage_start_date: str,
    coverage_end_date: str,
    api_key: str,
    entitlement_receipt_sha256: str,
    capture_id: str,
    timeout_seconds: float = 30.0,
    response_limit_bytes: int = 32 * 1024 * 1024,
    maximum_pages: int = 10_000,
) -> MassiveEconomicRawRestCaptureV8:
    """Capture a complete current Massive surface with fixed query and clocks."""

    secret = _text("Massive API key", api_key)
    if any(character.isspace() for character in secret):
        raise MassiveEconomicProviderCaptureV8Error("Massive API key is malformed")
    if timeout_seconds <= 0 or response_limit_bytes <= 0 or maximum_pages <= 0:
        raise MassiveEconomicProviderCaptureV8Error(
            "capture transport limits must be positive"
        )
    params = build_massive_economic_query_parameters_v8(
        surface_id=surface_id,
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
    )
    expected_path = MASSIVE_ECONOMIC_REST_SURFACES_V8[surface_id]
    initial = (
        MASSIVE_ECONOMIC_REST_BASE_V8 + expected_path + "?" + parse.urlencode(params)
    )
    current_url: str | None = initial
    pages: list[MassiveEconomicRawRestPageV8] = []
    opener = request.build_opener(_RejectAllRedirectsV8())
    while current_url is not None:
        if len(pages) >= maximum_pages:
            raise MassiveEconomicProviderCaptureV8Error(
                "provider pagination exceeded cap"
            )
        current_url = _safe_massive_url(current_url, expected_path=expected_path)
        requested_at_ms = time.time_ns() // 1_000_000
        web_request = request.Request(
            current_url,
            headers={
                "Authorization": f"Bearer {secret}",
                "Accept": "application/json",
                "User-Agent": "QuantTrade-Massive-Economic-V8",
            },
            method="GET",
        )
        try:
            with opener.open(web_request, timeout=timeout_seconds) as response:
                status = int(response.status)
                final_url = str(response.geturl())
                body = response.read(response_limit_bytes + 1)
                content_type = response.headers.get("Content-Type", "")
                header_request_id = response.headers.get("X-Request-ID")
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
        except MassiveEconomicProviderCaptureV8Error:
            raise
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise MassiveEconomicProviderCaptureV8Error(
                "Massive economic REST request failed"
            ) from exc
        completed_at_ms = time.time_ns() // 1_000_000
        if status != 200 or len(body) > response_limit_bytes:
            raise MassiveEconomicProviderCaptureV8Error(
                "Massive response status or size is invalid"
            )
        body_value = _json_without_duplicate_keys(body)
        body_request_id = (
            body_value.get("request_id") if isinstance(body_value, dict) else None
        )
        selected_request_id = header_request_id or body_request_id
        page = _build_page(
            page_index=len(pages),
            request_url=current_url,
            final_response_url=final_url,
            expected_path=expected_path,
            requested_at_ms=requested_at_ms,
            completed_at_ms=completed_at_ms,
            provider_request_id=_request_id(selected_request_id),
            content_type=content_type,
            provider_etag=etag,
            provider_last_modified=last_modified,
            body=body,
        )
        pages.append(page)
        current_url = page.next_url
    parsed_capture = _publish_capture(
        root=root,
        surface_id=surface_id,
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
        query_parameters=params,
        pages=tuple(pages),
        capture_kind="fixed-massive-rest-production-v8",
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        capture_id=capture_id,
    )
    result = replace(parsed_capture, fixed_runtime_captured=True)
    result.validate()
    return result


def capture_massive_economic_rest_surface_for_test_v8(
    *,
    root: str | Path,
    surface_id: str,
    coverage_start_date: str,
    coverage_end_date: str,
    raw_page_bodies: Sequence[bytes],
    requested_at_ms: int,
    completed_at_ms: int,
    entitlement_receipt_sha256: str,
    capture_id: str,
) -> MassiveEconomicRawRestCaptureV8:
    """Publish deterministic response bytes; output is always nonauthorizing."""

    start = _nonnegative_int("test request time", requested_at_ms)
    finish = _nonnegative_int("test completion time", completed_at_ms)
    if finish < start or not raw_page_bodies:
        raise MassiveEconomicProviderCaptureV8Error("test capture chronology differs")
    params = build_massive_economic_query_parameters_v8(
        surface_id=surface_id,
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
    )
    expected_path = MASSIVE_ECONOMIC_REST_SURFACES_V8[surface_id]
    initial = (
        MASSIVE_ECONOMIC_REST_BASE_V8 + expected_path + "?" + parse.urlencode(params)
    )
    pages: list[MassiveEconomicRawRestPageV8] = []
    expected_url = initial
    for index, body in enumerate(raw_page_bodies):
        page_completed_at_ms = finish - (len(raw_page_bodies) - index - 1)
        page_requested_at_ms = start if not pages else pages[-1].completed_at_ms
        if page_completed_at_ms < page_requested_at_ms:
            raise MassiveEconomicProviderCaptureV8Error(
                "test capture interval is too short for its page inventory"
            )
        page = _build_page(
            page_index=index,
            request_url=expected_url,
            final_response_url=expected_url,
            expected_path=expected_path,
            requested_at_ms=page_requested_at_ms,
            completed_at_ms=page_completed_at_ms,
            provider_request_id=f"TEST-V8-{surface_id}-{index}",
            content_type="application/json",
            provider_etag=None,
            provider_last_modified=None,
            body=body,
        )
        pages.append(page)
        if page.next_url is not None:
            expected_url = page.next_url
    return _publish_capture(
        root=root,
        surface_id=surface_id,
        coverage_start_date=coverage_start_date,
        coverage_end_date=coverage_end_date,
        query_parameters=params,
        pages=tuple(pages),
        capture_kind="synthetic-test-response-v8",
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        capture_id=capture_id,
    )


MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA,
        "base": MASSIVE_ECONOMIC_REST_BASE_V8,
        "surface_inventory": MASSIVE_ECONOMIC_REST_SURFACES_V8,
        "query": "frozen-all-market-date-range-limit-and-sort",
        "redirects": "all-rejected-before-authorization-forwarding",
        "production_clock": "noninjectable-time-time-ns",
        "production_transport": "noninjectable-urllib-opener",
        "generic_parse": "always-fixed-runtime-captured-false",
        "source_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SHA256,
    }
)


__all__ = [
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V8_DATASET",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V8_OBJECT_PREFIX",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SCHEMA",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V8_SPEC_SHA256",
    "MASSIVE_ECONOMIC_REST_BASE_V8",
    "MASSIVE_ECONOMIC_REST_DATE_FIELDS_V8",
    "MASSIVE_ECONOMIC_REST_SORTS_V8",
    "MASSIVE_ECONOMIC_REST_SURFACES_V8",
    "MassiveEconomicProviderCaptureV8Error",
    "MassiveEconomicRawRestCaptureV8",
    "MassiveEconomicRawRestPageV8",
    "build_massive_economic_query_parameters_v8",
    "capture_massive_economic_rest_surface_for_test_v8",
    "capture_massive_economic_rest_surface_v8",
    "parse_massive_economic_raw_rest_capture_v8",
]
