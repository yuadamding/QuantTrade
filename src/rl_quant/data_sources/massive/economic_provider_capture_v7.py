"""Fixed raw REST-response capture for Massive economic reference surfaces V7."""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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

MASSIVE_ECONOMIC_REST_BASE_V7 = "https://api.massive.com"
MASSIVE_ECONOMIC_REST_SURFACES_V7 = {
    "massive-dividends": "/v3/reference/dividends",
    "massive-splits": "/v3/reference/splits",
    "massive-tickers": "/v3/reference/tickers",
}
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA = "rl-quant.massive-economic-raw-rest-capture-v7"
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET = "massive-economic-raw-rest-capture-v7"
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_OBJECT_PREFIX = (
    "massive-profitability-p0/raw-economic-rest-v7/"
)
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA,
        "transport": "fixed-HTTPS-GET-bearer-header",
        "base_url": MASSIVE_ECONOMIC_REST_BASE_V7,
        "surfaces": MASSIVE_ECONOMIC_REST_SURFACES_V7,
        "raw_body": "exact-base64-and-physical-sha256",
        "pagination": "provider-next-url-complete-chain",
        "credentials": "never-persisted",
    }
)
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA,
        "base": MASSIVE_ECONOMIC_REST_BASE_V7,
        "surface_inventory": MASSIVE_ECONOMIC_REST_SURFACES_V7,
        "production_clock": "noninjectable-time-time-ns",
        "production_transport": "noninjectable-urllib-request-urlopen",
        "test_capture": "separate-and-provider-runtime-qualified-false",
        "adapter_source_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SHA256,
    }
)


class MassiveEconomicProviderCaptureV7Error(ValueError):
    """Raw Massive response capture or its immutable source chain differs."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEconomicProviderCaptureV7Error(f"{name} must be canonical text")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEconomicProviderCaptureV7Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEconomicProviderCaptureV7Error(f"{name} must be nonnegative")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise MassiveEconomicProviderCaptureV7Error(f"{name} fields differ")


def _safe_massive_url(value: object) -> str:
    url = _text("Massive request URL", value)
    parsed = parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.massive.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "apikey" in parsed.query.lower()
    ):
        raise MassiveEconomicProviderCaptureV7Error(
            "Massive request URL is outside the fixed secret-free endpoint"
        )
    return url


def _json_without_duplicate_keys(raw: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MassiveEconomicProviderCaptureV7Error(
                    "raw provider response contains a duplicate JSON key"
                )
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicProviderCaptureV7Error(
            "raw provider response is not JSON"
        ) from exc


@dataclass(frozen=True, slots=True)
class MassiveEconomicRawRestPageV7:
    page_index: int
    request_url: str
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
            raise MassiveEconomicProviderCaptureV7Error(
                "raw provider response is not canonical base64"
            ) from exc

    def parsed_body(self) -> Mapping[str, object]:
        raw = self.raw_body()
        value = _json_without_duplicate_keys(raw)
        if not isinstance(value, dict):
            raise MassiveEconomicProviderCaptureV7Error(
                "raw provider response root is not an object"
            )
        return value

    def validate(self, *, page_count: int) -> None:
        if (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or not 0 <= self.page_index < page_count
        ):
            raise MassiveEconomicProviderCaptureV7Error("page index is invalid")
        _safe_massive_url(self.request_url)
        if self.http_method != "GET" or self.http_status != 200:
            raise MassiveEconomicProviderCaptureV7Error(
                "raw provider response did not come from a successful fixed GET"
            )
        requested = _nonnegative_int("page request time", self.requested_at_ms)
        completed = _nonnegative_int("page completion time", self.completed_at_ms)
        if completed < requested:
            raise MassiveEconomicProviderCaptureV7Error(
                "provider page completion predates request"
            )
        _text("provider request ID", self.provider_request_id)
        _text("provider response content type", self.response_content_type)
        if "json" not in self.response_content_type.lower():
            raise MassiveEconomicProviderCaptureV7Error(
                "provider response content type is not JSON"
            )
        for name in ("provider_etag", "provider_last_modified"):
            value = getattr(self, name)
            if value is not None:
                _text(name, value)
        raw = self.raw_body()
        _digest("raw response body", self.raw_response_body_sha256)
        if (
            hashlib.sha256(raw).hexdigest() != self.raw_response_body_sha256
            or len(raw) != self.raw_response_content_length
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "raw provider response bytes differ"
            )
        body = self.parsed_body()
        if body.get("status") != "OK":
            raise MassiveEconomicProviderCaptureV7Error(
                "provider response status is not OK"
            )
        results = body.get("results")
        if not isinstance(results, list) or any(
            not isinstance(row, dict) for row in results
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "provider result inventory is malformed"
            )
        if self.result_count != len(
            results
        ) or self.result_inventory_sha256 != semantic_sha256(results):
            raise MassiveEconomicProviderCaptureV7Error(
                "provider result inventory differs"
            )
        provider_request_id = body.get("request_id")
        if (
            provider_request_id is not None
            and provider_request_id != self.provider_request_id
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "provider body/header request IDs differ"
            )
        body_next = body.get("next_url")
        if body_next is not None:
            body_next = _safe_massive_url(body_next)
        if body_next != self.next_url:
            raise MassiveEconomicProviderCaptureV7Error(
                "provider next URL differs from raw body"
            )
        if (self.next_url is None) is not (self.page_index == page_count - 1):
            raise MassiveEconomicProviderCaptureV7Error(
                "provider pagination did not close exactly"
            )
        _digest("raw REST page receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicProviderCaptureV7Error("raw REST page receipt differs")


@dataclass(frozen=True, slots=True)
class MassiveEconomicRawRestCaptureV7:
    surface_id: str
    provider_id: str
    provider_dataset: str
    endpoint_base: str
    initial_request_url: str
    query_parameters: tuple[tuple[str, str], ...]
    requested_at_ms: int
    completed_at_ms: int
    pages: tuple[MassiveEconomicRawRestPageV7, ...]
    page_count: int
    provider_request_ids: tuple[str, ...]
    raw_page_inventory_sha256: str
    pagination_complete: bool
    capture_kind: str
    provider_runtime_qualified: bool
    adapter_source_sha256: str
    adapter_spec_sha256: str
    loaded_source: LoadedMassiveSourceObject
    receipt_sha256: str
    schema: str = MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA:
            raise MassiveEconomicProviderCaptureV7Error("capture schema differs")
        if self.surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V7:
            raise MassiveEconomicProviderCaptureV7Error(
                "Massive REST surface is unsupported"
            )
        if (
            self.provider_id != "massive"
            or self.provider_dataset != self.surface_id
            or self.endpoint_base != MASSIVE_ECONOMIC_REST_BASE_V7
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "Massive provider identity differs"
            )
        initial = _safe_massive_url(self.initial_request_url)
        if (
            parse.urlsplit(initial).path
            != MASSIVE_ECONOMIC_REST_SURFACES_V7[self.surface_id]
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "initial request path differs from fixed surface"
            )
        initial_parameters = tuple(
            sorted(parse.parse_qsl(parse.urlsplit(initial).query))
        )
        if initial_parameters != self.query_parameters:
            raise MassiveEconomicProviderCaptureV7Error(
                "initial request query differs from committed parameters"
            )
        if self.query_parameters != tuple(sorted(set(self.query_parameters))):
            raise MassiveEconomicProviderCaptureV7Error(
                "request query parameters are not canonical"
            )
        for key, value in self.query_parameters:
            _text("query parameter name", key)
            _text("query parameter value", value)
            if key.lower() == "apikey":
                raise MassiveEconomicProviderCaptureV7Error(
                    "credential entered request query evidence"
                )
        requested = _nonnegative_int("capture request time", self.requested_at_ms)
        completed = _nonnegative_int("capture completion time", self.completed_at_ms)
        if completed < requested:
            raise MassiveEconomicProviderCaptureV7Error(
                "capture completion predates request"
            )
        if (
            not self.pagination_complete
            or self.page_count <= 0
            or len(self.pages) != self.page_count
            or len(self.provider_request_ids) != self.page_count
            or len(set(self.provider_request_ids)) != self.page_count
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "capture pagination inventory differs"
            )
        previous_completed = requested
        expected_url = self.initial_request_url
        for index, page in enumerate(self.pages):
            page.validate(page_count=self.page_count)
            if (
                page.page_index != index
                or page.provider_request_id != self.provider_request_ids[index]
                or page.request_url != expected_url
                or page.requested_at_ms < previous_completed
                or parse.urlsplit(page.request_url).path
                != MASSIVE_ECONOMIC_REST_SURFACES_V7[self.surface_id]
            ):
                raise MassiveEconomicProviderCaptureV7Error(
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
            raise MassiveEconomicProviderCaptureV7Error(
                "capture page inventory differs"
            )
        if self.capture_kind not in {
            "fixed-massive-rest-production",
            "synthetic-test-response",
        } or not isinstance(self.provider_runtime_qualified, bool):
            raise MassiveEconomicProviderCaptureV7Error("capture kind is invalid")
        if self.provider_runtime_qualified is not (
            self.capture_kind == "fixed-massive-rest-production"
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "provider runtime qualification differs from capture kind"
            )
        if (
            self.adapter_source_sha256 != MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SHA256
            or self.adapter_spec_sha256 != MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "capture adapter implementation differs"
            )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SCHEMA_SHA256
            or not self.loaded_source.receipt.source_object_key.startswith(
                MASSIVE_ECONOMIC_RAW_CAPTURE_V7_OBJECT_PREFIX
            )
            or self.loaded_source.receipt.request_id != self.provider_request_ids[-1]
            or self.loaded_source.receipt.downloaded_at_ms != self.completed_at_ms
        ):
            raise MassiveEconomicProviderCaptureV7Error(
                "capture immutable source transaction differs"
            )
        _digest("raw capture receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEconomicProviderCaptureV7Error("raw capture receipt differs")


def _page_payload(page: MassiveEconomicRawRestPageV7) -> dict[str, object]:
    return asdict(page)


def _capture_payload(capture: MassiveEconomicRawRestCaptureV7) -> dict[str, object]:
    return {
        "schema": capture.schema,
        "surface_id": capture.surface_id,
        "provider_id": capture.provider_id,
        "provider_dataset": capture.provider_dataset,
        "endpoint_base": capture.endpoint_base,
        "initial_request_url": capture.initial_request_url,
        "query_parameters": capture.query_parameters,
        "requested_at_ms": capture.requested_at_ms,
        "completed_at_ms": capture.completed_at_ms,
        "pages": tuple(_page_payload(page) for page in capture.pages),
        "page_count": capture.page_count,
        "provider_request_ids": capture.provider_request_ids,
        "raw_page_inventory_sha256": capture.raw_page_inventory_sha256,
        "pagination_complete": capture.pagination_complete,
        "capture_kind": capture.capture_kind,
        "provider_runtime_qualified": capture.provider_runtime_qualified,
        "adapter_source_sha256": capture.adapter_source_sha256,
        "adapter_spec_sha256": capture.adapter_spec_sha256,
    }


def _capture_payload_from_parts(
    *,
    surface_id: str,
    query_parameters: tuple[tuple[str, str], ...],
    pages: tuple[MassiveEconomicRawRestPageV7, ...],
    capture_kind: str,
    provider_runtime_qualified: bool,
) -> dict[str, object]:
    return {
        "schema": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA,
        "surface_id": surface_id,
        "provider_id": "massive",
        "provider_dataset": surface_id,
        "endpoint_base": MASSIVE_ECONOMIC_REST_BASE_V7,
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
        "provider_runtime_qualified": provider_runtime_qualified,
        "adapter_source_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SHA256,
        "adapter_spec_sha256": MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256,
    }


def parse_massive_economic_raw_rest_capture_v7(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveEconomicRawRestCaptureV7:
    """Reopen one exact committed raw REST capture."""

    loaded_source.validate()
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MassiveEconomicProviderCaptureV7Error(
            "raw capture source is not JSON"
        ) from exc
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveEconomicProviderCaptureV7Error(
            "raw capture source is not canonical JSON"
        )
    _exact_keys(
        payload,
        {
            "schema",
            "surface_id",
            "provider_id",
            "provider_dataset",
            "endpoint_base",
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
            "provider_runtime_qualified",
            "adapter_source_sha256",
            "adapter_spec_sha256",
        },
        name="raw capture source",
    )
    raw_pages = payload["pages"]
    if not isinstance(raw_pages, list) or any(
        not isinstance(page, dict) for page in raw_pages
    ):
        raise MassiveEconomicProviderCaptureV7Error("capture pages are malformed")
    pages: list[MassiveEconomicRawRestPageV7] = []
    for page in raw_pages:
        _exact_keys(
            page,
            set(MassiveEconomicRawRestPageV7.__dataclass_fields__),
            name="raw capture page",
        )
        pages.append(MassiveEconomicRawRestPageV7(**page))
    query_raw = payload["query_parameters"]
    if not isinstance(query_raw, list) or any(
        not isinstance(row, list) or len(row) != 2 for row in query_raw
    ):
        raise MassiveEconomicProviderCaptureV7Error(
            "capture query parameters are malformed"
        )
    request_ids_raw = payload["provider_request_ids"]
    if not isinstance(request_ids_raw, list):
        raise MassiveEconomicProviderCaptureV7Error("capture request IDs are malformed")
    provisional = MassiveEconomicRawRestCaptureV7(
        schema=payload["schema"],
        surface_id=payload["surface_id"],
        provider_id=payload["provider_id"],
        provider_dataset=payload["provider_dataset"],
        endpoint_base=payload["endpoint_base"],
        initial_request_url=payload["initial_request_url"],
        query_parameters=tuple((row[0], row[1]) for row in query_raw),
        requested_at_ms=payload["requested_at_ms"],
        completed_at_ms=payload["completed_at_ms"],
        pages=tuple(pages),
        page_count=payload["page_count"],
        provider_request_ids=tuple(request_ids_raw),
        raw_page_inventory_sha256=payload["raw_page_inventory_sha256"],
        pagination_complete=payload["pagination_complete"],
        capture_kind=payload["capture_kind"],
        provider_runtime_qualified=payload["provider_runtime_qualified"],
        adapter_source_sha256=payload["adapter_source_sha256"],
        adapter_spec_sha256=payload["adapter_spec_sha256"],
        loaded_source=loaded_source,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def _build_page(
    *,
    page_index: int,
    request_url: str,
    requested_at_ms: int,
    completed_at_ms: int,
    provider_request_id: str,
    content_type: str,
    provider_etag: str | None,
    provider_last_modified: str | None,
    body: bytes,
) -> MassiveEconomicRawRestPageV7:
    parsed = _json_without_duplicate_keys(body)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("results"), list):
        raise MassiveEconomicProviderCaptureV7Error(
            "Massive response result inventory is malformed"
        )
    results = parsed["results"]
    next_url = parsed.get("next_url")
    if next_url is not None:
        next_url = _safe_massive_url(next_url)
    provisional = MassiveEconomicRawRestPageV7(
        page_index=page_index,
        request_url=_safe_massive_url(request_url),
        http_method="GET",
        http_status=200,
        requested_at_ms=requested_at_ms,
        completed_at_ms=completed_at_ms,
        provider_request_id=_text("provider request ID", provider_request_id),
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
    return replace(provisional, receipt_sha256=semantic_sha256(provisional.unsigned()))


def _publish_capture(
    *,
    root: str | Path,
    surface_id: str,
    query_parameters: tuple[tuple[str, str], ...],
    pages: tuple[MassiveEconomicRawRestPageV7, ...],
    capture_kind: str,
    provider_runtime_qualified: bool,
    entitlement_receipt_sha256: str,
    capture_id: str,
) -> MassiveEconomicRawRestCaptureV7:
    if not pages:
        raise MassiveEconomicProviderCaptureV7Error("capture contains no pages")
    payload = _capture_payload_from_parts(
        surface_id=surface_id,
        query_parameters=query_parameters,
        pages=pages,
        capture_kind=capture_kind,
        provider_runtime_qualified=provider_runtime_qualified,
    )
    capture_name = _text("capture ID", capture_id)
    relative = (
        f"{MASSIVE_ECONOMIC_RAW_CAPTURE_V7_OBJECT_PREFIX}"
        f"{surface_id}-{capture_name}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET,
        source_object_key=relative,
        requested_at_ms=pages[0].requested_at_ms,
        downloaded_at_ms=pages[-1].completed_at_ms,
        schema_sha256=MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SCHEMA_SHA256,
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
    return parse_massive_economic_raw_rest_capture_v7(root=root, loaded_source=loaded)


def capture_massive_economic_rest_surface_v7(
    *,
    root: str | Path,
    surface_id: str,
    query_parameters: Mapping[str, str],
    api_key: str,
    entitlement_receipt_sha256: str,
    capture_id: str,
    timeout_seconds: float = 30.0,
    response_limit_bytes: int = 32 * 1024 * 1024,
    maximum_pages: int = 10_000,
) -> MassiveEconomicRawRestCaptureV7:
    """Capture exact Massive bytes with fixed endpoint, transport, and real clocks."""

    if surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V7:
        raise MassiveEconomicProviderCaptureV7Error("REST surface is unsupported")
    secret = _text("Massive API key", api_key)
    if any(character.isspace() for character in secret):
        raise MassiveEconomicProviderCaptureV7Error("Massive API key is malformed")
    params = tuple(
        sorted(
            (_text("query key", key), _text("query value", value))
            for key, value in query_parameters.items()
        )
    )
    if any(key.lower() == "apikey" for key, _ in params):
        raise MassiveEconomicProviderCaptureV7Error(
            "API key cannot enter query parameters"
        )
    initial = (
        MASSIVE_ECONOMIC_REST_BASE_V7 + MASSIVE_ECONOMIC_REST_SURFACES_V7[surface_id]
    )
    if params:
        initial += "?" + parse.urlencode(params)
    current_url: str | None = initial
    pages: list[MassiveEconomicRawRestPageV7] = []
    while current_url is not None:
        if len(pages) >= maximum_pages:
            raise MassiveEconomicProviderCaptureV7Error(
                "provider pagination exceeded cap"
            )
        current_url = _safe_massive_url(current_url)
        requested_at_ms = time.time_ns() // 1_000_000
        web_request = request.Request(
            current_url,
            headers={
                "Authorization": f"Bearer {secret}",
                "Accept": "application/json",
                "User-Agent": "QuantTrade-Massive-Economic-V7",
            },
            method="GET",
        )
        try:
            with request.urlopen(web_request, timeout=timeout_seconds) as response:
                status = int(response.status)
                body = response.read(response_limit_bytes + 1)
                content_type = response.headers.get("Content-Type", "")
                header_request_id = response.headers.get("X-Request-ID")
                etag = response.headers.get("ETag")
                last_modified = response.headers.get("Last-Modified")
        except (error.HTTPError, error.URLError) as exc:
            raise MassiveEconomicProviderCaptureV7Error(
                "Massive economic REST request failed"
            ) from exc
        completed_at_ms = time.time_ns() // 1_000_000
        if status != 200 or len(body) > response_limit_bytes:
            raise MassiveEconomicProviderCaptureV7Error(
                "Massive response status or size is invalid"
            )
        body_value = _json_without_duplicate_keys(body)
        body_request_id = (
            body_value.get("request_id") if isinstance(body_value, dict) else None
        )
        request_id = header_request_id or body_request_id
        page = _build_page(
            page_index=len(pages),
            request_url=current_url,
            requested_at_ms=requested_at_ms,
            completed_at_ms=completed_at_ms,
            provider_request_id=_text("provider request ID", request_id),
            content_type=content_type,
            provider_etag=etag,
            provider_last_modified=last_modified,
            body=body,
        )
        pages.append(page)
        current_url = page.next_url
    return _publish_capture(
        root=root,
        surface_id=surface_id,
        query_parameters=params,
        pages=tuple(pages),
        capture_kind="fixed-massive-rest-production",
        provider_runtime_qualified=True,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        capture_id=capture_id,
    )


def capture_massive_economic_rest_surface_for_test_v7(
    *,
    root: str | Path,
    surface_id: str,
    query_parameters: Mapping[str, str],
    raw_page_bodies: Sequence[bytes],
    requested_at_ms: int,
    completed_at_ms: int,
    entitlement_receipt_sha256: str,
    capture_id: str,
) -> MassiveEconomicRawRestCaptureV7:
    """Publish deterministic test bytes; output is always nonauthorizing."""

    if surface_id not in MASSIVE_ECONOMIC_REST_SURFACES_V7:
        raise MassiveEconomicProviderCaptureV7Error("REST surface is unsupported")
    start = _nonnegative_int("test request time", requested_at_ms)
    finish = _nonnegative_int("test completion time", completed_at_ms)
    if finish < start or not raw_page_bodies:
        raise MassiveEconomicProviderCaptureV7Error("test capture chronology differs")
    params = tuple(
        sorted(
            (_text("query key", key), _text("query value", value))
            for key, value in query_parameters.items()
        )
    )
    initial = (
        MASSIVE_ECONOMIC_REST_BASE_V7 + MASSIVE_ECONOMIC_REST_SURFACES_V7[surface_id]
    )
    if params:
        initial += "?" + parse.urlencode(params)
    pages: list[MassiveEconomicRawRestPageV7] = []
    expected_url = initial
    for index, body in enumerate(raw_page_bodies):
        page_completed_at_ms = finish - (len(raw_page_bodies) - index - 1)
        page_requested_at_ms = start if not pages else pages[-1].completed_at_ms
        if page_completed_at_ms < page_requested_at_ms:
            raise MassiveEconomicProviderCaptureV7Error(
                "test capture interval is too short for its page inventory"
            )
        page = _build_page(
            page_index=index,
            request_url=expected_url,
            requested_at_ms=page_requested_at_ms,
            completed_at_ms=page_completed_at_ms,
            provider_request_id=f"TEST-REQUEST-{index}",
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
        query_parameters=params,
        pages=tuple(pages),
        capture_kind="synthetic-test-response",
        provider_runtime_qualified=False,
        entitlement_receipt_sha256=entitlement_receipt_sha256,
        capture_id=capture_id,
    )


__all__ = [
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V7_DATASET",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V7_OBJECT_PREFIX",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SCHEMA",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ECONOMIC_RAW_CAPTURE_V7_SPEC_SHA256",
    "MASSIVE_ECONOMIC_REST_BASE_V7",
    "MASSIVE_ECONOMIC_REST_SURFACES_V7",
    "MassiveEconomicProviderCaptureV7Error",
    "MassiveEconomicRawRestCaptureV7",
    "MassiveEconomicRawRestPageV7",
    "capture_massive_economic_rest_surface_for_test_v7",
    "capture_massive_economic_rest_surface_v7",
    "parse_massive_economic_raw_rest_capture_v7",
]
