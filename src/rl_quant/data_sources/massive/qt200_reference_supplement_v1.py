"""Bounded provider supplements for QT200-AGG-DEV-01, never identity promotion.

Issuer queries discover candidate securities; a CIK is not an issue identifier.
Current/provider-dated responses cannot establish historical availability or
continuous issue identity. Every returned class and conflict remains visible.
"""

from __future__ import annotations

import gzip
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from urllib import parse

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS

SCHEMA = "rl-quant.qt200-reference-supplement-v1"
MAX_BYTES, MAX_PAGES = 128_000_000, 8
ANCHORS = ("2017-01-03", "2022-01-03", "2026-08-31")
REVIEW = tuple("""AVGO ON MPWR CDNS DELL PLTR CRWD DDOG SNOW GOOGL META CMCSA T
ROKU PINS BKNG ABNB HLT RCL CCL MDLZ MDT ELV CI IQV MS SCHW BLK BX KKR BRK.B
COIN GE HON RTX ETN CMI WM XOM LIN DOW NEE VST""".split())
ALIASES = (("META", "FB", "2017-01-03", "2022-06-08"),
           ("ELV", "ANTM", "2017-01-03", "2022-06-27"),
           ("HON", "HONI", "2026-06-15", "2026-06-28"))


@dataclass(frozen=True)
class SupplementQuery(transport.PilotQuery):
    target: str = ""
    cik: str = ""

    def validate(self) -> None:
        allowed = False
        if self.product == "day":
            allowed = (self.target, self.ticker, self.start, self.end) in ALIASES and not self.cik
        elif self.product == "reference-ticker":
            allowed = (not self.cik and self.target == self.ticker and self.ticker in SYMBOLS
                       and self.start == self.end == ANCHORS[-1])
        elif self.product == "reference-alias":
            allowed = not self.cik and any(
                self.target == target and self.ticker == ticker and self.start == self.end
                and self.start in (start, end) for target, ticker, start, end in ALIASES)
        elif self.product == "reference-issuer":
            allowed = (self.target == self.ticker and self.ticker in REVIEW
                       and self.start == self.end and self.start in ANCHORS
                       and len(self.cik) == 10 and self.cik.isascii() and self.cik.isdigit())
        elif self.product in ("splits", "dividends"):
            allowed = (self.ticker == self.target == "market" and not self.cik
                       and self.start == "2026-08-27" and self.end == "2026-08-31")
        if not allowed:
            raise transport.ResearchCaptureError("Query outside frozen supplement scope")

    @property
    def name(self) -> str:
        self.validate()
        return f"{self.product}-{self.ticker}-{self.start}"

    @property
    def url(self) -> str:
        self.validate()
        if self.product == "day":
            return super().url
        if self.product.startswith("reference-"):
            path = "/v3/reference/tickers"
            fields = {"market": "stocks", "active": "true", "date": self.start,
                      "sort": "ticker", "order": "asc", "limit": "1000"}
            fields["cik" if self.product == "reference-issuer" else "ticker"] = self.cik or self.ticker
        else:
            path = "/stocks/v1/" + self.product
            field = "execution_date" if self.product == "splits" else "ex_dividend_date"
            fields = {field + ".gte": self.start, field + ".lte": self.end,
                      "sort": field + ".asc", "limit": "5000"}
        return f"https://{transport.HOST}{path}?{parse.urlencode(sorted(fields.items()))}"

    def validate_url(self, url: str) -> str:
        self.validate()
        if self.product == "day":
            return super().validate_url(url)
        if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
            raise transport.ResearchCaptureError("Noncanonical supplement URL")
        parts, original = parse.urlsplit(url), parse.urlsplit(self.url)
        if (parts.scheme != "https" or parts.netloc != transport.HOST
                or parts.path != original.path or parts.fragment or parts.username or parts.password):
            raise transport.ResearchCaptureError("Supplement cursor escaped exact endpoint")
        pairs = parse.parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True)
        fields, expected = dict(pairs), dict(parse.parse_qsl(original.query))
        if (len(fields) != len(pairs) or any(not value for _, value in pairs)
                or not set(fields) <= set(expected) | {"cursor"}
                or any(fields[key] != value for key, value in expected.items() if key in fields)
                or ("cursor" not in fields and url != self.url)):
            raise transport.ResearchCaptureError("Supplement cursor changed scope or contains secrets")
        return url

    def inspect_page(self, raw: bytes) -> dict:
        self.validate()
        if self.product == "day":
            return super().inspect_page(raw)
        body = transport.parse_json(raw)
        rows = body.get("results", [])
        limit = 1000 if self.product.startswith("reference-") else 5000
        if (body.get("status") != "OK" or not isinstance(body.get("request_id"), str)
                or not body["request_id"] or not isinstance(rows, list) or len(rows) > limit
                or any(not isinstance(row, dict) for row in rows)):
            raise transport.ResearchCaptureError("Unidentified or malformed supplement response")
        next_url = body.get("next_url")
        if next_url is not None:
            self.validate_url(next_url)
        for row in rows:
            if self.product.startswith("reference-"):
                if row.get("market") != "stocks" or row.get("active") is not True:
                    raise transport.ResearchCaptureError("Reference row outside requested market/status")
                if self.product == "reference-issuer":
                    if row.get("cik") != self.cik:
                        raise transport.ResearchCaptureError("Provider returned another issuer")
                elif row.get("ticker") != self.ticker:
                    raise transport.ResearchCaptureError("Provider returned another ticker")
            else:
                field = "execution_date" if self.product == "splits" else "ex_dividend_date"
                day = row.get(field)
                if not isinstance(day, str) or not self.start <= day <= self.end:
                    raise transport.ResearchCaptureError("Economic event outside requested dates")
        return {"request_id": body["request_id"], "result_count": len(rows), "next_url": next_url,
                "records_deduplicated": 0, "issue_links_established": 0,
                "point_in_time_qualified": False, "training_ready": False}


def load_evidence(path: Path, expected_sha256: str) -> dict:
    packed = transport.read_regular(path, 8 * 1024 * 1024)
    if transport.digest(packed) != expected_sha256:
        raise transport.ResearchCaptureError("Externally bound identity evidence changed")
    with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024 or stream.read(1):
            raise transport.ResearchCaptureError("Identity evidence exceeds bound")
    body = transport.parse_json(raw)
    if (body.get("ordered_tickers") != list(SYMBOLS)
            or body.get("historical_identity_qualified") is not False
            or body.get("training_ready_for_adaptive_v5") is not False):
        raise transport.ResearchCaptureError("Evidence scope/qualification differs")
    return body


def queries_from_evidence(body: dict) -> tuple[SupplementQuery, ...]:
    rows = body.get("security_resolutions")
    if not isinstance(rows, list) or [r.get("qt200_ticker") for r in rows] != list(SYMBOLS):
        raise transport.ResearchCaptureError("Identity resolution population differs")
    by_ticker = {r["qt200_ticker"]: r for r in rows}
    result = [SupplementQuery("day", ticker, start, end, target) for target, ticker, start, end in ALIASES]
    result.extend(SupplementQuery(product, "market", "2026-08-27", "2026-08-31", "market")
                  for product in ("splits", "dividends"))
    result.extend(SupplementQuery("reference-ticker", ticker, ANCHORS[-1], ANCHORS[-1], ticker)
                  for ticker in SYMBOLS)
    result.extend(SupplementQuery("reference-alias", ticker, day, day, target)
                  for target, ticker, start, end in ALIASES for day in (start, end))
    for ticker in REVIEW:
        cik = by_ticker[ticker]["current_reference_evidence"].get("current_issuer_cik")
        if not isinstance(cik, str):
            raise transport.ResearchCaptureError("No source-observed issuer query key")
        for day in ANCHORS:
            result.append(SupplementQuery("reference-issuer", ticker, day, day, ticker, cik))
    for query in result:
        query.validate()
    if len(result) != 340 or len({q.name for q in result}) != len(result):
        raise transport.ResearchCaptureError("Supplement query inventory differs")
    return tuple(result)


def plan_fields(*, evidence: Path, evidence_sha256: str) -> dict:
    queries = queries_from_evidence(load_evidence(evidence, evidence_sha256))
    return {"schema": SCHEMA + "-plan", "research_track": "QT200-AGG-DEV-01",
            "queries": [asdict(q) for q in queries], "identity_evidence_sha256": evidence_sha256,
            "maximum_raw_response_bytes": MAX_BYTES, "maximum_page_bytes": transport.MAX_PAGE_BYTES,
            "maximum_pages_per_query": MAX_PAGES, "maximum_elapsed_seconds": transport.MAX_CAPTURE_SECONDS,
            "retry_count": 0, "concurrent_requests": 1, "minimum_request_gap_seconds": 0.3,
            "issuer_query_is_issue_identity": False, "automatic_alias_stitching": False,
            "native_v5_qualified": False, "point_in_time_qualified": False, "training_ready": False}


def _check(root: Path, plan_sha256: str, evidence: Path, evidence_sha256: str,
           current: bool) -> tuple[SupplementQuery, ...]:
    raw = transport.read_regular(root / "plan.json", 1048576)
    plan = transport.parse_json(raw)
    if (transport.digest(raw) != plan_sha256 or any(
            plan.get(k) != v or type(plan.get(k)) is not type(v)
            for k, v in plan_fields(evidence=evidence, evidence_sha256=evidence_sha256).items())):
        raise transport.ResearchCaptureError("Supplement plan differs")
    if current and plan.get("supplement_implementation_sha256") != transport.digest(
            transport.read_regular(Path(__file__).resolve(), 1048576)):
        raise transport.ResearchCaptureError("Supplement implementation changed")
    return queries_from_evidence(load_evidence(evidence, evidence_sha256))


def capture_supplement(*, root: Path, api_key: str, plan_sha256: str,
                       evidence: Path, evidence_sha256: str) -> dict:
    queries = _check(root, plan_sha256, evidence, evidence_sha256, True)
    return transport._capture_queries(root=root, api_key=api_key, plan_sha256=plan_sha256,
        queries=queries, schema=SCHEMA, maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES)


def replay_supplement(*, root: Path, plan_sha256: str, completion_sha256: str,
                      evidence: Path, evidence_sha256: str) -> dict:
    queries = _check(root, plan_sha256, evidence, evidence_sha256, False)
    return transport._replay_queries(root=root, plan_sha256=plan_sha256,
        completion_sha256=completion_sha256, queries=queries, schema=SCHEMA,
        maximum_bytes=MAX_BYTES, maximum_pages=MAX_PAGES)
