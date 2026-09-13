"""One exact-day BKNG/PCLN acquisition route with reviewed rename evidence.

This is not a generic missing-FIGI exception. The original issuer-query chain,
modern provider wrapper, intended population, and SEC body AND receipt are
byte-pinned. The common-share certificates continued through the documented
name/ticker change; there was no AVGO-style mandatory share exchange here.

Historical FIGIs remain absent and all provider bytes/tickers remain literal.
Retrospective source review does not qualify continuous identity, availability,
corporate-action accounting, tradability, model inputs, or training. No caller
mapping, copied report, or matching keyword may replace the reviewed originals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timezone
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

SCHEMA = "rl-quant.raw-second-bkng-exact-day-documented-alias-v1"
_EASTERN = ZoneInfo("America/New_York")
_SESSION_DATE = "2017-01-03"
_MODERN_DATE = "2022-01-03"
_ISSUE = ("BBG000BLBVN4", "BBG001S89N72", "0001075531")
_HISTORICAL_SHA = "50c403b3bce22e6a17cde2a3f7f95247472fc479cd348b993d3b74b807dda820"
_MODERN_SHA = "e8234d36868911569c8a14dd5872f33dd92b1d1b92092355e64af07b6c0d0b76"
_INTENDED_SHA = "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0"
_INTENDED_BYTES = 746_927
_ORDERED_TICKERS_SHA = "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
_PUBLIC_KIND = "bkng-sec-20180227"
_PUBLIC_URL = "https://www.sec.gov/Archives/edgar/data/1075531/000107553118000008/a8-knamechange2018.htm"
_PUBLIC_BYTES = 36_573
_PUBLIC_SHA = "5ab36122e9379cac575e408b2ac6610ecac485c9be5f483eaacb40c798ea156b"
_RECEIPT_BYTES = 596
_RECEIPT_SHA = "0b5ff9e7588cceebbb267142eaec360bb87d37df7823d1c9dc3e818e1f3abf48"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: dict, cls: type) -> dict:
    _require(type(value) is dict and set(value) == {field.name for field in fields(cls)},
             "Documented alias serialization fields differ")
    return value


def _path(value: str) -> Path:
    _require(type(value) is str and value.startswith("/") and "\\" not in value
             and "\x00" not in value and all(part not in ("", ".", "..")
                                             for part in value.split("/")[1:]),
             "Documented alias requires a canonical absolute original path")
    return Path(value)


def _read(path: str, expected: str, cap: int,
          evidence_relocation: EvidenceRelocation | None) -> bytes:
    original = _path(path)
    if evidence_relocation is None:
        raw = transport.read_regular(original, cap)
    else:
        _require(type(evidence_relocation) is EvidenceRelocation,
                 "Documented alias requires explicit typed evidence relocation")
        raw = evidence_relocation.read(original, expected, cap)
    _require(transport.digest(raw) == expected, "Documented alias original evidence changed")
    return raw


@dataclass(frozen=True)
class ReviewedDocumentedAliasSourceRef:
    kind: str
    path: str
    sha256: str
    receipt_path: str
    receipt_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ReviewedDocumentedAliasSourceRef:
        return cls(**_mapping(value, cls))

    def validate(self, *, evidence_relocation: EvidenceRelocation | None = None) -> dict:
        _require(self.kind == _PUBLIC_KIND and self.sha256 == _PUBLIC_SHA
                 and self.receipt_sha256 == _RECEIPT_SHA,
                 "Documented alias requires the reviewed original SEC body and receipt")
        raw = _read(self.path, _PUBLIC_SHA, _PUBLIC_BYTES, evidence_relocation)
        receipt_raw = _read(self.receipt_path, _RECEIPT_SHA, _RECEIPT_BYTES, evidence_relocation)
        _require(len(raw) == _PUBLIC_BYTES and len(receipt_raw) == _RECEIPT_BYTES,
                 "Documented alias primary source size differs")
        receipt = transport.parse_json(receipt_raw)
        _require(receipt.get("source") == _PUBLIC_KIND and receipt.get("request_url") == _PUBLIC_URL
                 and receipt.get("final_url") == _PUBLIC_URL and receipt.get("http_status") == 200
                 and receipt.get("capture_success") is True
                 and receipt.get("original_response_complete") is True
                 and receipt.get("body_truncated") is False
                 and receipt.get("body") == dict(path=_PUBLIC_KIND + ".html",
                                                bytes=_PUBLIC_BYTES, sha256=_PUBLIC_SHA)
                 and type(receipt.get("requested_at_ns")) is int
                 and type(receipt.get("received_at_ns")) is int
                 and 0 < receipt["requested_at_ns"] <= receipt["received_at_ns"],
                 "Documented alias original SEC receipt scope/completeness differs")
        return dict(kind=self.kind, path=self.path, sha256=self.sha256, bytes=_PUBLIC_BYTES,
                    url=_PUBLIC_URL, receipt_path=self.receipt_path,
                    receipt_sha256=self.receipt_sha256, receipt_bytes=_RECEIPT_BYTES,
                    capture_time_is_historical_availability=False)


def _query_clock(query: SecondQuery, session_date: str) -> None:
    _require(type(query) is SecondQuery, "Documented alias requires a literal SecondQuery")
    SecondQuery(query.ticker, query.start_ms, query.end_ms)
    _require(type(session_date) is str and session_date == _SESSION_DATE
             and query.ticker == "PCLN" and query.end_ms - query.start_ms < 3_600_000,
             "Documented alias query outside reviewed exact-date/ticker/hour scope")
    for stamp in (query.start_ms, query.end_ms):
        clock = datetime.fromtimestamp(stamp // 1000, timezone.utc).astimezone(_EASTERN)
        _require(clock.date().isoformat() == session_date
                 and time(9, 30) <= clock.time() < time(16),
                 "Documented alias query outside exact-date regular clock hours")
    # Clock bounds do not supply the independent corpus trading-session calendar.


def _provider(identity: AliasIdentityRef, *, modern: bool,
              evidence_relocation: EvidenceRelocation | None) -> dict:
    expected_kind = "dated-ticker-wrapper" if modern else "reference-supplement"
    expected_sha = _MODERN_SHA if modern else _HISTORICAL_SHA
    _require(type(identity) is AliasIdentityRef and identity.kind == expected_kind
             and identity.sha256 == expected_sha,
             "Documented alias provider reference is not the reviewed original")
    _path(identity.path)
    rows = identity.rows(fixed_slot="BKNG", provider_ticker="BKNG" if modern else "PCLN",
                         session_date=_MODERN_DATE if modern else _SESSION_DATE,
                         evidence_relocation=evidence_relocation)
    _require(len(rows) == 1 and type(rows[0]) is dict, "Ambiguous documented alias provider issue")
    row = rows[0]
    _require(row.get("ticker") == ("BKNG" if modern else "PCLN")
             and row.get("name") == ("Booking Holdings Inc. Common Stock" if modern
                                      else "The Priceline Group Inc.")
             and row.get("cik") == _ISSUE[2] and row.get("primary_exchange") == "XNAS"
             and row.get("type") == "CS" and row.get("active") is True
             and row.get("market") == "stocks" and row.get("locale") == "us"
             and row.get("currency_name") == "usd",
             "Documented alias original does not identify the reviewed common stock")
    if modern:
        _require((row.get("composite_figi"), row.get("share_class_figi")) == _ISSUE[:2],
                 "Documented alias modern issue identifiers differ")
    else:
        _require("composite_figi" not in row and "share_class_figi" not in row,
                 "Historical PCLN FIGI absence must be preserved, not backfilled")
    return row


def _intended(source: str, expected_sha256: str,
              evidence_relocation: EvidenceRelocation | None) -> None:
    _require(expected_sha256 == _INTENDED_SHA,
             "Documented alias intended issue must use the reviewed fixed population")
    packed = _read(source, _INTENDED_SHA, _INTENDED_BYTES, evidence_relocation)
    _require(len(packed) == _INTENDED_BYTES, "Documented alias intended source size differs")
    # Same bounded original gzip/JSON semantics as load_evidence, with all reads
    # routed through explicit relocation when requested; never reopen originals.
    with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
        raw = stream.read(32 * 1024 * 1024 + 1)
        _require(len(raw) <= 32 * 1024 * 1024 and not stream.read(1),
                 "Documented alias intended evidence exceeds bound")
    evidence = transport.parse_json(raw)
    tickers, resolutions = evidence.get("ordered_tickers"), evidence.get("security_resolutions")
    _require(evidence.get("schema") == "rl-quant.qt200-issue-resolution-evidence-v1"
             and evidence.get("historical_identity_qualified") is False
             and evidence.get("training_ready_for_adaptive_v5") is False
             and tickers == list(SYMBOLS) and len(tickers) == 200 and tickers[56] == "BKNG"
             and transport.digest(transport.canonical(tickers)) == _ORDERED_TICKERS_SHA
             and type(resolutions) is list and all(type(row) is dict for row in resolutions)
             and [row.get("qt200_ticker") for row in resolutions] == tickers,
             "Documented alias must preserve the exact fixed QT200 issue population")
    intended = resolutions[56].get("current_reference_evidence")
    _require(type(intended) is dict and intended.get("ticker") == "BKNG"
             and intended.get("current_security_type") == "CS"
             and intended.get("current_primary_exchange") == "XNAS"
             and type(intended.get("active_exact_reference_count")) is int
             and intended["active_exact_reference_count"] == 1
             and intended.get("current_issue_identifier_observed") is True
             and (intended.get("current_composite_figi"), intended.get("current_share_class_figi"),
                  intended.get("current_issuer_cik")) == _ISSUE,
             "Documented alias intended common-stock issue differs")


@dataclass(frozen=True)
class SecondDocumentedAliasRoute:
    fixed_slot_ticker: str
    provider_ticker: str
    session_date: str
    identity: AliasIdentityRef
    modern_identity: AliasIdentityRef
    modern_observation_date: str
    intended_issue_source: str
    intended_issue_sha256: str
    sec_source: ReviewedDocumentedAliasSourceRef

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict) -> SecondDocumentedAliasRoute:
        _require(type(value) is dict and value.get("schema") == SCHEMA
                 and set(value) == {"schema", *(field.name for field in fields(cls))},
                 "Documented alias route schema/serialization fields differ")
        value = {key: item for key, item in value.items() if key != "schema"}
        return cls(**{
            **value,
            "identity": AliasIdentityRef(**_mapping(value["identity"], AliasIdentityRef)),
            "modern_identity": AliasIdentityRef(**_mapping(value["modern_identity"], AliasIdentityRef)),
            "sec_source": ReviewedDocumentedAliasSourceRef.from_dict(value["sec_source"]),
        })

    def validate(self, query: SecondQuery, *,
                 evidence_relocation: EvidenceRelocation | None = None) -> dict:
        _require(self.fixed_slot_ticker == "BKNG" and self.provider_ticker == "PCLN"
                 and self.modern_observation_date == _MODERN_DATE,
                 "Only the reviewed BKNG/PCLN documented rename route is supported")
        _query_clock(query, self.session_date)
        _require(evidence_relocation is None or type(evidence_relocation) is EvidenceRelocation,
                 "Documented alias requires explicit typed evidence relocation")
        _require(type(self.sec_source) is ReviewedDocumentedAliasSourceRef,
                 "Documented alias primary evidence requires a typed original reference")
        public = self.sec_source.validate(evidence_relocation=evidence_relocation)
        old = _provider(self.identity, modern=False, evidence_relocation=evidence_relocation)
        new = _provider(self.modern_identity, modern=True, evidence_relocation=evidence_relocation)
        _intended(self.intended_issue_source, self.intended_issue_sha256, evidence_relocation)
        return dict(
            schema=SCHEMA, fixed_slot_ticker="BKNG", fixed_slot_index=56, provider_ticker="PCLN",
            ordered_tickers_sha256=_ORDERED_TICKERS_SHA, session_date=self.session_date,
            identity=asdict(self.identity), modern_identity=asdict(self.modern_identity),
            modern_observation_date=self.modern_observation_date,
            intended_issue_source=self.intended_issue_source,
            intended_issue_sha256=self.intended_issue_sha256,
            historical_provider_name=old["name"], historical_provider_cik=old["cik"],
            historical_provider_composite_figi=None, historical_provider_share_class_figi=None,
            historical_figi_fields_absent=True, intended_cik=new["cik"],
            intended_composite_figi=new["composite_figi"],
            intended_share_class_figi=new["share_class_figi"], security_class="common stock",
            relationship="documented-continuing-common-stock-name-ticker-change",
            legal_name_effective_date="2018-02-21", marketplace_effective_date="2018-02-27",
            new_common_stock_cusip="09857L108", original_common_stock_cusip=None,
            outstanding_certificates_remain_valid=True, certificate_exchange_required=False,
            mandatory_share_exchange=False, ticker_event_bracket_start=None,
            historical_interval_interpolated=False, primary_originals=[public],
            acquisition_route_only=True, provider_identifiers_rewritten=False,
            provider_tickers_rewritten=False, capture_time_is_historical_availability=False,
            query_clock_is_exchange_calendar_qualification=False,
            continuous_identity_qualified=False, historical_issue_identity_qualified=False,
            economic_conversion_qualified=False, economic_accounting_qualified=False,
            economic_inputs_qualified=False, tradability_qualified=False,
            point_in_time_qualified=False, native_catalog_activated=False, training_ready=False,
        )
