"""Three exact-day, original-evidence-backed acquisition-class routes.

HLT distributions/share basis, RCL common-class evidence, and CCL's paired
stock/redomiciliation are different relationships. They are not a generic
same-CIK or missing-FIGI exemption. These pins represent a review of exact
originals; caller-authored reports or rehashed replacements cannot authorize
another class, date or ticker. Provider identifiers and OHLCV stay untouched.

Only January 3, 2017 regular-session acquisition is covered. Neither the
relationship nor a later endpoint observation establishes intervening issue
continuity, tradability, event accounting, historical availability or training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timezone
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as io
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

SCHEMA = "rl-quant.raw-second-exact-day-reviewed-class-v1"
_DATES = ("2017-01-03", "2022-01-03", "2026-08-26")
_EASTERN = ZoneInfo("America/New_York")
_INTENDED_SHA = "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0"
_INTENDED_BYTES = 746_927
_ORDERED_SHA = "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
# URL, original body size/hash, original receipt size/hash. Receipt pins bind
# the original acquisition, not a claim that capture time was historical time.
_PUBLIC = {
    "hlt-sec-distribution-2016": (
        "https://www.sec.gov/Archives/edgar/data/1585689/000119312516784419/d293689dex991.htm",
        17_898, "8ec2fa1b7c306ed0dbaf7aca634ab302ce096402f2218da6b90c1fd1d2d8082b",
        602, "5f1b251787497070b251fcebb38ae6780aea683b79a93853abfae2a0095910d6"),
    "hlt-sec-q12017": (
        "https://www.sec.gov/Archives/edgar/data/1585689/000158568917000131/q12017hwhfsandmda-10q.htm",
        2_171_393, "59e2feea536d6f3ce096791b1de614ec231272d6bdda68a6f5e1eb59eab15a69",
        598, "4f684bd5fd4c65e8b1c1de49ba13a10ce53c0360c90654432c98632f0350cbc2"),
    "rcl-sec-fy2016": (
        "https://www.sec.gov/Archives/edgar/data/884887/000088488717000020/rcl-20161231x10k.htm",
        2_877_835, "6a427dfde34e4c8d75a9b10837546068199ce3b3e6b921ab9dbc9ba4af3ae903",
        794, "e1e151f6725b998664ee03ac37da69dda99adaffa2432e7dc2b6200bbb1583f0"),
    "rcl-sec-name-20200730": (
        "https://www.sec.gov/Archives/edgar/data/884887/000110465920088285/tm2025989d2_ex99-1.htm",
        7_759, "20fd16d95b50b45fd144dac4378d14aba6b2bbd03a1279e96eab6c4333c6bc1e",
        601, "09e835768740b5aa4f53c7f23266ce81c777f98f13f6ddf478bf024822bc9197"),
    "rcl-sec-fy2025": (
        "https://www.sec.gov/Archives/edgar/data/884887/000088488726000007/rcl-20251231.htm",
        2_829_280, "fbddfaa08099ac0e67a198144a679222767317f6ea491ae5949570631d2cfe48",
        786, "8d3b3cabe094e7239e40615d76ebf5037d5e55bb2c017990b38568f4782e6141"),
    "rcl-sec-capital-stock-2024": (
        "https://www.sec.gov/Archives/edgar/data/884887/000088488724000075/a2023q4exhibit42.htm",
        22_895, "4b839f22bfd0a98dcf7d7c6e6c703540be3b2ec9486ae3c912341838332301f6",
        819, "01fbc96f2a09d5c84b30b1d81ca13c78ce851ad7c50395a30081bf998a8afa47"),
    "rcl-sec-q22026": (
        "https://www.sec.gov/Archives/edgar/data/884887/000088488726000038/rcl-20260630.htm",
        1_773_101, "42c749f62b918d766a5306ac6facbd109bbd50ea5cc6c61566cf6a6e59106520",
        786, "c1971f85a9759a580d019fc8842a09a63ec0010232d6511d9afd92374f05ade5"),
    "ccl-sec-20260507": (
        "https://www.sec.gov/Archives/edgar/data/815097/000110465926057200/tm2613680d1_8k.htm",
        60_217, "a4ff798f333173826fba6187871311e4fd163644c6b59af379c330911e695a61",
        584, "6d181512f3168faae1d9517d40e5c2c94a6d271e3b9d9ddcce345f0a6d07ac5b"),
    "ccl-nyse-removal-2026": (
        "https://www.sec.gov/Archives/edgar/data/876661/000087666126000397/ruleprovisionnotice.htm",
        1_689, "cfe01a5e3c7c26509e27c3f7e3352baf3a228826e5c0d03a1170a69b86c7715a",
        603, "888d8cc40f6cd65c14024735deabb68037ea86b067a7bb5a44b35216dc0d602b"),
}
_HLT_FIGIS = ("BBG0058KMH30", "BBG0058KMH49")
_CCL_FIGIS = ("BBG000BF6LY3", "BBG001S5PL01")
_CLASSES = {
    "HLT": dict(
        slot=59, cik="0001585689", names=("Hilton Worldwide Holdings Inc.",) * 3,
        figis=(None, _HLT_FIGIS, _HLT_FIGIS),
        wrappers=("512fb80b07518a4d2397d19e5db613eaf1a5502adcc576e0a5ac55d15806a7eb",
                  "72e18249576213428bec1ea93648d28ef70839ba4d6f48266fa14b620839dbb8",
                  "84fc24df5c8709af6a44e7963c8037849637f6d045ee4310a6429c9d41766e0e"),
        sources=("hlt-sec-distribution-2016", "hlt-sec-q12017"),
        relationship="regular-way-common-stock-before-distributions-and-reverse-split",
        details=dict(legal_registrant="Hilton Worldwide Holdings Inc.",
            historical_class="regular-way-common-stock", par_value_usd="0.01",
            excluded_provider_symbols=["HLT WI", "PK", "HGV"],
            reviewed_event_date="2017-01-03", marketplace_post_event_date="2017-01-04",
            reverse_split_boundary="after-market-close-2017-01-03",
            reverse_split_ratio="one-new-for-three-old",
            distribution_record_date="2016-12-15",
            distribution_description="one-PK-per-five-HLT-and-one-HGV-per-ten-HLT",
            fractional_cash_and_distribution_ledger_implemented=False)),
    "RCL": dict(
        slot=60, cik="0000884887",
        names=("Royal Caribbean Cruises", "Royal Caribbean Group", "Royal Caribbean Group"),
        figis=(None, None, None),
        wrappers=("aa9598ec6c09ce319bfd1286952bd75ddab11b8d66afcf222273d385a0ac883e",
                  "ce2b44fb3d203db09cb56f129b36146f99a57970e256693345226bd1c049528c",
                  "86e844e4ddc68bb6f96d22b1b6d2db0fa9afe29c359d66bb3cd58e668c0b8301"),
        sources=("rcl-sec-fy2016", "rcl-sec-name-20200730", "rcl-sec-fy2025",
                 "rcl-sec-capital-stock-2024", "rcl-sec-q22026"),
        relationship="reviewed-listed-common-class-with-operating-name-change",
        details=dict(legal_registrant="Royal Caribbean Cruises Ltd.",
            jurisdiction="Republic of Liberia", commission_file="1-11884",
            historical_class="common-stock", par_value_usd="0.01",
            exchange="New York Stock Exchange", parent_incorporation_date="1985-07-23",
            operating_name="Royal Caribbean Group", synthetic_issue_identifier=None,
            preferred_stock_is_distinct=True, mandatory_share_exchange_asserted=False)),
    "CCL": dict(
        slot=61, cik="0000815097",
        names=("Carnival", "Carnival Corporation", "Carnival Corporation Ltd."),
        figis=(_CCL_FIGIS, _CCL_FIGIS, None),
        wrappers=("4df3e71f314a02bcad60f6589f1d58a3724a1fc74e47aa18dfde00d6d7cdeb6a",
                  "33c194f5a2940a7542510bd4a8c44cdd2b5586c32f402c1ccd76e5c5475560a3",
                  "eab4a8ef55abed71d247be76dd1962a875be15080eba7a8169d6f78305878df6"),
        sources=("ccl-sec-20260507", "ccl-nyse-removal-2026"),
        relationship="old-CCL-paired-stock-to-redomiciled-common-shares",
        details=dict(legal_registrant_before="Carnival Corporation",
            legal_registrant_after="Carnival Corporation Ltd.",
            jurisdiction_before="Panama", jurisdiction_after="Bermuda",
            historical_class="old-CCL-common-stock-paired-stock",
            successor_class="Bermuda-common-shares", par_value_usd="0.01",
            original_public_cusip="143658300", successor_public_cusip="G2004J103",
            reviewed_event_date="2026-05-07", existing_CCL_shares_remain_outstanding=True,
            successor_share_count_ratio="one-for-one", CUK_is_historical_alias=False,
            changed_shareholder_rights=True, intraday_conversion_timestamp_ms=None)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: dict, cls: type) -> dict:
    _require(type(value) is dict and set(value) == {field.name for field in fields(cls)},
             "Reviewed class serialization fields differ")
    return value


def _path(value: str) -> Path:
    _require(type(value) is str and value.startswith("/") and "\\" not in value
             and "\x00" not in value and all(p not in ("", ".", "..") for p in value.split("/")[1:]),
             "Reviewed class requires a canonical absolute original path")
    return Path(value)


def _read(path: str, expected: str, cap: int, relocation: EvidenceRelocation | None) -> bytes:
    original = _path(path)
    body = (io.read_regular(original, cap) if relocation is None
            else relocation.read(original, expected, cap))
    _require(io.digest(body) == expected, "Reviewed class original evidence changed")
    return body


@dataclass(frozen=True)
class ReviewedClassSourceRef:
    kind: str
    path: str
    sha256: str
    receipt_path: str
    receipt_sha256: str

    @classmethod
    def from_dict(cls, value: dict) -> ReviewedClassSourceRef:
        return cls(**_mapping(value, cls))

    def validate(self, *, evidence_relocation: EvidenceRelocation | None = None) -> dict:
        _require(evidence_relocation is None or type(evidence_relocation) is EvidenceRelocation,
                 "Reviewed class requires explicit typed evidence relocation")
        _require(type(self.kind) is str and self.kind in _PUBLIC, "Unsupported reviewed class original")
        url, size, expected, receipt_size, receipt_sha = _PUBLIC[self.kind]
        _require(self.sha256 == expected and self.receipt_sha256 == receipt_sha,
                 "Reviewed class requires the exact original body and receipt")
        body = _read(self.path, expected, size, evidence_relocation)
        receipt_raw = _read(self.receipt_path, receipt_sha, receipt_size, evidence_relocation)
        _require(len(body) == size and len(receipt_raw) == receipt_size, "Reviewed class source size differs")
        receipt = io.parse_json(receipt_raw)
        _require(receipt.get("source") == self.kind and receipt.get("request_url") == url
                 and receipt.get("final_url") == url and receipt.get("http_status") == 200
                 and receipt.get("capture_success") is True and receipt.get("body_truncated") is False
                 and receipt.get("original_response_complete") is True
                 and receipt.get("body") == dict(path=self.kind + ".html", bytes=size, sha256=expected)
                 and type(receipt.get("requested_at_ns")) is int
                 and type(receipt.get("received_at_ns")) is int
                 and 0 < receipt["requested_at_ns"] <= receipt["received_at_ns"],
                 "Reviewed class original capture scope/completeness differs")
        return dict(**asdict(self), bytes=size, receipt_bytes=receipt_size, url=url,
                    capture_time_is_historical_availability=False)


def _provider(identity, ticker, spec, index, relocation):
    _require(type(identity) is AliasIdentityRef and identity.kind == "dated-ticker-wrapper"
             and identity.sha256 == spec["wrappers"][index],
             "Reviewed class requires the exact dated provider wrapper")
    _path(identity.path)
    rows = identity.rows(fixed_slot=ticker, provider_ticker=ticker,
                         session_date=_DATES[index], evidence_relocation=relocation)
    _require(len(rows) == 1 and type(rows[0]) is dict, "Ambiguous reviewed provider class")
    row = rows[0]
    _require(row.get("ticker") == ticker and row.get("name") == spec["names"][index]
             and row.get("cik") == spec["cik"] and row.get("primary_exchange") == "XNYS"
             and row.get("type") == "CS" and row.get("active") is True
             and row.get("currency_name") == "usd" and row.get("market") == "stocks"
             and row.get("locale") == "us", "Reviewed provider issuer/class differs")
    figis = spec["figis"][index]
    if figis is None:
        _require("composite_figi" not in row and "share_class_figi" not in row,
                 "Missing provider FIGI fields must remain absent")
    else:
        _require((row.get("composite_figi"), row.get("share_class_figi")) == figis,
                 "Reviewed provider FIGIs differ")
    return dict(observation_date=_DATES[index], identity=asdict(identity), name=row["name"],
        cik=row["cik"], composite_figi=row.get("composite_figi"),
        share_class_figi=row.get("share_class_figi"), figi_fields_absent=figis is None)


def _intended(source, expected, ticker, spec, relocation):
    _require(expected == _INTENDED_SHA, "Reviewed class intended population is not the reviewed original")
    packed = _read(source, expected, _INTENDED_BYTES, relocation)
    _require(len(packed) == _INTENDED_BYTES, "Reviewed intended population size differs")
    with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
        raw = stream.read(32 * 1024**2 + 1)
        _require(len(raw) <= 32 * 1024**2 and not stream.read(1), "Intended evidence exceeds bound")
    evidence = io.parse_json(raw)
    tickers, rows = evidence.get("ordered_tickers"), evidence.get("security_resolutions")
    _require(evidence.get("schema") == "rl-quant.qt200-issue-resolution-evidence-v1"
             and evidence.get("historical_identity_qualified") is False
             and evidence.get("training_ready_for_adaptive_v5") is False
             and tickers == list(SYMBOLS) and len(tickers) == 200
             and io.digest(io.canonical(tickers)) == _ORDERED_SHA and tickers[spec["slot"]] == ticker
             and type(rows) is list and all(type(r) is dict for r in rows)
             and [r.get("qt200_ticker") for r in rows] == tickers,
             "Reviewed class fixed ordered population differs")
    row = rows[spec["slot"]].get("current_reference_evidence")
    figis = spec["figis"][-1]
    _require(type(row) is dict and row.get("ticker") == ticker
             and row.get("current_security_type") == "CS" and row.get("current_primary_exchange") == "XNYS"
             and type(row.get("active_exact_reference_count")) is int
             and row["active_exact_reference_count"] == 1 and row.get("current_issuer_cik") == spec["cik"]
             and (row.get("current_composite_figi"), row.get("current_share_class_figi")) ==
                 (figis if figis is not None else (None, None))
             and row.get("current_issue_identifier_observed") is (figis is not None)
             and row.get("historical_identity_qualified") is False,
             "Reviewed class intended observation differs; do not backfill missing IDs")
    return dict(current_issue_identifier_observed=row["current_issue_identifier_observed"],
        current_composite_figi=row["current_composite_figi"], current_share_class_figi=row["current_share_class_figi"],
        current_issuer_cik=row["current_issuer_cik"])


@dataclass(frozen=True)
class SecondReviewedClassRoute:
    fixed_slot_ticker: str
    provider_ticker: str
    session_date: str
    identity: AliasIdentityRef
    modern_identity: AliasIdentityRef
    latest_identity: AliasIdentityRef
    intended_issue_source: str
    intended_issue_sha256: str
    primary_sources: tuple[ReviewedClassSourceRef, ...]

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self),
                "primary_sources": [asdict(ref) for ref in self.primary_sources]}

    @classmethod
    def from_dict(cls, value: dict) -> SecondReviewedClassRoute:
        _require(type(value) is dict and value.get("schema") == SCHEMA
                 and set(value) == {"schema", *(f.name for f in fields(cls))}
                 and type(value["primary_sources"]) is list,
                 "Reviewed class route schema/serialization fields differ")
        value = {k: v for k, v in value.items() if k != "schema"}
        return cls(**{**value,
            **{k: AliasIdentityRef(**_mapping(value[k], AliasIdentityRef))
               for k in ("identity", "modern_identity", "latest_identity")},
            "primary_sources": tuple(ReviewedClassSourceRef.from_dict(r) for r in value["primary_sources"])})

    def validate(self, query: SecondQuery, *, evidence_relocation: EvidenceRelocation | None = None) -> dict:
        _require(type(self.fixed_slot_ticker) is str and self.fixed_slot_ticker in _CLASSES
                 and self.provider_ticker == self.fixed_slot_ticker and self.session_date == _DATES[0],
                 "Only the reviewed HLT/RCL/CCL exact-day literal routes are supported")
        _require(type(query) is SecondQuery and query.ticker == self.provider_ticker,
                 "Reviewed class requires the unchanged literal SecondQuery")
        SecondQuery(query.ticker, query.start_ms, query.end_ms)
        _require(query.end_ms - query.start_ms < 3_600_000, "Reviewed class query exceeds one hour")
        for stamp in (query.start_ms, query.end_ms):
            clock = datetime.fromtimestamp(stamp // 1000, timezone.utc).astimezone(_EASTERN)
            _require(clock.date().isoformat() == self.session_date and time(9, 30) <= clock.time() < time(16),
                     "Reviewed class query outside exact-day regular clock hours")
        _require(evidence_relocation is None or type(evidence_relocation) is EvidenceRelocation,
                 "Reviewed class requires explicit typed evidence relocation")
        spec = _CLASSES[self.fixed_slot_ticker]
        _require(type(self.primary_sources) is tuple
                 and all(type(r) is ReviewedClassSourceRef for r in self.primary_sources)
                 and tuple(r.kind for r in self.primary_sources) == spec["sources"],
                 "Reviewed class original population missing, reordered or substituted")
        public = [r.validate(evidence_relocation=evidence_relocation) for r in self.primary_sources]
        observed = [_provider(ref, self.fixed_slot_ticker, spec, i, evidence_relocation)
                    for i, ref in enumerate((self.identity, self.modern_identity, self.latest_identity))]
        intended = _intended(self.intended_issue_source, self.intended_issue_sha256,
                              self.fixed_slot_ticker, spec, evidence_relocation)
        return dict(schema=SCHEMA, fixed_slot_ticker=self.fixed_slot_ticker, fixed_slot_index=spec["slot"],
            provider_ticker=self.provider_ticker, session_date=self.session_date,
            ordered_tickers_sha256=_ORDERED_SHA, provider_observations=observed,
            intended_issue_source=self.intended_issue_source, intended_issue_sha256=self.intended_issue_sha256,
            intended_observation=intended, primary_originals=public, relationship=spec["relationship"],
            relationship_details=io.parse_json(io.canonical(spec["details"])),
            proof_basis="reviewed-exact-original-class-evidence-not-nullable-identifier-equality",
            acquisition_route_only=True, historical_interval_interpolated=False,
            provider_identifiers_rewritten=False, provider_tickers_rewritten=False, raw_values_transformed=False,
            capture_time_is_historical_availability=False, query_clock_is_exchange_calendar_qualification=False,
            continuous_identity_qualified=False, historical_issue_identity_qualified=False,
            economic_conversion_qualified=False, economic_accounting_qualified=False, economic_inputs_qualified=False,
            tradability_qualified=False, point_in_time_qualified=False, native_catalog_activated=False,
            training_authorized=False, training_ready=False)
