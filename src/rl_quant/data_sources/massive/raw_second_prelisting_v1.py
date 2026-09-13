"""Eight Class-A intervals and one exact-session Class-C acquisition disposition.

This is a reason not to request a particular intended issue's market data,
not a provider-empty response, a capture, a tensor, or trading authorization.
The fixed QT200 axis is unchanged. A reused symbol may have real observations
for a different issuer (notably 2017 SNOW); none become intended-issue history.

The trust anchors are exact reviewed original SEC/Nasdaq bodies AND original
capture receipts, the retained 2022 provider wrappers, and the frozen intended
issue population. Matching keywords or caller-authored ID/date mappings are
not proof. These retrospective sources do not establish historical availability.
ROKU retains conflicting date statements; its cutoff is the earlier date, not
an adjudicated intraday first-trade instant. DELL Class C is supported ONLY on
2017-01-03; later when-issued/regular-way dates do not establish a whole missing
interval, and the separately traded Class-V tracker is never an alias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import load_evidence
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

SCHEMA = "rl-quant.raw-second-reviewed-prelisting-disposition-v1"
_EASTERN = ZoneInfo("America/New_York")
_REVIEWED_START = "2017-01-03"
_PROVIDER_OBSERVATION_DATE = "2022-01-03"
_INTENDED_SHA = "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0"
_ORDERED_TICKERS_SHA = "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"

# kind -> exact URL, body bytes/hash, original HTTP receipt hash.
# New source bytes require explicit review and another code/source generation.
_PUBLIC_SOURCES = {
    "pltr-sec-20201231": (
        "https://www.sec.gov/Archives/edgar/data/1321655/000119312521060650/d65934d10k.htm",
        1_856_891, "847a864869bca3e87c1206269ba27a28b07b4623413e8a38c957f4db67a0c61d",
        "a583bba599df995aa1ceb066855cf6aaa4e319af852fc6705ab3ed58662cfeeb"),
    "crwd-sec-securities-20200131": (
        "https://www.sec.gov/Archives/edgar/data/1535527/000153552720000006/crwdex44-descriptionof.htm",
        41_409, "ddce5cf4377d5a676d8df0a5e56531899af151986d7b3a2951e4fed1357cdd2d",
        "ad2812480b882c60bc9c0570098bf4452dfdf60a41adcbb6fc10736f26c9023c"),
    "ddog-sec-20191231": (
        "https://www.sec.gov/Archives/edgar/data/1561550/000156459020006422/ddog-10k_20191231.htm",
        4_876_570, "db82c79f1200f78b263481280a6677040a26ea17cbe0fa7a8c63f8a9b9fe8671",
        "1fcb26ca9f185d87f8604cf13f514b7ac3ac8435295b6212e7c117c7fbd3519e"),
    "snow-sec-20210131": (
        "https://www.sec.gov/Archives/edgar/data/1640147/000164014721000073/snow-20210131.htm",
        3_095_062, "af2c1b07631445594cebded417bd70eb4bf18385a23e7237b5a5ba69fa123750",
        "03337ab59f904cae89d110fe07b1287c8534933f5920f27d7e4e5abad4d175bb"),
    "roku-sec-20171231": (
        "https://www.sec.gov/Archives/edgar/data/1428439/000156459018004134/roku-10k_20171231.htm",
        3_948_178, "feec44cde1bfa63f20dd00f54486ae6a21e622ef35a3aaddb598518936412747",
        "8ce30a615957c861893b5906cab69f71d45c69a1177a013cc4902cebc107e929"),
    "roku-nasdaq-20170928": (
        "https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-welcomes-roku-inc-nasdaq-roku-nasdaq-stock-market",
        119_348, "2871c6c7b841b06568e86f3c76b8add68b228bff93e5ba663451cb7ff26bca1f",
        "834dc0ce36184ada351aa39f678a8df388cb6440ed0d3b06fc9600e88a3076c9"),
    "pins-sec-20191231": (
        "https://www.sec.gov/Archives/edgar/data/1506293/000150629320000013/pins-20191231.htm",
        3_152_419, "1eb0145981183938f8026cea3c8d037c7eb036e1a3112a5eb78d6cfa899e97b1",
        "31f27f2bca9020b9a5ec276b97c03d930ed791955caf1dc635dab0f49244bc98"),
    "abnb-sec-20201231": (
        "https://www.sec.gov/Archives/edgar/data/1559720/000155972021000010/airbnb-10k.htm",
        2_521_639, "38195ebef82f64d93c3f1def64bbeea8a147acbc9e5aa66ca3d0438bd55d735f",
        "db7faffc59162d14abcbfc141ae615f74010f4ecb2712984bdb1948c9e124c37"),
    "coin-sec-20211231": (
        "https://www.sec.gov/Archives/edgar/data/1679788/000167978822000031/coin-20211231.htm",
        3_556_916, "f08f550ef143b2e9b6fb34b822071d9054727853efab763c95baed36cb36b977",
        "aa01cb33bb860444e00f9718c33da1a87169c1af2dfa9b3b2ddc14bef23ff853"),
    "dell-sec-fy2017": (
        "https://www.sec.gov/Archives/edgar/data/1571996/000157199617000004/delltechnologiesfy1710k.htm",
        4_642_877, "faa8204b51330f0aa8b6d011d4f9acacef36299f10d63a2e8c15a315580285f2",
        "c261efd09d1042d32b8b8b0e3ffd3824463bbc1f234ece72253404376d077556"),
    "dell-sec-fy2019": (
        "https://www.sec.gov/Archives/edgar/data/1571996/000157199619000008/delltechnologiesfy1910k.htm",
        7_241_012, "c2e9bcf64b3a9a291bcea501f83b8eaba0e147834692fb0b115ae7b8a10b8d34",
        "459e34fa054eca388b53d00f9eefe71e047b1a1a205f399036c7b1f35153508d"),
}


@dataclass(frozen=True)
class _ListingRule:
    boundary: str
    cik: str
    composite_figi: str
    share_class_figi: str
    provider_name: str
    provider_exchange: str
    current_exchange: str
    provider_sha256: str
    source_kinds: tuple[str, ...]
    security_class: str = "Class A common stock"
    reviewed_dates: tuple[str, ...] = ()


_RULES = {
    "PLTR": _ListingRule("2020-09-30", "0001321655", "BBG000N7QR55", "BBG001T53796",
        "Palantir Technologies Inc. Class A Common Stock", "XNYS", "XNAS",
        "9a64e999f6588f0c0c68920d3e5d3481b0e22a3a4b5349bc2791a4e471c765a7", ("pltr-sec-20201231",)),
    "CRWD": _ListingRule("2019-06-12", "0001535527", "BBG00BLYKS03", "BBG00BLYKRZ7",
        "CrowdStrike Holdings, Inc. Class A Common Stock", "XNAS", "XNAS",
        "1a711222febfe60a09f088324fb0a85b6807287330231dd00f4276280cd4553a", ("crwd-sec-securities-20200131",)),
    "DDOG": _ListingRule("2019-09-19", "0001561550", "BBG003NJHZT9", "BBG003NJHZW5",
        "Datadog, Inc. Class A Common Stock", "XNAS", "XNAS",
        "1254c15ecccb7782f064adc036264ab67eba2c1ff8d763ebbb889485f914bd02", ("ddog-sec-20191231",)),
    "SNOW": _ListingRule("2020-09-16", "0001640147", "BBG007DHGNJ4", "BBG007DHGNK2",
        "Snowflake Inc.", "XNYS", "XNYS",
        "38bfc3331ad47db9154e4bf0416523aff459c1d31a6ff8e9ef57502219ace866", ("snow-sec-20210131",)),
    "ROKU": _ListingRule("2017-09-28", "0001428439", "BBG001ZZPQJ6", "BBG001ZZPQM2",
        "Roku, Inc. Class A Common Stock", "XNAS", "XNAS",
        "669cd981f45b074b62eabbe069e7da42a2e14eb1115a4dfead4abec38ecb224f",
        ("roku-sec-20171231", "roku-nasdaq-20170928")),
    "PINS": _ListingRule("2019-04-18", "0001506293", "BBG002583CV8", "BBG002583CW7",
        "Pinterest, Inc. Class A Common Stock", "XNYS", "XNYS",
        "cf7c3f8fd43cd9654d952a9e871d9e69fcaa336a7f0a7b87dd71a4b724fc47d4", ("pins-sec-20191231",)),
    "ABNB": _ListingRule("2020-12-10", "0001559720", "BBG001Y2XS07", "BBG001Y2XS16",
        "Airbnb, Inc. Class A Common Stock", "XNAS", "XNAS",
        "7f76fbfa1443243b1a1c16f7617b68ee114343585ada0888cdbf7462ddc20393", ("abnb-sec-20201231",)),
    "COIN": _ListingRule("2021-04-14", "0001679788", "BBG00ZGF7771", "BBG00ZGF7799",
        "Coinbase Global, Inc. Class A Common Stock", "XNAS", "XNAS",
        "c673395d465155021f74ec16e33f850603824f500622b3215dfbc4281e82982e", ("coin-sec-20211231",)),
    "DELL": _ListingRule("2018-12-26", "0001571996", "BBG00DW3SZS1", "BBG00DW3SZT0",
        "Dell Technologies Inc.", "XNYS", "XNYS",
        "dba313ec84042a83020156f6059d9b40e6981a9394f7632442d98f3f26e2028f",
        ("dell-sec-fy2017", "dell-sec-fy2019"), security_class="Class C common stock",
        reviewed_dates=("2017-01-03",)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _day(value: str) -> str:
    _require(type(value) is str and date.fromisoformat(value).isoformat() == value,
             "Prelisting date must be canonical")
    return value


def _mapping(value: dict, cls: type) -> dict:
    _require(type(value) is dict and set(value) == {field.name for field in fields(cls)},
             "Prelisting reference serialization fields differ")
    return value


@dataclass(frozen=True)
class ReviewedPrelistingSourceRef:
    kind: str
    path: str
    sha256: str
    receipt_path: str
    receipt_sha256: str

    @classmethod
    def from_dict(cls, value: dict) -> ReviewedPrelistingSourceRef:
        return cls(**_mapping(value, cls))

    def validate(self, *, kind: str) -> dict:
        _require(self.kind == kind and kind in _PUBLIC_SOURCES,
                 "Unsupported prelisting original source")
        url, size, expected, receipt_sha = _PUBLIC_SOURCES[kind]
        _require(type(self.path) is str and type(self.receipt_path) is str
                 and self.sha256 == expected and self.receipt_sha256 == receipt_sha,
                 "Prelisting source/receipt is not the reviewed original")
        raw = transport.read_regular(Path(self.path), size)
        receipt_raw = transport.read_regular(Path(self.receipt_path), 8192)
        _require(len(raw) == size and transport.digest(raw) == expected
                 and transport.digest(receipt_raw) == receipt_sha,
                 "Prelisting original bytes changed")
        receipt = transport.parse_json(receipt_raw)
        _require(receipt.get("source") == kind and receipt.get("request_url") == url
                 and receipt.get("final_url") == url and receipt.get("http_status") == 200
                 and receipt.get("original_response_complete") is True
                 and receipt.get("capture_success") is True and receipt.get("body_truncated") is False
                 and receipt.get("body") == dict(path=kind + ".html", bytes=size, sha256=expected)
                 and type(receipt.get("requested_at_ns")) is int
                 and type(receipt.get("received_at_ns")) is int
                 and 0 < receipt["requested_at_ns"] <= receipt["received_at_ns"],
                 "Primary source capture identity/completeness differs")
        return dict(kind=kind, path=self.path, url=url, bytes=size, sha256=expected,
                    receipt_path=self.receipt_path, receipt_sha256=receipt_sha,
                    capture_time_is_historical_availability=False)


def _query_clock(query: SecondQuery, ticker: str, session_date: str, boundary: str) -> None:
    _require(type(query) is SecondQuery, "Prelisting requires a literal SecondQuery")
    SecondQuery(query.ticker, query.start_ms, query.end_ms)
    _require(query.ticker == ticker and query.end_ms - query.start_ms < 3_600_000
             and _REVIEWED_START <= _day(session_date) < boundary,
             "Prelisting query outside reviewed date/issue/hour boundary")
    for stamp in (query.start_ms, query.end_ms):
        clock = datetime.fromtimestamp(stamp // 1000, timezone.utc).astimezone(_EASTERN)
        _require(clock.date().isoformat() == session_date and time(9, 30) <= clock.time() < time(16),
                 "Prelisting query must remain within exact-date regular clock hours")


def _intended(disposition: SecondPrelistingDisposition, rule: _ListingRule) -> int:
    _require(type(disposition.intended_issue_source) is str
             and disposition.intended_issue_sha256 == _INTENDED_SHA,
             "Prelisting intended issue must use the reviewed fixed population")
    evidence = load_evidence(Path(disposition.intended_issue_source), _INTENDED_SHA)
    _require(evidence.get("schema") == "rl-quant.qt200-issue-resolution-evidence-v1",
             "Wrong intended issue evidence schema")
    tickers = evidence["ordered_tickers"]
    resolutions = evidence.get("security_resolutions")
    _require(len(tickers) == len(set(tickers)) == 200 and isinstance(resolutions, list)
             and all(isinstance(row, dict) for row in resolutions)
             and [row.get("qt200_ticker") for row in resolutions] == tickers,
             "Prelisting must preserve the complete fixed QT200 axis")
    # load_evidence checks the exact native SYMBOLS tuple; bind its canonical
    # ordered-list identity in the returned proof as well, not a compacted axis.
    _require(transport.digest(transport.canonical(tickers)) == _ORDERED_TICKERS_SHA,
             "Fixed QT200 ordered-list identity differs")
    slot = tickers.index(disposition.fixed_slot_ticker)
    intended = resolutions[slot].get("current_reference_evidence")
    _require(isinstance(intended, dict) and intended.get("ticker") == disposition.fixed_slot_ticker
             and intended.get("current_security_type") == "CS"
             and intended.get("current_primary_exchange") == rule.current_exchange
             and type(intended.get("active_exact_reference_count")) is int
             and intended["active_exact_reference_count"] == 1
             and intended.get("current_issue_identifier_observed") is True
             and (intended.get("current_issuer_cik"), intended.get("current_composite_figi"),
                  intended.get("current_share_class_figi")) ==
                 (rule.cik, rule.composite_figi, rule.share_class_figi),
             "Prelisting intended issue/class identity differs")
    return slot


@dataclass(frozen=True)
class SecondPrelistingDisposition:
    fixed_slot_ticker: str
    session_date: str
    public_trading_start_date: str
    identity: AliasIdentityRef
    identity_observation_date: str
    intended_issue_source: str
    intended_issue_sha256: str
    primary_sources: tuple[ReviewedPrelistingSourceRef, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["primary_sources"] = [asdict(source) for source in self.primary_sources]
        return value

    @classmethod
    def from_dict(cls, value: dict) -> SecondPrelistingDisposition:
        value = _mapping(value, cls)
        _require(type(value["primary_sources"]) is list, "Prelisting sources must serialize as a list")
        return cls(**{**value, "identity": AliasIdentityRef(**_mapping(value["identity"], AliasIdentityRef)),
                      "primary_sources": tuple(ReviewedPrelistingSourceRef.from_dict(source)
                                               for source in value["primary_sources"])})

    def validate(self, query: SecondQuery) -> dict:
        _require(type(self.fixed_slot_ticker) is str and self.fixed_slot_ticker in _RULES,
                 "No reviewed prelisting rule for this intended issue")
        rule = _RULES[self.fixed_slot_ticker]
        _require(_day(self.public_trading_start_date) == rule.boundary,
                 "Caller cannot move the reviewed public-trading boundary")
        _query_clock(query, self.fixed_slot_ticker, self.session_date, rule.boundary)
        _require(not rule.reviewed_dates or self.session_date in rule.reviewed_dates,
                 "This class has only an exact reviewed-session disposition, not an interpolated interval")
        _require(type(self.primary_sources) is tuple
                 and all(type(source) is ReviewedPrelistingSourceRef for source in self.primary_sources)
                 and tuple(source.kind for source in self.primary_sources) == rule.source_kinds,
                 "Missing, reordered, or unrelated affirmative listing evidence")
        public = [source.validate(kind=kind) for source, kind in zip(self.primary_sources, rule.source_kinds)]
        slot = _intended(self, rule)
        _require(type(self.identity) is AliasIdentityRef and self.identity.kind == "dated-ticker-wrapper"
                 and self.identity.sha256 == rule.provider_sha256
                 and self.identity_observation_date == _PROVIDER_OBSERVATION_DATE,
                 "Prelisting requires the reviewed original intended-issue provider observation")
        rows = self.identity.rows(fixed_slot=self.fixed_slot_ticker, provider_ticker=self.fixed_slot_ticker,
                                  session_date=_PROVIDER_OBSERVATION_DATE)
        _require(len(rows) == 1 and isinstance(rows[0], dict), "Ambiguous intended issue provider response")
        row = rows[0]
        _require(row.get("ticker") == self.fixed_slot_ticker and row.get("name") == rule.provider_name
                 and row.get("type") == "CS" and row.get("active") is True
                 and row.get("market") == "stocks" and row.get("locale") == "us"
                 and row.get("currency_name") == "usd" and row.get("primary_exchange") == rule.provider_exchange
                 and (row.get("cik"), row.get("composite_figi"), row.get("share_class_figi")) ==
                     (rule.cik, rule.composite_figi, rule.share_class_figi),
                 "Provider observation is not the reviewed intended security class")
        unavailable_end = ((date.fromisoformat(self.session_date) + timedelta(days=1)).isoformat()
                           if rule.reviewed_dates else rule.boundary)
        return dict(
            schema=SCHEMA, **self.to_dict(), query=asdict(query), fixed_slot_index=slot,
            fixed_axis_size=200, ordered_tickers_sha256=_ORDERED_TICKERS_SHA,
            intended_cik=rule.cik, intended_composite_figi=rule.composite_figi,
            intended_share_class_figi=rule.share_class_figi, intended_security_class=rule.security_class,
            intended_provider_name=row["name"], provider_list_date=row.get("list_date"),
            reviewed_start_date=_REVIEWED_START, unavailable_end_date_exclusive=unavailable_end,
            reviewed_session_dates=list(rule.reviewed_dates),
            disposition_scope=("exact-reviewed-session" if rule.reviewed_dates else "bounded-pre-public-interval"),
            public_trading_start_date_is_exclusion_interval_boundary=not bool(rule.reviewed_dates),
            later_trading_dates_are_corroboration_only=bool(rule.reviewed_dates),
            public_trading_conventions=({"when_issued": "2018-12-26", "regular_way": "2018-12-28"}
                                       if self.fixed_slot_ticker == "DELL" else {}),
            tracking_stock_alias_authorized=False,
            boundary_intraday_timestamp_ms=None,
            boundary_conflicting_source_dates=(["2017-09-28", "2017-09-29"] if self.fixed_slot_ticker == "ROKU" else []),
            boundary_policy=("reviewed-exact-session-with-later-listing-corroboration" if rule.reviewed_dates else
                             "conservative-earliest-source-date-with-Nasdaq-corroboration"
                             if self.fixed_slot_ticker == "ROKU" else "reviewed-first-public-trading-date"),
            primary_originals=public, reason="intended-class-before-public-trading",
            market_observation_status="known-unavailable", acquisition_required=False,
            fixed_axis_slot_preserved=True, unavailable_mask_required=True, known_empty_interval=False,
            synthetic_provider_response_created=False, synthetic_market_records_created=False,
            market_tensor_constructed=False, provider_capture_reference=None,
            no_same_ticker_records_claimed=False, private_security_nonexistence_claimed=False,
            all_historical_listing_statuses_qualified=False, point_in_time_qualified=False,
            first_trading_day_intraday_qualified=False, query_clock_is_exchange_calendar_qualification=False,
            continuous_identity_qualified=False, economic_accounting_qualified=False,
            tradability_qualified=False, native_catalog_activated=False,
            acquisition_disposition_only=True, training_authorized=False, training_ready=False,
        )
