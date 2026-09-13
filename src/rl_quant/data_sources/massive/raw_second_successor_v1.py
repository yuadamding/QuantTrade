"""One reviewed AVGO predecessor route, for acquisition only.

The old ordinary share is not assigned the successor's FIGIs. A reviewed
mandatory 1:1 exchange explains the issuer/issue change, but does not qualify
continuous identity, an economic conversion, observations, or training.

The semantic trust anchor is the two *exact original* SEC/Nasdaq documents
reviewed for this event. Caller-supplied mappings, report text, and matching
keywords cannot replace them. New public bytes require another source review
and code generation; hashes authenticate retained bytes, not the Internet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import load_evidence
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

SCHEMA = "rl-quant.raw-second-avgo-exact-day-successor-route-v1"
_EASTERN = ZoneInfo("America/New_York")
_OLD_CIK = "0001649338"
_NEW_ISSUE = ("BBG00KHY5S69", "BBG00KHY5SY8", "0001730168")
_FIRST_REVIEWED_DATE = "2017-01-03"
_MARKET_EFFECTIVE_DATE = "2018-04-05"
# Exact originals, not extracted prose or a user-authored decision receipt.
_PUBLIC_SOURCES = {
    "sec-20180404": (
        "https://www.sec.gov/Archives/edgar/data/1649338/000119312518107587/d548692d8k.htm",
        49_712,
        "dd5a6bf29b945d982449a0b471ad685257f80c71fdb2b220919f721b0053339e",
    ),
    "nasdaq-20180405": (
        "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2018-59",
        63_201,
        "fa36bd667d907a9ce09f0c33499117b7fd7ef3608c68eb77f2d087ee320d8d49",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _day(value: str) -> str:
    _require(type(value) is str and date.fromisoformat(value).isoformat() == value,
             "Successor observation date must be canonical")
    return value


def _mapping(value: dict, cls: type) -> dict:
    _require(type(value) is dict and set(value) == {field.name for field in fields(cls)},
             "Successor reference serialization fields differ")
    return value


@dataclass(frozen=True)
class ReviewedSuccessorSourceRef:
    kind: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: dict) -> ReviewedSuccessorSourceRef:
        return cls(**_mapping(value, cls))

    def validate(self, *, kind: str) -> dict:
        _require(self.kind == kind and kind in _PUBLIC_SOURCES,
                 "Unsupported successor primary source")
        url, size, expected = _PUBLIC_SOURCES[kind]
        _require(type(self.path) is str and self.sha256 == expected,
                 "Successor source is not the reviewed original")
        raw = transport.read_regular(Path(self.path), size)
        _require(len(raw) == size and transport.digest(raw) == expected,
                 "Successor original primary bytes changed")
        return dict(kind=kind, path=self.path, url=url, bytes=size, sha256=expected)


def _query_clock(query: SecondQuery, session_date: str) -> None:
    _require(isinstance(query, SecondQuery), "Successor route requires a SecondQuery")
    # Reconstruct to enforce the native integral-second contract even if an
    # object was deserialized unsafely. No numeric casts or timestamp rounding.
    SecondQuery(query.ticker, query.start_ms, query.end_ms)
    _require(query.ticker == "AVGO"
             and _FIRST_REVIEWED_DATE <= _day(session_date) < _MARKET_EFFECTIVE_DATE
             and query.end_ms - query.start_ms < 3_600_000,
             "Successor query outside the reviewed predecessor/date/hour scope")
    for stamp in (query.start_ms, query.end_ms):
        clock = datetime.fromtimestamp(stamp // 1000, timezone.utc).astimezone(_EASTERN)
        _require(clock.date().isoformat() == session_date
                 and time(9, 30) <= clock.time() < time(16),
                 "Successor query must remain within exact-date regular clock hours")
    # This is a clock bound, not a replacement for the corpus's session calendar.
    # In particular, no after-close April 4 timestamp is manufactured.


def _original_issue(identity: AliasIdentityRef, session_date: str, *, successor: bool) -> dict:
    _require(type(identity) is AliasIdentityRef and identity.kind == "dated-ticker-wrapper",
             "Successor route requires an original dated provider wrapper")
    rows = identity.rows(fixed_slot="AVGO", provider_ticker="AVGO", session_date=_day(session_date))
    _require(len(rows) == 1 and isinstance(rows[0], dict), "Ambiguous successor provider issue")
    row = rows[0]
    observed = (row.get("composite_figi"), row.get("share_class_figi"), row.get("cik"))
    expected = _NEW_ISSUE if successor else (None, None, _OLD_CIK)
    name = "Broadcom Inc. Common Stock" if successor else "Broadcom Limited Ordinary Shares"
    _require(row.get("ticker") == "AVGO" and row.get("name") == name
             and row.get("primary_exchange") == "XNAS" and row.get("type") == "CS"
             and row.get("market") == "stocks" and row.get("locale") == "us"
             and row.get("currency_name") == "usd" and row.get("active") is True
             and observed == expected,
             "Original provider response is not the reviewed predecessor/successor issue")
    if row.get("list_date") is not None:
        _require(_day(row["list_date"]) <= session_date, "Provider issue has a future listing date")
    # The source's list_date (including a carried 2009 AVGO date) is preserved,
    # but is not used to establish the April 2018 legal or marketplace boundary.
    return row


def _intended(source: str, expected_sha256: str) -> dict:
    _require(type(source) is str, "Intended issue source must have an explicit path")
    evidence = load_evidence(Path(source), expected_sha256)
    _require(evidence.get("schema") == "rl-quant.qt200-issue-resolution-evidence-v1",
             "Wrong intended-issue evidence schema")
    resolutions = evidence.get("security_resolutions")
    _require(isinstance(resolutions, list)
             and all(isinstance(row, dict) for row in resolutions)
             and [row.get("qt200_ticker") for row in resolutions] == evidence["ordered_tickers"],
             "Intended issue population differs")
    rows = [row for row in resolutions if row["qt200_ticker"] == "AVGO"]
    _require(len(rows) == 1, "AVGO is not a unique fixed slot")
    intended = rows[0].get("current_reference_evidence")
    _require(isinstance(intended, dict) and intended.get("ticker") == "AVGO"
             and intended.get("current_security_type") == "CS"
             and intended.get("current_primary_exchange") == "XNAS"
             and type(intended.get("active_exact_reference_count")) is int
             and intended["active_exact_reference_count"] == 1
             and intended.get("current_issue_identifier_observed") is True
             and (intended.get("current_composite_figi"), intended.get("current_share_class_figi"),
                  intended.get("current_issuer_cik")) == _NEW_ISSUE,
             "Frozen intended issue is not the reviewed Broadcom Inc. common stock")
    return intended


@dataclass(frozen=True)
class SecondSuccessorRoute:
    fixed_slot_ticker: str
    provider_ticker: str
    session_date: str
    identity: AliasIdentityRef
    successor_identity: AliasIdentityRef
    successor_observation_date: str
    intended_issue_source: str
    intended_issue_sha256: str
    sec_source: ReviewedSuccessorSourceRef
    nasdaq_source: ReviewedSuccessorSourceRef

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SecondSuccessorRoute:
        value = _mapping(value, cls)
        return cls(**{
            **value,
            "identity": AliasIdentityRef(**_mapping(value["identity"], AliasIdentityRef)),
            "successor_identity": AliasIdentityRef(
                **_mapping(value["successor_identity"], AliasIdentityRef)),
            "sec_source": ReviewedSuccessorSourceRef.from_dict(value["sec_source"]),
            "nasdaq_source": ReviewedSuccessorSourceRef.from_dict(value["nasdaq_source"]),
        })

    def validate(self, query: SecondQuery) -> dict:
        _require(self.fixed_slot_ticker == self.provider_ticker == "AVGO",
                 "Only the reviewed AVGO successor acquisition route is supported")
        _query_clock(query, self.session_date)
        _require(_day(self.successor_observation_date) >= _MARKET_EFFECTIVE_DATE,
                 "Successor observation predates the marketplace exchange")
        _require(type(self.sec_source) is ReviewedSuccessorSourceRef
                 and type(self.nasdaq_source) is ReviewedSuccessorSourceRef,
                 "Successor primary evidence requires typed references")
        public = [self.sec_source.validate(kind="sec-20180404"),
                  self.nasdaq_source.validate(kind="nasdaq-20180405")]
        old = _original_issue(self.identity, self.session_date, successor=False)
        new = _original_issue(self.successor_identity, self.successor_observation_date, successor=True)
        _intended(self.intended_issue_source, self.intended_issue_sha256)
        return dict(
            schema=SCHEMA, fixed_slot_ticker="AVGO", provider_ticker="AVGO",
            session_date=self.session_date, identity=asdict(self.identity),
            historical_provider_cik=old["cik"], historical_provider_name=old["name"],
            historical_provider_composite_figi=old.get("composite_figi"),
            historical_provider_share_class_figi=old.get("share_class_figi"),
            successor_identity=asdict(self.successor_identity),
            successor_observation_date=self.successor_observation_date,
            intended_issue_source=self.intended_issue_source,
            intended_issue_sha256=self.intended_issue_sha256,
            intended_cik=new["cik"], intended_composite_figi=new["composite_figi"],
            intended_share_class_figi=new["share_class_figi"],
            relationship="mandatory-one-for-one-successor-exchange",
            historical_issue_type="ordinary-shares", successor_issue_type="common-stock",
            original_public_cusip="Y09827109", successor_public_cusip="11135F101",
            exchange_ratio_numerator=1, exchange_ratio_denominator=1,
            legal_effective_boundary="after-market-close-2018-04-04",
            legal_effective_timestamp_ms=None, marketplace_effective_date=_MARKET_EFFECTIVE_DATE,
            primary_originals=public, acquisition_route_only=True,
            provider_identifiers_rewritten=False, provider_tickers_rewritten=False,
            query_clock_is_exchange_calendar_qualification=False,
            continuous_identity_qualified=False, historical_issue_identity_qualified=False,
            economic_conversion_qualified=False, economic_accounting_qualified=False,
            tradability_qualified=False, point_in_time_qualified=False, training_ready=False,
        )
