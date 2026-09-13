"""Two explicit historical aliases, bound to original exact-date evidence.

This authorizes a source-acquisition route, not continuous issue identity,
tradability, economic accounting, or training. Provider bytes/tickers stay raw.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import gzip
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_reference_supplement_v1 import SupplementQuery, load_evidence
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

SCHEMA = "rl-quant.raw-second-exact-day-alias-v1"
# No general caller-supplied ticker/issuer allowlist. These are two reviewed
# same-issue ticker changes, not XOM-style successor accounting or ticker reuse.
_ISSUES = {
    ("META", "FB"): ("BBG000MM2P62", "BBG001SQCQC5", "0001326801", "2022-06-09"),
    ("ELV", "ANTM"): ("BBG000BCG930", "BBG001S6KBQ8", "0001156039", "2022-06-28"),
}
_EASTERN = ZoneInfo("America/New_York")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path, expected: str, cap: int = 1024 * 1024,
          evidence_relocation: EvidenceRelocation | None = None) -> bytes:
    if evidence_relocation is not None:
        _require(type(evidence_relocation) is EvidenceRelocation, "Expected an explicit typed evidence relocation")
        return evidence_relocation.read(path, expected, cap)
    raw = transport.read_regular(path, cap)
    _require(transport.digest(raw) == expected, "Alias source evidence changed")
    return raw


def _body(row: dict) -> dict:
    raw = base64.b64decode(row["raw_response_body_base64"], validate=True)
    _require(len(raw) <= 1024 * 1024 and len(raw) == row["raw_response_content_length"]
             and transport.digest(raw) == row["raw_response_body_sha256"],
             "Alias original response body changed")
    result = transport.parse_json(raw)
    _require(result.get("status") == "OK"
             and isinstance(result.get("request_id"), str) and result["request_id"]
             and result["request_id"] == row["provider_request_id"],
             "Unidentified alias reference response")
    return result


@dataclass(frozen=True)
class AliasIdentityRef:
    """Original dated wrapper, or original supplemental query COMPLETE chain."""

    kind: str
    path: str
    sha256: str

    def rows(self, *, fixed_slot: str, provider_ticker: str, session_date: str,
             evidence_relocation: EvidenceRelocation | None = None) -> list[dict]:
        if evidence_relocation is not None:
            _require(isinstance(self.path, str) and Path(self.path).is_absolute()
                     and str(Path(self.path)) == self.path, "Noncanonical original evidence reference")
        if self.kind == "dated-ticker-wrapper":
            wrapper = transport.parse_json(_read(Path(self.path), self.sha256,
                                                 evidence_relocation=evidence_relocation))
            _require(wrapper.get("ticker") == provider_ticker
                     and wrapper.get("date") == session_date and wrapper.get("http_status") == 200
                     and wrapper.get("request_url") ==
                     f"https://api.massive.com/v3/reference/tickers/{provider_ticker}?date={session_date}"
                     and wrapper.get("historical_known_at") is None
                     and wrapper.get("capture_time_is_historical_availability") is False,
                     "Alias identity must be an original exact-date provider query")
            payload = _body(wrapper)
            _require(payload.get("results") == wrapper.get("results")
                     and isinstance(payload.get("results"), dict), "Alias wrapper/body disagreement")
            return [payload["results"]]
        _require(self.kind == "reference-supplement", "Unsupported alias identity evidence kind")
        root = Path(self.path)
        complete = transport.parse_json(_read(root / "COMPLETE.json", self.sha256,
                                              evidence_relocation=evidence_relocation))
        query = SupplementQuery(**complete["query"])
        query.validate()
        _require(query.product in ("reference-alias", "reference-issuer", "reference-ticker")
                 and query.target == fixed_slot and query.start == query.end == session_date
                 and (query.product == "reference-issuer" or query.ticker == provider_ticker)
                 and complete.get("schema") == "rl-quant.qt200-reference-supplement-v1-query-complete"
                 and complete.get("pagination_complete") is True
                 and complete.get("point_in_time_qualified") is False
                 and complete.get("training_ready") is False,
                 "Alias supplement scope differs")
        proofs = complete["pages"]
        _require(isinstance(proofs, list) and 1 <= len(proofs) <= 8
                 and complete.get("page_count") == len(proofs), "Alias reference pagination missing")
        rows, seen, expected_url, last_received = [], set(), query.url, 0
        names = {"COMPLETE.json"}
        for index, proof in enumerate(proofs):
            receipt_name, body_name = f"page-{index:04d}.receipt.json", f"page-{index:04d}.json.gz"
            names.update((receipt_name, body_name))
            _require(proof.get("path") == receipt_name, "Alias receipt path escaped query")
            raw = _read(root / receipt_name, proof["sha256"], evidence_relocation=evidence_relocation)
            _require(len(raw) == proof["bytes"], "Alias receipt size differs")
            receipt = transport.parse_json(raw)
            _require(expected_url is not None and expected_url not in seen
                     and receipt.get("request_url") == expected_url
                     and receipt.get("http_status") == 200 and receipt.get("page_index") == index
                     and receipt.get("query") == asdict(query)
                     and receipt.get("plan_sha256") == complete["plan_sha256"]
                     and receipt.get("predecessor_page_sha256") ==
                     (proofs[index - 1]["sha256"] if index else None)
                     and receipt.get("capture_time_is_historical_availability") is False,
                     "Alias reference page chain differs")
            query.validate_url(expected_url)
            seen.add(expected_url)
            requested, received = receipt.get("requested_at_ns"), receipt.get("received_at_ns")
            _require(type(requested) is int and type(received) is int
                     and last_received <= requested <= received, "Alias capture chronology differs")
            last_received = received
            _require(receipt["body"].get("path") == body_name, "Alias body path escaped query")
            packed = _read(root / body_name, receipt["body"]["sha256"], transport.MAX_PAGE_BYTES + 1048576,
                           evidence_relocation=evidence_relocation)
            _require(len(packed) == receipt["body"]["bytes"], "Alias packed size differs")
            with gzip.GzipFile(fileobj=BytesIO(packed)) as stream:
                raw = stream.read(transport.MAX_PAGE_BYTES + 1)
                _require(len(raw) <= transport.MAX_PAGE_BYTES and not stream.read(1),
                         "Alias reference expanded size exceeds bound")
            _require(len(raw) == receipt["raw_body_bytes"]
                     and transport.digest(raw) == receipt["raw_body_sha256"], "Alias raw bytes differ")
            summary = query.inspect_page(raw)
            rows.extend(transport.parse_json(raw).get("results", []))
            expected_url = summary["next_url"]
        actual_names = ({p.name for p in root.iterdir()} if evidence_relocation is None
                        else evidence_relocation.names(root))
        _require(expected_url is None and complete.get("result_count") == len(rows)
                 and actual_names == names, "Alias reference coverage incomplete")
        return rows


def _exact_issue(rows: list[dict], provider_ticker: str, session_date: str,
                 issue: tuple[str, ...]) -> dict:
    matches = [row for row in rows if row.get("ticker") == provider_ticker]
    _require(len(matches) == 1, "Missing or ambiguous exact-date provider issue")
    row = matches[0]
    _require(row.get("type") == "CS" and row.get("active") is True
             and row.get("market") == "stocks" and row.get("locale") == "us"
             and all(isinstance(value, str) and value for value in issue[:3])
             and (row.get("composite_figi"), row.get("share_class_figi"), row.get("cik")) == issue[:3]
             and (row.get("list_date") is None or row["list_date"] <= session_date),
             "Exact-date reference does not identify the reviewed common-stock issue")
    return row


def validate_fixed_slot_identity(*, fixed_slot_ticker: str, provider_ticker: str,
                                  session_date: str, identity: AliasIdentityRef,
                                  intended_issue_source: Path, intended_issue_sha256: str) -> dict:
    """Ordinary ticker route against the frozen intended-issue observation.

    Same ticker is not enough: historical ETFs, other issuers, missing FIGIs
    and prelisting conflicts reject. Aliases require SecondAliasRoute instead.
    Matching a daily finalized response still does not establish continuous
    tradability or source vintage. No freely supplied expected-ID override.
    """
    _require(fixed_slot_ticker == provider_ticker and type(identity) is AliasIdentityRef
             and date.fromisoformat(session_date).isoformat() == session_date,
             "Fixed-slot identity requires an exact-date literal-ticker route")
    evidence = load_evidence(intended_issue_source, intended_issue_sha256)
    _require(evidence.get("schema") == "rl-quant.qt200-issue-resolution-evidence-v1",
             "Wrong intended-issue evidence schema")
    resolutions = evidence["security_resolutions"]
    _require([row["qt200_ticker"] for row in resolutions] == evidence["ordered_tickers"],
             "Intended issue population differs")
    matches = [row for row in resolutions if row["qt200_ticker"] == fixed_slot_ticker]
    _require(len(matches) == 1, "Fixed slot is not in the frozen QT200 issue population")
    intended = matches[0]["current_reference_evidence"]
    _require(intended.get("current_security_type") == "CS"
             and intended.get("active_exact_reference_count") == 1
             and intended.get("current_issue_identifier_observed") is True,
             "No unique intended common-stock issue observation")
    issue = (intended.get("current_composite_figi"), intended.get("current_share_class_figi"),
             intended.get("current_issuer_cik"))
    rows = identity.rows(fixed_slot=fixed_slot_ticker, provider_ticker=provider_ticker,
                         session_date=session_date)
    _exact_issue(rows, provider_ticker, session_date, issue)
    return dict(schema=SCHEMA + "-fixed-slot", fixed_slot_ticker=fixed_slot_ticker,
                provider_ticker=provider_ticker, session_date=session_date,
                composite_figi=issue[0], share_class_figi=issue[1], cik=issue[2],
                identity=asdict(identity), intended_issue_source=str(intended_issue_source),
                intended_issue_sha256=intended_issue_sha256, acquisition_route_only=True,
                continuous_identity_qualified=False, tradability_qualified=False,
                historical_issue_identity_qualified=False, point_in_time_qualified=False,
                training_ready=False)


@dataclass(frozen=True)
class AliasEventRef:
    path: str
    sha256: str
    row_index: int

    def interval(self, *, fixed_slot: str, provider_ticker: str, issue: tuple[str, ...],
                 evidence_relocation: EvidenceRelocation | None = None) -> tuple[str, str]:
        if evidence_relocation is not None:
            _require(isinstance(self.path, str) and Path(self.path).is_absolute()
                     and str(Path(self.path)) == self.path, "Noncanonical original evidence reference")
        capture = transport.parse_json(_read(Path(self.path), self.sha256, 16 * 1024 * 1024,
                                             evidence_relocation=evidence_relocation))
        _require(capture.get("schema") == "quanttrade-massive-ticker-events-capture-v1"
                 and type(self.row_index) is int and 0 <= self.row_index < len(capture["rows"]),
                 "Unsupported original ticker-event source")
        row = capture["rows"][self.row_index]
        payload = _body(row)
        result = payload["results"]
        _require(isinstance(result, dict)
                 and row.get("identifier") == issue[0] and row.get("response_status") == 200
                 and result.get("composite_figi") == issue[0] and result.get("cik") == issue[2],
                 "Ticker-event source identifies another issue")
        events = result["events"]
        _require(len(events) == row.get("event_count") and len(events) >= 2
                 and all(e.get("type") == "ticker_change"
                         and date.fromisoformat(e["date"]).isoformat() == e["date"] for e in events),
                 "Unknown alias event family or noncanonical event date")
        ordered = sorted((date.fromisoformat(e["date"]).isoformat(), e["ticker_change"]["ticker"])
                         for e in events)
        _require(len({day for day, _ in ordered}) == len(ordered), "Ambiguous alias event dates")
        positions = [i for i, (_, ticker) in enumerate(ordered) if ticker == provider_ticker]
        _require(len(positions) == 1 and positions[0] + 1 < len(ordered), "Unbounded alias event bracket")
        index = positions[0]
        _require(ordered[index + 1] == (issue[3], fixed_slot), "Unsupported alias transition")
        return ordered[index][0], ordered[index + 1][0]


@dataclass(frozen=True)
class SecondAliasRoute:
    fixed_slot_ticker: str
    provider_ticker: str
    session_date: str
    identity: AliasIdentityRef
    event: AliasEventRef

    @classmethod
    def from_dict(cls, value: dict) -> SecondAliasRoute:
        return cls(**{**value, "identity": AliasIdentityRef(**value["identity"]),
                      "event": AliasEventRef(**value["event"])})

    def validate(self, query: SecondQuery, *, evidence_relocation: EvidenceRelocation | None = None) -> dict:
        """Resolve only this exact date; never interpolate endpoint observations."""
        _require(type(self.identity) is AliasIdentityRef and type(self.event) is AliasEventRef,
                 "Alias evidence requires typed original references")
        _require(isinstance(query, SecondQuery), "Alias requires a validated integral-ms SecondQuery")
        issue = _ISSUES.get((self.fixed_slot_ticker, self.provider_ticker))
        _require(issue is not None, "Only reviewed META/FB and ELV/ANTM routes are supported")
        _require(date.fromisoformat(self.session_date).isoformat() == self.session_date
                 and query.ticker == self.provider_ticker, "Alias query identity differs")
        days = [datetime.fromtimestamp(stamp / 1000, timezone.utc).astimezone(_EASTERN).date().isoformat()
                for stamp in (query.start_ms, query.end_ms)]
        _require(days == [self.session_date, self.session_date], "Alias query escapes its exact observed date")
        start, end = self.event.interval(fixed_slot=self.fixed_slot_ticker,
                                        provider_ticker=self.provider_ticker, issue=issue,
                                        evidence_relocation=evidence_relocation)
        _require(start <= self.session_date < end, "Alias query outside sourced ticker-event bracket")
        rows = self.identity.rows(fixed_slot=self.fixed_slot_ticker, provider_ticker=self.provider_ticker,
                                  session_date=self.session_date, evidence_relocation=evidence_relocation)
        _exact_issue(rows, self.provider_ticker, self.session_date, issue)
        return dict(schema=SCHEMA, fixed_slot_ticker=self.fixed_slot_ticker,
                    provider_ticker=self.provider_ticker, session_date=self.session_date,
                    composite_figi=issue[0], share_class_figi=issue[1], cik=issue[2],
                    ticker_event_bracket_start=start, ticker_event_bracket_end_exclusive=end,
                    identity=asdict(self.identity), event=asdict(self.event),
                    acquisition_route_only=True, continuous_identity_qualified=False,
                    tradability_qualified=False, historical_issue_identity_qualified=False,
                    point_in_time_qualified=False, training_ready=False)
