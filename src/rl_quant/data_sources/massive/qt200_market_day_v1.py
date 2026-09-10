"""Disk-backed, source-selected market-day processing; never V5 authorization.

The original whole gzip remains a mandatory durable replay dependency. This
bridge is not a native whole-market partition manifest or a cold-replay-ready
training dataset. All selected events, including errors and complete correction
chains, remain in the provisional SQLite spool; successful cleanup is a separate
caller-owned, receipt-gated operation. This module never deletes a spool.

Use one spool in a private empty attempt directory. SQLite indices are created
while empty, and large reads must use indexed order: temporary sort plans are
rejected. DELETE journaling, a fixed page cache, and a conservative page-count
bound cover database plus rollback journal inside the caller's allocation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
from zoneinfo import ZoneInfo
import zlib

from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.corrections import MassiveCorrectionAuthority
from rl_quant.data_sources.massive.finalized_listing import coverage_session_from_massive_trade_key
from rl_quant.data_sources.massive.selected_trade_scan_v1 import (
    MassiveSelectedOriginalTradeRowV1,
    MassiveSelectedTradeFileScanEvidenceV1,
    _SequenceDigest,
)
from rl_quant.data_sources.massive.session_calendar import MassiveExchangeSession, MassiveSessionAuthority
from rl_quant.data_sources.massive.trade_canonicalization import MassiveCanonicalTradeSourceRecord
from rl_quant.data_sources.massive.trade_extraction import MassiveExtractedTradeRow
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.protocol.canonical_artifact import canonical_json_payload, file_sha256, semantic_sha256


QT200_MARKET_DAY_V1_SCHEMA = "rl-quant.qt200-source-selected-market-day-v1"
QT200_MARKET_DAY_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
QT200_MARKET_DAY_V1_SPEC_SHA256 = semantic_sha256({
    "source": "complete-original-gzip-exact-ticker-selection-v1",
    "replay_key": "source-ticker,exchange,trf-or-minus-one,trade-id",
    "replay_order": "sip,sequence,exchange,trf,trade-id,physical-source-row",
    "regular_domain": "participant-time-in-[calendar-open,calendar-close)-after-full-replay",
    "bars": "native-V0-separate-open-close,high-low,volume-condition-populations",
    "legacy_tape": "native-V0-all-known-condition-terminal-active-regular-trades",
    "volume_flow": "native-V2-volume-forming-terminal-active-regular-trades",
    "price_volume_flow": "separately-labeled-price-and-volume-forming-terminal-active-regular-trades",
    "fill": "V5-[09:35,09:45)-ET-terminal-active-price-and-volume-forming",
    "size_quantiles": "exact-positive-decimal-order;median;nearest-rank-p90",
    "decimal_context": "precision28-half-even",
    "bad_ticker_day": "retain-all-events;zero-plus-false-output-masks",
    "identity_qualified": False, "whole_source_canonical_scan_qualified": False,
    "training_ready": False,
})
_DECIMAL_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN)
_ZERO = Decimal(0)
_MAX_I64 = (1 << 63) - 1
_DERIVED = (
    "ticker", "event_key", "kind", "participant", "sip", "sequence", "exchange_id",
    "trf_id", "trade_id", "regular", "price", "size", "size_digits", "size_whole",
    "size_fraction", "open_close", "high_low", "volume", "tape", "special",
    "native_receipt", "canonical_receipt", "issue",
)


class Qt200MarketDayError(ValueError):
    """A provisional market day cannot be sealed or used."""


def _owned_directory(path: Path) -> None:
    info = path.lstat()
    if (not path.is_absolute() or path.resolve() != path or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()):
        raise Qt200MarketDayError("attempt directory must be absolute, worker-owned and no-follow")


def _flow_add(flow: dict, price: Decimal, size: Decimal, receipt: str) -> None:
    prior = flow["last"]
    sign = (0 if prior is None else 1 if price > prior else -1 if price < prior else flow["sign"])
    value = price * size
    flow["count"] += 1
    flow["dollars"] += value
    flow["shares"] += size
    flow["signed"] += Decimal(sign) * value
    flow["first"] = price if prior is None else flow["first"]
    flow["last"] = price
    if sign:
        flow["sign"] = sign
    flow["inventory"].add(receipt)


def _flow() -> dict:
    return {"count": 0, "dollars": _ZERO, "shares": _ZERO, "signed": _ZERO,
            "first": None, "last": None, "sign": 0, "inventory": _SequenceDigest()}


def _flow_result(flow: dict, population: str, usable: bool = True) -> dict:
    valid = usable and flow["count"] > 0 and flow["dollars"] > 0
    return {"population": population, "valid": valid,
            "trade_count": flow["count"] if usable else 0,
            "share_volume": float(flow["shares"]) if valid else 0.0,
            "dollar_volume": float(flow["dollars"]) if valid else 0.0,
            "signed_dollar_flow": float(flow["signed"]) if valid else 0.0,
            "absolute_signed_flow_imbalance": float(abs(flow["signed"]) / flow["dollars"]) if valid else 0.0,
            "native_row_inventory_sha256": flow["inventory"].hexdigest()}


class Qt200MarketDaySpoolV1:
    """One process-local, bounded SQLite sink for one complete selected source day.

    ``append`` accepts provisional scanner callback rows. ``finalize`` requires
    the scanner's successful completion receipt and independently rereads every
    stored original/canonical row before replay. A returned report is still
    source-selected engineering data, not identity-qualified V5 input.
    The caller publishes the report, checks its retained source dependencies,
    and owns exact successful-spool cleanup and overall project quota.
    """

    def __init__(
        self, *, database_path: str | Path, ordered_tickers: Sequence[str],
        session_authority: MassiveSessionAuthority, session: MassiveExchangeSession,
        condition_authority: MassiveConditionAuthority, correction_authority: MassiveCorrectionAuthority,
        allocation_bytes: int, maximum_row_bytes: int = 8 * 1024 * 1024,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ):
        self.path = Path(database_path)
        _owned_directory(self.path.parent)
        if self.path.name in {"", ".", ".."} or any(self.path.parent.iterdir()):
            raise Qt200MarketDayError("spool requires a private empty attempt directory")
        if type(allocation_bytes) is not int or allocation_bytes < 4 * 1024 * 1024:
            raise Qt200MarketDayError("explicit spool allocation must allow at least four MiB")
        if type(maximum_row_bytes) is not int or maximum_row_bytes <= 0:
            raise Qt200MarketDayError("explicit row memory allocation is invalid")
        self.tickers = tuple(ordered_tickers)
        if (not self.tickers or len(set(self.tickers)) != len(self.tickers)
                or any(not isinstance(t, str) or not t or t != t.strip() for t in self.tickers)):
            raise Qt200MarketDayError("ordered ticker selection is malformed")
        session_authority.validate()
        condition_authority.validate()
        correction_authority.validate()
        if session.exchange != "XNYS" or session_authority.resolve(exchange="XNYS", session_date=session.session_date) != session:
            raise Qt200MarketDayError("session is not exactly authority-resolved XNYS")
        self.session, self.sessions = session, session_authority
        self.conditions, self.corrections = condition_authority, correction_authority
        self.rules = {rule.condition_id: rule for rule in condition_authority.rules}
        self.kinds = {rule.correction_code: rule.semantic_kind for rule in correction_authority.rules}
        self.allocation = allocation_bytes
        self.maximum_row_bytes = maximum_row_bytes
        self.progress = progress_callback
        self.count = 0
        self.previous = 1
        self.phase = "append"
        self.issues = {ticker: Counter() for ticker in self.tickers}
        self.samples = {ticker: [] for ticker in self.tickers}
        eastern = ZoneInfo("America/New_York")
        day = date.fromisoformat(session.session_date)
        self.day_start = int(datetime.combine(day, time(), eastern).timestamp()) * 1_000_000_000
        self.day_end = int(datetime.combine(day + timedelta(days=1), time(), eastern).timestamp()) * 1_000_000_000
        self.fill_start, self.fill_end = (value * 1_000_000 for value in adaptive_fill_clock_v1(session.session_date))
        if not session.regular_open_ns <= self.fill_start < self.fill_end <= session.regular_close_ns:
            raise Qt200MarketDayError("V5 fill window is outside the authority-resolved session")
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        info = os.fstat(descriptor)
        self.inode = (info.st_dev, info.st_ino)
        os.close(descriptor)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA page_size=4096")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA cache_size=-16384")
        self.connection.execute("PRAGMA mmap_size=0")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA automatic_index=OFF")
        # At most one third is DB pages, leaving two thirds for DELETE journal
        # and overhead. No WAL or external temporary database is permitted.
        self.connection.execute(f"PRAGMA max_page_count={allocation_bytes // (3 * 4096)}")
        self.connection.executescript("""
            CREATE TABLE events (
                ordinal INTEGER PRIMARY KEY, row_blob BLOB NOT NULL,
                ticker TEXT NOT NULL, event_key TEXT, kind TEXT, participant INTEGER,
                sip INTEGER, sequence INTEGER, exchange_id INTEGER, trf_id INTEGER,
                trade_id TEXT, regular INTEGER, price TEXT, size TEXT, size_digits INTEGER,
                size_whole TEXT, size_fraction TEXT, open_close INTEGER, high_low INTEGER,
                volume INTEGER, tape INTEGER, special INTEGER, native_receipt TEXT,
                canonical_receipt TEXT, issue TEXT, active INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX replay_order ON events(ticker,sip,sequence,exchange_id,trf_id,trade_id,ordinal);
            CREATE INDEX active_order ON events(ticker,participant,sip,sequence,ordinal) WHERE active=1 AND regular=1;
            CREATE INDEX size_order ON events(ticker,size_digits,size_whole,size_fraction,ordinal) WHERE active=1 AND regular=1;
            CREATE INDEX venue_order ON events(ticker,exchange_id,participant,sip,sequence,ordinal) WHERE active=1 AND regular=1;
            CREATE TABLE active_keys(event_key TEXT PRIMARY KEY, ordinal INTEGER NOT NULL);
            CREATE TABLE venue_sums(ticker TEXT,exchange_id INTEGER,participant INTEGER,sip INTEGER,
                                    sequence INTEGER,ordinal INTEGER,dollars TEXT,PRIMARY KEY(ticker,exchange_id));
            CREATE INDEX venue_first_order ON venue_sums(ticker,participant,sip,sequence,ordinal);
        """)
        self.connection.commit()
        self._check("created", 0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _check(self, phase: str, rows: int) -> None:
        _owned_directory(self.path.parent)
        total = 0
        allowed = {self.path.name, self.path.name + "-journal"}
        for path in self.path.parent.iterdir():
            info = path.lstat()
            if path.name not in allowed or not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise Qt200MarketDayError("unexpected or nonregular file in bounded spool attempt")
            if path == self.path and (info.st_dev, info.st_ino) != self.inode:
                raise Qt200MarketDayError("spool inode changed")
            total += info.st_size
        if total > self.allocation:
            raise Qt200MarketDayError("database plus journal exceed caller allocation")
        if self.progress is not None:
            self.progress(phase, rows, total)

    def _rows(self, sql: str, parameters=()):
        # A large temporary B-tree would evade the fixed page-cache bound.
        for plan in self.connection.execute("EXPLAIN QUERY PLAN " + sql, parameters):
            if "TEMP B-TREE" in plan[-1].upper() or "AUTOMATIC" in plan[-1].upper():
                raise Qt200MarketDayError("unbounded temporary query plan is forbidden")
        return self.connection.execute(sql, parameters)

    def _issue(self, ticker: str, code: str, ordinal: int) -> None:
        self.issues[ticker][code] += 1
        if len(self.samples[ticker]) < 6:
            self.samples[ticker].append({"source_row_number": ordinal, "issue": code})

    def _derive(self, row: MassiveSelectedOriginalTradeRowV1) -> tuple:
        ticker = row.original_values[0]
        if row.extracted_row is None:
            return (ticker,) + (None,) * (len(_DERIVED) - 2) + ("canonicalization_error",)
        trade = row.extracted_row.canonical_record
        integers = (trade.participant_timestamp_ns, trade.sip_timestamp_ns, trade.sequence_number,
                    trade.exchange_id, trade.trf_id, trade.tape_id)
        if any(value is not None and not 0 <= value <= _MAX_I64 for value in integers):
            return (ticker,) + (None,) * (len(_DERIVED) - 2) + ("numeric_field_outside_sqlite_int64",)
        kind = self.kinds.get(0 if trade.correction_code is None else trade.correction_code)
        issue = "unknown_correction" if kind is None else None
        rules = [self.rules.get(code) for code in trade.conditions]
        if any(rule is None for rule in rules):
            issue = "unknown_condition" if issue is None else "unknown_condition_and_correction"
            flags = (False, False, False)
        else:
            flags = (all(rule.updates_open_close for rule in rules), all(rule.updates_high_low for rule in rules),
                     all(rule.updates_volume for rule in rules))
        if not (self.day_start <= trade.participant_timestamp_ns < self.day_end
                and self.day_start <= trade.sip_timestamp_ns < self.day_end):
            issue = "source_calendar_date_mismatch" if issue is None else issue + "_and_bad_date"
        regular = self.session.regular_open_ns <= trade.participant_timestamp_ns < self.session.regular_close_ns
        whole, _, fraction = trade.size_decimal.partition(".")
        key = canonical_json_payload((ticker, trade.exchange_id, -1 if trade.trf_id is None else trade.trf_id, trade.trade_id)).decode("ascii")
        return (ticker, key, kind, trade.participant_timestamp_ns, trade.sip_timestamp_ns,
                trade.sequence_number, trade.exchange_id, -1 if trade.trf_id is None else trade.trf_id,
                trade.trade_id, int(regular), trade.price_decimal, trade.size_decimal, len(whole), whole,
                fraction, *(int(flag) for flag in flags), trade.tape_id, int(bool(trade.conditions)),
                row.extracted_row.receipt_sha256, trade.receipt_sha256, issue)

    def append(self, row: MassiveSelectedOriginalTradeRowV1) -> None:
        if self.phase != "append":
            raise Qt200MarketDayError("spool no longer accepts provisional rows")
        row.validate()
        if (row.original_values[0] not in self.issues or not self.previous < row.source_row_number <= _MAX_I64):
            raise Qt200MarketDayError("selected row is outside ordered scope or physical order")
        raw = canonical_json_payload(asdict(row))
        if len(raw) > self.maximum_row_bytes:
            raise Qt200MarketDayError("selected row exceeds explicit memory allocation")
        derived = self._derive(row)
        self.connection.execute(
            f"INSERT INTO events(ordinal,row_blob,{','.join(_DERIVED)}) VALUES({','.join('?' for _ in range(2 + len(_DERIVED)))})",
            (row.source_row_number, zlib.compress(raw, level=3), *derived),
        )
        self.previous = row.source_row_number
        self.count += 1
        if self.count % 1024 == 0:
            self.connection.commit()
            self._check("append", self.count)

    def _decode(self, blob: bytes) -> MassiveSelectedOriginalTradeRowV1:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(blob, self.maximum_row_bytes + 1)
        if len(raw) > self.maximum_row_bytes or not decoder.eof or decoder.unused_data:
            raise Qt200MarketDayError("bounded spool row compression is malformed")
        value = json.loads(raw)
        value["original_values"] = tuple(value["original_values"])
        if value["extracted_row"] is not None:
            extracted = value["extracted_row"]
            canonical = extracted["canonical_record"]
            canonical["conditions"] = tuple(canonical["conditions"])
            extracted["canonical_record"] = MassiveCanonicalTradeSourceRecord(**canonical)
            value["extracted_row"] = MassiveExtractedTradeRow(**extracted)
        row = MassiveSelectedOriginalTradeRowV1(**value)
        row.validate()
        return row

    def _reconcile(self, scan: MassiveSelectedTradeFileScanEvidenceV1) -> None:
        rows_hash, corrections_hash = _SequenceDigest(), _SequenceDigest()
        counts, invalid = dict.fromkeys(self.tickers, 0), dict.fromkeys(self.tickers, 0)
        seen = 0
        for stored in self._rows(f"SELECT ordinal,row_blob,{','.join(_DERIVED)} FROM events ORDER BY ordinal"):
            row = self._decode(stored[1])
            if row.source_row_number != stored[0] or self._derive(row) != tuple(stored[2:]):
                raise Qt200MarketDayError("spool fields differ from original bound row")
            ticker = row.original_values[0]
            if ticker not in counts:
                raise Qt200MarketDayError("spool ticker is outside completed selection")
            counts[ticker] += 1
            invalid[ticker] += row.extracted_row is None
            rows_hash.add(row.receipt_sha256)
            corrections_hash.add((row.source_row_number, ticker, row.original_values[1], row.original_values[2]))
            if stored[-1] is not None:
                self._issue(ticker, stored[-1], row.source_row_number)
            seen += 1
            if seen % 1024 == 0:
                self._check("reconcile", seen)
        if (seen != scan.selected_row_count or seen != self.count
                or tuple(counts.items()) != scan.selected_ticker_counts
                or tuple(invalid.items()) != scan.selected_ticker_canonicalization_error_counts
                or rows_hash.hexdigest() != scan.selected_row_inventory_sha256
                or corrections_hash.hexdigest() != scan.selected_correction_condition_inventory_sha256):
            raise Qt200MarketDayError("spool does not reconcile to complete original-gzip selection")

    def _replay(self):
        summaries = {ticker: {"timeline": _SequenceDigest(), "corrections": _SequenceDigest(),
                              "counts": Counter(), "regular_counts": Counter()} for ticker in self.tickers}
        cursor = self._rows("SELECT ordinal,ticker,event_key,kind,regular,native_receipt,canonical_receipt FROM events "
                            "INDEXED BY replay_order ORDER BY ticker,sip,sequence,exchange_id,trf_id,trade_id,ordinal")
        for index, (ordinal, ticker, key, kind, regular, native_sha, canonical_sha) in enumerate(cursor, 1):
            summary = summaries[ticker]
            previous = None
            applied = False
            if key is not None and kind is not None:
                prior = self.connection.execute("SELECT ordinal FROM active_keys WHERE event_key=?", (key,)).fetchone()
                previous = None if prior is None else prior[0]
                if kind in {"new-trade", "late-report"} and previous is not None:
                    self._issue(ticker, "conflicting_finalized_duplicate", ordinal)
                elif kind in {"replacement", "cancellation"} and previous is None:
                    self._issue(ticker, kind + "_lacks_predecessor", ordinal)
                else:
                    if previous is not None:
                        self.connection.execute("UPDATE events SET active=0 WHERE ordinal=?", (previous,))
                    if kind == "cancellation":
                        self.connection.execute("DELETE FROM active_keys WHERE event_key=?", (key,))
                    else:
                        self.connection.execute("INSERT OR REPLACE INTO active_keys VALUES(?,?)", (key, ordinal))
                        self.connection.execute("UPDATE events SET active=1 WHERE ordinal=?", (ordinal,))
                    applied = True
            summary["counts"][kind or "unknown-or-unparsed"] += 1
            if regular:
                summary["regular_counts"][kind or "unknown-or-unparsed"] += 1
            summary["timeline"].add((ordinal, key, kind, native_sha, previous, applied))
            if kind in {"replacement", "cancellation", "late-report"}:
                summary["corrections"].add({"source_row_number": ordinal, "correction_kind": kind,
                                            "event_key": tuple(json.loads(key)),
                                            "canonical_record_receipt_sha256": canonical_sha})
            if index % 1024 == 0:
                self._check("replay", index)
        return summaries

    def _size_at(self, ticker: str, offset: int) -> Decimal:
        self._check("quantile", offset)
        row = self._rows("SELECT size_whole,size_fraction FROM events INDEXED BY size_order "
                         "WHERE ticker=? AND active=1 AND regular=1 "
                         "ORDER BY size_digits,size_whole,size_fraction,ordinal LIMIT 1 OFFSET ?", (ticker, offset)).fetchone()
        if row is None:
            raise Qt200MarketDayError("exact size quantile lost active rows")
        return Decimal(row[0] + ("." + row[1] if row[1] else ""))

    def _aggregate(self, ticker: str, summary: dict) -> dict:
        legacy, volume_flow, price_volume_flow, fill = (_flow() for _ in range(4))
        opening = closing = high = low = None
        shares = dollars = large_dollars = trf_dollars = _ZERO
        volume_count = special_count = 0
        tape_dollars = {1: _ZERO, 2: _ZERO, 3: _ZERO}
        active_hash = _SequenceDigest()
        sql = ("SELECT ordinal,participant,price,size,trf_id,tape,special,open_close,high_low,volume,native_receipt "
               "FROM events INDEXED BY active_order WHERE ticker=? AND active=1 AND regular=1 "
               "ORDER BY participant,sip,sequence,ordinal")
        for ordinal, participant, raw_price, raw_size, trf, tape, special, oc, hl, vol, native_sha in self._rows(sql, (ticker,)):
            price, size = Decimal(raw_price), Decimal(raw_size)
            notional = price * size
            active_hash.add(native_sha)
            _flow_add(legacy, price, size, native_sha)
            if oc:
                opening = price if opening is None else opening
                closing = price
            if hl:
                high = price if high is None else max(high, price)
                low = price if low is None else min(low, price)
            if vol:
                volume_count += 1
                shares += size
                dollars += notional
                _flow_add(volume_flow, price, size, native_sha)
            if oc and vol:
                _flow_add(price_volume_flow, price, size, native_sha)
                if self.fill_start <= participant < self.fill_end:
                    _flow_add(fill, price, size, native_sha)
            if notional >= Decimal("100000"):
                large_dollars += notional
            if trf >= 0:
                trf_dollars += notional
            if tape in tape_dollars:
                tape_dollars[tape] += notional
            special_count += special
            if legacy["count"] % 1024 == 0:
                self._check("aggregate", ordinal)
        combined = opening is not None and high is not None
        bars = [opening or _ZERO, high or _ZERO, low or _ZERO, closing or _ZERO, shares, dollars,
                (high - low) / closing if combined else _ZERO,
                (_ZERO if not combined else Decimal("0.5") if high == low else (closing - low) / (high - low))]
        bar_mask = [opening is not None, high is not None, high is not None, opening is not None,
                    volume_count > 0, volume_count > 0, combined, combined]
        n, total = legacy["count"], legacy["dollars"]
        tape_values = [0.0] * len(MASSIVE_DAILY_TAPE_V0_FIELDS)
        if n:
            median = self._size_at(ticker, n // 2)
            if n % 2 == 0:
                median = (self._size_at(ticker, n // 2 - 1) + median) / Decimal(2)
            p90 = self._size_at(ticker, (9 * n + 9) // 10 - 1)
            entropy = largest = 0.0
            venue, venue_sum, first = None, _ZERO, None
            venue_rows = self._rows(
                "SELECT exchange_id,price,size,participant,sip,sequence,ordinal FROM events INDEXED BY venue_order "
                "WHERE ticker=? AND active=1 AND regular=1 ORDER BY exchange_id,participant,sip,sequence,ordinal", (ticker,),
            )
            for seen, (exchange, raw_price, raw_size, participant, sip, sequence, ordinal) in enumerate(venue_rows, 1):
                if venue is not None and exchange != venue:
                    self.connection.execute("INSERT INTO venue_sums VALUES(?,?,?,?,?,?,?)",
                                            (ticker, venue, *first, str(venue_sum)))
                    venue_sum = _ZERO
                if venue is None or exchange != venue:
                    first = (participant, sip, sequence, ordinal)
                venue = exchange
                venue_sum += Decimal(raw_price) * Decimal(raw_size)
                if seen % 1024 == 0:
                    self._check("venue", seen)
            self.connection.execute("INSERT INTO venue_sums VALUES(?,?,?,?,?,?,?)", (ticker, venue, *first, str(venue_sum)))
            # Native V0 accumulates venue shares in first-observed participant
            # order. Preserve that float summation order without a venue dict.
            for (raw_dollars,) in self._rows(
                "SELECT dollars FROM venue_sums INDEXED BY venue_first_order WHERE ticker=? "
                "ORDER BY participant,sip,sequence,ordinal", (ticker,),
            ):
                fraction = float(Decimal(raw_dollars) / total)
                if fraction > 0:
                    entropy -= fraction * math.log(fraction)
                largest = max(largest, fraction)
            signed = legacy["signed"]
            correction_count = sum(summary["counts"][kind] for kind in ("replacement", "cancellation", "late-report"))
            tape_values = [math.log1p(n), float(median), float(p90), float(large_dollars / total), float(signed),
                           float(abs(signed) / total), float((legacy["last"] - legacy["first"]) / abs(signed)) if signed else 0.0,
                           float(trf_dollars / total), entropy, largest,
                           *(float(tape_dollars[t] / total) for t in (1, 2, 3)), special_count / n, correction_count / n]
        usable = not self.issues[ticker]
        bars_float = [float(value) for value in bars]
        if not all(math.isfinite(value) for value in (*bars_float, *tape_values)):
            self._issue(ticker, "nonfinite_aggregate", 0)
            usable = False
        fill_valid = usable and fill["count"] > 0 and fill["shares"] > 0 and fill["dollars"] > 0
        result = {
            "ticker": ticker, "security_id": None, "identity_qualified": False,
            "source_selected_market_day_valid": usable, "observed_source_rows": sum(summary["counts"].values()),
            "errors": dict(self.issues[ticker]), "bounded_error_samples": self.samples[ticker],
            "event_counts": dict(summary["counts"]), "regular_participant_event_counts": dict(summary["regular_counts"]),
            "complete_timeline_inventory_sha256": summary["timeline"].hexdigest(),
            "native_correction_inventory_sha256": summary["corrections"].hexdigest(),
            "observed_terminal_active_regular_rows": n, "terminal_active_state_valid": usable,
            "terminal_active_regular_native_inventory_sha256": active_hash.hexdigest(),
            "bars_values": bars_float if usable else [0.0] * len(bars_float),
            "bars_valid": bar_mask if usable else [False] * len(bar_mask),
            "legacy_tape_values": tape_values if usable else [0.0] * len(tape_values),
            "legacy_tape_valid": [usable and n > 0] * len(tape_values),
            "volume_forming_flow": _flow_result(volume_flow, "native-V2-volume-forming", usable),
            "price_volume_forming_flow": _flow_result(price_volume_flow, "price-and-volume-forming", usable),
            "fill": {"window": "[09:35,09:45)-America/New_York", "valid": fill_valid,
                     "start_ns": self.fill_start, "end_ns": self.fill_end,
                     "vwap": float(fill["dollars"] / fill["shares"]) if fill_valid else 0.0,
                     "share_volume": float(fill["shares"]) if fill_valid else 0.0,
                     "dollar_volume": float(fill["dollars"]) if fill_valid else 0.0,
                     "trade_count": fill["count"] if usable else 0,
                     "native_row_inventory_sha256": fill["inventory"].hexdigest()},
            "training_ready": False,
        }
        # All derived public numbers must remain representable, even when a
        # nonauthorizing raw decimal lies outside float64's finite domain.
        try:
            canonical_json_payload(result)
        except ValueError as exc:
            raise Qt200MarketDayError("nonfinite derived flow or fill; day cannot seal") from exc
        return result

    def finalize(self, scan: MassiveSelectedTradeFileScanEvidenceV1) -> dict:
        if self.phase != "append":
            raise Qt200MarketDayError("spool may finalize only once")
        self.phase = "finalizing"
        scan.validate()
        if (scan.ordered_tickers != self.tickers
                or coverage_session_from_massive_trade_key(scan.loaded_source.receipt.source_object_key) != self.session.session_date):
            raise Qt200MarketDayError("completed source selection and market-day scope differ")
        self.connection.commit()
        self.connection.execute("BEGIN EXCLUSIVE")
        with localcontext(_DECIMAL_CONTEXT):
            self._reconcile(scan)
            summaries = self._replay()
            rows = [self._aggregate(ticker, summaries[ticker]) for ticker in self.tickers]
        self.connection.commit()
        self._check("finalized", self.count)
        self.close()
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != self.inode:
                raise Qt200MarketDayError("final spool path no longer names the processed database")
            physical = hashlib.sha256()
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                blocks = 0
                while block := handle.read(8 * 1024 * 1024):
                    physical.update(block)
                    blocks += 1
                    if blocks % 8 == 0:
                        self._check("hash", blocks * 8 * 1024 * 1024)
            after = os.fstat(descriptor)
            if any(getattr(before, name) != getattr(after, name) for name in (
                    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_gid", "st_mode", "st_nlink")):
                raise Qt200MarketDayError("final spool changed during physical hashing")
            spool_sha = physical.hexdigest()
        finally:
            os.close(descriptor)
        self._check("sealed", self.count)
        report = {
            "schema": QT200_MARKET_DAY_V1_SCHEMA, "session_date": self.session.session_date,
            "source_selected_processing_complete": True,
            "whole_source_canonical_scan_qualified": False, "identity_qualified": False,
            "native_daily_input_authority": False, "cold_replay_ready_without_original_gzip": False,
            "training_ready": False, "ordered_tickers": self.tickers,
            "selection_scan": asdict(scan), "selection_scan_receipt_sha256": scan.receipt_sha256,
            "session_authority_receipt_sha256": self.sessions.receipt_sha256,
            "condition_authority_receipt_sha256": self.conditions.receipt_sha256,
            "correction_authority_receipt_sha256": self.corrections.receipt_sha256,
            "implementation_sha256": QT200_MARKET_DAY_V1_SOURCE_SHA256,
            "spec_sha256": QT200_MARKET_DAY_V1_SPEC_SHA256,
            "bars_fields": MASSIVE_DAILY_BARS_V0_FIELDS, "legacy_tape_fields": MASSIVE_DAILY_TAPE_V0_FIELDS,
            "rows": rows, "row_inventory_sha256": semantic_sha256(rows),
            "spool": {"path": str(self.path), "sha256": spool_sha, "bytes": self.path.stat().st_size,
                      "allocation_bytes": self.allocation, "maximum_row_bytes": self.maximum_row_bytes,
                      "retained": True, "cleanup_authorized_by_library": False},
            "durable_replay_dependency": "original full gzip and its exact transaction; compact report is not a native partition archive",
        }
        report["receipt_sha256"] = semantic_sha256(report)
        self.phase = "complete"
        return report


__all__ = ["Qt200MarketDaySpoolV1", "Qt200MarketDayError", "QT200_MARKET_DAY_V1_SCHEMA",
           "QT200_MARKET_DAY_V1_SOURCE_SHA256", "QT200_MARKET_DAY_V1_SPEC_SHA256"]
