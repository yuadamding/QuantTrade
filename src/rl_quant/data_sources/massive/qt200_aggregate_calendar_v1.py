"""Pinned aggregate-research calendar and explicit lagged input/execution clocks.

Reproduces every accepted predecessor session before extending its coverage.
Clock plans are not observed historical data availability or executable fills.
"""

from __future__ import annotations

import gzip
from datetime import datetime, time, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_daily_reference_v1 import _calendar

SCHEMA = "rl-quant.qt200-aggregate-research-calendar-v1"
PINNED = {"exchange-calendars": "4.13.2", "pyluach": "2.3.0", "toolz": "1.0.0",
          "korean-lunar-calendar": "0.3.1", "tzdata": "2025.2"}
START, OLD_END, END = "2017-01-03", "2026-08-26", "2026-08-31"
ET = ZoneInfo("America/New_York")


def clock_rows(sessions: list[dict]) -> list[dict]:
    """Feature inputs lag one session; execution window is strictly next-session."""
    result = []
    for index, session in enumerate(sessions):
        day = session["session_date"]
        decision = session["regular_close_ns"] + 1
        previous = sessions[index - 1] if index else None
        following = sessions[index + 1] if index + 1 < len(sessions) else None
        window = None
        if following is not None:
            execution_day = datetime.fromisoformat(following["session_date"]).date()
            stamps = [int(datetime.combine(execution_day, time(9, minute), ET).timestamp()) * 10**9
                      for minute in (35, 45)]
            if not decision < following["regular_open_ns"] < stamps[0] < stamps[1] < following["regular_close_ns"]:
                raise transport.ResearchCaptureError("Execution clock not strictly subsequent")
            window = dict(start_inclusive_ns=stamps[0], end_exclusive_ns=stamps[1])
        if previous is not None and not previous["regular_close_ns"] < decision:
            raise transport.ResearchCaptureError("Feature clock not strictly earlier")
        result.append(dict(decision_session=day, decision_after_close_ns=decision,
            latest_feature_session=previous["session_date"] if previous else None,
            latest_feature_close_ns=previous["regular_close_ns"] if previous else None,
            execution_session=following["session_date"] if following else None,
            execution_window=window, availability_is_modeling_assumption=True,
            historical_availability_observed=False, execution_inputs_observed=False,
            boundary_support_complete=previous is not None and following is not None,
            training_eligible=False))
    return result


def extend_calendar(*, predecessor: Path, predecessor_sha256: str, output: Path) -> dict:
    before = transport.read_regular(predecessor, 1048576)
    if transport.digest(before) != predecessor_sha256:
        raise transport.ResearchCaptureError("Predecessor calendar bytes differ")
    old = _calendar(before)
    if (old["coverage_start_date"] != START or old["coverage_end_date"] != OLD_END
            or old["exchange_calendars_version"] != PINNED["exchange-calendars"]):
        raise transport.ResearchCaptureError("Predecessor calendar profile differs")
    observed_versions = {name: version(name) for name in PINNED}
    if observed_versions != PINNED:
        raise transport.ResearchCaptureError("Calendar dependency versions differ")
    import exchange_calendars

    calendar = exchange_calendars.get_calendar("XNYS", start=START, end=END)
    sessions = []
    for session in calendar.sessions_in_range(START, END):
        opened, closed = int(calendar.session_open(session).value), int(calendar.session_close(session).value)
        sessions.append(dict(session_date=str(session.date()), exchange="XNYS",
            regular_open_ns=opened, regular_close_ns=closed,
            scheduled_five_minute_intervals=(closed - opened) // (300 * 10**9),
            special_session_reason="early-close" if closed - opened < 23400 * 10**9 else None))
    if [r for r in sessions if r["session_date"] <= OLD_END] != old["sessions"]:
        raise transport.ResearchCaptureError("Pinned calendar does not reproduce every prior session")
    added = [r["session_date"] for r in sessions if r["session_date"] > OLD_END]
    if len(sessions) != 2428 or added != ["2026-08-27", "2026-08-28", "2026-08-31"]:
        raise transport.ResearchCaptureError("Calendar extension population differs")
    # Reuse the established strict clock/serialization validator on the newly
    # generated rows; no native exchange/session authority is published.
    check = {**old, "sessions": sessions, "coverage_end_date": END}
    check.pop("receipt_sha256")
    check["receipt_sha256"] = transport.digest(transport.canonical(check))
    _calendar(transport.canonical(check))
    clocks = clock_rows(sessions)
    if transport.read_regular(predecessor, 1048576) != before:
        raise transport.ResearchCaptureError("Predecessor changed during calendar generation")
    output.mkdir(mode=0o700)
    artifacts = {
        "sessions.json.gz": dict(calendar="XNYS", sessions=sessions, versions=observed_versions),
        "decision-clock-plan.json.gz": dict(clocks=clocks, realized_execution_or_availability=False),
    }
    files = {name: transport.write_once(output / name, gzip.compress(transport.canonical(body), mtime=0))
             for name, body in artifacts.items()}
    result = dict(schema=SCHEMA, files=files, predecessor_sha256=predecessor_sha256,
        dependency_versions=observed_versions, coverage_start=START, coverage_end=END,
        sessions=len(sessions), added_sessions=added, predecessor_sessions_reproduced=len(old["sessions"]),
        feature_lag_sessions=1, execution_lag_sessions=1, generated_at=datetime.now(timezone.utc).isoformat(),
        historic_availability_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
