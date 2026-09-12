"""Calendar/clock regressions run only under the pinned remote LSF environment."""

from datetime import datetime

import pytest

from rl_quant.data_sources.massive.qt200_aggregate_calendar_v1 import ET, clock_rows
from rl_quant.data_sources.massive.qt200_research_capture_v1 import ResearchCaptureError


def _session(day, close="16:00"):
    return dict(session_date=day,
        regular_open_ns=int(datetime.fromisoformat(day + "T09:30").replace(tzinfo=ET).timestamp()) * 10**9,
        regular_close_ns=int(datetime.fromisoformat(day + "T" + close).replace(tzinfo=ET).timestamp()) * 10**9)


def test_weekend_dst_and_lag_do_not_compress_time():
    rows = clock_rows([_session("2026-03-05"), _session("2026-03-06"), _session("2026-03-09")])
    middle = rows[1]
    assert middle["latest_feature_session"] == "2026-03-05"
    assert middle["execution_session"] == "2026-03-09"
    assert middle["execution_window"]["end_exclusive_ns"] - middle["execution_window"]["start_inclusive_ns"] == 600 * 10**9
    assert middle["latest_feature_close_ns"] < middle["decision_after_close_ns"] < middle["execution_window"]["start_inclusive_ns"]
    assert all(r["training_eligible"] is r["historical_availability_observed"] is False for r in rows)
    assert rows[0]["boundary_support_complete"] is rows[-1]["boundary_support_complete"] is False
    assert rows[0]["latest_feature_session"] is rows[-1]["execution_session"] is None


def test_early_close_controls_decision_cutoff():
    rows = clock_rows([_session("2026-11-25"), _session("2026-11-27", "13:00"), _session("2026-11-30")])
    assert rows[1]["decision_after_close_ns"] == _session("2026-11-27", "13:00")["regular_close_ns"] + 1
    assert rows[1]["latest_feature_session"] == "2026-11-25"


def test_reversed_session_order_cannot_open_execution():
    with pytest.raises(ResearchCaptureError):
        clock_rows([_session("2026-03-09"), _session("2026-03-06")])
