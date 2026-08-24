from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import math

import pytest

from rl_quant.alpha.accounting import EconomicPosition
from rl_quant.alpha.contracts import (
    CorporateActionKind,
    CorporateActionRecord,
    TerminalEventKind,
    TerminalEventRecord,
)
from rl_quant.features.massive_daily_bars_v0 import MASSIVE_DAILY_BARS_V0_FIELDS
from rl_quant.features.massive_daily_tape_v0 import MASSIVE_DAILY_TAPE_V0_FIELDS
from rl_quant.features.massive_economic_return_index_v1 import (
    MassiveEconomicReturnRowV1,
    build_massive_economic_event_authority_v1,
    build_massive_economic_return_rows_v1,
)
from rl_quant.features.massive_profitability_features_v1 import (
    BARS_MIN_V1_FIELDS,
    TAPE_MIN_V1_FIELDS,
    build_massive_profitability_feature_rows_v1,
)
from rl_quant.features.massive_session_panel_v1 import MassiveSessionPanelRowV1
from rl_quant.protocol.canonical_artifact import semantic_sha256
from rl_quant.protocol.massive_finalized_profitability_p0 import (
    MASSIVE_FINALIZED_PROFITABILITY_P0,
)


DIGEST = "a" * 64
DAY_MS = 86_400_000
BASE_MS = 1_704_067_200_000


def _date(index: int) -> str:
    return (date(2024, 1, 2) + timedelta(days=index)).isoformat()


def _panel_row(
    index: int,
    security_id: str = "SEC-A",
    *,
    close: float = 100.0,
    dollar_volume: float = 1_000_000.0,
    observed: bool = True,
    close_valid: bool = True,
    listed: bool = True,
    pit_member: bool = True,
    event_count: int = 10,
    replacement_count: int = 2,
    cancellation_count: int = 1,
    late_count: int = 3,
) -> MassiveSessionPanelRowV1:
    bars = {
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "share_volume": dollar_volume / close,
        "dollar_volume": dollar_volume,
        "high_low_range": 0.02,
        "close_location": 0.5,
    }
    tape = {
        "log_trade_count": math.log1p(100),
        "median_trade_size": 100.0,
        "p90_trade_size": 500.0,
        "large_trade_dollar_fraction": 0.2,
        "quote_free_signed_dollar_flow_proxy": dollar_volume * 0.1,
        "absolute_signed_flow_imbalance": 0.1,
        "price_response_per_signed_dollar": 99.0,
        "trf_off_exchange_dollar_fraction": 0.3,
        "venue_entropy": 1.2,
        "largest_venue_share": 0.4,
        "tape_a_fraction": 0.5,
        "tape_b_fraction": 0.3,
        "tape_c_fraction": 0.2,
        "special_condition_fraction": 0.7,
        "correction_replacement_fraction": 0.9,
    }
    bars_values = tuple(bars[name] for name in MASSIVE_DAILY_BARS_V0_FIELDS)
    tape_values = tuple(tape[name] for name in MASSIVE_DAILY_TAPE_V0_FIELDS)
    bars_valid = tuple(
        observed and (name != "close" or close_valid)
        for name in MASSIVE_DAILY_BARS_V0_FIELDS
    )
    tape_valid = (observed,) * len(MASSIVE_DAILY_TAPE_V0_FIELDS)
    if not observed:
        bars_values = (0.0,) * len(bars_values)
        tape_values = (0.0,) * len(tape_values)
        event_count = replacement_count = cancellation_count = late_count = 0
    regular_close_ns = (BASE_MS + index * DAY_MS + 20 * 3_600_000) * 1_000_000
    body = {
        "source_session_index": index,
        "source_session_date": _date(index),
        "regular_open_ns": regular_close_ns - 6 * 3_600_000 * 1_000_000,
        "regular_close_ns": regular_close_ns,
        "security_id": security_id,
        "pit_member": pit_member,
        "listed": listed,
        "tradable": pit_member and listed and observed and close_valid,
        "observed_regular_trade": observed,
        "halt_or_no_print": pit_member and listed and not observed,
        "bars_values": bars_values,
        "bars_valid": bars_valid,
        "tape_values": tape_values,
        "tape_valid": tape_valid,
        "event_timeline_count": event_count,
        "replacement_event_count": replacement_count,
        "cancellation_event_count": cancellation_count,
        "late_report_event_count": late_count,
        "daily_bars_row_receipt_sha256": DIGEST if observed else None,
        "daily_tape_row_receipt_sha256": "b" * 64 if observed else None,
        "event_counts_receipt_sha256": "c" * 64 if event_count else None,
    }
    provisional = MassiveSessionPanelRowV1(
        **body,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


def _economic_row(
    panel: MassiveSessionPanelRowV1,
    *,
    value: float,
    valid: bool = True,
) -> MassiveEconomicReturnRowV1:
    body = {
        "source_session_index": panel.source_session_index,
        "source_session_date": panel.source_session_date,
        "security_id": panel.security_id,
        "listed": panel.listed,
        "economic_value": value if valid else 0.0,
        "economic_value_valid": valid,
        "terminal": False,
        "position": EconomicPosition.from_mapping({panel.security_id: 1.0}),
        "applied_cash_return_receipts": (),
        "session_panel_row_receipt_sha256": panel.receipt_sha256,
        "economic_authority_receipt_sha256": "d" * 64,
    }
    provisional = MassiveEconomicReturnRowV1(
        **body,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        receipt_sha256=semantic_sha256(provisional.unsigned()),
    )
    result.validate()
    return result


def _value(row: object, names: tuple[str, ...], name: str) -> tuple[float, bool]:
    values = getattr(
        row, "bars_values" if names is BARS_MIN_V1_FIELDS else "tape_values"
    )
    valid = getattr(row, "bars_valid" if names is BARS_MIN_V1_FIELDS else "tape_valid")
    index = names.index(name)
    return values[index], valid[index]


def test_p0_protocol_is_research_only_with_two_session_staleness() -> None:
    protocol = MASSIVE_FINALIZED_PROFITABILITY_P0
    protocol.validate()
    assert protocol.source_staleness_sessions == 2
    assert protocol.minimum_vendor_lead_time_hours == 18
    assert protocol.production_equivalence is False
    assert protocol.panel_materialization_authorized is False
    assert protocol.predictive_training_authorized is False
    assert protocol.portfolio_evaluation_authorized is False


def test_exact_session_offsets_do_not_compress_a_missing_session() -> None:
    panels = tuple(_panel_row(index, observed=index != 30) for index in range(70))
    economics = tuple(
        _economic_row(
            panel,
            value=100.0 * (1.01**panel.source_session_index),
            valid=panel.source_session_index != 30,
        )
        for panel in panels
    )
    rows = build_massive_profitability_feature_rows_v1(
        panel_rows=panels,
        economic_rows=economics,
    )

    return_after_gap, valid_after_gap = _value(
        rows[31], BARS_MIN_V1_FIELDS, "economic_total_return_1"
    )
    next_return, next_valid = _value(
        rows[32], BARS_MIN_V1_FIELDS, "economic_total_return_1"
    )
    assert return_after_gap == 0.0
    assert valid_after_gap is False
    assert next_valid is True
    assert next_return == pytest.approx(0.01)
    assert rows[30].halt_or_no_print is True
    assert _value(rows[30], TAPE_MIN_V1_FIELDS, "log_trade_count") == (0.0, False)


def test_validity_masks_and_short_listing_history_remain_explicit() -> None:
    panels = tuple(
        _panel_row(
            index,
            close_valid=index != 64,
            listed=index >= 60,
            pit_member=index >= 60,
        )
        for index in range(65)
    )
    economics = tuple(
        _economic_row(
            panel,
            value=100.0 + panel.source_session_index,
            valid=panel.listed and panel.source_session_index != 64,
        )
        for panel in panels
    )
    rows = build_massive_profitability_feature_rows_v1(
        panel_rows=panels,
        economic_rows=economics,
    )
    final = rows[-1]

    assert final.tradable is False
    assert _value(final, BARS_MIN_V1_FIELDS, "economic_total_return_1") == (
        0.0,
        False,
    )
    assert _value(final, BARS_MIN_V1_FIELDS, "listing_age_sessions") == (
        5.0,
        True,
    )
    history_fraction, history_valid = _value(
        final, BARS_MIN_V1_FIELDS, "valid_history_fraction_63"
    )
    assert history_valid is True
    assert history_fraction == pytest.approx(4 / 63)


def test_tape_features_use_complete_event_timeline_denominators() -> None:
    panels = tuple(_panel_row(index) for index in range(2))
    economics = tuple(
        _economic_row(panel, value=100.0 + panel.source_session_index)
        for panel in panels
    )
    row = build_massive_profitability_feature_rows_v1(
        panel_rows=panels,
        economic_rows=economics,
    )[-1]

    assert _value(row, TAPE_MIN_V1_FIELDS, "signed_dollar_flow_fraction") == (
        pytest.approx(0.1),
        True,
    )
    assert _value(row, TAPE_MIN_V1_FIELDS, "replacement_event_fraction") == (
        pytest.approx(0.2),
        True,
    )
    assert _value(row, TAPE_MIN_V1_FIELDS, "cancellation_event_fraction") == (
        pytest.approx(0.1),
        True,
    )
    assert _value(row, TAPE_MIN_V1_FIELDS, "late_report_event_fraction") == (
        pytest.approx(0.3),
        True,
    )
    assert "special_condition_fraction" not in TAPE_MIN_V1_FIELDS
    assert "price_response_per_signed_dollar" not in TAPE_MIN_V1_FIELDS


def test_causal_economic_index_neutralizes_actions_and_carries_losses() -> None:
    securities = ("SEC-A", "SEC-B")
    panels: list[MassiveSessionPanelRowV1] = []
    closes = {
        "SEC-A": (100.0, 50.0, 50.0, 100.0, 100.0, 100.0, 1.0),
        "SEC-B": (20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0),
    }
    for index in range(7):
        for security_id in securities:
            panels.append(
                _panel_row(
                    index,
                    security_id,
                    close=closes[security_id][index],
                    close_valid=not (index == 6 and security_id in {"SEC-A", "SEC-B"}),
                    listed=not (security_id == "SEC-A" and index == 6),
                )
            )
    close_ms = {
        index: panels[index * len(securities)].regular_close_ns // 1_000_000
        for index in range(7)
    }
    actions = (
        CorporateActionRecord(
            event_id="split",
            security_id="SEC-A",
            kind=CorporateActionKind.SPLIT,
            effective_at_ms=close_ms[1] - 1,
            available_at_ms=close_ms[1] - 1,
            share_ratio=2.0,
        ),
        CorporateActionRecord(
            event_id="dividend",
            security_id="SEC-A",
            kind=CorporateActionKind.CASH_DIVIDEND,
            effective_at_ms=close_ms[2] - 1,
            available_at_ms=close_ms[2] - 1,
            cash_per_share=1.0,
        ),
        CorporateActionRecord(
            event_id="reverse-split",
            security_id="SEC-A",
            kind=CorporateActionKind.REVERSE_SPLIT,
            effective_at_ms=close_ms[3] - 2,
            available_at_ms=close_ms[3] - 2,
            share_ratio=0.5,
        ),
        CorporateActionRecord(
            event_id="ticker-transition",
            security_id="SEC-A",
            kind=CorporateActionKind.TICKER_EXCHANGE_CHANGE,
            effective_at_ms=close_ms[3] - 1,
            available_at_ms=close_ms[3] - 1,
        ),
        CorporateActionRecord(
            event_id="spin-off",
            security_id="SEC-A",
            kind=CorporateActionKind.SPIN_OFF,
            effective_at_ms=close_ms[4] - 1,
            available_at_ms=close_ms[4] - 1,
            successor_security_id="SEC-B",
            successor_ratio=0.5,
        ),
        CorporateActionRecord(
            event_id="cash-merger",
            security_id="SEC-A",
            kind=CorporateActionKind.MERGER_CASH,
            effective_at_ms=close_ms[5] - 1,
            available_at_ms=close_ms[5] - 1,
            cash_per_share=110.0,
        ),
    )
    terminals = (
        TerminalEventRecord(
            event_id="worthless-delisting",
            security_id="SEC-B",
            kind=TerminalEventKind.WORTHLESS,
            effective_at_ms=close_ms[6] - 1,
            available_at_ms=close_ms[6] - 1,
        ),
    )
    authority = build_massive_economic_event_authority_v1(
        security_ids=securities,
        corporate_actions=actions,
        terminal_events=terminals,
        cash_returns=(),
        source_receipts=(DIGEST,),
    )
    rows = build_massive_economic_return_rows_v1(
        panel_rows=tuple(panels),
        economic_authority=authority,
    )
    a = {row.source_session_index: row for row in rows if row.security_id == "SEC-A"}

    assert a[0].economic_value == pytest.approx(100.0)
    assert a[1].economic_value == pytest.approx(100.0)  # split is not -50%
    assert a[2].economic_value == pytest.approx(102.0)  # dividend included
    assert a[3].economic_value == pytest.approx(102.0)  # reverse split + ticker change
    assert a[4].economic_value == pytest.approx(112.0)  # spin-off value included
    assert a[5].economic_value == pytest.approx(122.0)  # cash merger + successor
    assert a[6].economic_value == pytest.approx(112.0)  # successor total loss
    assert a[6].economic_value_valid is True
    assert a[6].terminal is True
    feature_rows = build_massive_profitability_feature_rows_v1(
        panel_rows=tuple(panels),
        economic_rows=rows,
    )
    split_feature = next(
        row
        for row in feature_rows
        if row.security_id == "SEC-A" and row.source_session_index == 1
    )
    assert _value(split_feature, BARS_MIN_V1_FIELDS, "economic_total_return_1") == (
        pytest.approx(0.0),
        True,
    )


def test_future_economic_mutation_cannot_change_earlier_features() -> None:
    panels = tuple(_panel_row(index) for index in range(66))
    economics = tuple(
        _economic_row(panel, value=100.0 + panel.source_session_index)
        for panel in panels
    )
    before = build_massive_profitability_feature_rows_v1(
        panel_rows=panels,
        economic_rows=economics,
    )
    last = economics[-1]
    changed = replace(last, economic_value=999.0, receipt_sha256="0" * 64)
    changed = replace(
        changed,
        receipt_sha256=semantic_sha256(changed.unsigned()),
    )
    after = build_massive_profitability_feature_rows_v1(
        panel_rows=panels,
        economic_rows=(*economics[:-1], changed),
    )

    assert tuple(row.receipt_sha256 for row in before[:-1]) == tuple(
        row.receipt_sha256 for row in after[:-1]
    )
    assert before[-1].receipt_sha256 != after[-1].receipt_sha256
