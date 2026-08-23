from __future__ import annotations

import gzip
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive.source_receipts import (
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.data_sources.massive.trade_extraction import (
    MassiveTradeExtractionError,
    extract_massive_flat_file_security_session,
)
from rl_quant.data_sources.massive.trade_replay import MassiveResolvedSecurityIdentity
from test_massive_trade_replay import _entitlement


HEADER = (
    "ticker,conditions,correction,exchange,id,participant_timestamp,price,"
    "sequence_number,sip_timestamp,size,tape,trf_id,trf_timestamp\n"
)


def _loaded(tmp_path: Path, csv_payload: str):
    payload = gzip.compress(csv_payload.encode())
    entitlement = _entitlement()
    publish_massive_source_object(
        stream=BytesIO(payload),
        root=tmp_path,
        relative_payload_path="trades/day.csv.gz",
        dataset_id="us_stocks_sip/trades_v1",
        source_object_key="us_stocks_sip/trades_v1/2026/08/2026-08-20.csv.gz",
        requested_at_ms=1,
        downloaded_at_ms=2,
        committed_at_ms=3,
        schema_sha256="a" * 64,
        entitlement_receipt_sha256=entitlement.receipt_sha256,
    )
    return load_massive_source_bundle(
        root=tmp_path,
        relative_payload_path="trades/day.csv.gz",
        verified_at_ms=4,
    )


def _identity():
    return MassiveResolvedSecurityIdentity.build(
        security_id="SEC-A",
        source_ticker="AAA",
        primary_exchange="XNYS",
        session_date="2026-08-20",
        valid_from_ns=0,
        valid_to_ns=None,
        identity_authority_receipt_sha256="b" * 64,
        ticker_history_receipt_sha256="c" * 64,
    )


def test_flat_file_extraction_reads_every_committed_row(tmp_path: Path) -> None:
    sip = int(
        datetime(
            2026, 8, 20, 10, 0, tzinfo=ZoneInfo("America/New_York")
        ).timestamp()
        * 1_000_000_000
    )
    loaded = _loaded(
        tmp_path,
        HEADER
        + f"AAA,[1],0,4,T1,{sip - 1},10,1,{sip},100,1,,\n"
        + f"BBB,[1],0,4,T2,{sip - 1},20,1,{sip},200,1,,\n",
    )

    records, evidence = extract_massive_flat_file_security_session(
        root=tmp_path,
        loaded_source=loaded,
        identity_resolution=_identity(),
    )

    assert len(records) == 1
    assert evidence.source_row_count == 2
    assert evidence.selected_row_count == 1
    assert evidence.unselected_row_count == 1
    assert evidence.complete_for_security_session


def test_flat_file_extraction_rejects_schema_drift(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path, "ticker,price\nAAA,10\n")

    with pytest.raises(MassiveTradeExtractionError, match="schema"):
        extract_massive_flat_file_security_session(
            root=tmp_path,
            loaded_source=loaded,
            identity_resolution=_identity(),
        )
