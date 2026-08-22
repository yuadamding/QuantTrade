from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.data_sources.massive.websocket_capture import (
    MassiveDelayedWebSocketEvent,
    build_massive_delayed_websocket_capture_authority,
)
from rl_quant.evaluation.massive_replay_parity import (
    REQUIRED_MASSIVE_REPLAY_CANARIES,
    MassiveReplayParityError,
    MassiveReplayParityRow,
    build_massive_delayed_replay_authority,
)


def _row(*, feature_exact: bool = True) -> MassiveReplayParityRow:
    event_sha = "a" * 64
    delayed_feature = "b" * 64
    return MassiveReplayParityRow(
        security_id="SEC-A",
        session_date="2026-08-20",
        delayed_event_inventory_sha256=event_sha,
        finalized_replay_inventory_sha256=event_sha,
        delayed_feature_sha256=delayed_feature,
        finalized_feature_sha256=delayed_feature if feature_exact else "c" * 64,
        event_exact=True,
        feature_exact=feature_exact,
        failure_reason=None if feature_exact else "feature-mismatch",
    )


def _authority(*, row: MassiveReplayParityRow | None = None, canaries=REQUIRED_MASSIVE_REPLAY_CANARIES):
    return build_massive_delayed_replay_authority(
        (_row() if row is None else row,),
        entitlement_receipt_sha256="1" * 64,
        websocket_capture_receipts=("2" * 64,),
        finalized_flat_file_receipts=("3" * 64,),
        correction_semantics_receipt_sha256="4" * 64,
        condition_authority_receipt_sha256="5" * 64,
        canary_kinds_present=canaries,
    )


def test_exact_event_feature_and_canary_parity_authorizes_historical_replay_only() -> None:
    authority = _authority()

    assert authority.development_asof_replay_authorized
    assert authority.historical_asof_replay_authorized
    assert not authority.predictive_training_authorized
    authority.validate()


def test_feature_mismatch_or_missing_canary_blocks_historical_replay() -> None:
    mismatch = _authority(row=_row(feature_exact=False))
    missing = _authority(canaries=("normal-session",))

    assert not mismatch.historical_asof_replay_authorized
    assert mismatch.failed_feature_symbol_days == ("SEC-A:2026-08-20",)
    assert not missing.historical_asof_replay_authorized


def test_parity_flag_cannot_disagree_with_hashes() -> None:
    with pytest.raises(MassiveReplayParityError, match="flag differs"):
        replace(_row(), finalized_feature_sha256="c" * 64).validate()


def test_delayed_capture_is_secret_free_and_content_addressed() -> None:
    event = MassiveDelayedWebSocketEvent.from_payload(
        {
            "ev": "T",
            "sym": "AAPL",
            "t": 1_700_000_000_000,
            "q": 123,
            "p": 200.0,
            "s": 10,
        },
        received_at_ns=1_700_000_900_000_000_000,
    )
    capture = build_massive_delayed_websocket_capture_authority(
        (event,),
        session_date="2026-08-20",
        subscribed_tickers=("AAPL",),
        entitlement_receipt_sha256="d" * 64,
    )

    assert capture.event_count == 1
    assert not capture.secret_material_persisted
    capture.validate()
