from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from rl_quant.data_sources.massive.entitlement import (
    MassiveEntitlementError,
    MassiveEntitlementObservation,
    build_massive_developer_entitlement_authority,
    documented_massive_surface,
)


def _observation(
    surface_id: str, access_state: str, status: int | None
) -> MassiveEntitlementObservation:
    body = b"{}" if status is not None else b""
    return MassiveEntitlementObservation(
        surface_id=surface_id,
        request_path=f"/probe/{surface_id}",
        observed_at_ms=1_000,
        access_state=access_state,  # type: ignore[arg-type]
        http_status=status,
        response_content_length=len(body),
        response_body_sha256=hashlib.sha256(body).hexdigest(),
        request_id=None,
    )


def _authority():
    observations = (
        documented_massive_surface(
            surface_id="delayed-websocket",
            request_path="/documented/delayed-websocket",
            observed_at_ms=1_000,
        ),
        documented_massive_surface(
            surface_id="financials-and-ratios",
            request_path="/documented/financials-and-ratios",
            observed_at_ms=1_000,
        ),
        documented_massive_surface(
            surface_id="flat-files",
            request_path="/documented/flat-files",
            observed_at_ms=1_000,
        ),
        _observation("historical-quotes", "forbidden", 403),
        _observation("reference-rest", "available", 200),
        _observation("trades-rest", "available", 200),
    )
    return build_massive_developer_entitlement_authority(
        observations, observed_at_ms=1_001
    )


def test_developer_entitlement_is_secret_free_and_non_authorizing() -> None:
    authority = _authority()
    payload = json.dumps(asdict(authority), sort_keys=True)

    assert authority.trades_rest_available
    assert authority.delayed_websocket_documented
    assert authority.flat_files_documented
    assert not authority.historical_quotes_available
    assert not authority.financials_and_ratios_available
    assert not authority.predictive_training_authorized
    assert not authority.historical_performance_authorized
    assert "7zwE_" not in payload
    assert "apiKey=" not in payload
    authority.validate()


def test_entitlement_rejects_credential_in_request_path() -> None:
    with pytest.raises(MassiveEntitlementError, match="credential"):
        replace(
            _observation("trades-rest", "available", 200),
            request_path="/v3/trades/AAPL?apiKey=secret",
        ).validate()


def test_entitlement_receipt_detects_mutation() -> None:
    with pytest.raises(MassiveEntitlementError, match="drifted|differs"):
        replace(_authority(), history_years=11).validate()
