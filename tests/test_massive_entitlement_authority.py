from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from rl_quant.data_sources.massive.entitlement import (
    MASSIVE_ENTITLEMENT_EVIDENCE_KINDS,
    MassiveEntitlementError,
    MassiveEntitlementObservation,
    MassiveEntitlementSemanticEvidence,
    build_massive_developer_entitlement_authority,
    documented_massive_surface,
)
from rl_quant.workflows.massive_entitlement_canary import _parser


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
            surface_id="corporate-actions",
            request_path="/documented/corporate-actions",
            observed_at_ms=1_000,
        ),
        documented_massive_surface(
            surface_id="day-aggregates",
            request_path="/documented/day-aggregates",
            observed_at_ms=1_000,
        ),
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
        documented_massive_surface(
            surface_id="history-boundary",
            request_path="/documented/history-boundary",
            observed_at_ms=1_000,
        ),
        _observation("historical-quotes", "forbidden", 403),
        documented_massive_surface(
            surface_id="minute-aggregates",
            request_path="/documented/minute-aggregates",
            observed_at_ms=1_000,
        ),
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
    assert not authority.runtime_entitlement_qualified
    assert not authority.predictive_training_authorized
    assert not authority.historical_performance_authorized
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


def test_canary_credential_environment_name_cannot_drift() -> None:
    destinations = {action.dest for action in _parser()._actions}

    assert "api_key_env" not in destinations
    assert _authority().credential_source == "environment:MASSIVE_API_KEY"


def test_http_200_inventory_cannot_qualify_runtime_without_semantic_evidence() -> None:
    observations = tuple(
        _observation(surface, "available", 200)
        for surface in (
            "corporate-actions",
            "day-aggregates",
            "delayed-websocket",
            "financials-and-ratios",
            "flat-files",
            "historical-quotes",
            "history-boundary",
            "minute-aggregates",
            "reference-rest",
            "trades-rest",
        )
    )

    authority = build_massive_developer_entitlement_authority(
        observations, observed_at_ms=1_001
    )

    assert all(row.access_state == "available" for row in authority.observations)
    assert not authority.runtime_entitlement_qualified


def test_runtime_qualification_requires_bound_typed_semantic_evidence() -> None:
    required = tuple(sorted(MASSIVE_ENTITLEMENT_EVIDENCE_KINDS))
    evidence = tuple(
        MassiveEntitlementSemanticEvidence.build(
            surface_id=surface,
            evidence_kind=MASSIVE_ENTITLEMENT_EVIDENCE_KINDS[surface],
            source_receipts=(hashlib.sha256(surface.encode()).hexdigest(),),
            observed_schema_sha256=hashlib.sha256(
                f"schema:{surface}".encode()
            ).hexdigest(),
            result_count=1,
        )
        for surface in required
    )
    receipts = {row.surface_id: row.receipt_sha256 for row in evidence}
    observations = tuple(
        replace(
            _observation(surface, "available", 200),
            semantic_evidence_receipt_sha256=receipts.get(surface),
        )
        for surface in (
            "corporate-actions",
            "day-aggregates",
            "delayed-websocket",
            "financials-and-ratios",
            "flat-files",
            "historical-quotes",
            "history-boundary",
            "minute-aggregates",
            "reference-rest",
            "trades-rest",
        )
    )

    authority = build_massive_developer_entitlement_authority(
        observations,
        observed_at_ms=1_001,
        semantic_evidence=evidence,
    )

    assert authority.runtime_entitlement_qualified
