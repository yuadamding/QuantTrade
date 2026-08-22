"""Secret-free evidence for the observed Massive Stocks access surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Literal, Sequence
from urllib import error, request

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_ENTITLEMENT_OBSERVATION_SCHEMA = (
    "rl-quant.massive-entitlement-observation-v1"
)
MASSIVE_ENTITLEMENT_AUTHORITY_SCHEMA = "rl-quant.massive-entitlement-authority-v1"

MassiveAccessState = Literal[
    "available",
    "forbidden",
    "not-found",
    "transport-failed",
    "documented-not-runtime-probed",
]


class MassiveEntitlementError(ValueError):
    """An entitlement observation is incomplete, secret-bearing, or inconsistent."""


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise MassiveEntitlementError(f"{name} must be a canonical nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveEntitlementError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveEntitlementError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class MassiveEntitlementObservation:
    surface_id: str
    request_path: str
    observed_at_ms: int
    access_state: MassiveAccessState
    http_status: int | None
    response_content_length: int
    response_body_sha256: str
    request_id: str | None
    schema: str = MASSIVE_ENTITLEMENT_OBSERVATION_SCHEMA

    def validate(self) -> None:
        if self.schema != MASSIVE_ENTITLEMENT_OBSERVATION_SCHEMA:
            raise MassiveEntitlementError("entitlement observation schema drifted")
        _text("surface ID", self.surface_id)
        path = _text("request path", self.request_path)
        if not path.startswith("/") or "apiKey=" in path or "apikey=" in path.lower():
            raise MassiveEntitlementError(
                "request path must be relative and contain no credential"
            )
        _nonnegative_int("observation timestamp", self.observed_at_ms)
        if self.access_state not in {
            "available",
            "forbidden",
            "not-found",
            "transport-failed",
            "documented-not-runtime-probed",
        }:
            raise MassiveEntitlementError("entitlement access state is unsupported")
        if self.http_status is not None:
            if (
                isinstance(self.http_status, bool)
                or not isinstance(self.http_status, int)
                or not 100 <= self.http_status <= 599
            ):
                raise MassiveEntitlementError("HTTP status is invalid")
        if self.access_state == "available" and self.http_status != 200:
            raise MassiveEntitlementError("available observations require HTTP 200")
        if self.access_state == "forbidden" and self.http_status not in {401, 403}:
            raise MassiveEntitlementError("forbidden observations require 401 or 403")
        if self.access_state == "not-found" and self.http_status != 404:
            raise MassiveEntitlementError("not-found observations require HTTP 404")
        if self.access_state == "documented-not-runtime-probed" and self.http_status is not None:
            raise MassiveEntitlementError("unprobed surfaces cannot claim an HTTP status")
        _nonnegative_int("response content length", self.response_content_length)
        _digest("response body SHA", self.response_body_sha256)
        if self.request_id is not None:
            _text("request ID", self.request_id)

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return semantic_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MassiveEntitlementAuthority:
    plan_id: str
    observed_at_ms: int
    credential_source: str
    observations: tuple[MassiveEntitlementObservation, ...]
    trades_rest_available: bool
    delayed_websocket_documented: bool
    flat_files_documented: bool
    historical_quotes_available: bool
    financials_and_ratios_available: bool
    history_years: int
    entitlement_delay_minutes: int
    secret_material_persisted: bool
    predictive_training_authorized: bool
    historical_performance_authorized: bool
    receipt_sha256: str
    schema: str = MASSIVE_ENTITLEMENT_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "observed_at_ms": self.observed_at_ms,
            "credential_source": self.credential_source,
            "observations": [row.payload() for row in self.observations],
            "trades_rest_available": self.trades_rest_available,
            "delayed_websocket_documented": self.delayed_websocket_documented,
            "flat_files_documented": self.flat_files_documented,
            "historical_quotes_available": self.historical_quotes_available,
            "financials_and_ratios_available": self.financials_and_ratios_available,
            "history_years": self.history_years,
            "entitlement_delay_minutes": self.entitlement_delay_minutes,
            "secret_material_persisted": self.secret_material_persisted,
            "predictive_training_authorized": self.predictive_training_authorized,
            "historical_performance_authorized": self.historical_performance_authorized,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_ENTITLEMENT_AUTHORITY_SCHEMA:
            raise MassiveEntitlementError("entitlement authority schema drifted")
        if self.plan_id != "stocks-developer":
            raise MassiveEntitlementError("only Stocks Developer is frozen for V1")
        _nonnegative_int("authority timestamp", self.observed_at_ms)
        if self.credential_source != "environment:MASSIVE_API_KEY":
            raise MassiveEntitlementError("credential source must remain secret-free")
        if not self.observations:
            raise MassiveEntitlementError("entitlement authority has no observations")
        surface_ids = tuple(row.surface_id for row in self.observations)
        if surface_ids != tuple(sorted(set(surface_ids))):
            raise MassiveEntitlementError(
                "entitlement observations must be sorted and unique by surface"
            )
        for row in self.observations:
            row.validate()
            if row.observed_at_ms > self.observed_at_ms:
                raise MassiveEntitlementError("authority predates one observation")
        by_surface = {row.surface_id: row for row in self.observations}
        required = {
            "delayed-websocket",
            "flat-files",
            "financials-and-ratios",
            "historical-quotes",
            "reference-rest",
            "trades-rest",
        }
        if set(by_surface) != required:
            raise MassiveEntitlementError("entitlement surface inventory drifted")
        expected = {
            "trades_rest_available": by_surface["trades-rest"].access_state
            == "available",
            "delayed_websocket_documented": by_surface[
                "delayed-websocket"
            ].access_state
            in {"available", "documented-not-runtime-probed"},
            "flat_files_documented": by_surface["flat-files"].access_state
            in {"available", "documented-not-runtime-probed"},
            "historical_quotes_available": by_surface[
                "historical-quotes"
            ].access_state
            == "available",
            "financials_and_ratios_available": by_surface[
                "financials-and-ratios"
            ].access_state
            == "available",
        }
        for field, value in expected.items():
            if getattr(self, field) is not value:
                raise MassiveEntitlementError(f"{field} differs from observations")
        if self.history_years != 10 or self.entitlement_delay_minutes != 15:
            raise MassiveEntitlementError("Stocks Developer plan terms drifted")
        if any(
            (
                self.secret_material_persisted,
                self.predictive_training_authorized,
                self.historical_performance_authorized,
            )
        ):
            raise MassiveEntitlementError(
                "entitlement evidence cannot persist secrets or authorize science"
            )
        _digest("entitlement authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveEntitlementError("entitlement receipt differs from payload")


def build_massive_developer_entitlement_authority(
    observations: Sequence[MassiveEntitlementObservation],
    *,
    observed_at_ms: int,
) -> MassiveEntitlementAuthority:
    """Build a secret-free authority from observed/documented access rows."""

    ordered = tuple(sorted(observations, key=lambda row: row.surface_id))
    by_surface = {row.surface_id: row for row in ordered}
    body = {
        "schema": MASSIVE_ENTITLEMENT_AUTHORITY_SCHEMA,
        "plan_id": "stocks-developer",
        "observed_at_ms": observed_at_ms,
        "credential_source": "environment:MASSIVE_API_KEY",
        "observations": [row.payload() for row in ordered],
        "trades_rest_available": by_surface.get("trades-rest") is not None
        and by_surface["trades-rest"].access_state == "available",
        "delayed_websocket_documented": by_surface.get("delayed-websocket")
        is not None
        and by_surface["delayed-websocket"].access_state
        in {"available", "documented-not-runtime-probed"},
        "flat_files_documented": by_surface.get("flat-files") is not None
        and by_surface["flat-files"].access_state
        in {"available", "documented-not-runtime-probed"},
        "historical_quotes_available": by_surface.get("historical-quotes")
        is not None
        and by_surface["historical-quotes"].access_state == "available",
        "financials_and_ratios_available": by_surface.get(
            "financials-and-ratios"
        )
        is not None
        and by_surface["financials-and-ratios"].access_state == "available",
        "history_years": 10,
        "entitlement_delay_minutes": 15,
        "secret_material_persisted": False,
        "predictive_training_authorized": False,
        "historical_performance_authorized": False,
    }
    value = MassiveEntitlementAuthority(
        plan_id="stocks-developer",
        observed_at_ms=observed_at_ms,
        credential_source="environment:MASSIVE_API_KEY",
        observations=ordered,
        trades_rest_available=bool(body["trades_rest_available"]),
        delayed_websocket_documented=bool(body["delayed_websocket_documented"]),
        flat_files_documented=bool(body["flat_files_documented"]),
        historical_quotes_available=bool(body["historical_quotes_available"]),
        financials_and_ratios_available=bool(
            body["financials_and_ratios_available"]
        ),
        history_years=10,
        entitlement_delay_minutes=15,
        secret_material_persisted=False,
        predictive_training_authorized=False,
        historical_performance_authorized=False,
        receipt_sha256=semantic_sha256(body),
    )
    value.validate()
    return value


def observe_massive_rest_surface(
    *,
    surface_id: str,
    request_path: str,
    api_key: str,
    timeout_seconds: float = 30.0,
    response_limit_bytes: int = 2 * 1024 * 1024,
) -> MassiveEntitlementObservation:
    """Probe one REST surface without putting the credential in the URL."""

    _text("API key", api_key)
    if any(character.isspace() for character in api_key):
        raise MassiveEntitlementError("API key contains whitespace")
    if not request_path.startswith("/") or "apikey=" in request_path.lower():
        raise MassiveEntitlementError("REST probe path is unsafe")
    url = "https://api.massive.com" + request_path
    web_request = request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "QuantTrade-Massive-Entitlement-V1",
        },
        method="GET",
    )
    observed_at_ms = time.time_ns() // 1_000_000
    status: int | None = None
    request_id: str | None = None
    body = b""
    access_state: MassiveAccessState
    try:
        with request.urlopen(web_request, timeout=timeout_seconds) as response:
            status = int(response.status)
            request_id = response.headers.get("X-Request-ID")
            body = response.read(response_limit_bytes + 1)
            if len(body) > response_limit_bytes:
                raise MassiveEntitlementError("entitlement response exceeded its cap")
        access_state = "available" if status == 200 else "transport-failed"
    except error.HTTPError as exc:
        status = int(exc.code)
        request_id = exc.headers.get("X-Request-ID") if exc.headers is not None else None
        body = exc.read(response_limit_bytes + 1)
        if len(body) > response_limit_bytes:
            raise MassiveEntitlementError("entitlement error response exceeded its cap")
        if status in {401, 403}:
            access_state = "forbidden"
        elif status == 404:
            access_state = "not-found"
        else:
            access_state = "transport-failed"
    except error.URLError:
        access_state = "transport-failed"
    return MassiveEntitlementObservation(
        surface_id=surface_id,
        request_path=request_path,
        observed_at_ms=observed_at_ms,
        access_state=access_state,
        http_status=status,
        response_content_length=len(body),
        response_body_sha256=hashlib.sha256(body).hexdigest(),
        request_id=request_id,
    )


def documented_massive_surface(
    *, surface_id: str, request_path: str, observed_at_ms: int
) -> MassiveEntitlementObservation:
    """Record a documented plan surface that was not runtime-probed."""

    return MassiveEntitlementObservation(
        surface_id=surface_id,
        request_path=request_path,
        observed_at_ms=observed_at_ms,
        access_state="documented-not-runtime-probed",
        http_status=None,
        response_content_length=0,
        response_body_sha256=hashlib.sha256(b"").hexdigest(),
        request_id=None,
    )


__all__ = [
    "MASSIVE_ENTITLEMENT_AUTHORITY_SCHEMA",
    "MASSIVE_ENTITLEMENT_OBSERVATION_SCHEMA",
    "MassiveEntitlementAuthority",
    "MassiveEntitlementError",
    "MassiveEntitlementObservation",
    "build_massive_developer_entitlement_authority",
    "documented_massive_surface",
    "observe_massive_rest_surface",
]
