"""Issue a secret-free Massive Stocks Developer entitlement authority."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import time

from rl_quant.data_sources.massive.entitlement import (
    MassiveEntitlementError,
    build_massive_developer_entitlement_authority,
    documented_massive_surface,
    observe_massive_rest_surface,
)
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes


def _write_once(path: Path, payload: object) -> str:
    data = canonical_json_file_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    return hashlib.sha256(data).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    key = os.environ.get("MASSIVE_API_KEY")
    if key is None:
        raise MassiveEntitlementError(
            "required secret environment variable is absent: MASSIVE_API_KEY"
        )
    observed = (
        observe_massive_rest_surface(
            surface_id="reference-rest",
            request_path="/v3/reference/tickers?active=true&limit=1",
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="trades-rest",
            request_path="/v3/trades/AAPL?order=desc&limit=1&sort=timestamp",
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="historical-quotes",
            request_path="/v3/quotes/AAPL?order=desc&limit=1&sort=timestamp",
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="minute-aggregates",
            request_path=(
                "/v2/aggs/ticker/AAPL/range/1/minute/2026-08-20/2026-08-20"
                "?adjusted=false&limit=1"
            ),
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="day-aggregates",
            request_path=(
                "/v2/aggs/ticker/AAPL/range/1/day/2016-08-22/2016-08-22"
                "?adjusted=false&limit=1"
            ),
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="corporate-actions",
            request_path="/v3/reference/dividends?ticker=AAPL&limit=1",
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
        observe_massive_rest_surface(
            surface_id="history-boundary",
            request_path=(
                "/v2/aggs/ticker/AAPL/range/1/day/2016-08-22/2016-08-22"
                "?adjusted=false&limit=1"
            ),
            api_key=key,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    observed_at_ms = max(
        time.time_ns() // 1_000_000, *(row.observed_at_ms for row in observed)
    )
    observations = observed + (
        documented_massive_surface(
            surface_id="delayed-websocket",
            request_path="/documented-plan/delayed-websocket",
            observed_at_ms=observed_at_ms,
        ),
        documented_massive_surface(
            surface_id="flat-files",
            request_path="/documented-plan/flat-files",
            observed_at_ms=observed_at_ms,
        ),
        documented_massive_surface(
            surface_id="financials-and-ratios",
            request_path="/documented-plan/financials-and-ratios-not-included",
            observed_at_ms=observed_at_ms,
        ),
    )
    authority = build_massive_developer_entitlement_authority(
        observations, observed_at_ms=observed_at_ms
    )
    file_sha256 = _write_once(args.output, asdict(authority))
    print(
        f"authority={args.output} file_sha256={file_sha256} "
        f"receipt_sha256={authority.receipt_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
