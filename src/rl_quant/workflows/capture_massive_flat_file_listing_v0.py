"""Capture a committed Massive S3 listing without persisting credentials."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from rl_quant.data_sources.massive.finalized_listing_acquisition import (
    MASSIVE_FLAT_FILE_ENDPOINT,
    MassiveFlatFileListingAcquisitionError,
    capture_massive_flat_file_listing_v0,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    parser.add_argument("--entitlement-receipt", required=True)
    parser.add_argument("--access-key-env", default="MASSIVE_S3_ACCESS_KEY_ID")
    parser.add_argument("--secret-key-env", default="MASSIVE_S3_SECRET_ACCESS_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    access_key = os.environ.get(args.access_key_env)
    secret_key = os.environ.get(args.secret_key_env)
    if not access_key or not secret_key:
        raise MassiveFlatFileListingAcquisitionError(
            "required Massive S3 credential environment variables are absent"
        )
    try:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise MassiveFlatFileListingAcquisitionError(
            "boto3 and botocore are required for Massive listing capture"
        ) from exc
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client = session.client(
        "s3",
        endpoint_url=MASSIVE_FLAT_FILE_ENDPOINT,
        config=Config(signature_version="s3v4"),
    )
    result = capture_massive_flat_file_listing_v0(
        s3_client=client,
        root=args.root,
        year=args.year,
        month=args.month,
        entitlement_receipt_sha256=args.entitlement_receipt,
        access_key_environment_variable=args.access_key_env,
        secret_key_environment_variable=args.secret_key_env,
    )
    print(
        json.dumps(
            {
                "acquisition_receipt_sha256": result.acquisition_evidence.receipt_sha256,
                "committed_listing_receipt_sha256": result.committed_listing.receipt_sha256,
                "object_count": result.acquisition_evidence.object_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
