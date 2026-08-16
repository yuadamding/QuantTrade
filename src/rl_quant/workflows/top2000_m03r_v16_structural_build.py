"""Build the V16 structural slab from package-owned immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v9_projection import (
    load_m03r_v9_projector_manifest,
)
from rl_quant.training.top2000_m03r_v9_risk_materialization import (
    load_top2000_m03r_v9_risk_source,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    build_m03r_v16_structural_slab,
    write_m03r_v16_structural_slab,
)

M03R_V16_STRUCTURAL_BUILD_RECEIPT_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-structural-build-receipt-v1"
)


class M03RV16StructuralBuildError(ValueError):
    """Package-owned V16 structural construction failed closed."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def build_package_owned_m03r_v16_structural_slab(
    *,
    cache_path: str | Path,
    cache_sha256: str,
    cache_manifest_sha256: str,
    risk_manifest_path: str | Path,
    risk_manifest_file_sha256: str,
    projector_manifest_path: str | Path,
    projector_manifest_file_sha256: str,
    source_manifest_sha256: str,
    operator_source_sha256: str,
    risk_artifact_file_sha256: str,
    output_slab_path: str | Path,
    output_receipt_path: str | Path,
) -> dict[str, Any]:
    cache = load_verified_top2000_hold30_development_cache(
        cache_path,
        expected_cache_sha256=cache_sha256,
        acknowledgement=DEVELOPMENT_ACK,
    )
    risk_source, written = load_top2000_m03r_v9_risk_source(
        Path(risk_manifest_path),
        expected_manifest_file_sha256=risk_manifest_file_sha256,
    )
    projector, binding = load_m03r_v9_projector_manifest(
        Path(projector_manifest_path),
        expected_file_sha256=projector_manifest_file_sha256,
    )
    if (
        written.artifact_file_sha256 != risk_artifact_file_sha256
        or written.manifest_file_sha256 != risk_manifest_file_sha256
        or binding.source_materialization_receipt_sha256
        != risk_source.receipt_sha256
        or binding.source_exposure_receipt_sha256
        != risk_source.exposures.receipt_sha256
        or binding.source_artifact_file_sha256 != risk_artifact_file_sha256
        or binding.source_artifact_manifest_file_sha256
        != risk_manifest_file_sha256
        or binding.projector_manifest_sha256 != projector.manifest_sha256
    ):
        raise M03RV16StructuralBuildError(
            "package-owned V16 risk and projector binding drifted"
        )
    slab = build_m03r_v16_structural_slab(
        cache,
        risk_source,
        cache_manifest_sha256=cache_manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        operator_source_sha256=operator_source_sha256,
        risk_artifact_file_sha256=risk_artifact_file_sha256,
        risk_source_manifest_file_sha256=risk_manifest_file_sha256,
        projector_manifest_file_sha256=projector_manifest_file_sha256,
        projector_manifest_sha256=projector.manifest_sha256,
        projector_binding_sha256=binding.binding_sha256,
    )
    slab.receipt.validate_for_package(
        cache_sha256=cache_sha256,
        cache_manifest_sha256=cache_manifest_sha256,
        asset_axis_sha256=cache.action_hash,
        source_manifest_sha256=source_manifest_sha256,
        operator_source_sha256=operator_source_sha256,
        risk_artifact_file_sha256=risk_artifact_file_sha256,
        risk_source_manifest_file_sha256=risk_manifest_file_sha256,
        risk_source_receipt_sha256=risk_source.receipt_sha256,
        exposure_receipt_sha256=risk_source.exposures.receipt_sha256,
        projector_manifest_file_sha256=projector_manifest_file_sha256,
        projector_manifest_sha256=projector.manifest_sha256,
        projector_binding_sha256=binding.binding_sha256,
    )
    slab_file_sha256 = write_m03r_v16_structural_slab(output_slab_path, slab)
    unsigned = {
        "schema": M03R_V16_STRUCTURAL_BUILD_RECEIPT_SCHEMA,
        "slab_file_sha256": slab_file_sha256,
        "slab_receipt_sha256": slab.receipt.receipt_sha256,
        "cache_sha256": cache_sha256,
        "asset_axis_sha256": cache.action_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "operator_source_sha256": operator_source_sha256,
        "risk_source_receipt_sha256": risk_source.receipt_sha256,
        "exposure_receipt_sha256": risk_source.exposures.receipt_sha256,
        "projector_manifest_sha256": projector.manifest_sha256,
        "projector_binding_sha256": binding.binding_sha256,
        "action_operator_root_sha256": slab.receipt.action_operator_root_sha256,
        "common_target_operator_root_sha256": (
            slab.receipt.common_target_operator_root_sha256
        ),
        "target_root_sha256": slab.receipt.target_root_sha256,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256(unsigned)}
    target = Path(output_receipt_path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(receipt))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--risk-manifest", required=True)
    parser.add_argument("--risk-manifest-file-sha256", required=True)
    parser.add_argument("--projector-manifest", required=True)
    parser.add_argument("--projector-manifest-file-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--operator-source-sha256", required=True)
    parser.add_argument("--risk-artifact-file-sha256", required=True)
    parser.add_argument("--output-slab", required=True)
    parser.add_argument("--output-receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_package_owned_m03r_v16_structural_slab(
        cache_path=args.cache,
        cache_sha256=args.cache_sha256,
        cache_manifest_sha256=args.cache_manifest_sha256,
        risk_manifest_path=args.risk_manifest,
        risk_manifest_file_sha256=args.risk_manifest_file_sha256,
        projector_manifest_path=args.projector_manifest,
        projector_manifest_file_sha256=args.projector_manifest_file_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        operator_source_sha256=args.operator_source_sha256,
        risk_artifact_file_sha256=args.risk_artifact_file_sha256,
        output_slab_path=args.output_slab,
        output_receipt_path=args.output_receipt,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_STRUCTURAL_BUILD_RECEIPT_SCHEMA",
    "M03RV16StructuralBuildError",
    "build_package_owned_m03r_v16_structural_slab",
    "main",
]
