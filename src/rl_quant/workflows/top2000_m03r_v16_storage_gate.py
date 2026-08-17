"""Zero-GPU append-only PVC semantics gate for M03R-v16."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_lifecycle import (
    qualify_m03r_v16_append_only_storage,
    write_m03r_v16_storage_semantics_evidence,
)

M03R_V16_STORAGE_GATE_TERMINAL_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v16-storage-gate-terminal-v1"
)


class M03RV16StorageGateError(RuntimeError):
    """The zero-GPU cross-mount storage contract was not established."""


def _write_exclusive(path: Path, payload: dict[str, object]) -> str:
    raw = canonical_json_file_bytes(payload)
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def run_m03r_v16_storage_gate(
    *,
    authority_root: str | Path,
    observer_root: str | Path,
    output_path: str | Path,
    terminal_path: str | Path,
) -> dict[str, object]:
    """Prove immutable publication through two mounts of the authority PVC."""

    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise M03RV16StorageGateError(
            "V16 storage qualification requires NVIDIA_VISIBLE_DEVICES=none"
        )
    if any(Path("/dev").glob("nvidia[0-9]*")):
        raise M03RV16StorageGateError(
            "V16 storage qualification unexpectedly observes a GPU device"
        )
    authority = Path(authority_root).resolve()
    observer = Path(observer_root).resolve()
    output = Path(output_path).resolve()
    try:
        output.relative_to(authority)
    except ValueError as exc:
        raise M03RV16StorageGateError(
            "V16 storage evidence must be published under the authority root"
        ) from exc
    evidence = qualify_m03r_v16_append_only_storage(
        authority,
        observer_root=observer,
    )
    evidence_file_sha256 = write_m03r_v16_storage_semantics_evidence(
        output,
        evidence,
        authority_root=authority,
        observer_root=observer,
    )
    unsigned: dict[str, object] = {
        "schema": M03R_V16_STORAGE_GATE_TERMINAL_SCHEMA,
        "storage_semantics_file_sha256": evidence_file_sha256,
        "storage_semantics_receipt_sha256": evidence.receipt_sha256,
        "storage_authority_root_sha256": evidence.authority_root_sha256,
        "storage_observer_root_sha256": evidence.observer_root_sha256,
        "distinct_observer_mount": evidence.distinct_observer_mount,
        "hard_link_supported": evidence.hard_link_supported,
        "directory_fsync_supported": evidence.directory_fsync_supported,
        "observer_read_matched": evidence.observer_read_matched,
        "observer_same_file": evidence.observer_same_file,
        "duplicate_publication_rejected": (
            evidence.duplicate_publication_rejected
        ),
        "gpu_requested": False,
        "gpu_visible": False,
        "training_performed": False,
        "development_only": True,
        "reportable": False,
        "promotion_eligible": False,
    }
    terminal = {**unsigned, "receipt_sha256": semantic_sha256(unsigned)}
    terminal_file_sha256 = _write_exclusive(Path(terminal_path), terminal)
    return {**terminal, "terminal_file_sha256": terminal_file_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True)
    parser.add_argument("--observer-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--terminal", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_m03r_v16_storage_gate(
        authority_root=args.authority_root,
        observer_root=args.observer_root,
        output_path=args.output,
        terminal_path=args.terminal,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03R_V16_STORAGE_GATE_TERMINAL_SCHEMA",
    "M03RV16StorageGateError",
    "main",
    "run_m03r_v16_storage_gate",
]
