"""Aggregate CPU qualification input closure before any V16 outer access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    _issue_m03r_v16_qualification_outer_access_authority,
    load_m03r_v16_qualification_activation,
    load_m03r_v16_qualification_outer_access_authority,
    write_m03r_v16_qualification_outer_access_authority,
)
from rl_quant.training.top2000_m03r_v16_package import (
    load_m03r_v16_execution_authorization,
    load_m03r_v16_package_plan,
)


class M03RV16QualificationPreflightAggregateError(RuntimeError):
    """The complete zero-GPU qualification preflight did not close."""


def _read_self_hashed(path: Path) -> tuple[dict[str, object], str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise M03RV16QualificationPreflightAggregateError(
            "V16 qualification preflight artifact is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 4 * 1024**2:
            raise M03RV16QualificationPreflightAggregateError(
                "V16 qualification preflight artifact type or size drifted"
            )
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
    ):
        raise M03RV16QualificationPreflightAggregateError(
            "V16 qualification preflight artifact changed while read"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M03RV16QualificationPreflightAggregateError(
            "V16 qualification preflight artifact is malformed"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_file_bytes(value):
        raise M03RV16QualificationPreflightAggregateError(
            "V16 qualification preflight artifact is not canonical"
        )
    return value, hashlib.sha256(raw).hexdigest()


def aggregate_m03r_v16_qualification_preflight(
    *,
    package_plan_path: str | Path,
    package_plan_file_sha256: str,
    execution_authorization_path: str | Path,
    execution_authorization_file_sha256: str,
    qualification_activation_path: str | Path,
    qualification_activation_file_sha256: str,
    training_panel_path: str | Path,
    training_root: str | Path,
    preflight_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    package = load_m03r_v16_package_plan(
        package_plan_path, expected_file_sha256=package_plan_file_sha256
    )
    authorization = load_m03r_v16_execution_authorization(
        execution_authorization_path,
        expected_file_sha256=execution_authorization_file_sha256,
        package=package,
    )
    training = Path(training_root)
    activation = load_m03r_v16_qualification_activation(
        qualification_activation_path,
        expected_file_sha256=qualification_activation_file_sha256,
        package=package,
        authorization=authorization,
        training_panel_path=training_panel_path,
        prequalification_closure_path=(
            Path(training_panel_path).parent / "prequalification-closure.json"
        ),
        training_terminal_paths=tuple(
            training
            / f"completion-{index:02d}-setting-{index:02d}"
            / "training-terminal.json"
            for index in range(3)
        ),  # type: ignore[arg-type]
    )
    root = Path(preflight_root)
    closure_paths = tuple(
        root
        / f"completion-{index:02d}-setting-{index:02d}"
        / "qualification-inputs-complete.json"
        for index in range(3)
    )
    # The authority loader performs the exact no-follow reads.  Issuance first
    # binds the file/receipt/risk roots assembled from the three immutable
    # closures; the subsequent load is the independent verification boundary.
    closure_file_sha256: list[str] = []
    closure_receipt_sha256: list[str] = []
    terminal_file_sha256: list[str] = []
    terminal_receipt_sha256: list[str] = []
    risk_roots: list[str] = []
    for index, path in enumerate(closure_paths):
        row, observed_file_sha = _read_self_hashed(path)
        if row.get("setting_index") != index:
            raise M03RV16QualificationPreflightAggregateError(
                "V16 qualification preflight setting inventory drifted"
            )
        closure_file_sha256.append(observed_file_sha)
        closure_receipt_sha256.append(str(row.get("receipt_sha256")))
        risk_roots.append(str(row.get("qualification_risk_input_root_sha256")))
        terminal_path = path.parent / "qualification-preflight-terminal.json"
        terminal, terminal_file_sha = _read_self_hashed(terminal_path)
        if terminal.get("setting_index") != index:
            raise M03RV16QualificationPreflightAggregateError(
                "V16 qualification preflight terminal inventory drifted"
            )
        terminal_file_sha256.append(terminal_file_sha)
        terminal_receipt_sha256.append(str(terminal.get("receipt_sha256")))
    authority = _issue_m03r_v16_qualification_outer_access_authority(
        package=package,
        authorization=authorization,
        activation=activation,
        setting_input_closure_file_sha256=tuple(closure_file_sha256),  # type: ignore[arg-type]
        setting_input_closure_receipt_sha256=tuple(closure_receipt_sha256),  # type: ignore[arg-type]
        setting_preflight_terminal_file_sha256=tuple(terminal_file_sha256),  # type: ignore[arg-type]
        setting_preflight_terminal_receipt_sha256=tuple(terminal_receipt_sha256),  # type: ignore[arg-type]
        qualification_risk_input_root_sha256=semantic_sha256(tuple(risk_roots)),
    )
    destination = Path(output_path)
    temporary = destination.with_name(
        f".{destination.name}.{secrets.token_hex(8)}.tmp"
    )
    try:
        file_sha = write_m03r_v16_qualification_outer_access_authority(
            temporary, authority
        )
        reloaded = load_m03r_v16_qualification_outer_access_authority(
            temporary,
            expected_file_sha256=file_sha,
            expected_receipt_sha256=authority.receipt_sha256,
            package=package,
            authorization=authorization,
            activation=activation,
            setting_input_closure_paths=closure_paths,  # type: ignore[arg-type]
            setting_preflight_terminal_paths=tuple(
                path.parent / "qualification-preflight-terminal.json"
                for path in closure_paths
            ),  # type: ignore[arg-type]
        )
        if reloaded != authority:
            raise M03RV16QualificationPreflightAggregateError(
                "V16 qualification outer-access authority roundtrip drifted"
            )
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise M03RV16QualificationPreflightAggregateError(
            "V16 qualification outer-access authority already exists"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "authority_file_sha256": file_sha,
        "authority_receipt_sha256": authority.receipt_sha256,
        "outer_access_authorized": True,
        "outer_qualification_access_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-plan", required=True)
    parser.add_argument("--package-plan-file-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--execution-authorization-file-sha256", required=True)
    parser.add_argument("--qualification-activation", required=True)
    parser.add_argument("--qualification-activation-file-sha256", required=True)
    parser.add_argument("--training-panel", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--preflight-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    aggregate_m03r_v16_qualification_preflight(
        package_plan_path=args.package_plan,
        package_plan_file_sha256=args.package_plan_file_sha256,
        execution_authorization_path=args.execution_authorization,
        execution_authorization_file_sha256=(
            args.execution_authorization_file_sha256
        ),
        qualification_activation_path=args.qualification_activation,
        qualification_activation_file_sha256=(
            args.qualification_activation_file_sha256
        ),
        training_panel_path=args.training_panel,
        training_root=args.training_root,
        preflight_root=args.preflight_root,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M03RV16QualificationPreflightAggregateError",
    "aggregate_m03r_v16_qualification_preflight",
]
