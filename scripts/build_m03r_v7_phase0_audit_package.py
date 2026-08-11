#!/usr/bin/env python3
"""Build the immutable M03R-v7 Phase-0 audit source package.

The builder starts from the exact A05 training source archive and overlays only
the reviewed inference/audit modules.  It deliberately excludes every
other working-tree file so unfinished experiments cannot enter the GPU Job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_ARCHIVE_SHA256 = (
    "468df12c9679e97d413d02943717770251bd74e963b669fb967ceb751009d5cd"
)
CACHE_SHA256 = "0ba73414c3adea7712f7a68b1e76d934a17694a27671f35b8aa191bcc6aa1ee0"
PACKAGE_PLAN_SHA256 = (
    "5ad68ddfe97205e18ba34816a41782bd5eacda35654bfa325310ecf7111f4f12"
)
IMAGE = (
    "hpcharbor.mdanderson.edu/yding41/ml2:quanttrade-ppo-cu124-py311-85cf781d3e08"
    "@sha256:7cff8faedcfb44ad25e1001d7e1634569f7cd3f5365bbd8ff8caa9b10d8bcdf9"
)
PARENT_RUN_ID = "qt-m03r-v7-t2k12-s17-20260808-a05q2"
PARENT_TERMINAL_EVIDENCE_SHA256 = (
    "688c212aebc42e04109be1eaf6bb4d80c76080d0e3b826a7619366e7a891deac"
)
PARENT_CLEANUP_RECEIPT_SHA256 = (
    "ac58cffface2df9317780d1ce7052051416d1c3b60765c08a0a92712cb0df6ef"
)

OVERLAY_PATHS = (
    "src/rl_quant/evaluation/top2000_m03r_v7_dev.py",
    "src/rl_quant/evaluation/m03r_v7_trace_audit.py",
    "src/rl_quant/evaluation/m03r_cost_ladder_evaluator.py",
    "src/rl_quant/evaluation/m03r_alpha_head_diagnostics.py",
    "src/rl_quant/evaluation/m03r_projection_attribution.py",
    "src/rl_quant/evaluation/m03r_setting9_risk_audit.py",
    "src/rl_quant/workflows/top2000_m03r_v7_forensic_audit.py",
    "src/rl_quant/workflows/top2000_m03r_v7_forensic_audit_worker.py",
)

SETTING_INPUTS = (
    (0, 0, "bfa535aff335a8bbb8981e0602e5222fa093926c4f88a37c5c0ac0319321c20a"),
    (1, 1, "f60025dac05a83c553f1b349033ea13b2a09b52d5fe074bf36b19cbf0dae8587"),
    (2, 2, "0e897a014b013a2ade6d2c0b598a2074d95afe9d1f1f11bff8ba42d45045b5ce"),
    (3, 3, "61073377ed13258b6048baa6440d290104dbf4a69fbeb161e13e6b31dd677aeb"),
    (4, 4, "ceb15cc1e04e2b2dbc3241e89fd545ca22f145b6d81a06485ff5c4da4872732c"),
    (5, 5, "4fa34f86a536da9dee7240a5d54eca833251a257337de68d0f4be400c33a2530"),
    (6, 6, "c03d35748779481a914755735cd8235773318a6c6e1d2b2bb0207b102d002e14"),
    (7, 8, "a7f885ad32a917d99b66e1795995e26944af41ff18324de055bc3d080cf2b016"),
    (8, 7, "cd222d3a7075676cae2fec2cd8b14da95d51e8480afcf5a0c36e5ef5021b4141"),
    (9, 9, "0134b596ab8d469abffbf4552d58181e589b8dd7d43b9cdc654b1ba10d3a77e2"),
    (10, 10, "6a1324fb10d1d497a38e99b945de4a6d8b05b7b4f539f0b97231b4d23188775d"),
    (11, 11, "b0410d202a6014ea02d3f6396ac9c69125513de2937eef80a5902a183b0faf23"),
)


class PackageBuildError(RuntimeError):
    """The frozen input or package boundary failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _safe_extract_source(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        if not members:
            raise PackageBuildError("source archive is empty")
        paths: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or not pure.parts
                or pure.parts[0] != "source"
                or ".." in pure.parts
                or member.name in paths
                or not (member.isfile() or member.isdir())
            ):
                raise PackageBuildError(f"unsafe source archive member: {member.name}")
            paths.add(member.name)
        bundle.extractall(destination, filter="data")


def _inventory(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise PackageBuildError(f"package contains a non-regular file: {relative}")
        result.append(
            {
                "path": relative,
                "size": metadata.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return result


def _deterministic_tar(package_root: Path, output: Path) -> None:
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as bundle:
        for path in (package_root, *sorted(package_root.rglob("*"))):
            relative = path.relative_to(package_root.parent).as_posix()
            info = bundle.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_dir():
                info.mode = 0o555
                bundle.addfile(info)
            elif path.is_file() and not path.is_symlink():
                info.mode = 0o444
                with path.open("rb") as source:
                    bundle.addfile(info, source)
            else:
                raise PackageBuildError(f"unsupported archive path: {path}")


def build(original_package: Path, repository: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise PackageBuildError(f"output root already exists: {output_root}")
    source_archive = original_package / "source.tar"
    cache = original_package / "cache.pt"
    package_plan = json.loads((original_package / "package-plan.json").read_bytes())
    if _sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise PackageBuildError("original source archive hash drifted")
    if _sha256_file(cache) != CACHE_SHA256:
        raise PackageBuildError("original cache hash drifted")
    if package_plan.get("package_plan_sha256") != PACKAGE_PLAN_SHA256:
        raise PackageBuildError("original package-plan semantic hash drifted")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary_name:
        temporary = Path(temporary_name)
        _safe_extract_source(source_archive, temporary)
        source_root = temporary / "source"
        overlays: list[dict[str, Any]] = []
        for relative_name in OVERLAY_PATHS:
            source = repository / relative_name
            if not source.is_file() or source.is_symlink():
                raise PackageBuildError(f"overlay is not a regular file: {relative_name}")
            destination = source_root / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
            overlays.append(
                {
                    "path": relative_name,
                    "size": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            )

        inventory = _inventory(source_root)
        inventory_sha256 = hashlib.sha256(_canonical_bytes(inventory)).hexdigest()
        overlay_inventory_sha256 = hashlib.sha256(
            _canonical_bytes(overlays)
        ).hexdigest()
        manifest: dict[str, Any] = {
            "schema": "rl-quant.top2000-dev.m03r-v7-phase0-audit-package-v1",
            "parent_run_id": PARENT_RUN_ID,
            "parent_terminal_evidence_file_sha256": PARENT_TERMINAL_EVIDENCE_SHA256,
            "parent_cleanup_receipt_file_sha256": PARENT_CLEANUP_RECEIPT_SHA256,
            "original_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "original_cache_sha256": CACHE_SHA256,
            "original_package_plan_sha256": PACKAGE_PLAN_SHA256,
            "image": IMAGE,
            "overlay_files": overlays,
            "overlay_inventory_sha256": overlay_inventory_sha256,
            "source_inventory": inventory,
            "source_inventory_sha256": inventory_sha256,
            "setting_inputs": [
                {
                    "completion_index": completion,
                    "setting_index": setting,
                    "training_plan_file_sha256": plan_sha,
                }
                for completion, setting, plan_sha in SETTING_INPUTS
            ],
            "setting_count": 12,
            "fold_count_per_setting": 6,
            "retraining_authorized": False,
            "checkpoint_selection_authorized": False,
            "development_only": True,
            "future_selected_universe": True,
            "reportable": False,
            "promotable": False,
        }
        manifest["receipt_sha256"] = hashlib.sha256(
            _canonical_bytes(manifest)
        ).hexdigest()

        package_root = temporary / "package"
        package_root.mkdir()
        source_root.rename(package_root / "source")
        manifest_path = package_root / "audit-package-manifest.json"
        manifest_path.write_bytes(_canonical_bytes(manifest) + b"\n")
        output_root.mkdir(mode=0o755)
        final_package = output_root / "package"
        package_root.rename(final_package)
        for path in sorted(final_package.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(final_package, 0o555)
        archive = output_root / "audit-package.tar"
        _deterministic_tar(final_package, archive)
        result = {
            "package_root": str(final_package),
            "manifest_file_sha256": _sha256_file(
                final_package / "audit-package-manifest.json"
            ),
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "source_inventory_sha256": inventory_sha256,
            "overlay_inventory_sha256": overlay_inventory_sha256,
            "archive_path": str(archive),
            "archive_sha256": _sha256_file(archive),
        }
        (output_root / "local-build-receipt.json").write_bytes(
            _canonical_bytes(result) + b"\n"
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-package", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.original_package, args.repository, args.output_root)
    print(_canonical_bytes(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
