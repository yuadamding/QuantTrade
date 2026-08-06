"""Focused tests for the deterministic TOP2000 M03R-v7 package builder."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_quant.workflows.top2000_m03r_v7_package_builder import (
    PINNED_IMAGE_PYTHON,
    PINNED_QUANTTRADE_IMAGE,
    Top2000M03RV7CacheContract,
    Top2000M03RV7PackageBuildError,
    build_top2000_m03r_v7_package,
    validate_top2000_m03r_v7_package,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(label: str) -> str:
    return _sha256_bytes(label.encode("utf-8"))


def _axis_digest(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    )


def _write_repository(tmp_path: Path, *, tracked_symlink: bool = False) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Package Builder Test")
    _git(repository, "config", "user.email", "package-builder@example.invalid")
    (repository / "pyproject.toml").write_text(
        "[project]\nname = \"fake-quanttrade\"\nversion = \"0.0.0\"\n",
        encoding="utf-8",
    )
    source = repository / "src"
    source.mkdir()
    (source / "demo.py").write_text("VALUE = 7\n", encoding="utf-8")
    if tracked_symlink:
        os.symlink("demo.py", source / "linked.py")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--message", "test fixture")
    return repository


def _write_cache(
    tmp_path: Path,
    *,
    days: int = 70,
) -> tuple[Path, Top2000M03RV7CacheContract]:
    actions = ("CASH", "A1", "A2", "A3")
    dates = tuple(
        (dt.date(2024, 1, 2) + dt.timedelta(days=index)).isoformat()
        for index in range(days)
    )
    bars = torch.zeros((days, len(actions), 5), dtype=torch.float32)
    day = torch.arange(days, dtype=torch.float32).view(-1, 1)
    asset = torch.arange(1, len(actions), dtype=torch.float32).view(1, -1)
    close = 100.0 + 0.01 * day * asset
    bars[:, 1:, 0] = close - 0.01
    bars[:, 1:, 1] = close + 0.02
    bars[:, 1:, 2] = close - 0.02
    bars[:, 1:, 3] = close
    bars[:, 1:, 4] = 1_000_000.0
    availability = torch.ones((days, len(actions)), dtype=torch.bool)
    action_hash = _axis_digest(actions)
    date_hash = _axis_digest(dates)
    identities = {
        "cache_identity": _digest("cache"),
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
    }
    payload = {
        "schema_version": 1,
        "feature_cache_version": 1,
        "label": "development-only",
        "development_only": True,
        "bars_only": True,
        "bar_seconds": 300,
        **identities,
        "actions": actions,
        "action_hash": action_hash,
        "exchange_dates": dates,
        "date_hash": date_hash,
        "daily_ohlcv": bars,
        "availability": availability,
    }
    cache = tmp_path / "cache.pt"
    torch.save(payload, cache)
    cache_sha256 = _sha256_bytes(cache.read_bytes())
    return cache, Top2000M03RV7CacheContract(
        cache_sha256=cache_sha256,
        cache_identity=identities["cache_identity"],
        search_identity=identities["search_identity"],
        base_dataset_identity=identities["base_dataset_identity"],
        lockbox_partition_names_hash=identities["lockbox_partition_names_hash"],
        action_hash=action_hash,
        date_hash=date_hash,
        state_rows=days,
        action_count=len(actions),
        first_exchange_date=dates[0],
        last_exchange_date=dates[-1],
    )


def _build(
    repository: Path,
    cache: Path,
    output: Path,
    contract: Top2000M03RV7CacheContract,
):
    return build_top2000_m03r_v7_package(
        repository_root=repository,
        cache_path=cache,
        output_root=output,
        cache_contract=contract,
        critical_source_paths=("src/demo.py",),
    )


def test_builder_is_deterministic_and_self_validating(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path)
    cache, contract = _write_cache(tmp_path)

    first = _build(repository, cache, tmp_path / "package-a", contract)
    second = _build(repository, cache, tmp_path / "package-b", contract)

    first_root = Path(first.output_root)
    second_root = Path(second.output_root)
    first_files = {
        path.relative_to(first_root).as_posix(): path.read_bytes()
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_root).as_posix(): path.read_bytes()
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    assert {
        path.name
        for path in (first_root / "source").rglob("*")
        if path.is_file()
    } == {
        "demo.py",
        "pyproject.toml",
    }
    assert (first_root / "cache.pt").read_bytes() == cache.read_bytes()

    plan = validate_top2000_m03r_v7_package(first_root, cache_contract=contract)
    runtime = json.loads((first_root / "runtime-manifest.json").read_text())
    data = json.loads((first_root / "data-manifest.json").read_text())
    assert plan.runtime_profile.token_dim == 512
    assert plan.artifacts.image_reference == PINNED_QUANTTRADE_IMAGE
    assert runtime["runtime_profile"]["token_dim"] == 512
    assert plan.source_pythonpath == "/mnt/package/source/src"
    assert runtime["source_pythonpath"] == plan.source_pythonpath
    assert (first_root / "source" / "src").is_dir()
    assert runtime["gpu_count_per_worker"] == 2
    assert runtime["worker_argv_prefix"][0] == PINNED_IMAGE_PYTHON
    assert "--max-restarts=1" in runtime["worker_argv_prefix"]
    assert runtime["torchrun_max_restarts"] == 1
    assert data["data_role"] == "development-only-nonreportable"
    assert data["first_exchange_date"] == contract.first_exchange_date
    assert data["last_exchange_date"] == contract.last_exchange_date
    assert not data["promotion_eligible"]
    assert not data["outer_evaluation_authorized"]
    assert first.package_plan_sha256 == second.package_plan_sha256
    assert first.source_archive_sha256 == second.source_archive_sha256


def test_builder_rejects_dirty_repository_without_publishing(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path)
    cache, contract = _write_cache(tmp_path)
    (repository / "untracked.txt").write_text("must fail closed\n", encoding="utf-8")
    output = tmp_path / "dirty-package"

    with pytest.raises(Top2000M03RV7PackageBuildError, match="completely clean"):
        _build(repository, cache, output, contract)

    assert not output.exists()


def test_builder_rejects_overwrite_symlinks_and_cache_geometry_drift(
    tmp_path: Path,
) -> None:
    repository = _write_repository(tmp_path)
    cache, contract = _write_cache(tmp_path)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(Top2000M03RV7PackageBuildError, match="never overwrite"):
        _build(repository, cache, existing, contract)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(Top2000M03RV7PackageBuildError, match="symlink"):
        _build(repository, cache, linked_parent / "package", contract)

    wrong_geometry = replace(contract, action_count=contract.action_count + 1)
    output = tmp_path / "wrong-geometry"
    with pytest.raises(
        Top2000M03RV7PackageBuildError,
        match="tensor, date, or action geometry drifted",
    ):
        _build(repository, cache, output, wrong_geometry)
    assert not output.exists()


def test_builder_rejects_tracked_symlink(tmp_path: Path) -> None:
    repository = _write_repository(tmp_path, tracked_symlink=True)
    cache, contract = _write_cache(tmp_path)
    output = tmp_path / "symlink-source-package"

    with pytest.raises(
        Top2000M03RV7PackageBuildError,
        match="only stage-zero regular tracked files",
    ):
        _build(repository, cache, output, contract)

    assert not output.exists()
