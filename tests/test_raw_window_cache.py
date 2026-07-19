from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import rl_quant.datasets.raw_window as raw_window_module
from rl_quant.datasets.raw_window import (
    RAW_WINDOW_CACHE_VERSION,
    RawWindowConfig,
    raw_window_cache_key,
    raw_window_dependency_paths,
    raw_window_source_signature,
)


W1 = "2024-01-02_to_2024-01-03"
W2 = "2024-01-04_to_2024-01-05"


def _write(root: Path, relative: str, payload: bytes = b"x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _dataset(root: Path) -> None:
    _write(root, "universe.json", b'{"actions":["CASH","AAA"],"cash_index":0}')
    _write(root, f"partitions/{W1}/bars.parquet", b"bars-one")
    _write(root, f"partitions/{W1}/covariates.parquet", b"cov-one")
    _write(root, f"partitions/{W2}/bars.parquet", b"bars-two")
    _write(root, f"partitions/{W2}/covariates.parquet", b"cov-two")
    _write(root, f"partitions/{W2}/news.jsonl", b"news-two")
    _write(root, "universe_membership.parquet", b"membership")


def test_dependencies_cover_carried_covariates_and_membership_inputs(tmp_path: Path) -> None:
    _dataset(tmp_path)
    cfg = RawWindowConfig(cov_carry_days=400, use_news=True)
    relative = {path.relative_to(tmp_path).as_posix() for path in raw_window_dependency_paths(tmp_path, W2, cfg)}

    assert relative == {
        "universe.json",
        "universe_membership.parquet",
        f"partitions/{W2}/bars.parquet",
        f"partitions/{W1}/covariates.parquet",
        f"partitions/{W2}/covariates.parquet",
        f"partitions/{W2}/news.jsonl",
    }


def test_changing_carried_covariate_invalidates_later_window(tmp_path: Path) -> None:
    _dataset(tmp_path)
    cfg = RawWindowConfig(cov_carry_days=400)
    before = raw_window_source_signature(tmp_path, W2, cfg)
    _write(tmp_path, f"partitions/{W1}/covariates.parquet", b"changed-carried-covariate")
    after = raw_window_source_signature(tmp_path, W2, cfg)
    assert after != before


def test_adding_a_new_eligible_prior_partition_invalidates_cache(tmp_path: Path) -> None:
    _write(tmp_path, "universe.json", b"universe")
    _write(tmp_path, f"partitions/{W2}/bars.parquet", b"bars-two")
    cfg = RawWindowConfig(cov_carry_days=400)
    before = raw_window_source_signature(tmp_path, W2, cfg)

    _write(tmp_path, f"partitions/{W1}/bars.parquet", b"bars-one")
    _write(tmp_path, f"partitions/{W1}/covariates.parquet", b"new-prior-cov")
    after = raw_window_source_signature(tmp_path, W2, cfg)
    assert after != before


def test_missing_dependency_sentinel_detects_later_covariate_creation(tmp_path: Path) -> None:
    _dataset(tmp_path)
    prior_cov = tmp_path / "partitions" / W1 / "covariates.parquet"
    prior_cov.unlink()
    cfg = RawWindowConfig(cov_carry_days=400)
    before = raw_window_source_signature(tmp_path, W2, cfg)
    prior_cov.write_bytes(b"created-after-cache")
    after = raw_window_source_signature(tmp_path, W2, cfg)
    assert after != before


def test_dependencies_outside_carry_horizon_do_not_cause_false_invalidation(tmp_path: Path) -> None:
    _dataset(tmp_path)
    old_window = "2020-01-02_to_2020-01-03"
    _write(tmp_path, f"partitions/{old_window}/bars.parquet", b"old-bars")
    old_cov = _write(tmp_path, f"partitions/{old_window}/covariates.parquet", b"old-cov")
    cfg = RawWindowConfig(cov_carry_days=30)
    before = raw_window_source_signature(tmp_path, W2, cfg)
    old_cov.write_bytes(b"changed-but-still-outside-the-horizon")
    after = raw_window_source_signature(tmp_path, W2, cfg)
    assert after == before


def test_news_only_invalidates_news_enabled_cache(tmp_path: Path) -> None:
    _dataset(tmp_path)
    news = tmp_path / "partitions" / W2 / "news.jsonl"
    disabled = RawWindowConfig(use_news=False)
    disabled_before = raw_window_source_signature(tmp_path, W2, disabled)
    news.write_bytes(b"changed-news")
    disabled_after = raw_window_source_signature(tmp_path, W2, disabled)
    assert disabled_after == disabled_before

    enabled = RawWindowConfig(use_news=True)
    enabled_before = raw_window_source_signature(tmp_path, W2, enabled)
    news.write_bytes(b"changed-news-again-and-longer")
    enabled_after = raw_window_source_signature(tmp_path, W2, enabled)
    assert enabled_after != enabled_before


def test_universe_declaration_and_membership_events_invalidate_cache(tmp_path: Path) -> None:
    _dataset(tmp_path)
    cfg = RawWindowConfig()
    initial = raw_window_source_signature(tmp_path, W2, cfg)
    (tmp_path / "universe_membership.parquet").write_bytes(b"changed-membership")
    membership_changed = raw_window_source_signature(tmp_path, W2, cfg)
    assert membership_changed != initial

    (tmp_path / "universe.json").write_bytes(b"changed-universe-declaration")
    universe_changed = raw_window_source_signature(tmp_path, W2, cfg)
    assert universe_changed != membership_changed


def test_atomic_same_size_replacement_is_detected_without_content_hashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _dataset(tmp_path)
    cfg = RawWindowConfig()
    bars = tmp_path / "partitions" / W2 / "bars.parquet"
    original = bars.stat()
    before = raw_window_source_signature(tmp_path, W2, cfg)

    replacement = bars.with_suffix(".replacement")
    replacement.write_bytes(b"BARS-TWO")  # same length as b"bars-two"
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(bars)
    assert bars.stat().st_size == original.st_size
    assert bars.stat().st_mtime_ns == original.st_mtime_ns

    # Cache lookup must remain metadata-only even for large source artifacts.
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError(self)))
    monkeypatch.setattr(Path, "read_text", lambda self: (_ for _ in ()).throw(AssertionError(self)))
    after = raw_window_source_signature(tmp_path, W2, cfg)
    assert after != before


def test_cache_key_covers_version_config_universe_schedule_and_source(tmp_path: Path, monkeypatch) -> None:
    _dataset(tmp_path)
    cfg = RawWindowConfig()
    assert cfg.cache_version == RAW_WINDOW_CACHE_VERSION == 11
    base = raw_window_cache_key(tmp_path, W2, cfg, universe_signature="CASH|AAA")
    assert f"_v{RAW_WINDOW_CACHE_VERSION}_" in base

    assert raw_window_cache_key(
        tmp_path,
        W2,
        replace(cfg, bar_seconds=60),
        universe_signature="CASH|AAA",
    ) != base
    assert raw_window_cache_key(tmp_path, W2, cfg, universe_signature="CASH|BBB") != base
    assert raw_window_cache_key(
        tmp_path,
        W2,
        replace(cfg, cache_version=RAW_WINDOW_CACHE_VERSION + 1),
        universe_signature="CASH|AAA",
    ) != base
    audited_schedule = raw_window_module.XNYS_EARLY_CLOSE_DATES_2022_2026
    monkeypatch.setattr(
        raw_window_module,
        "XNYS_EARLY_CLOSE_DATES_2022_2026",
        (*audited_schedule, "2027-11-26"),
    )
    assert raw_window_cache_key(tmp_path, W2, cfg, universe_signature="CASH|AAA") != base
    monkeypatch.setattr(raw_window_module, "XNYS_EARLY_CLOSE_DATES_2022_2026", audited_schedule)

    (tmp_path / "universe_membership.parquet").write_bytes(b"new-membership-events")
    assert raw_window_cache_key(tmp_path, W2, cfg, universe_signature="CASH|AAA") != base
