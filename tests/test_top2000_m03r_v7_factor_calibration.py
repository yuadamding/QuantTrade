"""Causality tests for TOP2000 v7 observation-warmup factor calibration."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    build_top2000_hold30_development_sequence_from_loaded_cache,
    load_verified_top2000_hold30_development_cache,
)
from rl_quant.training.top2000_m03r_v7_factor_calibration import (
    TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS,
    TOP2000_M03R_V7_FACTOR_NAMES,
    Top2000M03RV7FactorCalibrationError,
    fit_top2000_m03r_v7_warmup_factor_calibration,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cache_axis_digest(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_cache(tmp_path: Path, *, identity: str = "cache-a") -> tuple[Path, str]:
    days = 400
    actions = ("CASH", *(f"A{index}" for index in range(1, 9)))
    dates = tuple(
        (dt.date(2023, 1, 2) + dt.timedelta(days=index)).isoformat()
        for index in range(days)
    )
    bars = torch.zeros((days, len(actions), 5), dtype=torch.float64)
    day = torch.arange(days, dtype=torch.float64).view(-1, 1)
    asset = torch.arange(1, len(actions), dtype=torch.float64).view(1, -1)
    close = 100.0 + 0.01 * day * asset + 0.1 * torch.sin(day / (asset + 2.0))
    bars[:, 1:, 0] = close - 0.02
    bars[:, 1:, 1] = close + 0.05
    bars[:, 1:, 2] = close - 0.05
    bars[:, 1:, 3] = close
    bars[:, 1:, 4] = 1_000_000.0 + 1_000.0 * asset + 10.0 * day
    availability = torch.ones((days, len(actions)), dtype=torch.bool)
    payload = {
        "schema_version": 1,
        "feature_cache_version": 1,
        "label": "development-only",
        "development_only": True,
        "bars_only": True,
        "bar_seconds": 300,
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "cache_identity": _digest(identity),
        "actions": actions,
        "action_hash": _cache_axis_digest(actions),
        "exchange_dates": dates,
        "date_hash": _cache_axis_digest(dates),
        "daily_ohlcv": bars,
        "availability": availability,
    }
    path = tmp_path / f"{identity}.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _loaded(tmp_path: Path, *, identity: str = "cache-a"):
    path, digest = _write_cache(tmp_path, identity=identity)
    return load_verified_top2000_hold30_development_cache(
        path,
        expected_cache_sha256=digest,
        acknowledgement=DEVELOPMENT_ACK,
    )


def _slice(cache, start: int, stop: int):
    return build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=start,
        state_stop_index_exclusive=stop,
    )


def test_factor_controls_use_only_observation_warmup_before_first_score(
    tmp_path: Path,
) -> None:
    cache = _loaded(tmp_path)
    calibration = _slice(cache, 0, 64)
    episode = _slice(cache, 0, 378)

    fitted = fit_top2000_m03r_v7_warmup_factor_calibration(
        calibration,
        episode,
    )

    assert fitted.calibration_transition_count == 63
    assert (
        fitted.calibration_transition_count
        == TOP2000_M03R_V7_FACTOR_CALIBRATION_TRANSITIONS
    )
    assert fitted.factor_names == TOP2000_M03R_V7_FACTOR_NAMES
    assert fitted.loadings.shape == (9, 4)
    assert torch.equal(fitted.loadings[0], torch.zeros(4, dtype=torch.float64))
    assert fitted.calibration_state_stop_index_exclusive == 64
    assert fitted.episode_state_start_index == 0
    assert fitted.episode_state_stop_index_exclusive == 378
    assert fitted.first_loss_origin_state_index == 251
    assert fitted.calibration_last_date < fitted.first_loss_origin_date
    assert fitted.development_only
    assert not fitted.outer_evaluation_authorized
    assert not fitted.promotion_eligible
    assert len(fitted.receipt_sha256) == 64


def test_nonprefix_factor_calibration_is_rejected(tmp_path: Path) -> None:
    cache = _loaded(tmp_path)
    calibration = _slice(cache, 1, 65)
    episode = _slice(cache, 0, 378)

    with pytest.raises(
        Top2000M03RV7FactorCalibrationError,
        match="exact observation-warmup prefix",
    ):
        fit_top2000_m03r_v7_warmup_factor_calibration(
            calibration,
            episode,
        )


def test_future_factor_calibration_is_rejected(tmp_path: Path) -> None:
    cache = _loaded(tmp_path)
    episode = _slice(cache, 0, 378)
    future = _slice(cache, 251, 315)

    with pytest.raises(Top2000M03RV7FactorCalibrationError, match="prefix"):
        fit_top2000_m03r_v7_warmup_factor_calibration(future, episode)


def test_factor_calibration_rejects_noncanonical_episode_geometry(
    tmp_path: Path,
) -> None:
    cache = _loaded(tmp_path)
    calibration = _slice(cache, 0, 64)
    short_episode = _slice(cache, 0, 377)

    with pytest.raises(Top2000M03RV7FactorCalibrationError, match="378-state"):
        fit_top2000_m03r_v7_warmup_factor_calibration(
            calibration,
            short_episode,
        )


def test_factor_calibration_rejects_a_different_cache_identity(
    tmp_path: Path,
) -> None:
    first = _loaded(tmp_path, identity="cache-a")
    second = _loaded(tmp_path, identity="cache-b")
    calibration = _slice(first, 0, 64)
    episode = _slice(second, 0, 378)

    with pytest.raises(
        Top2000M03RV7FactorCalibrationError, match="one verified cache"
    ):
        fit_top2000_m03r_v7_warmup_factor_calibration(
            calibration,
            episode,
        )
