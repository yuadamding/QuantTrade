"""Focused official-factor retrieval/materialization tests for 2026."""

from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

from rl_quant.evaluation import top2000_m03r_v7_2026_factor_data as factor_module
from rl_quant.evaluation.top2000_m03r_v7_2026_factor_data import (
    TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
    TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
    Top2000M03RV72026FactorDataError,
    build_top2000_m03r_v7_2026_factor_data,
    load_top2000_m03r_v7_2026_factor_data,
    load_top2000_m03r_v7_2026_official_factor_retrieval,
    retrieve_top2000_m03r_v7_2026_official_factor_archives,
    write_top2000_m03r_v7_2026_factor_data,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)


def _score_dates() -> tuple[str, ...]:
    values: list[str] = []
    current = date(2026, 1, 2)
    stop = date(2026, 6, 23)
    while current <= stop:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


def _yyyymmdd(value: str) -> str:
    return value.replace("-", "")


def _zip_bytes(member: str, text: str, *, extra_member: bool = False) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(member, text)
        if extra_member:
            archive.writestr("unexpected.txt", "drift")
    return target.getvalue()


def _archive_payloads(
    *,
    omit_last_momentum: bool = False,
    extra_member: bool = False,
    unused_june_24_row: bool = False,
) -> dict[str, bytes]:
    dates = _score_dates()
    five_lines = [
        "Official synthetic fixture",
        ",Mkt-RF,SMB,HML,RMW,CMA,RF",
        *(
            f"{_yyyymmdd(value)},1.0,0.2,-0.1,0.3,0.4,0.01"
            for value in dates
        ),
    ]
    momentum_dates = dates[:-1] if omit_last_momentum else dates
    momentum_lines = [
        "Official synthetic fixture",
        ",Mom",
        *(f"{_yyyymmdd(value)},0.5" for value in momentum_dates),
    ]
    if unused_june_24_row:
        # Deliberately nonnumeric: exact-date extraction must never parse it.
        five_lines.append("20260624,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED,UNUSED")
        momentum_lines.append("20260624,UNUSED")
    five_lines.append("")
    momentum_lines.append("")
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    return {
        contract.five_factor_download_url: _zip_bytes(
            TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
            "\n".join(five_lines),
            extra_member=extra_member,
        ),
        contract.momentum_download_url: _zip_bytes(
            TOP2000_M03R_V7_2026_MOMENTUM_MEMBER,
            "\n".join(momentum_lines),
        ),
    }


class _MockHTTPSResponse:
    def __init__(self, raw: bytes, url: str) -> None:
        self._stream = io.BytesIO(raw)
        self._url = url
        self.status = 200

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        return self._stream.read(size)


def _retrieve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **payload_options: bool,
):
    payloads = _archive_payloads(**payload_options)
    requested: list[str] = []

    def fake_urlopen(request: Any, *, timeout: int) -> _MockHTTPSResponse:
        assert timeout == 30
        requested.append(str(request.full_url))
        return _MockHTTPSResponse(payloads[str(request.full_url)], request.full_url)

    monkeypatch.setattr(factor_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(factor_module, "_utc_now", lambda: "2026-06-25T12:00:00Z")
    evidence, receipt_file_sha256 = (
        retrieve_top2000_m03r_v7_2026_official_factor_archives(
            output_directory=tmp_path / "official-factor-archives",
            output_receipt_path=tmp_path / "retrieval-receipt.json",
            frozen_plan_file_sha256="a" * 64,
            frozen_plan_receipt_sha256="b" * 64,
        )
    )
    return evidence, receipt_file_sha256, requested


def test_mocked_official_retrieval_is_bound_and_converts_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, retrieval_file_sha256, requested = _retrieve(tmp_path, monkeypatch)
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT.factors
    assert requested == [
        contract.five_factor_download_url,
        contract.momentum_download_url,
    ]
    assert evidence.official_source_verified
    assert not evidence.caller_staged_archives
    assert evidence.default_tls_verification
    assert evidence.frozen_plan_file_sha256 == "a" * 64
    assert (
        load_top2000_m03r_v7_2026_official_factor_retrieval(
            tmp_path / "retrieval-receipt.json",
            expected_file_sha256=retrieval_file_sha256,
        )
        == evidence
    )

    built = build_top2000_m03r_v7_2026_factor_data(
        retrieval_evidence=evidence,
        score_dates=_score_dates(),
    )
    assert built.score_dates == _score_dates()
    np.testing.assert_allclose(built.market_excess_returns, 0.01)
    np.testing.assert_allclose(built.risk_free_returns, 0.0001)
    np.testing.assert_allclose(np.asarray(built.factor_returns)[:, -1], 0.005)
    assert (
        built.source_receipt["retrieval_receipt_sha256"]
        == evidence.receipt_sha256
    )
    assert built.source_receipt["official_source_verified"] is True
    assert built.source_receipt["caller_staged_archives"] is False
    assert built.coverage_receipt["imputed_value_count"] == 0
    assert built.coverage_receipt["score_window_shortened"] is False
    assert built.manifest.missing_value_policy == "no-imputation"
    assert not built.scientific_reporting_eligible
    assert not built.promotion_eligible

    output = tmp_path / "factor-data.json"
    first_sha = write_top2000_m03r_v7_2026_factor_data(built, output)
    assert write_top2000_m03r_v7_2026_factor_data(built, output) == first_sha
    assert (
        load_top2000_m03r_v7_2026_factor_data(
            output, expected_file_sha256=first_sha
        )
        == built
    )
    output.write_text("drift", encoding="utf-8")
    with pytest.raises(Top2000M03RV72026FactorDataError, match="overwrite"):
        write_top2000_m03r_v7_2026_factor_data(built, output)


def test_official_builder_rejects_unverified_or_mutated_caller_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, _receipt_sha, _requested = _retrieve(tmp_path, monkeypatch)
    with pytest.raises(
        Top2000M03RV72026FactorDataError,
        match="caller-staged archives are unverified",
    ):
        build_top2000_m03r_v7_2026_factor_data(
            retrieval_evidence=object(),  # type: ignore[arg-type]
            score_dates=_score_dates(),
        )

    arbitrary = _zip_bytes(
        TOP2000_M03R_V7_2026_FIVE_FACTOR_MEMBER,
        "\n".join(
            [
                "Caller-staged arbitrary fixture",
                ",Mkt-RF,SMB,HML,RMW,CMA,RF",
                *(f"{_yyyymmdd(value)},0,0,0,0,0,0" for value in _score_dates()),
                "",
            ]
        ),
    )
    Path(evidence.five_factor_archive_path).write_bytes(arbitrary)
    with pytest.raises(
        Top2000M03RV72026FactorDataError,
        match="do not match the package-owned retrieval evidence",
    ):
        build_top2000_m03r_v7_2026_factor_data(
            retrieval_evidence=evidence,
            score_dates=_score_dates(),
        )


def test_factor_data_rejects_missing_dates_without_shortening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, _receipt_sha, _requested = _retrieve(
        tmp_path,
        monkeypatch,
        omit_last_momentum=True,
    )
    with pytest.raises(
        Top2000M03RV72026FactorDataError,
        match="do not cover every scored date",
    ):
        build_top2000_m03r_v7_2026_factor_data(
            retrieval_evidence=evidence,
            score_dates=_score_dates(),
        )


def test_retrieval_rejects_archive_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        Top2000M03RV72026FactorDataError,
        match="member inventory",
    ):
        _retrieve(tmp_path, monkeypatch, extra_member=True)


def test_june_24_container_rows_are_bound_but_never_parsed_or_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, _receipt_sha, _requested = _retrieve(
        tmp_path,
        monkeypatch,
        unused_june_24_row=True,
    )
    built = build_top2000_m03r_v7_2026_factor_data(
        retrieval_evidence=evidence,
        score_dates=_score_dates(),
    )

    assert built.coverage_receipt["five_factor_last_source_date"] == "2026-06-24"
    assert built.coverage_receipt["momentum_last_source_date"] == "2026-06-24"
    assert (
        built.coverage_receipt[
            "five_factor_unused_post_end_source_row_count"
        ]
        == 1
    )
    assert (
        built.coverage_receipt[
            "momentum_unused_post_end_source_row_count"
        ]
        == 1
    )
    assert built.coverage_receipt["post_end_source_rows_used"] == 0
    assert built.coverage_receipt["post_end_source_values_parsed"] is False
    assert (
        built.coverage_receipt[
            "post_end_source_rows_may_enter_evaluator_arrays"
        ]
        is False
    )
    np.testing.assert_allclose(built.market_excess_returns, 0.01)
    np.testing.assert_allclose(np.asarray(built.factor_returns)[:, -1], 0.005)
