from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive import source_receipts
from rl_quant.data_sources.massive.source_receipts import (
    MassiveSourceObjectError,
    load_massive_source_object,
    publish_massive_source_object,
)


def _publish(root: Path, payload: bytes = b"massive-source-bytes"):
    return publish_massive_source_object(
        stream=BytesIO(payload),
        root=root,
        relative_payload_path="bronze/trades/2026/08/2026-08-20.csv.gz",
        dataset_id="us_stocks_sip/trades_v1",
        source_object_key="us_stocks_sip/trades_v1/2026/08/2026-08-20.csv.gz",
        requested_at_ms=1_000,
        downloaded_at_ms=1_100,
        schema_sha256="a" * 64,
        entitlement_receipt_sha256="b" * 64,
        committed_at_ms=1_200,
        etag="etag-v1",
        request_id="request-v1",
    )


def test_source_publication_round_trips_payload_receipt_and_commit(tmp_path: Path) -> None:
    receipt, commit = _publish(tmp_path)
    loaded_receipt, loaded_commit = load_massive_source_object(
        root=tmp_path,
        relative_payload_path="bronze/trades/2026/08/2026-08-20.csv.gz",
    )

    assert loaded_receipt == receipt
    assert loaded_commit == commit
    assert receipt.content_length == len(b"massive-source-bytes")
    assert commit.source_receipt_sha256 == receipt.receipt_sha256


def test_source_publication_is_create_only(tmp_path: Path) -> None:
    _publish(tmp_path)
    with pytest.raises(MassiveSourceObjectError, match="already exists"):
        _publish(tmp_path)


def test_source_hash_mismatch_publishes_nothing(tmp_path: Path) -> None:
    with pytest.raises(MassiveSourceObjectError, match="hash mismatch"):
        publish_massive_source_object(
            stream=BytesIO(b"wrong"),
            root=tmp_path,
            relative_payload_path="object.bin",
            dataset_id="dataset",
            source_object_key="remote/object.bin",
            requested_at_ms=1,
            downloaded_at_ms=2,
            schema_sha256="a" * 64,
            entitlement_receipt_sha256="b" * 64,
            committed_at_ms=3,
            expected_physical_sha256="c" * 64,
        )
    assert not (tmp_path / "object.bin").exists()


def test_invalid_metadata_is_rejected_before_final_publication(tmp_path: Path) -> None:
    with pytest.raises(MassiveSourceObjectError, match="predates"):
        publish_massive_source_object(
            stream=BytesIO(b"payload"),
            root=tmp_path,
            relative_payload_path="object.bin",
            dataset_id="dataset",
            source_object_key="remote/object.bin",
            requested_at_ms=20,
            downloaded_at_ms=10,
            schema_sha256="a" * 64,
            entitlement_receipt_sha256="b" * 64,
            committed_at_ms=30,
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_receipt_write_failure_rolls_back_the_linked_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write_once(path: Path, payload: object) -> str:
        raise OSError(f"injected receipt failure: {path.name}")

    monkeypatch.setattr(source_receipts, "_canonical_write_once", fail_write_once)

    with pytest.raises(OSError, match="injected receipt failure"):
        _publish(tmp_path)

    assert tuple(tmp_path.rglob("*.*")) == ()


def test_source_mutation_fails_reopen(tmp_path: Path) -> None:
    _publish(tmp_path)
    payload = tmp_path / "bronze/trades/2026/08/2026-08-20.csv.gz"
    payload.chmod(0o644)
    payload.write_bytes(b"mutated")
    with pytest.raises(MassiveSourceObjectError, match="bytes changed"):
        load_massive_source_object(
            root=tmp_path,
            relative_payload_path="bronze/trades/2026/08/2026-08-20.csv.gz",
        )


@pytest.mark.parametrize(
    ("dataset_id", "relative_path"),
    (
        ("us_stocks_sip/trades_v1", "bronze/trades/one.csv.gz"),
        ("us_stocks_sip/minute_aggs_v1", "bronze/minute_aggs/one.csv.gz"),
        ("us_stocks_sip/day_aggs_v1", "bronze/day_aggs/one.csv.gz"),
    ),
)
def test_trade_minute_and_day_source_kinds_share_one_byte_authority(
    tmp_path: Path, dataset_id: str, relative_path: str
) -> None:
    receipt, _ = publish_massive_source_object(
        stream=BytesIO(dataset_id.encode()),
        root=tmp_path,
        relative_payload_path=relative_path,
        dataset_id=dataset_id,
        source_object_key=f"{dataset_id}/2026/08/one.csv.gz",
        requested_at_ms=1,
        downloaded_at_ms=2,
        schema_sha256="a" * 64,
        entitlement_receipt_sha256="b" * 64,
        committed_at_ms=3,
    )
    loaded, _ = load_massive_source_object(
        root=tmp_path, relative_payload_path=relative_path
    )

    assert loaded == receipt
