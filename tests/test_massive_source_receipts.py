from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive import source_receipts
from rl_quant.data_sources.massive.source_receipts import (
    MassiveSourceObjectError,
    load_massive_source_bundle,
    load_massive_source_object,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
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
    def fail_write_once(directory_fd: int, name: str, payload: object):
        raise OSError(f"injected receipt failure: {name}")

    monkeypatch.setattr(
        source_receipts, "_canonical_write_once_at", fail_write_once
    )

    with pytest.raises(OSError, match="injected receipt failure"):
        _publish(tmp_path)

    assert tuple(tmp_path.rglob("*.*")) == ()


def test_interruption_after_sidecar_install_is_preserved_and_never_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write_once = source_receipts._canonical_write_once_at
    installed_sidecar: Path | None = None

    def install_then_interrupt(directory_fd: int, name: str, payload: object):
        nonlocal installed_sidecar
        result = real_write_once(directory_fd, name, payload)
        if installed_sidecar is None:
            installed_sidecar = (
                tmp_path / "bronze/trades/2026/08" / name
            )
            raise OSError("injected interruption after sidecar install")
        return result

    monkeypatch.setattr(
        source_receipts, "_canonical_write_once_at", install_then_interrupt
    )
    with pytest.raises(OSError, match="after sidecar install"):
        _publish(tmp_path)

    assert installed_sidecar is not None
    assert installed_sidecar.is_file()
    preserved = installed_sidecar.read_bytes()
    assert not (
        tmp_path / "bronze/trades/2026/08/2026-08-20.csv.gz"
    ).exists()
    assert not installed_sidecar.with_name(
        "2026-08-20.csv.gz.commit.json"
    ).exists()

    monkeypatch.setattr(
        source_receipts, "_canonical_write_once_at", real_write_once
    )
    with pytest.raises(MassiveSourceObjectError, match="already exists"):
        _publish(tmp_path)
    assert installed_sidecar.read_bytes() == preserved


@pytest.mark.parametrize(
    "present_suffixes",
    (("",), ("", ".receipt.json"), (".receipt.json", ".commit.json")),
)
def test_source_publication_never_repairs_preexisting_partial_transaction(
    tmp_path: Path, present_suffixes: tuple[str, ...]
) -> None:
    payload = tmp_path / "bronze/trades/2026/08/2026-08-20.csv.gz"
    payload.parent.mkdir(parents=True)
    for suffix in present_suffixes:
        payload.with_name(payload.name + suffix).write_bytes(
            f"partial:{suffix}".encode()
        )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(MassiveSourceObjectError, match="already exists"):
        _publish(tmp_path)

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_source_publication_rejects_an_orphaned_temporary_payload(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "bronze/trades/2026/08"
    parent.mkdir(parents=True)
    orphan = parent / ".2026-08-20.csv.gz.interrupted.partial"
    orphan.write_bytes(b"uncommitted payload fragment")

    with pytest.raises(MassiveSourceObjectError, match="incomplete"):
        _publish(tmp_path)

    assert orphan.read_bytes() == b"uncommitted payload fragment"
    assert not (parent / "2026-08-20.csv.gz").exists()
    assert not (parent / "2026-08-20.csv.gz.receipt.json").exists()
    assert not (parent / "2026-08-20.csv.gz.commit.json").exists()


def test_source_publication_rejects_intermediate_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "bronze").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MassiveSourceObjectError, match="symlink"):
        _publish(tmp_path)

    assert tuple(outside.iterdir()) == ()


def test_source_loader_rejects_final_symlink(tmp_path: Path) -> None:
    payload = tmp_path / "target.bin"
    payload.write_bytes(b"not-authority")
    nested = tmp_path / "bronze/trades/2026/08"
    nested.mkdir(parents=True)
    (nested / "2026-08-20.csv.gz").symlink_to(payload)

    with pytest.raises(MassiveSourceObjectError, match="no-follow"):
        load_massive_source_object(
            root=tmp_path,
            relative_payload_path="bronze/trades/2026/08/2026-08-20.csv.gz",
        )


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


def test_loaded_bundle_rejects_same_path_replacement(tmp_path: Path) -> None:
    _publish(tmp_path)
    relative = "bronze/trades/2026/08/2026-08-20.csv.gz"
    loaded = load_massive_source_bundle(
        root=tmp_path, relative_payload_path=relative, verified_at_ms=1_300
    )
    payload = tmp_path / relative
    payload.unlink()
    payload.write_bytes(b"massive-source-bytes")

    with pytest.raises(MassiveSourceObjectError, match="inode was replaced"):
        read_loaded_massive_source_bytes(root=tmp_path, loaded_source=loaded)


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
