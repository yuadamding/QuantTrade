"""Publication interruption/tamper regressions; executed only on LSF GPUs."""

import hashlib
import json

import pytest

from rl_quant.data_sources.massive.qt200_minute_recovery_v1 import recovery_partition, verify_published_bundle


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "TTWO"
    (root / "capture").mkdir(parents=True)
    plan, complete = raw({"fixture": "source plan"}), raw({"fixture": "complete source"})
    binding = raw(dict(ticker="TTWO", plan_sha256=sha(plan), completion_sha256=sha(complete)))
    files = {"capture/plan.json": plan, "capture/COMPLETE.json": complete,
             "acquisition-binding.json": binding, "capture/page.json.gz": b"retained bytes"}
    for path, body in files.items():
        (root / path).write_bytes(body)
    inv = raw(dict(ticker="TTWO", training_ready=False, acquisition_binding_sha256=sha(binding),
        files=[dict(path=p, bytes=len(b), sha256=sha(b)) for p, b in sorted(files.items())]))
    pub = raw(dict(ticker="TTWO", training_ready=False, remote_root=str(root),
                   inventory_sha256=sha(inv), acquisition_binding_sha256=sha(binding)))
    (root / "transfer-inventory.json").write_bytes(inv)
    (root / "PUBLICATION.json").write_bytes(pub)
    return root, dict(root=root, ticker="TTWO", inventory_sha256=sha(inv))


def test_lost_ack_recovers_complete_publication_without_writes(publication):
    root, args = publication
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = verify_published_bundle(**args)
    assert result["publication_sha256"] == sha((root / "PUBLICATION.json").read_bytes())
    assert result["training_ready"] is False
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["PUBLICATION.json", "capture/COMPLETE.json", "capture/page.json.gz"])
def test_partial_or_missing_publication_cannot_be_reused(publication, name):
    root, args = publication
    (root / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_published_bundle(**args)
    assert not (root / name).exists()


@pytest.mark.parametrize("name", ["capture/page.json.gz", "acquisition-binding.json", "transfer-inventory.json"])
def test_corrupted_committed_source_is_rejected(publication, name):
    root, args = publication
    (root / name).write_bytes(b"altered")
    with pytest.raises(ValueError):
        verify_published_bundle(**args)


def test_wrong_returned_ack_is_not_accepted(publication):
    _, args = publication
    with pytest.raises(ValueError):
        verify_published_bundle(**args, publication_sha256="0" * 64)


def test_wrong_ticker_is_not_accepted(publication):
    _, args = publication
    with pytest.raises(ValueError):
        verify_published_bundle(**{**args, "ticker": "AAPL"})


def test_complete_capture_is_not_downloaded_again():
    result = recovery_partition(ordered=("AAPL", "TTWO", "ROKU"), published=("AAPL",), captured=("AAPL", "TTWO"))
    assert result["reuse_publications"] == ("AAPL",)
    assert result["publish_existing_captures"] == ("TTWO",)
    assert result["acquire_missing"] == ("ROKU",)


@pytest.mark.parametrize("published,captured", [(("AAPL",), ()), (("GOOG",), ("GOOG",)),
    (("AAPL", "AAPL"), ("AAPL",)), ((), ("TTWO", "AAPL"))])
def test_ambiguous_recovery_population_fails(published, captured):
    with pytest.raises(ValueError):
        recovery_partition(ordered=("AAPL", "TTWO"), published=published, captured=captured)
