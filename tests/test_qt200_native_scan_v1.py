"""H100-target native scan prerequisites; never run a model or grant authority."""

from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import os
import stat

import pytest

from rl_quant.data_sources.massive import qt200_native_scan_v1 as candidate
from rl_quant.data_sources.massive.corrections import build_massive_correction_authority
from rl_quant.data_sources.massive.finalized_daily_scan import scan_massive_daily_trade_file_v0
from rl_quant.data_sources.massive.selected_trade_scan_v1 import scan_massive_selected_trade_file_v1
from rl_quant.data_sources.massive.session_calendar import build_massive_session_authority
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, file_sha256, semantic_sha256
from test_massive_selected_trade_scan_v1 import HEADER, _source
from test_qt200_market_day_v1 import _authorities, _trade


def _arguments(tmp_path: Path, content: bytes, *, compressed=None):
    source = _source(tmp_path / "raw", content, compressed=compressed)
    sessions, session, _, corrections = _authorities()
    return dict(source_request=source, output_directory=tmp_path / "result",
                output_budget_bytes=1024 * 1024, session_authority=sessions,
                session=session, correction_authority=corrections)


def _publish(args):
    return candidate.scan_and_publish_qt200_native_trade_file_v1(**args)


def _reconstruct(args, **changes):
    values = dict(root=args["source_request"]["root"], output_directory=args["output_directory"],
                  expected_completion_sha256=file_sha256(args["output_directory"] / "COMPLETE.json"),
                  session_authority=args["session_authority"], session=args["session"],
                  correction_authority=args["correction_authority"], verified_at_ms=5)
    values.update(changes)
    return candidate.reconstruct_qt200_native_trade_file_v1(**values)


def test_repeated_create_only_publications_are_durable_sealed_single_links(tmp_path):
    for index in range(64):
        name = f"receipt-{index:03d}.json"
        value = dict(index=index, diagnostic_only=True)
        entry = candidate._publish(tmp_path, name, value, 1024)
        replayed, _ = candidate._read(tmp_path / name, entry["sha256"], entry["bytes"])
        assert replayed == value
        assert (tmp_path / name).stat().st_nlink == 1
        assert not (tmp_path / (name + ".partial")).exists()


def test_publisher_closes_its_temporary_handle_before_unlink(tmp_path, monkeypatch):
    original_open, original_unlink = os.open, Path.unlink
    owned = []

    def capture(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if Path(path) == tmp_path / "receipt.json.partial":
            owned.append(fd)
        return fd

    def closed_unlink(path, *args, **kwargs):
        if path == tmp_path / "receipt.json.partial":
            assert len(owned) == 1
            with pytest.raises(OSError):
                os.fstat(owned[0])
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(candidate.os, "open", capture)
    monkeypatch.setattr(Path, "unlink", closed_unlink)
    candidate._publish(tmp_path, "receipt.json", dict(diagnostic_only=True), 1024)
    assert (tmp_path / "receipt.json").stat().st_nlink == 1


def test_generic_replay_still_rejects_an_extra_hardlink(tmp_path):
    entry = candidate._publish(tmp_path, "receipt.json", dict(diagnostic_only=True), 1024)
    os.link(tmp_path / "receipt.json", tmp_path / "not-owned-by-publisher")
    with pytest.raises(candidate.Qt200NativeScanError, match="nlink=2"):
        candidate._read(tmp_path / "receipt.json", entry["sha256"], entry["bytes"])


def test_full_scan_matches_native_and_reconstructs_with_fresh_loaded_time(tmp_path: Path, monkeypatch) -> None:
    lines = [
        _trade("P", 29, "10.000", "100.00", correction=7),
        _trade("U", 36, "20", "2", ticker="UNSELECTED", sequence=2),
        _trade("P", 29, "10.000", "100.00", correction=11, sip_hour=17, sip_minute=0, sequence=3),
        _trade("DOT", 36, "30", "3", ticker="BRK.B", sequence=4),
        _trade("SLASH", 36, "40", "4", ticker="BRK/B", sequence=5),
    ]
    content = HEADER + b"".join(lines)
    args = _arguments(tmp_path, content)
    calls = []
    original = candidate.scan_massive_daily_trade_file_v0

    def observed(**kwargs):
        assert kwargs["retain_rows"] is False
        calls.append(kwargs["loaded_source"])
        return original(**kwargs)

    monkeypatch.setattr(candidate, "scan_massive_daily_trade_file_v0", observed)
    completion = _publish(args)
    directory = args["output_directory"]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    proof = json.loads(before["prerequisite.json"])
    stored_native = json.loads(before["native-scan.json"])
    assert set(before) == {"native-scan.json", "prerequisite.json", "COMPLETE.json"}
    assert completion["source_rows"] == 5 and completion["selected_rows"] == 3
    assert proof["retained_native_row_count"] == 0
    assert proof["all_source_canonical_correction_code_counts"] == {"0": 3, "7": 1, "11": 1}
    assert proof["selection_preflight"]["decompressed_sha256"] == hashlib.sha256(content).hexdigest()
    assert all(completion[key] is False for key in candidate._CLAIMS)
    assert proof["original_gzip_required_for_replay"] is True
    assert stored_native["post_close_correction_row_count"] == 1
    for path in directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444 and path.stat().st_nlink == 1
    selected = scan_massive_selected_trade_file_v1(**args["source_request"], row_sink=lambda _: None)
    retained, reference = scan_massive_daily_trade_file_v0(
        root=args["source_request"]["root"], loaded_source=selected.loaded_source,
        session_authority=args["session_authority"], session=args["session"],
        correction_authority=args["correction_authority"], retain_rows=True,
    )
    assert len(retained) == 5
    assert [row.source_row_number for row in retained] == [2, 3, 4, 5, 6]
    assert [row.raw_row_sha256 for row in retained] == [hashlib.sha256(line).hexdigest() for line in lines]
    assert json.loads(canonical_json_file_bytes(asdict(reference))) == stored_native
    fresh = _reconstruct(args)
    assert len(calls) == 2
    assert fresh.loaded_source_receipt_sha256 != reference.loaded_source_receipt_sha256
    assert candidate._native_semantics(fresh) == candidate._native_semantics(reference)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert file_sha256(args["source_request"]["root"] / "trades/day.csv.gz") == args["source_request"]["expected_compressed_sha256"]


@pytest.mark.parametrize("bad", ["correction", "price"])
def test_unselected_bad_record_cannot_escape_full_native_validation(tmp_path: Path, bad: str) -> None:
    row = _trade("BAD", 36, "" if bad == "price" else "10", "1",
                 ticker="OTHER", correction=99 if bad == "correction" else 0)
    args = _arguments(tmp_path, HEADER + _trade("OK", 36, "1", "1") + row)
    with pytest.raises(ValueError):
        _publish(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_failure_histogram_distinguishes_all_observed_rejection_reasons(tmp_path: Path) -> None:
    from test_massive_selected_trade_scan_v1 import _line

    args = _arguments(tmp_path, HEADER + _line(conditions="true") * 5 + _line(price="") * 7)
    with pytest.raises(candidate.Qt200NativeScanError) as failure:
        _publish(args)
    message = str(failure.value)
    assert "count=12" in message and "error_counts=" in message
    assert "flat-file conditions are malformed': 5" in message
    assert "price must be decimal': 7" in message
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_unknown_condition_is_preserved_but_not_condition_qualified(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("C", 36, "1", "1", conditions=(999,)))
    completion = _publish(args)
    assert completion["source_rows"] == 1
    assert completion["condition_eligibility_qualified"] is False
    assert completion["correction_chains_replayed"] is False


@pytest.mark.parametrize("code", [1, 12])
@pytest.mark.parametrize("ticker", ["AAA", "UNSELECTED"])
def test_retrospective_corrected_values_cannot_enter_forward_replay(
    tmp_path: Path, code: int, ticker: str,
) -> None:
    # A nonblank ID and syntactically valid positive trade are insufficient:
    # the provider's 01/12 payload orientation is retrospective, not forward.
    args = _arguments(tmp_path, HEADER + _trade("PRESENT-ID", 36, "10", "100",
                                               correction=code, ticker=ticker))
    args["correction_authority"] = build_massive_correction_authority(
        [(0, "new-trade"), (1, "new-trade"), (12, "replacement")],
        canary_receipt_sha256="a" * 64,
    )
    with pytest.raises(candidate.Qt200NativeScanError, match="historical retrospective"):
        _publish(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_zero_volume_close_failure_is_classified_without_manufacturing_liquidity(tmp_path: Path) -> None:
    line = b'AAA,"38,41",0,12,34110,1483478100016366686,116.15,2303896,1483478100016390377,0,3,0,0\n'
    args = _arguments(tmp_path, HEADER + line)
    with pytest.raises(candidate.Qt200NativeScanError, match="zero-volume-corrected-close-candidate"):
        _publish(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_provider_singleton_and_comma_cells_complete_native_scan_and_replay(tmp_path: Path) -> None:
    from test_massive_selected_trade_scan_v1 import _line

    lines = (_line(conditions="12"), _line("UNSELECTED", conditions="12,37"),
             _line("BRK.B", conditions="0"))
    args = _arguments(tmp_path, HEADER + b"".join(lines))
    completion = _publish(args)
    assert completion["source_rows"] == 3 and completion["selected_rows"] == 2
    assert completion["condition_eligibility_qualified"] is False
    assert _reconstruct(args).source_row_count == 3


def test_failed_scan_reports_bounded_original_row_diagnostics(tmp_path: Path) -> None:
    from test_massive_selected_trade_scan_v1 import _line

    bad = _line(conditions="true")
    args = _arguments(tmp_path, HEADER + bad * 20)
    with pytest.raises(candidate.Qt200NativeScanError) as failure:
        _publish(args)
    message = str(failure.value)
    assert "count=20" in message
    assert message.count("source_row_number") == 4
    assert "flat-file conditions are malformed" in message
    assert hashlib.sha256(bad).hexdigest() in message
    assert len(message) < 1800
    assert not (args["output_directory"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("damage", ["truncated", "crc", "trailing"])
def test_gzip_damage_prevents_any_completion(tmp_path: Path, damage: str) -> None:
    content = HEADER + _trade("A", 36, "1", "1")
    compressed = gzip.compress(content, mtime=0)
    if damage == "truncated":
        compressed = compressed[:-4]
    elif damage == "crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    else:
        compressed += b"not-a-member"
    args = _arguments(tmp_path, content, compressed=compressed)
    with pytest.raises((ValueError, EOFError, OSError)):
        _publish(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_wrong_original_transaction_hash_cannot_publish(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    args["source_request"]["expected_receipt_file_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        _publish(args)
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_progress_cancellation_never_publishes_completion(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))

    def cancelled(phase, *_):
        if phase == "native-complete":
            raise RuntimeError("caller cancellation")

    with pytest.raises(RuntimeError, match="caller cancellation"):
        _publish(dict(args, progress_callback=cancelled))
    assert not (args["output_directory"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("operation", ["publish", "reconstruct"])
def test_completion_callback_source_mutation_is_detected(tmp_path: Path, operation: str) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    if operation == "reconstruct":
        _publish(args)
    before = {p.name: p.read_bytes() for p in args["output_directory"].iterdir()} if operation == "reconstruct" else {}

    def mutate(phase, *_):
        if phase == "native-complete":
            path = args["source_request"]["root"] / "trades/day.csv.gz"
            body = path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(body[:4] + bytes([body[4] ^ 1]) + body[5:])

    with pytest.raises(ValueError, match="source identity changed"):
        if operation == "publish":
            _publish(dict(args, progress_callback=mutate))
        else:
            _reconstruct(args, progress_callback=mutate)
    if operation == "publish":
        assert not (args["output_directory"] / "COMPLETE.json").exists()
    else:
        assert {p.name: p.read_bytes() for p in args["output_directory"].iterdir()} == before


def test_interrupted_publication_retains_partial_and_cannot_reconstruct(tmp_path: Path, monkeypatch) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))

    def fail_link(*_, **__):
        raise OSError("injected publication interruption")

    monkeypatch.setattr(candidate.os, "link", fail_link)
    with pytest.raises(OSError, match="publication interruption"):
        _publish(args)
    directory = args["output_directory"]
    assert {path.name for path in directory.iterdir()} == {"native-scan.json.partial"}
    assert stat.S_IMODE((directory / "native-scan.json.partial").stat().st_mode) == 0o444
    with pytest.raises(candidate.Qt200NativeScanError, match="incomplete"):
        candidate.reconstruct_qt200_native_trade_file_v1(
            root=args["source_request"]["root"], output_directory=directory,
            expected_completion_sha256="f" * 64, session_authority=args["session_authority"],
            session=args["session"], correction_authority=args["correction_authority"], verified_at_ms=5,
        )
    assert {path.name for path in directory.iterdir()} == {"native-scan.json.partial"}


def test_output_budget_failure_retains_unaccepted_attempt(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    with pytest.raises(candidate.Qt200NativeScanError, match="allocation"):
        _publish(dict(args, output_budget_bytes=1))
    assert args["output_directory"].is_dir()
    assert not (args["output_directory"] / "COMPLETE.json").exists()


def test_completed_attempt_cannot_be_overwritten(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    _publish(args)
    before = {p.name: p.read_bytes() for p in args["output_directory"].iterdir()}
    with pytest.raises(FileExistsError):
        _publish(args)
    assert {p.name: p.read_bytes() for p in args["output_directory"].iterdir()} == before


def test_reconstruction_rehashes_compressed_header_not_only_crc(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    _publish(args)
    path = args["source_request"]["root"] / "trades/day.csv.gz"
    body = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(body[:4] + bytes([body[4] ^ 1]) + body[5:])
    with pytest.raises(ValueError, match="compressed source hash"):
        _reconstruct(args)


@pytest.mark.parametrize("which", ["session_authority", "correction_authority"])
def test_reconstruction_rejects_invalid_native_support(tmp_path: Path, which: str) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    _publish(args)
    changed = replace(args[which], receipt_sha256="f" * 64)
    with pytest.raises(ValueError):
        _reconstruct(args, **{which: changed})


@pytest.mark.parametrize("which", ["calendar", "corrections"])
def test_valid_different_native_support_cannot_reconstruct_original(tmp_path: Path, which: str) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    _publish(args)
    if which == "calendar":
        session = replace(args["session"], calendar_source_receipt_sha256="c" * 64)
        changes = dict(session=session, session_authority=build_massive_session_authority(
            (session,), calendar_source_receipt_sha256="c" * 64))
    else:
        rules = [(row.correction_code, row.semantic_kind) for row in args["correction_authority"].rules]
        changes = dict(correction_authority=build_massive_correction_authority(
            rules, canary_receipt_sha256="c" * 64))
    with pytest.raises(candidate.Qt200NativeScanError, match="fresh reconstruction differs"):
        _reconstruct(args, **changes)


def test_read_only_reconstruction_does_not_recreate_missing_evidence(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    _publish(args)
    path = args["output_directory"] / "native-scan.json"
    path.unlink()
    with pytest.raises(candidate.Qt200NativeScanError, match="incomplete"):
        _reconstruct(args)
    assert not path.exists()


def test_even_resealed_metadata_cannot_claim_identity_qualification(tmp_path: Path) -> None:
    args = _arguments(tmp_path, HEADER + _trade("A", 36, "1", "1"))
    completion = _publish(args)
    proof_path = args["output_directory"] / "prerequisite.json"
    proof = json.loads(proof_path.read_bytes())
    proof["identity_qualified"] = True
    proof["receipt_sha256"] = semantic_sha256({k: v for k, v in proof.items() if k != "receipt_sha256"})
    body = canonical_json_file_bytes(proof)
    proof_path.chmod(0o600)
    proof_path.write_bytes(body)
    proof_path.chmod(0o444)
    completion["files"][1].update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    completion["receipt_sha256"] = semantic_sha256({k: v for k, v in completion.items() if k != "receipt_sha256"})
    path = args["output_directory"] / "COMPLETE.json"
    path.chmod(0o600)
    path.write_bytes(canonical_json_file_bytes(completion))
    path.chmod(0o444)
    with pytest.raises(candidate.Qt200NativeScanError, match="prerequisite contract"):
        _reconstruct(args)
