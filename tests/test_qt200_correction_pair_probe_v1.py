"""Run on approved remote H100; no historical semantic promotion."""

from dataclasses import replace
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl_quant.data_sources.massive import qt200_correction_pair_probe_v1 as probe


def _row(code="0", *, ticker="AAA", sequence="1", trade_id="", price="10.00",
         size="100", ordinal=2, participant="100", sip="101"):
    values = [ticker, "", code, "4", trade_id, participant, price, sequence,
              sip, size, "1", "12", "0"]
    return dict(zip(probe.COLUMNS, values), source_row_number=ordinal)


def _arguments(tmp_path, rows, *, schema=None):
    path = tmp_path / "trades.parquet"
    schema = schema or pa.schema([(name, pa.string()) for name in probe.COLUMNS]
        + [("source_row_number", pa.uint64())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    path.chmod(0o444)
    return dict(parquet_path=path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_bytes=path.stat().st_size, expected_rows=len(rows), source_date="2017-01-03")


def test_blank_id_code_pairs_retain_both_values_without_role_inference(tmp_path):
    args = _arguments(tmp_path, [
        _row("1", price="10.01", ordinal=2),
        _row("12", price="10.00", ordinal=3, participant="200", sip="201"),
        _row("8", sequence="2", ordinal=4), _row("10", sequence="2", ordinal=5),
    ])
    before = args["parquet_path"].read_bytes()
    result = probe.probe_qt200_correction_pairs_v1(**args)
    assert result["retained_rows"] == 4 and result["candidate_key_count"] == 2
    comparison = result["groups"][0]["comparisons"][0]
    assert comparison["code_order_in_source"] == ["1", "12"]
    assert comparison["different_original_fields"] == ["correction", "participant_timestamp", "price", "sip_timestamp"]
    assert comparison["right_minus_left_timestamp_ns"]["sip_timestamp"] == 100
    assert comparison["economic_role_assignment"] == "not-established"
    assert all(row["provider_trade_id"] is None and row["raw_line_sha256"] is None
        for group in result["groups"] for row in group["rows"])
    assert all(result[key] is value for key, value in probe.CLAIMS.items())
    assert args["parquet_path"].read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trades.parquet"]


def test_second_pass_includes_ordinary_and_unknown_code_collisions(tmp_path):
    result = probe.probe_qt200_correction_pairs_v1(**_arguments(tmp_path, [
        _row("0", ordinal=2), _row("8", ordinal=3), _row("10", ordinal=4),
        _row("99", ordinal=5), _row("0", ticker="OTHER", ordinal=6),
    ]))
    group = result["groups"][0]
    assert group["status"] == "ambiguous-candidate-group"
    assert group["code_counts"] == {"0": 1, "8": 1, "10": 1, "99": 1}
    assert group["comparisons"] == [] and result["source_rows"] == 5


def test_missing_and_unmatched_keys_are_not_resolved(tmp_path):
    result = probe.probe_qt200_correction_pairs_v1(**_arguments(tmp_path, [
        _row("10", ordinal=2), _row("12", sequence="", ordinal=3),
        _row("8", ticker="OTHER", ordinal=4),
    ]))
    assert result["status_counts"] == {
        "missing-or-malformed-group-key": 1, "unmatched-candidate": 2}


def test_raw_sequence_lexemes_are_not_silently_normalized(tmp_path):
    result = probe.probe_qt200_correction_pairs_v1(**_arguments(tmp_path, [
        _row("8", sequence="01", ordinal=2), _row("10", sequence="1", ordinal=3)]))
    assert result["candidate_key_count"] == 2
    assert result["status_counts"] == {"unmatched-candidate": 2}


def test_no_correction_candidates_still_scans_all_rows(tmp_path):
    result = probe.probe_qt200_correction_pairs_v1(**_arguments(tmp_path, [_row()]))
    assert result["groups"] == [] and result["passes"] == 2


@pytest.mark.parametrize("damage", ["hash", "bytes", "rows", "writable", "symlink", "ordinal", "null", "schema"])
def test_exact_input_contract_rejects_corruption(tmp_path, damage):
    rows = [_row("8", ordinal=2), _row("10", ordinal=3)]
    if damage == "ordinal":
        rows[1]["source_row_number"] = 2
    if damage == "null":
        rows[0]["price"] = None
    schema = None
    if damage == "schema":
        schema = pa.schema([(name, pa.large_string()) for name in probe.COLUMNS]
            + [("source_row_number", pa.uint64())])
    args = _arguments(tmp_path, rows, schema=schema)
    if damage == "hash":
        args["expected_sha256"] = "f" * 64
    if damage == "bytes":
        args["expected_bytes"] += 1
    if damage == "rows":
        args["expected_rows"] += 1
    if damage == "writable":
        args["parquet_path"].chmod(0o644)
    if damage == "symlink":
        link = tmp_path / "link.parquet"
        link.symlink_to(args["parquet_path"])
        args["parquet_path"] = link
    with pytest.raises((ValueError, OSError)):
        probe.probe_qt200_correction_pairs_v1(**args)


@pytest.mark.parametrize("limit", ["maximum_rows", "maximum_keys", "maximum_candidate_rows",
    "maximum_candidate_bytes", "maximum_report_bytes"])
def test_bounded_failure_returns_no_partial_report(tmp_path, limit):
    args = _arguments(tmp_path, [_row("8", ordinal=2), _row("10", sequence="2", ordinal=3)])
    args["limits"] = replace(probe.CorrectionPairProbeLimits(), **{limit: 1})
    with pytest.raises(ValueError, match="allocation"):
        probe.probe_qt200_correction_pairs_v1(**args)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trades.parquet"]


@pytest.mark.parametrize("mode", [0o400, 0o444, 0o600, 0o700])
def test_completion_metadata_is_hash_bound_and_not_treated_as_authority(tmp_path, mode):
    args = _arguments(tmp_path, [_row("8")])
    complete = tmp_path / "completion.json"
    body = {
        "schema": "quanttrade-qt200-full-history-extraction-pilot-completion-v1",
        "training_ready": False, "output_sha256": args["expected_sha256"],
        "output_bytes": args["expected_bytes"], "selected_row_count": 1,
        "source_columns": list(probe.COLUMNS), "output_file": "trades.parquet",
        "source_object_key": "us_stocks_sip/trades_v1/2017/01/2017-01-03.csv.gz",
        "ordered_values_reopen_verified": True, "full_gzip_integrity_verified": True,
    }
    body["receipt_sha256"] = hashlib.sha256(probe._canonical(body)).hexdigest()
    complete.write_text(json.dumps(body))
    complete.chmod(mode)
    before = (complete.read_bytes(), complete.stat().st_mode)
    args.update(expected_completion_path=complete,
        expected_completion_sha256=hashlib.sha256(complete.read_bytes()).hexdigest())
    result = probe.probe_qt200_correction_pairs_v1(**args)
    assert result["completion_metadata_reloaded"] is True
    assert result["source_data_qualified"] is False
    assert (complete.read_bytes(), complete.stat().st_mode) == before
    args["expected_completion_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        probe.probe_qt200_correction_pairs_v1(**args)


def test_arbitrary_completion_json_is_not_promoted_by_a_matching_file_hash(tmp_path):
    args = _arguments(tmp_path, [_row("8")])
    complete = tmp_path / "completion.json"
    complete.write_text(json.dumps({"training_ready": False}))
    complete.chmod(0o444)
    args.update(expected_completion_path=complete,
        expected_completion_sha256=hashlib.sha256(complete.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="binding or semantic receipt"):
        probe.probe_qt200_correction_pairs_v1(**args)


@pytest.mark.parametrize("damage", ["content", "mode", "replacement"])
def test_owner_writable_historical_metadata_still_rejects_in_read_changes(tmp_path, damage):
    complete = tmp_path / "completion.json"
    original = b'{"historical":true}\n'
    complete.write_bytes(original)
    complete.chmod(0o700)
    digest = hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="changed"):
        with probe._bound_file(complete, digest, len(original), metadata_only=True) as stream:
            assert stream.read() == original
            if damage == "content":
                complete.write_bytes(original + b" ")
            elif damage == "mode":
                complete.chmod(0o600)
            else:
                complete.rename(tmp_path / "preserved-original.json")
                complete.write_bytes(original)
                complete.chmod(0o700)
