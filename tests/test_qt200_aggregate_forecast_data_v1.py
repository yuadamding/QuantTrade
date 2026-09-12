"""Persisted-source forecast-reader tests; executed only on remote LSF GPUs."""

import shutil

import pytest

import test_qt200_aggregate_targets_v1 as target_fixtures
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.training.qt200_aggregate_forecast_data_v1 import prepare_fit_inputs

# Explicitly expose the existing pytest fixture without an unused-name import.
persisted_inputs = target_fixtures.persisted_inputs


def inputs(p, **kwargs):
    options = {k: p[k] for k in ("feature_root", "feature_sha256", "target_root", "target_sha256")}
    options.update(fit_sessions=p["days"][64:69], cutoff_session=p["days"][70], heldout_start=p["days"][71])
    options.update(kwargs)
    return prepare_fit_inputs(**options)


def test_mature_label_access_and_fit_only_scaling(persisted_inputs):
    p = persisted_inputs
    result = inputs(p)
    assert len(result["samples"]) == 1000
    assert result["normalizer"]["counts"] == [1000] * 12
    assert all(row["targets"][2:] == [None] * 5 for row in result["samples"])
    assert result["mature_labels_by_bucket"]["price_0_1"] == 800
    assert result["mature_labels_by_bucket"]["price_1_5"] == 0
    assert not result["training_authorized"] and not result["model_fitted"]
    assert result["samples"][-1]["target_loss_mask"] == [False] * 7


def test_cutoff_cannot_enter_heldout_or_compress_fit_sessions(persisted_inputs):
    p = persisted_inputs
    for kwargs in (dict(cutoff_session=p["days"][71]), dict(heldout_start=p["days"][70]),
                   dict(cutoff_session=p["days"][67]), dict(fit_sessions=[p["days"][64], p["days"][66]])):
        with pytest.raises(ValueError):
            inputs(p, **kwargs)


def test_forecast_read_is_nonmaterializing(persisted_inputs):
    p = persisted_inputs
    roots = (p["feature_root"], p["target_root"])
    before = {str(path): transport.digest(path.read_bytes()) for root in roots for path in root.iterdir()}
    inputs(p)
    after = {str(path): transport.digest(path.read_bytes()) for root in roots for path in root.iterdir()}
    assert before == after


def test_changed_source_is_rejected_not_rebuilt(persisted_inputs, tmp_path):
    p = persisted_inputs
    root = tmp_path / "changed"
    shutil.copytree(p["target_root"], root)
    with (root / "targets.parquet").open("ab") as stream:
        stream.write(b"tampered")
    before = sorted(str(x) for x in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="file"):
        inputs(p, target_root=root)
    assert before == sorted(str(x) for x in tmp_path.rglob("*"))


def changed_target(p, tmp_path, mutate):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "changed"
    shutil.copytree(p["target_root"], root)
    table = pq.read_table(root / "targets.parquet")
    rows = table.to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), root / "targets.parquet", compression="zstd")
    parent = transport.parse_json((root / "COMPLETE.json").read_bytes())
    body = (root / "targets.parquet").read_bytes()
    parent["files"]["targets.parquet"].update(bytes=len(body), sha256=transport.digest(body))
    # Synthetic source corruption test: this is not a production authority.
    (root / "COMPLETE.json").write_bytes(transport.canonical(parent))
    return root, transport.digest((root / "COMPLETE.json").read_bytes())


def test_future_label_values_cannot_affect_earlier_fit_view(persisted_inputs, tmp_path):
    p = persisted_inputs
    reference = inputs(p)

    def mutate(rows):
        for row in rows:
            for i, maturity in enumerate(row["label_maturity_sessions"]):
                if row["values"][i] is not None and maturity > p["days"][70]:
                    row["values"][i] = 1000.0

    root, digest = changed_target(p, tmp_path, mutate)
    actual = inputs(p, target_root=root, target_sha256=digest)
    assert actual["samples"] == reference["samples"]
    assert actual["normalizer"] == reference["normalizer"]
    assert actual["mature_labels_by_bucket"] == reference["mature_labels_by_bucket"]


@pytest.mark.parametrize("field", ["feature_row_index", "label_maturity_sessions", "bucket_end_sessions"])
def test_authenticating_bytes_does_not_excuse_wrong_clock_or_link(persisted_inputs, tmp_path, field):
    p = persisted_inputs

    def mutate(rows):
        if field == "feature_row_index":
            rows[65][field] = 64
        else:
            rows[65][field][0] = p["days"][65]

    root, digest = changed_target(p, tmp_path, mutate)
    with pytest.raises(ValueError, match="linkage|maturity"):
        inputs(p, target_root=root, target_sha256=digest)
