"""Persisted pretraining-input qualification; run only on approved LSF GPUs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import test_qt200_aggregate_targets_v1 as target_fixtures
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.training.qt200_aggregate_fit_package_v1 import (
    build_fit_plan, materialize_fit_package, verify_fit_package,
)

persisted_inputs = target_fixtures.persisted_inputs


def parents(p):
    return {key: p[key] for key in ("feature_root", "feature_sha256", "target_root", "target_sha256")}


def tree(*roots):
    return {str(path): transport.digest(path.read_bytes()) for root in roots
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def fit_package(persisted_inputs, tmp_path_factory):
    p = persisted_inputs
    plan = build_fit_plan(sessions=p["days"], fit_lengths=[68, 70])
    output = tmp_path_factory.mktemp("fit-package") / "prepared"
    report = materialize_fit_package(**parents(p), plan=plan, output=output)
    return dict(p=p, plan=plan, output=output, report=report,
                complete_sha256=transport.digest((output / "COMPLETE.json").read_bytes()),
                plan_sha256=transport.digest(transport.canonical(plan)))


def replay(f, **changes):
    options = dict(**parents(f["p"]), output=f["output"],
                   complete_sha256=f["complete_sha256"], plan_sha256=f["plan_sha256"])
    options.update(changes)
    return verify_fit_package(**options)


def test_plan_is_calendar_only_and_does_not_register_a_study(persisted_inputs):
    days = persisted_inputs["days"]
    plan = build_fit_plan(sessions=days, fit_lengths=[68, 70])
    assert plan["views"][0]["fit_sessions"] == days[:68]
    assert plan["views"][0]["cutoff_session"] == days[67]
    assert plan["views"][0]["heldout_start"] == days[68]
    assert not plan["training_authorized"] and not plan["outer_access_authorized"]
    assert not plan["study_selection_frozen"] and not plan["model_fitted"]


@pytest.mark.parametrize("lengths", [[], [True], [0], [-1], [80], [68, 68], [70, 68], [68.0]])
def test_invalid_fit_lengths_rejected(persisted_inputs, lengths):
    with pytest.raises(ValueError, match="lengths"):
        build_fit_plan(sessions=persisted_inputs["days"], fit_lengths=lengths)


def test_altered_cutoff_rejected_before_publication(persisted_inputs, tmp_path):
    p = persisted_inputs
    plan = build_fit_plan(sessions=p["days"], fit_lengths=[68])
    plan["views"][0]["cutoff_session"] = p["days"][68]
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="plan"):
        materialize_fit_package(**parents(p), plan=plan, output=output)
    assert not output.exists()


def test_persisted_rows_keep_missing_features_and_unmatured_samples(fit_package):
    import pyarrow.parquet as pq

    f = fit_package
    rows = pq.read_table(f["output"] / "fit-inputs.parquet").to_pylist()
    assert len(rows) == 200 * (68 + 70) == f["report"]["rows"]
    assert rows[0]["features"] == rows[0]["normalized_features"] == [None] * 12
    assert rows[67]["target_loss_mask"] == [False] * 7
    assert rows[67]["targets"] == [None] * 7
    assert rows[64]["target_loss_mask"][0]
    assert all(row["training_authorized"] is False for row in rows)
    assert all(v["rows_without_mature_targets_retained"] == 600 for v in f["report"]["views"])
    assert f["report"]["normalization_fitted"] and not f["report"]["forecast_fitted"]
    assert not f["report"]["training_ready"] and not f["report"]["native_v5_qualified"]
    normalizers = json.loads((f["output"] / "normalizers.json").read_bytes())
    n = normalizers["history-0068"]
    assert rows[64]["normalized_features"][0] == pytest.approx(
        (rows[64]["features"][0] - n["mean"][0]) / n["scale"][0])
    assert n["counts"][0] == 200 * 67  # Label maturity did not select scaler rows.


def test_exact_replay_is_read_only(fit_package):
    f = fit_package
    roots = (f["p"]["feature_root"], f["p"]["target_root"], f["output"])
    before = tree(*roots)
    result = replay(f)
    assert result["rows_reconstructed"] == 27600 and result["views_reconstructed"] == 2
    assert result["nonmaterializing"] and result["exact_rows_and_normalizers_reproduced"]
    assert not result["model_fitted"] and not result["training_authorized"]
    assert tree(*roots) == before


def test_bad_parent_blocks_before_creating_output(persisted_inputs, tmp_path):
    p = persisted_inputs
    options = parents(p)
    options["feature_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parent"):
        materialize_fit_package(**options, plan=build_fit_plan(sessions=p["days"], fit_lengths=[68]),
                                output=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_existing_output_is_never_overwritten(fit_package):
    f = fit_package
    before = tree(f["output"])
    with pytest.raises(FileExistsError):
        materialize_fit_package(**parents(f["p"]), plan=f["plan"], output=f["output"])
    assert tree(f["output"]) == before


def test_missing_evidence_is_not_regenerated(fit_package, tmp_path):
    root = tmp_path / "missing"
    shutil.copytree(fit_package["output"], root)
    (root / "fit-inputs.parquet").unlink()
    before = tree(root)
    with pytest.raises(FileNotFoundError):
        replay(fit_package, output=root)
    assert tree(root) == before


def test_changed_parent_blocks_replay(fit_package, tmp_path):
    root = tmp_path / "changed-targets"
    shutil.copytree(fit_package["p"]["target_root"], root)
    with (root / "targets.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file"):
        replay(fit_package, target_root=root)


def reseal(root, name):
    # Explicit synthetic corruption fixture; this cannot qualify real sources.
    report = json.loads((root / "COMPLETE.json").read_bytes())
    body = (root / name).read_bytes()
    report["files"][name].update(bytes=len(body), sha256=transport.digest(body))
    (root / "COMPLETE.json").chmod(0o600)  # Only the copied synthetic attack fixture.
    (root / "COMPLETE.json").write_bytes(transport.canonical(report))
    return transport.digest((root / "COMPLETE.json").read_bytes())


def test_rehashed_normalizer_must_still_reconstruct(fit_package, tmp_path):
    root = tmp_path / "changed-scale"
    shutil.copytree(fit_package["output"], root)
    obj = json.loads((root / "normalizers.json").read_bytes())
    obj["history-0068"]["mean"][0] += 1
    (root / "normalizers.json").chmod(0o600)
    (root / "normalizers.json").write_bytes(transport.canonical(obj))
    with pytest.raises(ValueError, match="normalizer"):
        replay(fit_package, output=root, complete_sha256=reseal(root, "normalizers.json"))


@pytest.mark.parametrize("field", ["normalized_features", "target_loss_mask", "training_authorized",
                                  "feature_row_index", "decision_session"])
def test_rehashed_rows_cannot_forge_values_masks_links_or_authorization(fit_package, tmp_path, field):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "changed-rows"
    shutil.copytree(fit_package["output"], root)
    table = pq.read_table(root / "fit-inputs.parquet")
    rows = table.to_pylist()
    if field == "normalized_features":
        rows[64][field][0] += 1
    elif field == "target_loss_mask":
        rows[67][field][0] = True
    elif field == "training_authorized":
        rows[64][field] = True
    elif field == "feature_row_index":
        rows[64][field] = 63
    else:
        rows[64][field] = fit_package["p"]["days"][63]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), root / "fit-inputs.parquet", compression="zstd")
    with pytest.raises(ValueError, match="reconstruction"):
        replay(fit_package, output=root, complete_sha256=reseal(root, "fit-inputs.parquet"))


def test_rehashed_summary_cannot_promote_preparation_to_training(fit_package, tmp_path):
    root = tmp_path / "changed-report"
    shutil.copytree(fit_package["output"], root)
    report = json.loads((root / "COMPLETE.json").read_bytes())
    report["historical_identity_qualified"] = True
    body = transport.canonical(report)
    (root / "COMPLETE.json").chmod(0o600)
    (root / "COMPLETE.json").write_bytes(body)
    with pytest.raises(ValueError, match="authorization"):
        replay(fit_package, output=root, complete_sha256=transport.digest(body))


def test_interrupted_completion_remains_uncommitted_and_is_not_overwritten(persisted_inputs, tmp_path, monkeypatch):
    p = persisted_inputs
    output = tmp_path / "interrupted"
    original = transport.write_once

    def interrupt(path, body):
        if path == output / "COMPLETE.json":
            raise OSError("injected publication interruption")
        return original(path, body)

    monkeypatch.setattr(transport, "write_once", interrupt)
    plan = build_fit_plan(sessions=p["days"], fit_lengths=[68])
    with pytest.raises(OSError, match="interruption"):
        materialize_fit_package(**parents(p), plan=plan, output=output)
    assert (output / "fit-inputs.parquet").is_file() and not (output / "COMPLETE.json").exists()
    before = tree(output)
    with pytest.raises(FileExistsError):
        materialize_fit_package(**parents(p), plan=plan, output=output)
    assert tree(output) == before


def test_fresh_process_reconstructs_without_writes(fit_package):
    f = fit_package
    args = dict(**parents(f["p"]), output=f["output"], complete_sha256=f["complete_sha256"],
                plan_sha256=f["plan_sha256"])
    args = {k: str(v) if isinstance(v, Path) else v for k, v in args.items()}
    code = (
        "import json,sys;from pathlib import Path;"
        "from rl_quant.training.qt200_aggregate_fit_package_v1 import verify_fit_package;"
        "a=json.loads(sys.argv[1]);"
        "a.update({k:Path(a[k]) for k in ('feature_root','target_root','output')});"
        "print(json.dumps(verify_fit_package(**a),sort_keys=True))"
    )
    roots = (f["p"]["feature_root"], f["p"]["target_root"], f["output"])
    before = tree(*roots)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(args)], env=env,
                           capture_output=True, text=True, timeout=120, check=True)
    assert json.loads(child.stdout) == replay(f)
    assert tree(*roots) == before
