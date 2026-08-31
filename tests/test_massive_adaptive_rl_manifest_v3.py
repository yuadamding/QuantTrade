from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
    MassiveAdaptiveRLExperimentManifestV3Error,
    build_massive_adaptive_rl_experiment_manifest_v3,
    load_massive_adaptive_rl_experiment_manifest_v3,
    write_massive_adaptive_rl_experiment_manifest_v3,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import main


def test_manifest_v3_freezes_final_report_gates_and_bootstrap(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="final-profitability-protocol"
    )
    assert manifest.final_gate_names == MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3
    assert "primary-net-log-return-lcb-positive" in manifest.final_gate_names
    assert manifest.bootstrap_replicates == 2_000
    assert manifest.bootstrap_block_sessions == 63
    assert manifest.bootstrap_seed == 0
    assert manifest.annualization_sessions == 252
    assert (
        manifest.risk_free_return_specification
        == "none-net-log-return-to-volatility-v1"
    )

    path = tmp_path / "manifest-v3.json"
    write_massive_adaptive_rl_experiment_manifest_v3(path=path, manifest=manifest)
    assert load_massive_adaptive_rl_experiment_manifest_v3(path) == manifest


def test_manifest_v3_rejects_post_registration_gate_changes() -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v3(
        experiment_id="tampered-final-profitability-protocol"
    )
    changed = replace(
        manifest,
        final_gate_names=tuple(
            name
            for name in manifest.final_gate_names
            if name != "primary-net-log-return-lcb-positive"
        ),
    )
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV3Error, match="differs"):
        changed.validate()


def test_manifest_v3_cli_dispatches_to_blocked_resumable_runner(
    tmp_path: Path, capsys
) -> None:
    manifest_path = tmp_path / "manifest-v3.json"
    assert (
        main(
            [
                "manifest-v3",
                "--experiment-id",
                "v3-cli-dispatch",
                "--output",
                str(manifest_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["validate", "--manifest", str(manifest_path)]) == 0
    capsys.readouterr()
    common = [
        "--manifest",
        str(manifest_path),
        "--source-root",
        str(tmp_path / "sources"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--device",
        "cpu",
    ]
    assert main(["run", *common]) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["current_stage"] == "blocked"
    assert first["blocker_code"] == "source-bundle-temporarily-absent"
    assert main(["resume", *common]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second["semantic_receipt_sha256"] == first["semantic_receipt_sha256"]
