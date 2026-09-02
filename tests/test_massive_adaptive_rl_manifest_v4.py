from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v4 import (
    MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256,
    MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4,
    MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1,
    MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256,
    MassiveAdaptiveRLExperimentManifestV4Error,
    build_massive_adaptive_rl_experiment_manifest_v4,
    load_massive_adaptive_rl_experiment_manifest_v4,
    write_massive_adaptive_rl_experiment_manifest_v4,
)
from rl_quant.workflows.massive_adaptive_rl_v2 import (
    MassiveAdaptiveRLWorkflowV2Error,
    main,
)


def test_manifest_v4_preregisters_selection_without_opening_outcomes(
    tmp_path: Path,
) -> None:
    first = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-selection-protocol",
        execution_device_specification="cuda:0",
    )
    second = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="validation-selection-protocol",
        execution_device_specification="cuda:0",
    )

    assert first == second
    assert first.base_manifest.final_gate_names == MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3
    assert first.validation_selection_specification_sha256 == (
        MASSIVE_ADAPTIVE_RL_VALIDATION_SELECTION_SPECIFICATION_V1_SHA256
    )
    assert first.candidate_ranking_specification_sha256 == (
        MASSIVE_ADAPTIVE_RL_CANDIDATE_RANKING_SPECIFICATION_V1_SHA256
    )
    assert (
        first.candidate_ranking_metric_names
        == MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1
    )
    assert first.candidate_tie_breaking_specification_sha256 == (
        MASSIVE_ADAPTIVE_RL_CANDIDATE_TIE_BREAKING_SPECIFICATION_V1_SHA256
    )
    assert (
        first.candidate_tie_breaking_rule_names
        == MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1
    )
    assert (
        first.validation_eligibility_criteria
        == MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1
    )
    assert (
        first.no_eligible_candidate_policy
        == MASSIVE_ADAPTIVE_RL_NO_ELIGIBLE_CANDIDATE_POLICY_V1
    )
    assert first.validation_gate_names == MASSIVE_ADAPTIVE_RL_VALIDATION_GATE_NAMES_V1
    assert first.final_gate_names == MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V4
    assert set(first.final_gate_names) == {
        *MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3,
        "all-selected-policies-validation-eligible",
    }
    assert not first.profitability_reporting_authorized
    assert not first.live_trading_authorized
    assert not first.lockbox_access_authorized
    assert first.execution_device_specification == "cuda:0"

    path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(path=path, manifest=first)
    assert load_massive_adaptive_rl_experiment_manifest_v4(path) == first
    assert tuple(tmp_path.iterdir()) == (path,)


def test_manifest_v4_freezes_candidate_ranking_and_total_tie_break() -> None:
    assert MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_RANKING_V1 == (
        "primary-incremental-rl-log-wealth-descending",
        "ppo-minus-fc06-log-wealth-descending",
        "primary-strategy-active-log-wealth-descending",
        "40bp-liquidation-adjusted-return-descending",
        "maximum-drawdown-ascending",
    )
    assert MASSIVE_ADAPTIVE_RL_VALIDATION_CANDIDATE_TIE_BREAKING_V1 == (
        "update-index-ascending",
        "checkpoint-receipt-sha256-lexicographic-ascending",
    )
    assert MASSIVE_ADAPTIVE_RL_VALIDATION_ELIGIBILITY_CRITERIA_V1 == (
        "primary-incremental-rl-log-wealth-strictly-positive",
        "ppo-minus-fc06-log-wealth-strictly-positive",
        "primary-strategy-active-log-wealth-strictly-positive",
        "40bp-liquidation-adjusted-return-nonnegative",
        "terminal-return-cost-ladder-low-ge-primary-ge-high",
        "maximum-drawdown-at-most-0.25",
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("validation_selection_specification_sha256", "0" * 64),
        ("candidate_ranking_specification_sha256", "1" * 64),
        ("candidate_ranking_metric_names", ("maximum-drawdown-ascending",)),
        ("candidate_tie_breaking_specification_sha256", "2" * 64),
        ("candidate_tie_breaking_rule_names", ("update-index-descending",)),
        ("validation_eligibility_criteria", ()),
        ("no_eligible_candidate_policy", "fail-experiment"),
        ("validation_gate_names", ()),
        ("final_gate_names", MASSIVE_ADAPTIVE_RL_FINAL_GATE_NAMES_V3),
        ("profitability_reporting_authorized", True),
    ),
)
def test_manifest_v4_rejects_post_registration_selection_changes(
    field_name: str,
    replacement: object,
) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="tampered-validation-selection-protocol"
    )
    changed = replace(manifest, **{field_name: replacement})
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV4Error, match="differs"):
        changed.validate()


def test_manifest_v4_is_create_only(tmp_path: Path) -> None:
    manifest = build_massive_adaptive_rl_experiment_manifest_v4(
        experiment_id="create-only-validation-selection-protocol"
    )
    path = tmp_path / "manifest-v4.json"
    write_massive_adaptive_rl_experiment_manifest_v4(path=path, manifest=manifest)
    with pytest.raises(MassiveAdaptiveRLExperimentManifestV4Error, match="create-only"):
        write_massive_adaptive_rl_experiment_manifest_v4(
            path=path,
            manifest=manifest,
        )


def test_manifest_v4_cli_creates_and_validates_but_execution_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest-v4.json"
    assert (
        main(
            [
                "manifest-v4",
                "--experiment-id",
                "v4-cli-preregistration",
                "--output",
                str(manifest_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    created_receipt = capsys.readouterr().out.strip()
    assert len(created_receipt) == 64
    assert main(["validate", "--manifest", str(manifest_path)]) == 0
    assert capsys.readouterr().out.strip() == created_receipt

    source_root = tmp_path / "sources-that-must-not-be-opened"
    artifact_root = tmp_path / "artifacts-that-must-not-be-created"
    with pytest.raises(
        MassiveAdaptiveRLWorkflowV2Error,
        match="Manifest V4 execution requires the package-owned validation backend",
    ):
        main(
            [
                "run",
                "--manifest",
                str(manifest_path),
                "--source-root",
                str(source_root),
                "--artifact-root",
                str(artifact_root),
                "--device",
                "cpu",
            ]
        )
    assert not source_root.exists()
    assert not artifact_root.exists()
    assert json.loads(manifest_path.read_text())["schema"].endswith("manifest-v4")
