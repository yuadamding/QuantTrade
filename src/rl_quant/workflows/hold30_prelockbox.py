"""Qualification and freeze CLI for ``prelockbox-hold30-mech8-v2``.

This command is intentionally unable to launch Kubernetes Jobs.  It produces
local software evidence and deterministic split/manifest artifacts that a
separate approved Seadragon launcher must verify before allocating GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.protocol.hold30 import HOLD30_MECH8_IDS, HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import (
    Hold30FreezeBindings,
    render_hold30_folds,
    render_hold30_manifest,
    sha256_payload,
)
from rl_quant.training.hold30_experiment import (
    build_hold30_context_config,
    build_hold30_policy_config,
    hold30_parameter_counts,
)

SOFTWARE_GATE_FILES = (
    "tests/test_hold30_accounting.py",
    "tests/test_hold30_actions.py",
    "tests/test_hold30_coordinator.py",
    "tests/test_hold30_controls.py",
    "tests/test_hold30_dataset.py",
    "tests/test_hold30_designs.py",
    "tests/test_hold30_distributed.py",
    "tests/test_hold30_driver.py",
    "tests/test_hold30_endpoints.py",
    "tests/test_hold30_ensemble.py",
    "tests/test_hold30_ensemble_runtime.py",
    "tests/test_hold30_freeze.py",
    "tests/test_hold30_folds.py",
    "tests/test_hold30_inference.py",
    "tests/test_hold30_mechanisms.py",
    "tests/test_hold30_metrics.py",
    "tests/test_hold30_null_rebuild.py",
    "tests/test_hold30_policy.py",
    "tests/test_hold30_qualification.py",
    "tests/test_hold30_runtime.py",
    "tests/test_hold30_sleeves.py",
    "tests/test_hold30_state.py",
    "tests/test_hold30_training.py",
    "tests/test_hold30_workflow.py",
)
COMPATIBILITY_GATE_FILES = (
    "tests/test_context_normalization.py",
    "tests/test_portfolio_env.py",
    "tests/test_target_weight_execution.py",
    "tests/test_daily_runtime_accounting.py",
    "tests/test_daily_policy_accum.py",
)

# Each package-owned Hold-30 component has an explicit blocking test.  A future
# component (including a package-owned driver) is admitted by discovery only
# when ``tests/test_<module-name>.py`` exists; an untested source file therefore
# cannot silently join a qualified tree.
HOLD30_COMPONENT_TESTS = (
    ("src/rl_quant/datasets/hold30_alpha.py", "tests/test_hold30_alpha.py"),
    ("src/rl_quant/datasets/hold30.py", "tests/test_hold30_dataset.py"),
    ("src/rl_quant/datasets/hold30_folds.py", "tests/test_hold30_folds.py"),
    (
        "src/rl_quant/datasets/hold30_qualification.py",
        "tests/test_hold30_qualification.py",
    ),
    (
        "src/rl_quant/datasets/hold30_null_rebuild.py",
        "tests/test_hold30_null_rebuild.py",
    ),
    ("src/rl_quant/envs/hold30.py", "tests/test_hold30_accounting.py"),
    ("src/rl_quant/execution/hold30.py", "tests/test_hold30_actions.py"),
    ("src/rl_quant/execution/hold30_sleeves.py", "tests/test_hold30_sleeves.py"),
    (
        "src/rl_quant/evaluation/hold30_ensemble_runtime.py",
        "tests/test_hold30_ensemble_runtime.py",
    ),
    ("src/rl_quant/evaluation/hold30_controls.py", "tests/test_hold30_controls.py"),
    (
        "src/rl_quant/evaluation/hold30_alpha_evaluation.py",
        "tests/test_hold30_alpha_evaluation.py",
    ),
    (
        "src/rl_quant/evaluation/hold30_alpha_m03r.py",
        "tests/test_hold30_alpha_m03r_evaluation.py",
    ),
    (
        "src/rl_quant/evaluation/hold30_alpha_m03r_v5.py",
        "tests/test_hold30_alpha_m03r_v5_evaluation.py",
    ),
    (
        "src/rl_quant/evaluation/hold30_alpha_m03r_v6.py",
        "tests/test_hold30_alpha_m03r_v6_evaluation.py",
    ),
    ("src/rl_quant/evaluation/hold30_endpoints.py", "tests/test_hold30_endpoints.py"),
    ("src/rl_quant/evaluation/hold30_inference.py", "tests/test_hold30_inference.py"),
    ("src/rl_quant/evaluation/hold30_metrics.py", "tests/test_hold30_metrics.py"),
    ("src/rl_quant/models/hold30_ensemble.py", "tests/test_hold30_ensemble.py"),
    ("src/rl_quant/models/hold30_alpha.py", "tests/test_hold30_alpha.py"),
    (
        "src/rl_quant/models/hold30_hazard.py",
        "tests/test_hold30_m03r_mechanism.py",
    ),
    (
        "src/rl_quant/models/hold30_exit_action_v6.py",
        "tests/test_hold30_m03r_v6_exit_action.py",
    ),
    (
        "src/rl_quant/models/hold30_confidence_v6.py",
        "tests/test_hold30_confidence_v6.py",
    ),
    (
        "src/rl_quant/models/hold30_m03r_ensemble.py",
        "tests/test_hold30_m03r_projection.py",
    ),
    (
        "src/rl_quant/models/hold30_m03r_ensemble_v5.py",
        "tests/test_hold30_m03r_v5_projection.py",
    ),
    ("src/rl_quant/protocol/hold30.py", "tests/test_hold30_designs.py"),
    (
        "src/rl_quant/protocol/hold30_alpha_v3.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_v3_freeze.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r.py",
        "tests/test_hold30_alpha_m03r_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v5.py",
        "tests/test_hold30_alpha_m03r_v5_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_m03r_confidence.py",
        "tests/test_hold30_m03r_v5_model_semantics.py",
    ),
    ("src/rl_quant/protocol/hold30_freeze.py", "tests/test_hold30_freeze.py"),
    ("src/rl_quant/training/hold30.py", "tests/test_hold30_training.py"),
    ("src/rl_quant/training/hold30_alpha.py", "tests/test_hold30_alpha.py"),
    (
        "src/rl_quant/training/hold30_alpha_m03r.py",
        "tests/test_hold30_alpha_m03r.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_selection.py",
        "tests/test_hold30_alpha_m03r_selection.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v5.py",
        "tests/test_hold30_m03r_v5_model_semantics.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v5_routes.py",
        "tests/test_hold30_alpha_m03r_v5_routes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v5_selection.py",
        "tests/test_hold30_alpha_m03r_v5_selection.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v6.py",
        "tests/test_hold30_alpha_m03r_v6_protocol.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v6.py",
        "tests/test_hold30_alpha_m03r_v6_objective.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v6_ledger.py",
        "tests/test_hold30_alpha_m03r_v6_ledger.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v6_routes.py",
        "tests/test_hold30_alpha_m03r_v6_routes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v6_selection.py",
        "tests/test_hold30_alpha_m03r_v6_selection.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7.py",
        "tests/test_hold30_alpha_m03r_v7_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_schedule.py",
        "tests/test_hold30_alpha_m03r_v7_schedule.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7.py",
        "tests/test_hold30_alpha_m03r_v7_objective.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_routes.py",
        "tests/test_hold30_alpha_m03r_v7_routes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_schedule.py",
        "tests/test_hold30_alpha_m03r_v7_schedule.py",
    ),
    (
        "src/rl_quant/models/hold30_alpha_m03r_v7_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v7_top2000_dev_model.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v7_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_kubernetes.py",
        "tests/test_hold30_m03r_v7_top2000_kubernetes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_package.py",
        "tests/test_hold30_m03r_v7_top2000_kubernetes.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_seed17_top2000_dev.py",
        "tests/test_hold30_m03r_v7_seed17_kubernetes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_kubernetes.py",
        "tests/test_hold30_m03r_v7_seed17_kubernetes.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_package.py",
        "tests/test_hold30_m03r_v7_seed17_kubernetes.py",
    ),
    (
        "src/rl_quant/training/hold30_m03r_confidence_fit.py",
        "tests/test_hold30_m03r_confidence_fit.py",
    ),
    (
        "src/rl_quant/training/hold30_m03r_confidence_objective_v6.py",
        "tests/test_hold30_m03r_confidence_objective_v6.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_plan.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/training/hold30_coordinator.py",
        "tests/test_hold30_coordinator.py",
    ),
    ("src/rl_quant/training/hold30_driver.py", "tests/test_hold30_driver.py"),
    ("src/rl_quant/training/hold30_experiment.py", "tests/test_hold30_designs.py"),
    ("src/rl_quant/training/hold30_runtime.py", "tests/test_hold30_runtime.py"),
    ("src/rl_quant/training/hold30_state.py", "tests/test_hold30_state.py"),
    (
        "src/rl_quant/execution/hold30_m03r_projection_v5.py",
        "tests/test_hold30_m03r_v5_projection.py",
    ),
    (
        "src/rl_quant/execution/hold30_m03r_soft_persistence_v6.py",
        "tests/test_hold30_m03r_v6_behavior.py",
    ),
    (
        "src/rl_quant/execution/hold30_exit_v6.py",
        "tests/test_hold30_m03r_v6_exit_action.py",
    ),
    ("src/rl_quant/workflows/hold30_prelockbox.py", "tests/test_hold30_workflow.py"),
    (
        "src/rl_quant/workflows/hold30_alpha_prelockbox.py",
        "tests/test_hold30_alpha_workflow.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v7_seed17_top2000_2026_ytd.py",
        "tests/test_hold30_alpha_m03r_v7_seed17_top2000_2026_ytd_protocol.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_2026_ytd_kubernetes.py",
        "tests/test_top2000_m03r_v7_seed17_2026_ytd_workflow.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_m03r_v7_seed17_2026_ytd_package.py",
        "tests/test_top2000_m03r_v7_seed17_2026_ytd_workflow.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v8_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v8_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v9_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v9_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v10_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v10_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v11_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v11_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v11_a15_inference_audit.py",
        "tests/test_top2000_m03r_v11_a15_inference_audit.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v12_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v12_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v12_posthoc_inference_audit.py",
        "tests/test_top2000_m03r_v12_posthoc_inference_audit.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v13_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v13_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v14_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v14_top2000_dev_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_m03r_v15_top2000_dev.py",
        "tests/test_hold30_alpha_m03r_v15_top2000_dev_protocol.py",
    ),
)
HOLD30_INTEGRATION_SOURCE_FILES = (
    "src/rl_quant/datasets/daily.py",
    "src/rl_quant/envs/__init__.py",
    "src/rl_quant/envs/portfolio.py",
    "src/rl_quant/models/context_encoder.py",
    "src/rl_quant/models/daily_policy.py",
    "src/rl_quant/training/context_pretrain.py",
    "src/rl_quant/training/designs.py",
)
HOLD30_EVIDENCE_FILES = (
    "README.md",
    "docs/adr/0006-daily-decision-soft-30-session-holding.md",
    "docs/adr/0007-benchmark-relative-hold30-alpha-objective.md",
    "docs/daily_hold30_policy_rfc.md",
    "docs/m03r_confidence_calibration_protocol.md",
    "docs/prelockbox_hold30_active_alpha_m03r_v4.md",
    "docs/prelockbox_hold30_active_alpha_m03r_v5.md",
    "docs/prelockbox_hold30_active_alpha_m03r_v6.md",
    "docs/prelockbox_hold30_active_alpha_m03r_v7.md",
    "docs/prelockbox_hold30_active_alpha_m03r_v7_experiment.md",
    "docs/prelockbox_hold30_alpha_evaluation_v3.md",
    "docs/prelockbox_hold30_alpha_mech8_v3.md",
    "docs/prelockbox_hold30_h0_h3_experiment.md",
    "docs/prelockbox_hold30_mech8_v2.md",
    "pyproject.toml",
)

HOLD30_CONTEXT_PARAMETER_MAX = 2_000_000
HOLD30_ACTOR_PARAMETER_MIN = 500_000
HOLD30_ACTOR_PARAMETER_MAX = 5_000_000
HOLD30_TOTAL_PARAMETER_MAX = 7_000_000


@dataclass(frozen=True, slots=True)
class Hold30QualificationInventory:
    """Resolved, regular-file-only software evidence boundary."""

    component_sources: tuple[str, ...]
    integration_sources: tuple[str, ...]
    hold30_tests: tuple[str, ...]
    compatibility_tests: tuple[str, ...]
    static_hygiene_files: tuple[str, ...]
    qualified_files: tuple[str, ...]


class Hold30QualificationError(RuntimeError):
    """Software qualification cannot produce a passing receipt."""


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _require_regular_files(root: Path, paths: Sequence[str]) -> None:
    absent: list[str] = []
    unsafe: list[str] = []
    for name in paths:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            unsafe.append(name)
            continue
        candidate = root / relative
        if not candidate.is_file():
            absent.append(name)
        elif candidate.is_symlink():
            unsafe.append(name)
    if absent:
        raise Hold30QualificationError(
            "required qualification files are absent: " + ", ".join(absent)
        )
    if unsafe:
        raise Hold30QualificationError(
            "qualification files must be regular in-repository files: "
            + ", ".join(unsafe)
        )


def _resolve_qualification_inventory(root: Path) -> Hold30QualificationInventory:
    """Resolve all landed Hold-30 modules without accepting untested additions."""

    registered = {source: test for source, test in HOLD30_COMPONENT_TESTS}
    component_sources = list(registered)
    hold30_tests = [
        *SOFTWARE_GATE_FILES,
        *(test for _source, test in HOLD30_COMPONENT_TESTS),
    ]

    source_root = root / "src" / "rl_quant"
    if not source_root.is_dir():
        raise Hold30QualificationError(
            "src/rl_quant is absent from the qualification root"
        )
    for candidate in sorted(source_root.rglob("*hold30*.py")):
        relative = candidate.relative_to(root).as_posix()
        if relative in registered:
            continue
        conventional_test = f"tests/test_{candidate.stem}.py"
        if not (root / conventional_test).is_file():
            raise Hold30QualificationError(
                f"unregistered Hold-30 component {relative} requires {conventional_test}"
            )
        component_sources.append(relative)
        hold30_tests.append(conventional_test)

    # Discover additional Hold-30 tests too: a newly landed mechanism or
    # driver test belongs to the blocking gate even before this tuple is
    # manually reordered for readability.
    test_root = root / "tests"
    if not test_root.is_dir():
        raise Hold30QualificationError("tests is absent from the qualification root")
    hold30_tests.extend(
        candidate.relative_to(root).as_posix()
        for candidate in sorted(test_root.glob("test_hold30*.py"))
    )

    component_tuple = _ordered_unique(component_sources)
    integration_tuple = _ordered_unique(HOLD30_INTEGRATION_SOURCE_FILES)
    hold30_test_tuple = _ordered_unique(hold30_tests)
    compatibility_tuple = _ordered_unique(COMPATIBILITY_GATE_FILES)
    static_files = _ordered_unique(
        (*component_tuple, *integration_tuple, *hold30_test_tuple)
    )
    qualified_files = _ordered_unique(
        (
            *component_tuple,
            *integration_tuple,
            *hold30_test_tuple,
            *compatibility_tuple,
            *HOLD30_EVIDENCE_FILES,
        )
    )
    _require_regular_files(root, qualified_files)

    for source, test in HOLD30_COMPONENT_TESTS:
        if source not in component_tuple or test not in hold30_test_tuple:
            raise Hold30QualificationError(
                f"Hold-30 component/test inventory omitted {source} -> {test}"
            )
    return Hold30QualificationInventory(
        component_sources=component_tuple,
        integration_sources=integration_tuple,
        hold30_tests=hold30_test_tuple,
        compatibility_tests=compatibility_tuple,
        static_hygiene_files=static_files,
        qualified_files=qualified_files,
    )


def _hold30_model_evidence() -> dict[str, Any]:
    """Return and enforce the compact-model contract used by every setting."""

    context_config = asdict(build_hold30_context_config())
    expected_context = {
        "bar_feature_dim": 5,
        "covariate_dim": 0,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "feedforward_dim": 256,
        "dropout": 0.0,
        "max_seconds": 390,
        "block_seconds": 5,
    }
    mismatches = {
        name: (context_config.get(name), expected)
        for name, expected in expected_context.items()
        if context_config.get(name) != expected
    }
    if mismatches:
        raise Hold30QualificationError(
            f"Hold-30 compact context configuration drifted: {mismatches}"
        )

    settings: list[dict[str, Any]] = []
    context_parameter_count: int | None = None
    for setting_id in HOLD30_MECH8_IDS:
        policy_config = asdict(build_hold30_policy_config(setting_id))
        if policy_config.get("hold30_setting") != setting_id:
            raise Hold30QualificationError(
                f"policy configuration is not bound to setting {setting_id}"
            )
        counts = hold30_parameter_counts(setting_id)
        values = (counts.context_encoder, counts.actor_path, counts.total_unique)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise Hold30QualificationError(
                f"Hold-30 parameter counts must be positive integers for {setting_id}: {values}"
            )
        if counts.total_unique != counts.context_encoder + counts.actor_path:
            raise Hold30QualificationError(
                f"Hold-30 parameter counts overlap or omit parameters for {setting_id}"
            )
        if context_parameter_count is None:
            context_parameter_count = counts.context_encoder
        elif counts.context_encoder != context_parameter_count:
            raise Hold30QualificationError(
                f"shared context parameter count changed for {setting_id}"
            )
        if not (
            counts.context_encoder < HOLD30_CONTEXT_PARAMETER_MAX
            and HOLD30_ACTOR_PARAMETER_MIN
            <= counts.actor_path
            <= HOLD30_ACTOR_PARAMETER_MAX
            and counts.total_unique <= HOLD30_TOTAL_PARAMETER_MAX
        ):
            raise Hold30QualificationError(
                f"Hold-30 compact parameter caps failed for {setting_id}: {values}"
            )
        settings.append(
            {
                "setting_id": setting_id,
                "policy_config": policy_config,
                "parameter_counts": asdict(counts),
            }
        )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "context_config": context_config,
        "parameter_caps": {
            "context_encoder_exclusive_max": HOLD30_CONTEXT_PARAMETER_MAX,
            "actor_path_inclusive_min": HOLD30_ACTOR_PARAMETER_MIN,
            "actor_path_inclusive_max": HOLD30_ACTOR_PARAMETER_MAX,
            "total_unique_inclusive_max": HOLD30_TOTAL_PARAMETER_MAX,
        },
        "settings": settings,
    }
    evidence["model_contract_sha256"] = sha256_payload(evidence)
    return evidence


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_gate(
    gate_id: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cwd / "src")
    # Local qualification is deliberately CPU deterministic.  Remote H100
    # numerical/capacity parity is a separate, receipt-bound gate.
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        tuple(argv),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_seconds,
    )
    output = completed.stdout
    return {
        "gate_id": gate_id,
        "argv": list(argv),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "output_sha256": _sha256_bytes(output),
        "output_tail": output.decode("utf-8", errors="replace")[-4000:],
    }


def _git_value(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise Hold30QualificationError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or f"git {' '.join(arguments)} failed"
        )
    return result.stdout.decode("utf-8").strip()


def qualify_hold30_software(
    repo: str | Path,
    *,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the fixed CPU software gates and return a content-addressed receipt."""

    root = Path(repo).resolve()
    inventory = _resolve_qualification_inventory(root)
    model_evidence = _hold30_model_evidence()
    commands = (
        (
            "hold30_blockers",
            (sys.executable, "-m", "pytest", "-q", *inventory.hold30_tests),
        ),
        (
            "compatibility_regressions",
            (sys.executable, "-m", "pytest", "-q", *inventory.compatibility_tests),
        ),
        (
            "static_hygiene",
            (
                sys.executable,
                "-m",
                "ruff",
                "check",
                *inventory.static_hygiene_files,
            ),
        ),
    )
    gates = tuple(
        _run_gate(gate_id, argv, cwd=root, timeout_seconds=timeout_seconds)
        for gate_id, argv in commands
    )
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--", *inventory.qualified_files),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if diff.returncode:
        raise Hold30QualificationError("could not bind the Hold-30 source diff")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_generation": HOLD30_PROTOCOL_GENERATION,
        "qualification_scope": "cpu_software_only",
        "scientific_qualification": False,
        "gpu_capacity_qualification": False,
        "launch_authorized": False,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "hold30_source_diff_sha256": _sha256_bytes(diff.stdout),
        "qualification_inventory": asdict(inventory),
        "model_contract": model_evidence,
        "qualified_file_sha256s": {
            name: _sha256_bytes((root / name).read_bytes())
            for name in inventory.qualified_files
        },
        "gates": list(gates),
        "passed": all(bool(gate["passed"]) for gate in gates),
    }
    payload["qualification_sha256"] = sha256_payload(payload)
    if not payload["passed"]:
        failed = ", ".join(gate["gate_id"] for gate in gates if not gate["passed"])
        raise Hold30QualificationError(
            f"Hold-30 software qualification failed: {failed}; receipt={json.dumps(payload)}"
        )
    return payload


def _read_axis(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise Hold30QualificationError(
            "decision-axis JSON must be an array of timestamps"
        )
    return tuple(payload)


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    qualify = subparsers.add_parser("qualify-software")
    qualify.add_argument("--repo", type=Path, default=Path.cwd())
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--timeout-seconds", type=int, default=1800)

    splits = subparsers.add_parser("render-splits")
    splits.add_argument("--decision-axis", type=Path, required=True)
    splits.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser("render-manifest")
    manifest.add_argument("--decision-axis", type=Path, required=True)
    manifest.add_argument("--bindings", type=Path, required=True)
    manifest.add_argument("--approval-state", default="dry_run")
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "qualify-software":
        receipt = qualify_hold30_software(
            args.repo, timeout_seconds=args.timeout_seconds
        )
        _write_new_json(args.output, receipt)
        print(receipt["qualification_sha256"])
        return 0
    if args.command == "render-splits":
        folds = render_hold30_folds(_read_axis(args.decision_axis))
        _write_new_json(args.output, [asdict(fold) for fold in folds])
        return 0
    if args.command == "render-manifest":
        bindings_payload = json.loads(args.bindings.read_text(encoding="utf-8"))
        bindings = Hold30FreezeBindings(**bindings_payload)
        manifest = render_hold30_manifest(
            _read_axis(args.decision_axis),
            bindings,
            approval_state=args.approval_state,
        )
        _write_new_json(args.output, manifest)
        print(manifest["manifest_sha256"])
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
