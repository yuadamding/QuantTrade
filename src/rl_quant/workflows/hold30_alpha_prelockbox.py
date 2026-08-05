"""Local CPU-only qualification for Hold-30 alpha mechanism-8 v3.

This command cannot render an executable manifest, inspect scientific data,
check Kubernetes/GPU capacity, or launch work.  Its receipt schema and
generation are disjoint from the superseded v2 workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rl_quant.models.hold30_alpha import (
    Hold30AlphaHead,
    Hold30AlphaHeadConfig,
    Hold30AlphaModelError,
)
from rl_quant.protocol.hold30_alpha_v3 import (
    HOLD30_ALPHA_MECH8_IDS,
    HOLD30_ALPHA_MECH8_SETTINGS,
    HOLD30_ALPHA_PROTOCOL_GENERATION,
    HOLD30_ALPHA_V3_CANONICAL_ID,
    HOLD30_ALPHA_V3_DESIGN,
    hold30_alpha_v3_design_payload,
)
from rl_quant.protocol.hold30_freeze import sha256_payload
from rl_quant.training.hold30_alpha import (
    Hold30AlphaObjectiveConfig,
    Hold30AlphaUnresolvedCoefficientError,
)
from rl_quant.training.hold30_alpha_plan import unresolved_hold30_alpha_training_plan

HOLD30_ALPHA_SOFTWARE_SCHEMA = "rl-quant.hold30-alpha-v3.software-qualification-v1"

V3_COMPONENT_SOURCE_TESTS = (
    ("src/rl_quant/datasets/hold30_alpha.py", "tests/test_hold30_alpha.py"),
    (
        "src/rl_quant/datasets/hold30_alpha_qualification.py",
        "tests/test_hold30_alpha_qualification.py",
    ),
    (
        "src/rl_quant/models/hold30_alpha.py",
        "tests/test_hold30_alpha_core.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha.py",
        "tests/test_hold30_alpha_core.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_plan.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_pilot_plan.py",
        "tests/test_hold30_alpha_pilot_plan.py",
    ),
    (
        "src/rl_quant/training/hold30_alpha_driver.py",
        "tests/test_hold30_alpha_driver.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_v3.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/protocol/hold30_alpha_v3_freeze.py",
        "tests/test_hold30_alpha_v3_protocol.py",
    ),
    (
        "src/rl_quant/evaluation/hold30_alpha_evaluation.py",
        "tests/test_hold30_alpha_evaluation.py",
    ),
    (
        "src/rl_quant/workflows/hold30_alpha_prelockbox.py",
        "tests/test_hold30_alpha_workflow.py",
    ),
)

V3_COMPONENT_TESTS = (
    "tests/test_hold30_alpha.py",
    "tests/test_hold30_alpha_a06.py",
    "tests/test_hold30_alpha_core.py",
    "tests/test_hold30_alpha_driver.py",
    "tests/test_hold30_alpha_distributed.py",
    "tests/test_hold30_alpha_pilot_plan.py",
    "tests/test_hold30_alpha_qualification.py",
    "tests/test_hold30_alpha_v3_protocol.py",
    "tests/test_hold30_alpha_evaluation.py",
    "tests/test_hold30_alpha_workflow.py",
)

V3_REUSED_NON_HOLD30_SOURCES = (
    "src/rl_quant/models/context_encoder.py",
    "src/rl_quant/models/daily_policy.py",
    "src/rl_quant/training/context_pretrain.py",
)

V3_EXCLUDED_RUNTIME_HISTORY = (
    # Superseded v2 command.  It is hashed as audit history and its tests run,
    # but it cannot enter the v3 runtime source set or issue a v3 receipt.
    "src/rl_quant/workflows/hold30_prelockbox.py",
)

V3_COMPATIBILITY_TESTS = (
    "tests/test_context_normalization.py",
    "tests/test_portfolio_env.py",
    "tests/test_target_weight_execution.py",
    "tests/test_daily_runtime_accounting.py",
    "tests/test_daily_policy_accum.py",
)

V3_EVIDENCE_FILES = (
    "docs/adr/0006-daily-decision-soft-30-session-holding.md",
    "docs/adr/0007-benchmark-relative-hold30-alpha-objective.md",
    "docs/daily_hold30_policy_rfc.md",
    "docs/prelockbox_hold30_mech8_v2.md",
    "docs/prelockbox_hold30_alpha_mech8_v3.md",
    "docs/prelockbox_hold30_alpha_evaluation_v3.md",
    "pyproject.toml",
)


class Hold30AlphaQualificationError(RuntimeError):
    """The v3 local software gate cannot issue a passing receipt."""


@dataclass(frozen=True, slots=True)
class Hold30AlphaQualificationInventory:
    component_sources: tuple[str, ...]
    component_tests: tuple[str, ...]
    integration_sources: tuple[str, ...]
    integration_tests: tuple[str, ...]
    compatibility_tests: tuple[str, ...]
    evidence_files: tuple[str, ...]
    excluded_runtime_history: tuple[str, ...]
    static_hygiene_files: tuple[str, ...]
    qualified_files: tuple[str, ...]


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
        raise Hold30AlphaQualificationError(
            "required v3 qualification files are absent: " + ", ".join(absent)
        )
    if unsafe:
        raise Hold30AlphaQualificationError(
            "v3 qualification files must be regular in-repository files: "
            + ", ".join(unsafe)
        )


def resolve_hold30_alpha_qualification_inventory(
    repo: str | Path,
) -> Hold30AlphaQualificationInventory:
    """Resolve every v3-named source and reject untested additions."""

    root = Path(repo).resolve()
    registered = {source: test for source, test in V3_COMPONENT_SOURCE_TESTS}
    source_root = root / "src" / "rl_quant"
    if not source_root.is_dir():
        raise Hold30AlphaQualificationError("src/rl_quant is absent")
    discovered = tuple(
        candidate.relative_to(root).as_posix()
        for candidate in sorted(source_root.rglob("*hold30*alpha*.py"))
    )
    missing_registration = sorted(set(discovered) - set(registered))
    if missing_registration:
        raise Hold30AlphaQualificationError(
            "unregistered v3 source requires an explicit blocking test: "
            + ", ".join(missing_registration)
        )
    stale_registration = sorted(set(registered) - set(discovered))
    if stale_registration:
        raise Hold30AlphaQualificationError(
            "registered v3 sources are absent: " + ", ".join(stale_registration)
        )

    test_root = root / "tests"
    discovered_tests = tuple(
        candidate.relative_to(root).as_posix()
        for candidate in sorted(test_root.glob("test_hold30_alpha*.py"))
    )
    expected_tests = _ordered_unique((*V3_COMPONENT_TESTS, *registered.values()))
    if set(discovered_tests) != set(expected_tests):
        missing = sorted(set(discovered_tests) - set(expected_tests))
        stale = sorted(set(expected_tests) - set(discovered_tests))
        raise Hold30AlphaQualificationError(
            f"v3 test inventory mismatch; unregistered={missing}, absent={stale}"
        )

    component_sources = tuple(registered)
    component_tests = expected_tests
    all_hold30_sources = tuple(
        candidate.relative_to(root).as_posix()
        for candidate in sorted(source_root.rglob("*hold30*.py"))
    )
    excluded_runtime_history = _ordered_unique(V3_EXCLUDED_RUNTIME_HISTORY)
    unexpected_exclusions = set(excluded_runtime_history) - set(all_hold30_sources)
    if unexpected_exclusions:
        raise Hold30AlphaQualificationError(
            "documented v2 runtime-history exclusion is absent: "
            + ", ".join(sorted(unexpected_exclusions))
        )
    integration_sources = _ordered_unique(
        (
            *(
                source
                for source in all_hold30_sources
                if source not in set(component_sources)
                and source not in set(excluded_runtime_history)
            ),
            *V3_REUSED_NON_HOLD30_SOURCES,
        )
    )
    all_hold30_tests = tuple(
        candidate.relative_to(root).as_posix()
        for candidate in sorted(test_root.glob("test_hold30*.py"))
    )
    integration_tests = tuple(
        test for test in all_hold30_tests if test not in set(component_tests)
    )
    compatibility_tests = _ordered_unique(V3_COMPATIBILITY_TESTS)
    evidence_files = _ordered_unique(V3_EVIDENCE_FILES)
    # Ruff qualifies v3-owned code only.  Reused v2 mechanics are fully
    # content-bound and all Hold30 tests execute, but this receipt does not
    # misrepresent their pre-existing style debt as a full-tree lint claim.
    static_hygiene = _ordered_unique((*component_sources, *component_tests))
    qualified = _ordered_unique(
        (
            *component_sources,
            *component_tests,
            *integration_sources,
            *integration_tests,
            *compatibility_tests,
            *evidence_files,
            *excluded_runtime_history,
        )
    )
    _require_regular_files(root, qualified)
    return Hold30AlphaQualificationInventory(
        component_sources=component_sources,
        component_tests=component_tests,
        integration_sources=integration_sources,
        integration_tests=integration_tests,
        compatibility_tests=compatibility_tests,
        evidence_files=evidence_files,
        excluded_runtime_history=excluded_runtime_history,
        static_hygiene_files=static_hygiene,
        qualified_files=qualified,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise Hold30AlphaQualificationError(
            result.stderr.decode("utf-8", errors="replace").strip()
            or f"git {' '.join(arguments)} failed"
        )
    return result.stdout.decode("utf-8").strip()


def _run_gate(
    gate_id: str,
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(cwd / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        tuple(argv),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_seconds,
    )
    return {
        "gate_id": gate_id,
        "argv": list(argv),
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "output_sha256": _sha256_bytes(completed.stdout),
        "output_tail": completed.stdout.decode("utf-8", errors="replace")[-4000:],
    }


def _model_evidence() -> dict[str, Any]:
    training_plan = unresolved_hold30_alpha_training_plan()
    rows: list[dict[str, Any]] = []
    unresolved_settings: list[str] = []
    for setting in HOLD30_ALPHA_MECH8_SETTINGS:
        row: dict[str, Any] = {
            "setting_id": setting.setting_id,
            "setting": asdict(setting),
            "alpha_head": None,
        }
        if setting.supervised_residual_alpha_heads:
            try:
                config = Hold30AlphaHeadConfig(
                    setting_id=setting.setting_id,
                    hidden_dim=128,
                )
            except Hold30AlphaModelError as exc:
                row["alpha_head_construction_resolved"] = False
                row["alpha_head_blocker"] = str(exc)
            else:
                head = Hold30AlphaHead(config)
                row["alpha_head_construction_resolved"] = True
                row["alpha_head"] = {
                    "config": asdict(config),
                    "parameter_count": head.parameter_count,
                    "has_uncertainty": config.use_uncertainty,
                    "has_separate_total_risk_overlay": config.use_total_risk_overlay,
                }
        if setting.setting_index >= 2:
            objective = Hold30AlphaObjectiveConfig(setting_id=setting.setting_id)
            try:
                objective.require_resolved()
            except Hold30AlphaUnresolvedCoefficientError:
                unresolved_settings.append(setting.setting_id)
                row["objective_coefficients_resolved"] = False
            else:
                row["objective_coefficients_resolved"] = True
        rows.append(row)
    overlay_rows = [
        setting.setting_id
        for setting in HOLD30_ALPHA_MECH8_SETTINGS
        if setting.sharpe_mode == "separate-total-risk-overlay"
    ]
    if overlay_rows != ["hold30a-a06-sharpe-overlay"]:
        raise Hold30AlphaQualificationError("only A06 may contain the separate Sharpe overlay")
    payload: dict[str, Any] = {
        "schema": "rl-quant.hold30-alpha-v3.model-contract-v1",
        "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
        "design": hold30_alpha_v3_design_payload(),
        "checkpoint_thresholds_complete": (
            HOLD30_ALPHA_V3_DESIGN.checkpoint.result_moving_thresholds_complete
        ),
        "scientific_plan_resolved": False,
        "typed_training_plan_resolved_for_executable": (
            training_plan.resolved_for_executable
        ),
        "typed_training_plan_receipt_id": training_plan.receipt_id,
        "unresolved_objective_settings": unresolved_settings,
        "A06_overlay_coefficient_and_routing_frozen": False,
        "settings": rows,
    }
    payload["receipt_sha256"] = sha256_payload(payload)
    return payload


def qualify_hold30_alpha_software(
    repo: str | Path,
    *,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run deterministic local gates and issue a non-authorizing v3 receipt."""

    root = Path(repo).resolve()
    inventory = resolve_hold30_alpha_qualification_inventory(root)
    commands = (
        (
            "v3_components",
            (sys.executable, "-m", "pytest", "-q", *inventory.component_tests),
        ),
        (
            "hold30_integration",
            (sys.executable, "-m", "pytest", "-q", *inventory.integration_tests),
        ),
        (
            "compatibility_regressions",
            (sys.executable, "-m", "pytest", "-q", *inventory.compatibility_tests),
        ),
        (
            "static_hygiene",
            (sys.executable, "-m", "ruff", "check", *inventory.static_hygiene_files),
        ),
    )
    gates = tuple(
        _run_gate(gate_id, argv, cwd=root, timeout_seconds=timeout_seconds)
        for gate_id, argv in commands
    )
    qualified_hashes = {
        name: _sha256_bytes((root / name).read_bytes()) for name in inventory.qualified_files
    }
    payload: dict[str, Any] = {
        "schema": HOLD30_ALPHA_SOFTWARE_SCHEMA,
        "schema_version": 1,
        "protocol_generation": HOLD30_ALPHA_PROTOCOL_GENERATION,
        "superseded_v2_receipts_accepted": False,
        "qualification_scope": "local_cpu_component_integration_only",
        "capabilities": {
            "reads_scientific_data": False,
            "performs_remote_access": False,
            "checks_gpu_capacity": False,
            "renders_executable_manifest": False,
            "launches_jobs": False,
        },
        "scientific_qualification": False,
        "data_qualification": False,
        "gpu_capacity_qualification": False,
        "launch_authorized": False,
        "scientific_plan_resolved": False,
        "executable_authorization_capable": False,
        "end_to_end_v3_training_driver_qualified": False,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_tree": _git_value(root, "rev-parse", "HEAD^{tree}"),
        "qualification_inventory": asdict(inventory),
        "qualified_file_sha256s": qualified_hashes,
        "qualified_content_sha256": sha256_payload(qualified_hashes),
        "protocol_contract": {
            "setting_ids": list(HOLD30_ALPHA_MECH8_IDS),
            "promotion_candidate": HOLD30_ALPHA_V3_CANONICAL_ID,
            "trial_inventory": {"settings": 8, "folds": 6, "seeds": 5, "trials": 240},
        },
        "model_contract": _model_evidence(),
        "gates": list(gates),
        "passed": all(bool(gate["passed"]) for gate in gates),
    }
    payload["qualification_sha256"] = sha256_payload(payload)
    if not payload["passed"]:
        failed = ", ".join(gate["gate_id"] for gate in gates if not gate["passed"])
        raise Hold30AlphaQualificationError(
            f"v3 software qualification failed: {failed}; receipt={json.dumps(payload)}"
        )
    return payload


def verify_hold30_alpha_software_receipt(receipt: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "protocol_generation",
        "superseded_v2_receipts_accepted",
        "qualification_scope",
        "capabilities",
        "scientific_qualification",
        "data_qualification",
        "gpu_capacity_qualification",
        "launch_authorized",
        "scientific_plan_resolved",
        "executable_authorization_capable",
        "end_to_end_v3_training_driver_qualified",
        "git_commit",
        "git_tree",
        "qualification_inventory",
        "qualified_file_sha256s",
        "qualified_content_sha256",
        "protocol_contract",
        "model_contract",
        "gates",
        "passed",
        "qualification_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise Hold30AlphaQualificationError("v3 software receipt has partial or unknown fields")
    if (
        receipt["schema"] != HOLD30_ALPHA_SOFTWARE_SCHEMA
        or receipt["schema_version"] != 1
        or receipt["protocol_generation"] != HOLD30_ALPHA_PROTOCOL_GENERATION
        or receipt["superseded_v2_receipts_accepted"] is not False
        or receipt["qualification_scope"] != "local_cpu_component_integration_only"
        or receipt["scientific_qualification"] is not False
        or receipt["data_qualification"] is not False
        or receipt["gpu_capacity_qualification"] is not False
        or receipt["launch_authorized"] is not False
        or receipt["scientific_plan_resolved"] is not False
        or receipt["executable_authorization_capable"] is not False
        or receipt["end_to_end_v3_training_driver_qualified"] is not False
        or receipt["passed"] is not True
    ):
        raise Hold30AlphaQualificationError("v3 software receipt identity/authority is invalid")
    if receipt["capabilities"] != {
        "reads_scientific_data": False,
        "performs_remote_access": False,
        "checks_gpu_capacity": False,
        "renders_executable_manifest": False,
        "launches_jobs": False,
    }:
        raise Hold30AlphaQualificationError("v3 software receipt claims forbidden capability")
    if receipt["protocol_contract"] != {
        "setting_ids": list(HOLD30_ALPHA_MECH8_IDS),
        "promotion_candidate": HOLD30_ALPHA_V3_CANONICAL_ID,
        "trial_inventory": {"settings": 8, "folds": 6, "seeds": 5, "trials": 240},
    }:
        raise Hold30AlphaQualificationError("v3 software receipt protocol contract drifted")
    if not isinstance(receipt["gates"], list) or [row.get("gate_id") for row in receipt["gates"]] != [
        "v3_components",
        "hold30_integration",
        "compatibility_regressions",
        "static_hygiene",
    ] or any(row.get("passed") is not True for row in receipt["gates"]):
        raise Hold30AlphaQualificationError("v3 software gate evidence is incomplete")
    qualified = receipt["qualified_file_sha256s"]
    if not isinstance(qualified, Mapping) or sha256_payload(qualified) != receipt[
        "qualified_content_sha256"
    ]:
        raise Hold30AlphaQualificationError("v3 qualified file digest map is invalid")
    claimed = receipt["qualification_sha256"]
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise Hold30AlphaQualificationError("v3 qualification digest is invalid")
    unsigned = dict(receipt)
    del unsigned["qualification_sha256"]
    if sha256_payload(unsigned) != claimed:
        raise Hold30AlphaQualificationError("v3 software receipt self-hash mismatch")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualify-software", choices=("qualify-software",))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = qualify_hold30_alpha_software(
        args.repo,
        timeout_seconds=args.timeout_seconds,
    )
    _write_new_json(args.output, receipt)
    print(receipt["qualification_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HOLD30_ALPHA_SOFTWARE_SCHEMA",
    "Hold30AlphaQualificationError",
    "Hold30AlphaQualificationInventory",
    "main",
    "qualify_hold30_alpha_software",
    "resolve_hold30_alpha_qualification_inventory",
    "verify_hold30_alpha_software_receipt",
]
