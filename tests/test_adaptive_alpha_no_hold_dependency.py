from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import subprocess
import sys

import pytest

from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    PROHIBITED_ADAPTIVE_HOLD_TOKENS,
    MassiveAdaptiveAlphaProtocolError,
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.training.massive_adaptive_profit_checkpoint_selection_v3 import (
    MassiveAdaptiveProfitCheckpointCandidateV3,
    MassiveAdaptiveProfitCheckpointSelectionV3,
)
from rl_quant.training.massive_adaptive_rl_policy_selection_v1 import (
    MassiveAdaptiveRLPolicyCandidateV1,
    MassiveAdaptiveRLPolicySelectionV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v1 import (
    MassiveAdaptiveRLOuterEvidenceV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v2 import (
    MassiveAdaptiveRLOuterEvidenceV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v3 import (
    MassiveAdaptiveRLOuterEvidenceV3,
)
from rl_quant.rl.massive_adaptive_rl_observation_v1 import (
    MassiveAdaptiveRLObservationV1,
)


def test_adaptive_protocol_payload_contains_no_forbidden_duration_key() -> None:
    assert_no_adaptive_hold_semantics(MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.payload())

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(child) for child in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not (
        keys(MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.payload())
        & FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS
    )


@pytest.mark.parametrize("field", sorted(FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS))
def test_adaptive_configuration_rejects_every_forbidden_duration_field(
    field: str,
) -> None:
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="forbidden"):
        assert_no_adaptive_hold_semantics({"model": {field: 30}})


@pytest.mark.parametrize("token", PROHIBITED_ADAPTIVE_HOLD_TOKENS)
def test_adaptive_semantic_payload_rejects_every_hold_token(token: str) -> None:
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="hold token"):
        assert_no_adaptive_hold_semantics(
            {"registered-description": f"adaptive-control-{token}-v1"}
        )


def test_import_firewall_rejects_historical_hold_modules(tmp_path: Path) -> None:
    forbidden = (
        "rl_quant.execution.age_aware_no_trade",
        "rl_quant.execution.hold30_sleeves",
        "rl_quant.envs.hold30",
        "rl_quant.models.hold30_exit_action_v6",
        "rl_quant.training.hold30_runtime",
        "rl_quant.protocol.hold30_alpha_m03r_v6",
    )
    for index, module in enumerate(forbidden):
        bad = tmp_path / f"bad_{index}.py"
        bad.write_text(f"from {module} import anything\n", encoding="utf-8")
        with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="forbidden"):
            assert_adaptive_import_firewall((bad,))


def test_no_hold30_imports_in_adaptive_namespace() -> None:
    root = Path(__file__).parents[1] / "src" / "rl_quant"
    paths = []
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if (
            "massive_adaptive" in path.name
            or "adaptive_alpha" in path.name
            or "massive_adaptive_alpha_v1" in source
        ):
            paths.append(path)
    paths.sort()
    assert paths
    assert_adaptive_import_firewall(paths)


def test_massive_adaptive_rl_has_no_hold_duration_dependency() -> None:
    prohibited_tokens = (
        *PROHIBITED_ADAPTIVE_HOLD_TOKENS,
        "holding_period",
        "holding-period",
    )
    package = Path(__file__).parents[1] / "src" / "rl_quant"
    adaptive_roots = tuple(
        package / name for name in ("rl", "training", "evaluation", "workflows")
    )
    adaptive_files = tuple(
        sorted(
            path
            for root in adaptive_roots
            for path in root.glob("massive_adaptive_*.py")
        )
    )
    violations = {
        str(path): tuple(token for token in prohibited_tokens if token in source)
        for path in adaptive_files
        if (source := path.read_text(encoding="utf-8").lower())
        and any(token in source for token in prohibited_tokens)
    }
    assert adaptive_files
    assert not violations, violations


def test_policy_selection_contains_no_duration_gate() -> None:
    forbidden = (
        "age",
        "duration",
        "persistence",
        "survival",
        "scheduled_exit",
        "hazard",
    )
    names = {
        field.name
        for artifact in (
            MassiveAdaptiveProfitCheckpointCandidateV3,
            MassiveAdaptiveProfitCheckpointSelectionV3,
            MassiveAdaptiveRLPolicyCandidateV1,
            MassiveAdaptiveRLPolicySelectionV1,
        )
        for field in fields(artifact)
    }
    assert not any(fragment in name for name in names for fragment in forbidden)


def test_observation_contains_no_target_duration() -> None:
    forbidden = (
        "age",
        "duration",
        "entry_session",
        "survival",
        "scheduled_exit",
    )
    names = {field.name for field in fields(MassiveAdaptiveRLObservationV1)}
    assert not any(fragment in name for name in names for fragment in forbidden)


def test_outer_evidence_contains_no_duration_gate() -> None:
    forbidden = (
        "age",
        "duration",
        "persistence",
        "survival",
        "scheduled_exit",
    )
    names = {
        field.name
        for artifact in (
            MassiveAdaptiveRLOuterEvidenceV1,
            MassiveAdaptiveRLOuterEvidenceV2,
            MassiveAdaptiveRLOuterEvidenceV3,
        )
        for field in fields(artifact)
    }
    assert not any(fragment in name for name in names for fragment in forbidden)


def test_adaptive_submodule_imports_do_not_load_legacy_runtime_modules() -> None:
    script = """
import sys
import rl_quant.evaluation.massive_adaptive_profitability_env_v1
import rl_quant.evaluation.massive_adaptive_economic_step_v1
forbidden = (
    'rl_quant.envs.hold30',
    'rl_quant.execution.age_aware_no_trade',
)
loaded = tuple(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('legacy modules loaded: ' + ','.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
