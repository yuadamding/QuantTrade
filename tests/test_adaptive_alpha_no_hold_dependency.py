from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS,
    MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL,
    MassiveAdaptiveAlphaProtocolError,
    assert_adaptive_import_firewall,
    assert_no_adaptive_hold_semantics,
)


def test_adaptive_protocol_payload_contains_no_forbidden_duration_key() -> None:
    assert_no_adaptive_hold_semantics(MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.payload())

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(child) for child in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not (keys(MASSIVE_ADAPTIVE_ALPHA_V1_PROTOCOL.payload()) & FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS)


@pytest.mark.parametrize("field", sorted(FORBIDDEN_ADAPTIVE_CONFIGURATION_FIELDS))
def test_adaptive_configuration_rejects_every_forbidden_duration_field(
    field: str,
) -> None:
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="forbidden"):
        assert_no_adaptive_hold_semantics({"model": {field: 30}})


def test_import_firewall_rejects_historical_hold_modules(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from rl_quant.execution.age_aware_no_trade import anything\n",
        encoding="utf-8",
    )
    with pytest.raises(MassiveAdaptiveAlphaProtocolError, match="forbidden"):
        assert_adaptive_import_firewall((bad,))


def test_all_current_adaptive_python_modules_pass_import_firewall() -> None:
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
