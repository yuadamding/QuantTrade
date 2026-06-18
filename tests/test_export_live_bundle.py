"""Producer-side parity gate: a freshly-exported bundle reproduces under rl_quant_live.

This is the cross-repo half of the offline↔online parity gate, run in the
QuantTrade (rl_quant) repo. It regenerates the deployment bundle from the REAL
rl_quant model via ``scripts/export_live_bundle.py``, then loads it with the
rl_quant_live consumer and asserts the live ``TorchScorer`` reproduces the golden
scores. This catches EXPORTER drift (the committed live-side fixture only catches
consumer drift). rl_quant_live is an offline build/test tool here; the live
runtime never imports rl_quant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("rl_quant_live")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_live_bundle import build_skeleton_bundle  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:enable_nested_tensor")


def test_exported_bundle_reproduces_under_live(tmp_path: Path) -> None:
    from rl_quant_live.artifact_contract.loader import load_bundle
    from rl_quant_live.model_serving import TorchScorer
    from rl_quant_live.panel.decision_tensor import DecisionTensorBatch
    from rl_quant_live.parity import assert_parity, load_golden
    from rl_quant_live.protocol.enums import Environment

    bundle = build_skeleton_bundle(tmp_path / "bundle", seed=0)

    loaded = load_bundle(bundle, environment=Environment.OBSERVE, require_approved=False)
    golden = load_golden(bundle / "parity", require_schema_hash=True)
    batch = DecisionTensorBatch(**golden.tensors)
    report = assert_parity(batch, golden, TorchScorer.from_bundle(loaded).score(batch))
    assert report.ok


def test_export_is_deterministic(tmp_path: Path) -> None:
    # Same seed -> same golden content hash (the exporter must be reproducible).
    from rl_quant_live.parity import load_golden

    a = build_skeleton_bundle(tmp_path / "a", seed=0)
    b = build_skeleton_bundle(tmp_path / "b", seed=0)
    assert load_golden(a / "parity").content_hash == load_golden(b / "parity").content_hash
