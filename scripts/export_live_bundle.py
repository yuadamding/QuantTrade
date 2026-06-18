"""Export a frozen rl_quant -> rl_quant_live deployment bundle + golden parity fixture.

This is the PRODUCER half of the offline↔online parity gate. It instantiates the
REAL rl_quant model (``SecondContextTransformerQNetwork``), runs its REAL forward
pass to produce golden scores, and writes a frozen bundle that rl_quant_live can
load and reproduce. It is an OFFLINE build tool: it imports both ``rl_quant`` (the
model) and ``rl_quant_live`` (the bundle/canonicalization + golden format) — that
is fine, because the no-import rule binds the live *runtime*, not an export script.
The live runtime still consumes only the file-based bundle.

Walking-skeleton scope: a deterministic, SEEDED model (random init — the parity
gate proves "live serving reproduces research serving for identical weights", which
holds for any weights) over a tiny SYNTHETIC-but-schema-valid decision batch, with
an IDENTITY normalizer (the bundle ships already-model-ready tensors). The real
trained checkpoint + the two-block z-score normalizer + feature-builder parity are
later slices; this proves the loop end-to-end with the real architecture + forward.

Run:  python scripts/export_live_bundle.py --output-dir <dir> [--seed 0]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from rl_quant.models.second_context import SecondContextTransformerQNetwork

from rl_quant_live.artifact_contract.action_schema import ActionSchema
from rl_quant_live.artifact_contract.decision_tensor_schema import DecisionTensorSchema
from rl_quant_live.artifact_contract.exporter import write_bundle
from rl_quant_live.artifact_contract.feature_schema import FeatureSchema
from rl_quant_live.artifact_contract.hashing import stable_json_hash
from rl_quant_live.artifact_contract.normalizer import Normalizer
from rl_quant_live.panel.decision_tensor import DecisionTensorBatch
from rl_quant_live.parity.golden import save_golden

_SCHEMA_VERSION = "stock_second_context_decision_v3"
_PROTOCOL_VERSION = "decision_tensor_v1"

# Real rl_quant second-context feature names (rl_quant.features.stock_second_context).
_MARKET_NAMES = (
    "active_symbol_count", "active_fraction", "equal_weight_return", "dollar_volume_weighted_return",
    "median_return", "return_std", "up_fraction", "down_fraction", "top_decile_return", "bottom_decile_return",
    "top_minus_bottom_return", "abs_return_dollar_volume_weighted", "dollar_volume_concentration",
    "transaction_concentration", "log_total_dollar_volume", "log_total_volume", "log_total_transactions",
    "mean_range_bps", "range_bps_std", "mean_active_seconds", "missing_symbol_fraction", "large_move_fraction",
    "quality_score", "is_premarket", "is_regular_session", "is_postmarket", "seconds_since_open", "seconds_to_close",
)
_ACTION_NAMES = (
    "action_index_scaled", "is_cash", "is_etf", "is_stock", "is_inverse", "is_leveraged", "leverage_factor",
    "target_weight", "valid_price_flag", "feature_staleness_seconds", "log_last_dollar_volume", "estimated_cost_bps",
)
_PORTFOLIO_NAMES = ("cash_weight", "gross_exposure", "previous_action_index_scaled")
_CONSTRAINT_NAMES = ("data_quality_score", "valid_action_fraction", "minutes_to_close_scaled")

# Tiny but real-width architecture (small d_model keeps the committed fixture ~tens of KB).
_ARCH = {
    "class": "SecondContextTransformerQNetwork",
    "market_feature_dim": len(_MARKET_NAMES),       # 28
    "action_feature_dim": len(_ACTION_NAMES),       # 12
    "portfolio_state_dim": len(_PORTFOLIO_NAMES),   # 3
    "constraint_state_dim": len(_CONSTRAINT_NAMES), # 3
    "d_model": 32,
    "n_heads": 4,
    "temporal_layers": 1,
    "feedforward_dim": 48,
    "dropout": 0.0,
    "max_lookback_blocks": 8,
    "action_count": 4,  # CASH + 3
}
_CTOR_KEYS = tuple(k for k in _ARCH if k != "class")

_MODEL_INPUT_KEYS = (
    "decision_timestamps_ms", "market_context", "market_context_mask", "market_context_available_ts",
    "action_features", "action_features_available_ts", "decision_action_valid_mask", "action_cost_bps",
    "action_target_weights", "action_ids", "portfolio_state", "constraint_state", "decision_quality_score",
    "force_cash_mask",
)


def _build_batch(seed: int) -> DecisionTensorBatch:
    """A tiny, schema-valid synthetic decision batch (B=2, A=4, L=8)."""
    rng = np.random.default_rng(seed)
    b, a, lb = 2, _ARCH["action_count"], _ARCH["max_lookback_blocks"]
    decision_ts = np.array([1_000_000, 2_000_000], dtype=np.int64)
    valid = np.array([[True, True, True, False], [True, True, True, True]])  # CASH(0) always valid
    avail = np.where(valid, decision_ts[:, None], -1).astype(np.int64)
    return DecisionTensorBatch(
        decision_timestamps_ms=decision_ts,
        market_context=rng.standard_normal((b, lb, len(_MARKET_NAMES))).astype(np.float32),
        market_context_mask=np.ones((b, lb), dtype=bool),
        market_context_available_ts=np.repeat(decision_ts[:, None], lb, axis=1),
        action_features=rng.standard_normal((b, a, len(_ACTION_NAMES))).astype(np.float32),
        action_features_available_ts=avail,
        decision_action_valid_mask=valid,
        action_cost_bps=np.zeros((b, a), dtype=np.float32),
        action_target_weights=np.zeros((b, a), dtype=np.float32),
        action_ids=np.broadcast_to(np.arange(a, dtype=np.int64), (b, a)).copy(),
        portfolio_state=rng.standard_normal((b, len(_PORTFOLIO_NAMES))).astype(np.float32),
        constraint_state=rng.standard_normal((b, len(_CONSTRAINT_NAMES))).astype(np.float32),
        decision_quality_score=np.array([0.8, 0.9], dtype=np.float32),
        force_cash_mask=np.array([False, False]),
    )


def _decision_tensor_schema() -> DecisionTensorSchema:
    return DecisionTensorSchema.from_dict({
        "schema_version": _SCHEMA_VERSION,
        "protocol_version": _PROTOCOL_VERSION,
        "lookback_blocks": _ARCH["max_lookback_blocks"],
        "market_context_dim": _ARCH["market_feature_dim"],
        "action_feature_dim": _ARCH["action_feature_dim"],
        "portfolio_state_dim": _ARCH["portfolio_state_dim"],
        "constraint_state_dim": _ARCH["constraint_state_dim"],
        "action_count": _ARCH["action_count"],
        "cash_action_index": 0,
        "bar_latency_ms": 1000,
        "decision_interval": "15m",
        "block_seconds": 300,
        "context_seconds": 3600,
        "min_active_symbols": 1,
        "max_action_staleness_seconds": 300,
        "model_input_keys": list(_MODEL_INPUT_KEYS),
        "label_keys": ["action_realized_return"],
        "forbidden_model_input_keys": ["action_realized_return", "fill_price"],
        "feature_names": {
            "market_context": list(_MARKET_NAMES),
            "action_features": list(_ACTION_NAMES),
            "portfolio_state": list(_PORTFOLIO_NAMES),
            "constraint_state": list(_CONSTRAINT_NAMES),
        },
        "execution_latency_ms": 1000,
    })


def _golden_scores(model: SecondContextTransformerQNetwork, batch: DecisionTensorBatch) -> np.ndarray:
    """Run the REAL rl_quant forward (CPU, eval, no-grad) and mask exactly as the
    live TorchScorer will: invalid actions -> -inf (force-cash all False here)."""
    model.eval()
    with torch.no_grad():
        raw = model(
            torch.from_numpy(batch.market_context),
            torch.from_numpy(batch.market_context_mask),
            torch.from_numpy(batch.action_features),
            torch.from_numpy(batch.portfolio_state),
            torch.from_numpy(batch.constraint_state),
            torch.from_numpy(batch.action_ids),
        )
    raw_np = raw.detach().cpu().numpy().astype(np.float32)
    return np.where(batch.decision_action_valid_mask, raw_np, -np.inf).astype(np.float32)


def build_skeleton_bundle(dest: str | Path, *, seed: int = 0) -> Path:
    dest = Path(dest)
    torch.manual_seed(seed)  # seed BEFORE init so weights are deterministic
    model = SecondContextTransformerQNetwork(**{k: _ARCH[k] for k in _CTOR_KEYS})

    batch = _build_batch(seed)
    dts = _decision_tensor_schema()
    batch.validate(dts)  # fail-closed: never export a batch the live validator would reject
    scores = _golden_scores(model, batch)

    flat_names = (*_MARKET_NAMES, *_ACTION_NAMES, *_PORTFOLIO_NAMES, *_CONSTRAINT_NAMES)
    feature_schema = FeatureSchema(
        schema_version=_SCHEMA_VERSION, feature_names=flat_names, dtypes=("float32",) * len(flat_names),
    )
    action_schema = ActionSchema(
        action_names=("CASH", "A1", "A2", "A3"), cash_action_name="CASH", cash_action_index=0,
    )
    normalizer = Normalizer(  # identity: the bundle ships model-ready tensors (skeleton)
        method="identity", feature_names=flat_names,
        center=tuple(0.0 for _ in flat_names), scale=tuple(1.0 for _ in flat_names),
    )

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = Path(tmp) / "model.safetensors"
        save_file(model.state_dict(), str(weights_path))
        write_bundle(
            dest,
            model_src=weights_path,
            model_format="safetensors",
            model_family="second_context",
            model_class="SecondContextTransformerQNetwork",
            feature_schema=feature_schema,
            action_schema=action_schema,
            normalizer=normalizer,
            constraints={},
            approved_for=("observe", "paper"),
            not_approved_for=("live", "live_tiny"),
            reportability_status="reportable",
            statistical_status="passed",
            decision_frequency="15m",
            requires_position_state=True,
            decision_tensor_schema=dts,
            extra_manifests={"model_architecture": _ARCH},
            extra_manifest_fields={
                "architecture_hash": stable_json_hash(_ARCH),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "parity_seed": seed,
            },
        )

    save_golden(
        dest / "parity",
        batch=batch,
        scores=scores,
        schema_version=_SCHEMA_VERSION,
        protocol_version=_PROTOCOL_VERSION,
        decision_tensor_schema_hash=dts.content_hash(),
    )
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a frozen rl_quant -> rl_quant_live parity bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = build_skeleton_bundle(args.output_dir, seed=args.seed)
    print(f"wrote parity bundle -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
