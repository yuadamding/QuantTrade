"""Cutoff-aware forecast inputs, not a trainer or native V5 authorization.

Full-history outcome preparation is separate from access by a fitting model.
This reader reveals only labels mature at the declared prior fit cutoff. It
does not select settings, fit a forecast, or open a validation/outer authority.
"""

from __future__ import annotations

import math
from pathlib import Path

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_features_v1 import (
    FEATURE_NAMES, SCHEMA as FEATURE_SCHEMA, _parquet_rows, feature_specification, fit_normalizer,
)
from rl_quant.data_sources.massive.qt200_aggregate_targets_v1 import (
    BOUNDARIES, BUCKET_NAMES, SCHEMA as TARGET_SCHEMA, read_completion, target_specification, verify_artifact,
)
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS


def prepare_fit_inputs(*, feature_root: Path, feature_sha256: str, target_root: Path,
                       target_sha256: str, fit_sessions: list[str], cutoff_session: str,
                       heldout_start: str) -> dict:
    """Read exact persisted rows and compute a fit-only scaler; no model updates.

    Samples retain feature missingness and are NOT selected using future
    profitability or future target completeness. A sample with no mature label
    remains in the fit view with a false loss mask. Consumers must honor both
    masks and the independent, currently false, training authorization.
    """
    fbody, feature = read_completion(feature_root, feature_sha256, FEATURE_SCHEMA)
    tbody, target = read_completion(target_root, target_sha256, TARGET_SCHEMA)
    if (target.get("feature_sha256") != feature_sha256 or target.get("ordered_tickers") != list(SYMBOLS)
            or target.get("panel_sha256") != feature.get("panel_sha256")
            or target.get("numeric_target_preparation_complete") is not True
            or feature.get("feature_names") != list(FEATURE_NAMES)
            or feature.get("numeric_feature_preparation_complete") is not True
            or feature.get("normalization_fitted") is not False):
        raise ValueError("Forecast input parent binding differs")
    days = target["sessions"]
    if not days or days != sorted(set(days)):
        raise ValueError("Forecast calendar differs")
    if (not fit_sessions or fit_sessions != sorted(set(fit_sessions))
            or not set(fit_sessions).issubset(days) or cutoff_session not in days or heldout_start not in days
            or not fit_sessions[-1] <= cutoff_session < heldout_start):
        raise ValueError("Forecast fitting crosses held-out/cutoff boundary")
    begin, end = days.index(fit_sessions[0]), days.index(fit_sessions[-1])
    if days[begin:end + 1] != fit_sessions:
        raise ValueError("Forecast fit interval compresses calendar time")
    for root, parent, name in ((feature_root, feature, "features.parquet"),
                               (target_root, target, "targets.parquet")):
        verify_artifact(root, parent, name)
    if (transport.parse_json(verify_artifact(feature_root, feature, "specification.json")) != feature_specification()
            or transport.parse_json(verify_artifact(target_root, target, "specification.json")) != target_specification()):
        raise ValueError("Forecast input specification differs")
    fit_rows, samples = [], []
    selected = set(fit_sessions)
    total = 0
    label_counts = [0] * len(BUCKET_NAMES)
    features = _parquet_rows(feature_root / "features.parquet")
    targets = _parquet_rows(target_root / "targets.parquet")
    for offset, (frow, trow) in enumerate(zip(features, targets, strict=True)):
        total += 1
        security_index, session_index = divmod(offset, len(days))
        if security_index >= len(SYMBOLS):
            raise ValueError("Forecast input has extra rows")
        key = (SYMBOLS[security_index], days[session_index])
        if ((frow["ticker"], frow["decision_session"]) != key
                or (trow["ticker"], trow["decision_session"]) != key
                or trow["feature_row_index"] != offset
                or frow["latest_feature_session"] != (days[session_index - 1] if session_index else None)
                or len(frow["values"]) != len(FEATURE_NAMES)
                or frow["observed_mask"] != [v is not None for v in frow["values"]]
                or frow["training_eligible"] is not False or trow["training_eligible"] is not False):
            raise ValueError("Forecast row linkage/feature clock differs")
        fields = ("values", "observed_mask", "reason_bits", "bucket_start_sessions",
                  "bucket_end_sessions", "label_maturity_sessions")
        if (any(len(trow[k]) != len(BUCKET_NAMES) for k in fields)
                or trow["observed_mask"] != [v is not None for v in trow["values"]]
                or trow["observed_mask"] != [r == 0 for r in trow["reason_bits"]]):
            raise ValueError("Forecast target masks differ")
        for bucket, (lo, hi) in enumerate(zip(BOUNDARIES, BOUNDARIES[1:])):
            for field, shift in (("bucket_start_sessions", lo + 1), ("bucket_end_sessions", hi + 1),
                                 ("label_maturity_sessions", hi + 2)):
                position = session_index + shift
                if trow[field][bucket] != (days[position] if position < len(days) else None):
                    raise ValueError("Forecast label maturity/geometry differs")
        if any(v is not None and (isinstance(v, bool) or not math.isfinite(v))
               for v in frow["values"] + trow["values"]):
            raise ValueError("Nonfinite forecast input")
        if key[1] not in selected:
            continue
        fit_rows.append(frow)
        losses, visible = [], []
        for value, maturity in zip(trow["values"], trow["label_maturity_sessions"], strict=True):
            valid = value is not None and maturity is not None and maturity <= cutoff_session
            losses.append(valid)
            visible.append(value if valid else None)
        label_counts = [n + valid for n, valid in zip(label_counts, losses, strict=True)]
        samples.append(dict(ticker=key[0], decision_session=key[1], feature_row_index=offset,
            features=frow["values"], feature_mask=frow["observed_mask"],
            targets=visible, target_loss_mask=losses))
    if total != len(SYMBOLS) * len(days) or total != target["counts"]["rows"] or total != feature["counts"]["rows"]:
        raise ValueError("Forecast source population incomplete")
    scaler = fit_normalizer(fit_rows, fit_sessions=fit_sessions, heldout_start=heldout_start)
    for root, parent, name in ((feature_root, feature, "features.parquet"),
                               (target_root, target, "targets.parquet"),
                               (feature_root, feature, "specification.json"),
                               (target_root, target, "specification.json")):
        verify_artifact(root, parent, name)
    if (transport.read_regular(feature_root / "COMPLETE.json", 4 * 1024 * 1024) != fbody
            or transport.read_regular(target_root / "COMPLETE.json", 4 * 1024 * 1024) != tbody):
        raise ValueError("Forecast source changed during preparation")
    return dict(schema="rl-quant.qt200-bars-forecast-fit-inputs-v1", samples=samples,
        feature_sha256=feature_sha256, target_sha256=target_sha256, normalizer=scaler,
        fit_sessions=fit_sessions, cutoff_session=cutoff_session, heldout_start=heldout_start,
        mature_labels_by_bucket=dict(zip(BUCKET_NAMES, label_counts, strict=True)),
        feature_missingness_retained=True, label_maturity_is_assumed=True,
        source_reverified_before_and_after=True, training_authorized=False, model_fitted=False)
