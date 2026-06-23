from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResearchProtocolError(ValueError):
    """Raised when a research artifact violates the protocol contract."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ResearchProtocolError(f"Invalid ISO timestamp {value!r}.") from exc
    if parsed.tzinfo is None:
        # A tz-naive timestamp has no absolute instant -- silently assuming UTC made the chronology checks
        # (fit/train/val/test windows, decision causality) depend on an implicit assumption. Reject it, as the
        # decision-log and second-context parsers already do; reportable artifacts must carry tz-aware ISO-8601.
        # Every timestamp this codebase emits is tz-aware (utc_now_iso -> datetime.now(timezone.utc)).
        raise ResearchProtocolError(f"Timestamp {value!r} must include timezone information.")
    return parsed


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_string_sequence(values: list[str]) -> str:
    return stable_json_hash(list(values))


@dataclass(frozen=True)
class FitWindow:
    fit_start: str
    fit_end: str
    feature_asof: str
    name: str = "feature_fit"

    def validate_prior_only(self) -> None:
        fit_start = parse_iso_timestamp(self.fit_start)
        fit_end = parse_iso_timestamp(self.fit_end)
        feature_asof = parse_iso_timestamp(self.feature_asof)
        if fit_start > fit_end:
            raise ResearchProtocolError(f"{self.name}: fit_start must be <= fit_end.")
        if fit_end >= feature_asof:
            raise ResearchProtocolError(
                f"{self.name}: fit_end {self.fit_end} must be before feature_asof {self.feature_asof}."
            )


@dataclass
class DatasetManifest:
    dataset_id: str
    created_at_utc: str
    source_vendor: str
    symbols: list[str]
    universe_selection_date: str | None
    bar_interval: str
    timezone: str
    adjustment: str
    feature_names: list[str]
    action_names: list[str]
    timestamps_hash: str
    next_timestamps_hash: str
    first_timestamp: str
    last_timestamp: str
    feature_fit_windows: list[FitWindow] = field(default_factory=list)
    source_manifest_hash: str | None = None
    known_limitations: list[str] = field(default_factory=list)
    # Canonical action-return basis recorded on the manifest -- not only in the .pt payload -- so the
    # reportability agreement check can compare the dataset's DECLARED basis against the evaluation's (the
    # ReturnBasis in rl_quant.protocol.action_return_basis reads these exact keys). All optional/None on legacy
    # manifests (default-preserving); they do not affect validate().
    action_return_weight_semantics: str | None = None
    action_return_formula: str | None = None
    action_return_clip_min: float | None = None
    action_return_clip_max: float | None = None
    action_return_semantics_version: str | None = None
    action_return_fill_convention: str | None = None
    # Structured v2 provenance (the precise economics behind fill_convention). Optional/None on v1 manifests.
    action_return_basis_version: str | None = None
    action_return_entry_fill_rule: str | None = None
    action_return_exit_fill_rule: str | None = None
    action_return_execution_latency_ms: int | None = None
    action_return_source_bar_interval: str | None = None
    action_return_price_source: str | None = None
    # The basis content hash (sha256 over the declared basis), persisted as a first-class field so it round-trips
    # through from_dict/to_dict and validate() can confirm it matches the declared basis (catching a stale /
    # hand-edited hash). Optional/None on manifests that do not record it. This is NOT an action_return_* key, so
    # it is not subject to the protected-prefix typo guard.
    return_basis_content_hash: str | None = None

    def validate(self) -> None:
        if not self.dataset_id:
            raise ResearchProtocolError("dataset_id is required.")
        if not self.symbols:
            raise ResearchProtocolError("symbols must not be empty.")
        if not self.feature_names:
            raise ResearchProtocolError("feature_names must not be empty.")
        if not self.action_names:
            raise ResearchProtocolError("action_names must not be empty.")
        if self.action_names[0] != "CASH":
            raise ResearchProtocolError("action_names must start with CASH.")
        if parse_iso_timestamp(self.first_timestamp) > parse_iso_timestamp(self.last_timestamp):
            raise ResearchProtocolError("first_timestamp must be <= last_timestamp.")
        if self.universe_selection_date is None:
            raise ResearchProtocolError("universe_selection_date is required for point-in-time universe validation.")
        selection_ts = parse_iso_timestamp(self.universe_selection_date)
        first_ts = parse_iso_timestamp(self.first_timestamp)
        if selection_ts > first_ts:
            raise ResearchProtocolError("universe_selection_date must be before or at first_timestamp.")
        for window in self.feature_fit_windows:
            window.validate_prior_only()
        # If a return-basis content hash is recorded, it MUST match the declared action_return_* basis -- this
        # catches a stale or hand-edited hash (the basis economics and their stamp must agree). A manifest that
        # records no hash is unaffected (default-preserving). Imported lazily to keep this module's import graph
        # free of the protocol-layer dependency at module load.
        if self.return_basis_content_hash is not None:
            from rl_quant.protocol.action_return_basis import ReturnBasis

            basis = ReturnBasis.from_mapping(self)
            # Recording a hash is opting into a COMPLETE, VALUE-VALID basis -- not merely "the hash equals
            # whatever fields happen to be declared". So a recorded hash requires the declared basis to be valid
            # (no corrupt clip / latency / blank string / unrecognized version) AND complete; otherwise a partial
            # or invalid basis could be hashed over and still pass. (A manifest with no hash is unaffected.)
            basis_errors = basis.validation_errors()
            if basis_errors:
                raise ResearchProtocolError(
                    f"return_basis_content_hash is recorded but the declared action-return basis is invalid: "
                    f"{basis_errors}."
                )
            if not basis.is_complete():
                raise ResearchProtocolError(
                    "return_basis_content_hash is recorded but the declared action-return basis is incomplete; "
                    "a hashed basis must be complete."
                )
            expected = basis.content_hash()
            if self.return_basis_content_hash != expected:
                raise ResearchProtocolError(
                    f"return_basis_content_hash {self.return_basis_content_hash!r} does not match the declared "
                    f"action-return basis (expected {expected!r}); the basis and its content hash must agree."
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        windows = [FitWindow(**item) for item in payload.get("feature_fit_windows", [])]
        # The on-disk manifest is intentionally a SUPERSET: the builder adds reportable / reportability_errors /
        # missing_* extras via update(). Filter to declared fields so from_dict round-trips an enriched manifest
        # instead of raising on those extras. Unknown keys are dropped here; callers that need them (e.g. the
        # reportability summary's basis-agreement check) read the raw JSON dict directly. Trade-off: a MISSPELLED
        # OPTIONAL field (e.g. a typo'd action_return_* key) is silently dropped and degrades to its default
        # rather than raising -- acceptable because the manifest is machine-generated from a single dict; a
        # misspelled REQUIRED field still fails (its positional arg ends up absent).
        known = {f.name for f in fields(cls)}
        # ...EXCEPT the basis-governance keys (action_return_* AND return_basis_*), which are the protected
        # reportability contract: a silently-dropped typo there (e.g. action_return_fill_conventon, or
        # return_basis_content_hahs) would degrade the manifest-side basis to all-None / disable the hash check,
        # making the return-basis governance vacuous -- so reject any unknown key with those prefixes.
        protected_typos = sorted(
            k for k in payload
            if (k.startswith("action_return_") or k.startswith("return_basis_")) and k not in known
        )
        if protected_typos:
            raise ResearchProtocolError(
                f"unknown protected basis key(s) on DatasetManifest (likely a typo of a basis/hash field): "
                f"{protected_typos}. The action-return basis and its content hash are the protected reportability "
                "contract; a silently-dropped typo would make the return-basis governance vacuous."
            )
        kwargs = {key: value for key, value in payload.items() if key in known}
        kwargs["feature_fit_windows"] = windows
        return cls(**kwargs)

    def write_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class BaselineResult:
    name: str
    total_return: float
    sharpe: float | None
    max_drawdown: float
    turnover: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class StressTestResult:
    name: str
    kind: str
    parameter: str
    value: float | str
    total_return: float
    sharpe: float | None
    max_drawdown: float


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    train_start: str | None
    train_end: str
    val_end: str
    test_start: str
    test_end: str | None
    purge_rule: str = "chronological_no_overlap"
    embargo_bars: int = 0
    benchmark_names: list[str] = field(default_factory=list)
    cost_stress_multipliers: list[float] = field(default_factory=lambda: [1.0, 2.0])
    frequency_stress_configs: list[str] = field(default_factory=lambda: ["base"])

    def validate(self) -> None:
        train_end = parse_iso_timestamp(self.train_end)
        val_end = parse_iso_timestamp(self.val_end)
        test_start = parse_iso_timestamp(self.test_start)
        if self.train_start is not None:
            # parse_iso_timestamp also rejects tz-naive / malformed values.
            if parse_iso_timestamp(self.train_start) > train_end:
                raise ResearchProtocolError("train_start must be <= train_end.")
        if train_end >= val_end:
            raise ResearchProtocolError("train_end must be before val_end.")
        if val_end >= test_start:
            raise ResearchProtocolError("val_end must be before test_start.")
        if self.test_end is not None and parse_iso_timestamp(self.test_end) < test_start:
            raise ResearchProtocolError("test_end must be >= test_start.")
        if not self.benchmark_names:
            raise ResearchProtocolError("At least one benchmark is required.")


@dataclass
class ModelManifest:
    model_id: str
    created_at_utc: str
    algorithm: str
    encoder: str
    training_dataset_id: str
    validation_protocol: EvaluationProtocol
    hyperparameter_search_space_hash: str
    hyperparameter_trials: int
    selected_by: str
    feature_names_hash: str
    action_names_hash: str
    # Structured selection record. selection_split is the anti-leakage GATE (must be "validation" for a
    # strict-reportable model -- a checkpoint chosen on the test split is leakage); it replaces the brittle
    # "test" in selected_by heuristic. selected_by stays a human-readable description, no longer the enforced
    # field. selection_metric / selection_artifact_hash are optional provenance (recorded, not enforced).
    selection_split: str | None = None
    selection_metric: str | None = None
    selection_artifact_hash: str | None = None
    baseline_results: list[BaselineResult] = field(default_factory=list)
    cost_stress_results: list[StressTestResult] = field(default_factory=list)
    frequency_stress_results: list[StressTestResult] = field(default_factory=list)
    allowed_use: str = "research only"
    not_allowed_use: str = "unattended live trading"

    def validate_reportable(self, *, strict: bool = True) -> None:
        if not self.model_id:
            raise ResearchProtocolError("model_id is required.")
        if self.hyperparameter_trials < 1:
            raise ResearchProtocolError("hyperparameter_trials must be positive.")
        if not self.hyperparameter_search_space_hash:
            raise ResearchProtocolError("hyperparameter_search_space_hash is required.")
        if not self.selected_by:
            raise ResearchProtocolError("selected_by is required.")
        if strict:
            # Structured anti-leakage gate: a reportable model MUST be selected on the validation split. A
            # non-validation (or missing) selection_split fails closed -- regardless of the free-text
            # selected_by label, and never INFERRED from that label.
            if self.selection_split != "validation":
                raise ResearchProtocolError(
                    f"reportable model must declare selection_split == 'validation' (got {self.selection_split!r}); "
                    "selecting a checkpoint on the test split is leakage. A legacy manifest lacking the field "
                    "must be re-validated with strict=False and migrated to the structured field."
                )
            for required_name in ("created_at_utc", "algorithm", "encoder", "training_dataset_id",
                                  "feature_names_hash", "action_names_hash"):
                if not getattr(self, required_name):
                    raise ResearchProtocolError(f"{required_name} is required for reportability.")
        elif "test" in self.selected_by.lower():
            # Legacy compatibility (no structured selection_split): fall back to the brittle selected_by text
            # heuristic so an old manifest still trips on an obvious test-split selection.
            raise ResearchProtocolError(
                "selected_by must reference validation, not test; selecting a checkpoint on the test "
                "split is leakage."
            )
        self.validation_protocol.validate()
        if not self.baseline_results:
            raise ResearchProtocolError("Reportable models require at least one baseline result.")
        baseline_names = {result.name for result in self.baseline_results}
        missing_benchmarks = [
            name for name in self.validation_protocol.benchmark_names if name not in baseline_names
        ]
        if missing_benchmarks:
            raise ResearchProtocolError(
                "Declared validation benchmark_names are missing from baseline_results: "
                f"{missing_benchmarks}. Every declared benchmark must have a produced baseline result."
            )
        if not self.cost_stress_results:
            raise ResearchProtocolError("Reportable models require at least one cost stress result.")
        if not self.frequency_stress_results:
            raise ResearchProtocolError("Reportable models require at least one frequency stress result.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelManifest":
        payload = dict(payload)
        payload["validation_protocol"] = EvaluationProtocol(**payload["validation_protocol"])
        payload["baseline_results"] = [BaselineResult(**item) for item in payload.get("baseline_results", [])]
        payload["cost_stress_results"] = [StressTestResult(**item) for item in payload.get("cost_stress_results", [])]
        payload["frequency_stress_results"] = [
            StressTestResult(**item) for item in payload.get("frequency_stress_results", [])
        ]
        return cls(**payload)


def default_benchmark_registry(action_names: list[str]) -> list[str]:
    out = ["CASH", "PreviousActionNoTrade", "RandomSameTurnover"]
    for candidate in ("QQQ", "SPY"):
        if candidate in action_names:
            out.append(f"BuyAndHold_{candidate}")
    if len(action_names) > 2:
        out.append("EqualWeight_ETFs")
    return list(dict.fromkeys(out))


class ExperimentRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as sink:
            sink.write(json.dumps(record, sort_keys=True, default=str) + "\n")
