"""Canonical action-return basis: the protocol-layer contract for HOW a dataset's action_returns are defined.

A reportable result must declare the FULL basis behind its returns -- the weight semantics (the PR-4 cost
basis), the return formula, clip bounds, semantics version, and fill convention -- so the evaluation's basis can
be checked for AGREEMENT against the dataset's declared basis. This lives in the protocol layer (stdlib-only, no
torch) so the dataset loader, the DatasetManifest, and the reportability/decision validators can all share ONE
definition instead of an evaluation validator reaching into a dataset module for it.

Pure, stdlib only; changes no backtest number.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

# The only accepted values for the action-return weight semantics (PR-4 gate). Anything else -- None,
# "unresolved", a typo, a vague string -- fail-closes use_execution_env_reward. See
# docs/execution_wiring_design.md §3: the value determines the execution-reward turnover cost basis.
ALLOWED_ACTION_RETURN_WEIGHT_SEMANTICS = frozenset(
    {"metadata_weighted_portfolio_returns", "full_capital_single_slot_returns"}
)


# Canonical field name -> the payload/split attribute key it wraps. The ReturnBasis object reads/writes these.
_RETURN_BASIS_FIELD_KEYS = {
    "weight_semantics": "action_return_weight_semantics",
    "formula": "action_return_formula",
    "clip_min": "action_return_clip_min",
    "clip_max": "action_return_clip_max",
    "semantics_version": "action_return_semantics_version",
    "fill_convention": "action_return_fill_convention",
    # Structured v2 provenance: the precise economics summarized by the fill_convention label. Optional/None on
    # a v1 (or version-less) basis -- a basis_version=="v2" basis declares all of them.
    "basis_version": "action_return_basis_version",
    "entry_fill_rule": "action_return_entry_fill_rule",
    "exit_fill_rule": "action_return_exit_fill_rule",
    "execution_latency_ms": "action_return_execution_latency_ms",
    "source_bar_interval": "action_return_source_bar_interval",
    "price_source": "action_return_price_source",
}

# The v1 core basis (the original flat fields). v2 ADDS the structured provenance fields below. Completeness is
# version-aware so an existing version-less complete basis stays complete (default-preserving).
_V1_BASIS_FIELDS = ("weight_semantics", "formula", "clip_min", "clip_max", "semantics_version", "fill_convention")
_V2_STRUCTURED_FIELDS = (
    "entry_fill_rule", "exit_fill_rule", "execution_latency_ms", "source_bar_interval", "price_source",
)
# Recognized basis-SCHEMA versions (None == version-less, treated as v1). basis_version versions the SCHEMA
# (which provenance fields are mandatory), which is ORTHOGONAL to semantics_version (the return-FORMULA
# economics) -- a v2-schema basis legitimately carries semantics_version "v1".
_KNOWN_BASIS_VERSIONS = frozenset({"v1", "v2"})


def _basis_value_differs(a: Any, b: Any) -> bool:
    """NaN-safe inequality for basis-field comparison: two NaN floats are treated as EQUAL so a basis never
    contradicts itself (NaN != NaN would otherwise surface a corrupt NaN clip bound as a spurious
    self-disagreement). Any other values use ordinary ``!=`` (so -0.0 == 0.0, -1.0 == -1, etc.)."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:  # both NaN
        return False
    return a != b


@dataclass(frozen=True)
class ReturnBasis:
    """Canonical, hashable wrapper for the FULL action-return basis -- the weight semantics (the PR-4 cost
    basis), the return formula, clip bounds, semantics version, and fill convention. It WRAPS the loose
    ``action_return_*`` fields the dataset already records (additive; it does not replace them). Used to declare
    the evaluation's basis in the reportability summary and to assert AGREEMENT between what the dataset
    declares and what the evaluation used. A field is "declared" when it is not None."""

    weight_semantics: str | None = None
    formula: str | None = None
    clip_min: float | None = None
    clip_max: float | None = None
    semantics_version: str | None = None
    fill_convention: str | None = None
    # Structured v2 provenance (declared together when basis_version == "v2"): the precise entry/exit fill
    # rules, execution latency, source bar interval, and price source behind the fill_convention summary label.
    basis_version: str | None = None
    entry_fill_rule: str | None = None
    exit_fill_rule: str | None = None
    execution_latency_ms: int | None = None
    source_bar_interval: str | None = None
    price_source: str | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "ReturnBasis":
        """Build from a mapping (dataset manifest dict) OR any object exposing the ``action_return_*`` attributes
        (a HourFromMinuteDataSplit). Missing keys/attributes resolve to None. A NESTED
        ``{"action_return_basis": {...}}`` surface (metadata.json, .pt["source"]) carries the basis one level
        down; it is read transparently so from_mapping works on every artifact surface, not only flat ones."""
        if hasattr(payload, "get") and isinstance(payload.get("action_return_basis"), dict):
            payload = payload["action_return_basis"]
        getter = payload.get if hasattr(payload, "get") else (lambda key, default=None: getattr(payload, key, default))
        return cls(**{name: getter(key, None) for name, key in _RETURN_BASIS_FIELD_KEYS.items()})

    @classmethod
    def from_canonical(cls, payload: Any) -> "ReturnBasis":
        """Build from a mapping keyed by the CANONICAL field names -- i.e. the round-trip of ``to_dict()`` as
        stored in the reportability summary's ``return_basis`` section. Missing keys resolve to None."""
        getter = payload.get if hasattr(payload, "get") else (lambda key, default=None: getattr(payload, key, default))
        return cls(**{name: getter(name, None) for name in _RETURN_BASIS_FIELD_KEYS})

    def to_dict(self) -> dict[str, Any]:
        """All canonical fields (including None), for the reportability summary."""
        return {name: getattr(self, name) for name in _RETURN_BASIS_FIELD_KEYS}

    def to_payload_mapping(self) -> dict[str, Any]:
        """The basis keyed by the ``action_return_*`` PAYLOAD/manifest keys (the inverse of from_mapping), so a
        builder can write it into a dataset payload / DatasetManifest without hand-spelling the keys."""
        return {key: getattr(self, name) for name, key in _RETURN_BASIS_FIELD_KEYS.items()}

    def declared(self) -> dict[str, Any]:
        """Only the fields that are declared (non-None)."""
        return {name: value for name, value in self.to_dict().items() if value is not None}

    def is_complete(self) -> bool:
        """True iff the basis is fully declared FOR ITS VERSION and the weight semantics is recognized. A v1
        (or version-less) basis requires the core flat fields; a basis_version == "v2" basis additionally
        requires the structured provenance (entry/exit fill rule, execution latency, source bar interval, price
        source). Default-preserving: an existing version-less complete basis is still complete."""
        if self.weight_semantics not in ALLOWED_ACTION_RETURN_WEIGHT_SEMANTICS:
            return False
        if any(getattr(self, name) is None for name in _V1_BASIS_FIELDS):
            return False
        # Fail-closed on the discriminator: a version-less or explicit "v1" basis needs only the core fields,
        # but ANY other version -- the structured "v2", a future version, or a typo ("V2", "v3", " v2") -- must
        # declare the structured provenance. A basis claiming a structured version must not pass with none.
        if self.basis_version not in (None, "v1"):
            return all(getattr(self, name) is not None for name in _V2_STRUCTURED_FIELDS)
        return True

    def content_hash(self) -> str:
        """A stable sha256 over the DECLARED basis fields -- a single provenance stamp for strict reportability
        (two economically-different bases hash differently; an undeclared field never affects the hash)."""
        payload = json.dumps(self.declared(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def invalid_weight_semantics(self) -> bool:
        """True iff a weight_semantics is declared but is NOT a recognized value (a typo / unresolved string
        reaching a reportable artifact). A None (undeclared) value is not "invalid" here."""
        return self.weight_semantics is not None and self.weight_semantics not in ALLOWED_ACTION_RETURN_WEIGHT_SEMANTICS

    def validation_errors(self) -> list[str]:
        """VALUE-level problems with the DECLARED fields, independent of completeness: a non-numeric /
        non-finite clip bound, or clip_min > clip_max. (Weight-semantics validity is reported separately via
        invalid_weight_semantics / the agreement check; formula and fill_convention are free-form descriptive
        strings, not validated against a closed allow-list.) An undeclared (None) field contributes no error,
        so this stays default-preserving for a partially-declared legacy basis while still catching a CORRUPT
        declared value -- 'complete' must not be mistaken for 'valid'."""
        errors: list[str] = []
        for name in ("clip_min", "clip_max"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"non_numeric_{name}:{value!r}")
            elif not math.isfinite(value):
                errors.append(f"non_finite_{name}:{value!r}")
        cmin, cmax = self.clip_min, self.clip_max
        if (
            isinstance(cmin, (int, float)) and not isinstance(cmin, bool) and math.isfinite(cmin)
            and isinstance(cmax, (int, float)) and not isinstance(cmax, bool) and math.isfinite(cmax)
            and cmin > cmax
        ):
            errors.append(f"clip_min_exceeds_clip_max:{cmin!r}>{cmax!r}")
        latency = self.execution_latency_ms
        if latency is not None:
            if not isinstance(latency, int) or isinstance(latency, bool):
                errors.append(f"non_integer_execution_latency_ms:{latency!r}")
            elif latency < 0:
                errors.append(f"negative_execution_latency_ms:{latency!r}")
        if self.basis_version is not None and self.basis_version not in _KNOWN_BASIS_VERSIONS:
            # An unrecognized schema version (typo / future) is fail-closed: we cannot know its requirements,
            # so it is invalid rather than silently treated as the laxer v1.
            errors.append(f"unrecognized_basis_version:{self.basis_version!r}")
        # Declared string fields must be non-blank: a DECLARED "" passes the None-based completeness check yet is
        # a degenerate value (formula / fill_convention / etc. are free-form, but "" is never a valid value).
        # weight_semantics and basis_version have their own recognized-value checks above.
        for name in ("formula", "semantics_version", "fill_convention",
                     "entry_fill_rule", "exit_fill_rule", "source_bar_interval", "price_source"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"blank_or_non_string_{name}:{value!r}")
        return errors

    def disagreements_with(self, other: "ReturnBasis") -> list[str]:
        """Fields that BOTH self and other declare (non-None) but with different values -- a genuine basis
        contradiction. Fields declared by only one side are not contradictions (nothing to compare)."""
        out: list[str] = []
        mine, theirs = self.declared(), other.declared()
        for name in _RETURN_BASIS_FIELD_KEYS:
            if name in mine and name in theirs and _basis_value_differs(mine[name], theirs[name]):
                out.append(name)
        return out


def return_basis_agreement_errors(eval_basis: ReturnBasis, declared_basis: ReturnBasis) -> list[str]:
    """Fail-closed agreement check for a reportable result. Returns reasons (empty == agree) when EITHER side
    declares an invalid weight semantics, OR the two bases CONTRADICT on a jointly-declared field. It is
    default-preserving: a basis that declares nothing (or only one side declares a field) yields no error --
    the check fires only on a real contradiction or an invalid declared value, never merely on absence."""
    errors: list[str] = []
    for basis, label in ((eval_basis, "eval"), (declared_basis, "dataset_manifest")):
        if basis.invalid_weight_semantics():
            errors.append(f"return_basis_invalid_weight_semantics[{label}]:{basis.weight_semantics!r}")
    for name in eval_basis.disagreements_with(declared_basis):
        errors.append(
            f"return_basis_disagreement:{name}"
            f"(eval={eval_basis.declared().get(name)!r},dataset_manifest={declared_basis.declared().get(name)!r})"
        )
    return errors


def validate_return_basis_surfaces(surfaces: dict[str, Any]) -> list[str]:
    """Cross-ARTIFACT agreement AND per-surface validity for the return basis as persisted to every surface of
    one built dataset (the ``.pt`` payload, ``dataset_manifest.json``, ``metadata.json``, ...). Returns reasons
    (empty == every declaring surface agrees, is value-valid, and is self-consistent with its persisted hash).

    Each payload is read with ReturnBasis.from_mapping (nested-``action_return_basis``-aware), so flat
    (.pt/manifest) and nested (metadata) surfaces are handled uniformly. Surfaces that declare NO basis (an empty
    ReturnBasis -- a dict that simply omits the keys, or a non-mapping value) are skipped: an absent basis is not
    a disagreement. The checks, all order-independent and fail-closed only on a real problem:

    * AGREEMENT -- FULL pairwise across declaring surfaces (not star-vs-one-reference), so a contradiction on a
      field that only two NON-reference surfaces declare is still caught the moment a third surface is compared.
    * VALIDITY -- each declaring surface must declare a VALID basis (recognized weight semantics + no corrupt
      value); a uniformly-corrupt-but-agreeing basis is still flagged, and on the metadata surface (not a
      DatasetManifest) this is the only basis validation it gets.
    * HASH SELF-CONSISTENCY -- a surface's persisted ``return_basis_content_hash`` must match its OWN declared
      basis (a stale / hand-edited hash tripwire).

    In a clean build all surfaces are written from ONE valid basis, so any reason here means a hand-edit, a stale
    or partial artifact, a cross-build mix, or a corrupt value (a tamper / divergence tripwire)."""
    bases = {label: ReturnBasis.from_mapping(payload) for label, payload in surfaces.items()}
    # Sorted so the verdict (and the labels in each reason) are deterministic regardless of dict insertion order.
    declaring = sorted(
        ((label, basis) for label, basis in bases.items() if basis.declared()),
        key=lambda item: item[0],
    )
    reasons: list[str] = []
    # AGREEMENT: full pairwise (not star) -- independent of which surface is the sparsest.
    for (label_a, basis_a), (label_b, basis_b) in combinations(declaring, 2):
        diffs = basis_a.disagreements_with(basis_b)
        if diffs:
            reasons.append(f"return_basis_surface_disagreement:{label_a}!={label_b}:{sorted(diffs)}")
    # VALIDITY: a declaring surface must declare a value-valid basis with recognized weight semantics.
    for label, basis in declaring:
        errors = basis.validation_errors()
        if basis.invalid_weight_semantics():
            errors = [*errors, f"invalid_weight_semantics:{basis.weight_semantics!r}"]
        if errors:
            reasons.append(f"return_basis_surface_invalid:{label}:{sorted(errors)}")
    # HASH SELF-CONSISTENCY: each surface's persisted hash must match its own declared basis.
    declaring_basis = dict(declaring)
    for label, payload in surfaces.items():
        if label not in declaring_basis:
            continue
        stored = payload.get("return_basis_content_hash") if hasattr(payload, "get") else None
        if stored is not None and stored != declaring_basis[label].content_hash():
            reasons.append(f"return_basis_content_hash_mismatch:{label}")
    return reasons
