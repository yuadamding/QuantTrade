"""Canonical action-return basis: the protocol-layer contract for HOW a dataset's action_returns are defined.

A reportable result must declare the FULL basis behind its returns -- the weight semantics (the PR-4 cost
basis), the return formula, clip bounds, semantics version, and fill convention -- so the evaluation's basis can
be checked for AGREEMENT against the dataset's declared basis. This lives in the protocol layer (stdlib-only, no
torch) so the dataset loader, the DatasetManifest, and the reportability/decision validators can all share ONE
definition instead of an evaluation validator reaching into a dataset module for it.

Pure, stdlib only; changes no backtest number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
}


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

    @classmethod
    def from_mapping(cls, payload: Any) -> "ReturnBasis":
        """Build from a mapping (dataset manifest dict) OR any object exposing the ``action_return_*`` attributes
        (a HourFromMinuteDataSplit). Missing keys/attributes resolve to None."""
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

    def declared(self) -> dict[str, Any]:
        """Only the fields that are declared (non-None)."""
        return {name: value for name, value in self.to_dict().items() if value is not None}

    def is_complete(self) -> bool:
        """True iff every field is declared AND the weight semantics is a recognized value."""
        return (
            len(self.declared()) == len(_RETURN_BASIS_FIELD_KEYS)
            and self.weight_semantics in ALLOWED_ACTION_RETURN_WEIGHT_SEMANTICS
        )

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
