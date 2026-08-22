"""Explicit canary-qualified Massive trade-correction semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_CORRECTION_AUTHORITY_SCHEMA = "rl-quant.massive-correction-authority-v1"
MassiveCorrectionKind = Literal[
    "new-trade", "replacement", "cancellation", "late-report"
]

class MassiveCorrectionError(ValueError):
    """Correction semantics were not explicitly qualified."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveCorrectionError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MassiveCorrectionRule:
    correction_code: int
    semantic_kind: MassiveCorrectionKind
    canary_receipt_sha256: str

    def validate(self) -> None:
        if (
            isinstance(self.correction_code, bool)
            or not isinstance(self.correction_code, int)
            or self.correction_code < 0
        ):
            raise MassiveCorrectionError("correction code must be nonnegative")
        if self.semantic_kind not in {
            "new-trade",
            "replacement",
            "cancellation",
            "late-report",
        }:
            raise MassiveCorrectionError("correction semantic kind is unsupported")
        _digest("correction canary receipt", self.canary_receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveCorrectionAuthority:
    rules: tuple[MassiveCorrectionRule, ...]
    canary_receipt_sha256: str
    unknown_correction_invalidates_symbol_day: bool
    receipt_sha256: str
    schema: str = MASSIVE_CORRECTION_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rules": [asdict(rule) for rule in self.rules],
            "canary_receipt_sha256": self.canary_receipt_sha256,
            "unknown_correction_invalidates_symbol_day": self.unknown_correction_invalidates_symbol_day,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_CORRECTION_AUTHORITY_SCHEMA:
            raise MassiveCorrectionError("correction authority schema drifted")
        if not self.rules:
            raise MassiveCorrectionError("correction authority has no rules")
        codes = tuple(rule.correction_code for rule in self.rules)
        if codes != tuple(sorted(set(codes))):
            raise MassiveCorrectionError("correction rules must be sorted and unique")
        for rule in self.rules:
            rule.validate()
            if rule.canary_receipt_sha256 != self.canary_receipt_sha256:
                raise MassiveCorrectionError("correction canary identities differ")
        if self.unknown_correction_invalidates_symbol_day is not True:
            raise MassiveCorrectionError("unknown correction states must fail closed")
        _digest("correction canary receipt", self.canary_receipt_sha256)
        _digest("correction authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveCorrectionError("correction authority receipt differs")

    def resolve(self, correction_code: int) -> MassiveCorrectionKind:
        self.validate()
        for rule in self.rules:
            if rule.correction_code == correction_code:
                return rule.semantic_kind
        raise MassiveCorrectionError(
            f"unqualified Massive correction code: {correction_code}"
        )


def build_massive_correction_authority(
    rules: Sequence[tuple[int, MassiveCorrectionKind]],
    *,
    canary_receipt_sha256: str,
) -> MassiveCorrectionAuthority:
    """Build only from empirically qualified delayed/final canary evidence."""

    canary = _digest("correction canary receipt", canary_receipt_sha256)
    rows = tuple(
        sorted(
            (
                MassiveCorrectionRule(code, kind, canary)
                for code, kind in rules
            ),
            key=lambda row: row.correction_code,
        )
    )
    body = {
        "schema": MASSIVE_CORRECTION_AUTHORITY_SCHEMA,
        "rules": [asdict(row) for row in rows],
        "canary_receipt_sha256": canary,
        "unknown_correction_invalidates_symbol_day": True,
    }
    authority = MassiveCorrectionAuthority(
        rules=rows,
        canary_receipt_sha256=canary,
        unknown_correction_invalidates_symbol_day=True,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_CORRECTION_AUTHORITY_SCHEMA",
    "MassiveCorrectionAuthority",
    "MassiveCorrectionError",
    "MassiveCorrectionKind",
    "MassiveCorrectionRule",
    "build_massive_correction_authority",
]
