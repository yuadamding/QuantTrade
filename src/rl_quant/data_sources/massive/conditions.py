"""Content-addressed Massive trade-condition semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_CONDITION_AUTHORITY_SCHEMA = "rl-quant.massive-condition-authority-v1"


class MassiveConditionError(ValueError):
    """Trade-condition semantics are missing or ambiguous."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveConditionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MassiveConditionError(f"{name} must be an integer")
    try:
        observed = int(value)
    except ValueError as exc:
        raise MassiveConditionError(f"{name} must be an integer") from exc
    if observed < 0:
        raise MassiveConditionError(f"{name} must be nonnegative")
    return observed


def _string_sequence(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MassiveConditionError(f"{name} must be a sequence")
    return tuple(str(item) for item in value)


@dataclass(frozen=True, slots=True)
class MassiveTradeConditionRule:
    condition_id: int
    name: str
    data_types: tuple[str, ...]
    updates_high_low: bool
    updates_open_close: bool
    updates_volume: bool
    source_receipt_sha256: str

    def validate(self) -> None:
        if (
            isinstance(self.condition_id, bool)
            or not isinstance(self.condition_id, int)
            or self.condition_id < 0
        ):
            raise MassiveConditionError("condition ID must be nonnegative")
        if not self.name or self.name != self.name.strip():
            raise MassiveConditionError("condition name must be canonical")
        if (
            not self.data_types
            or tuple(sorted(set(self.data_types))) != self.data_types
            or "trade" not in self.data_types
        ):
            raise MassiveConditionError(
                "condition data types must be sorted, unique, and include trade"
            )
        if any(
            not isinstance(value, bool)
            for value in (
                self.updates_high_low,
                self.updates_open_close,
                self.updates_volume,
            )
        ):
            raise MassiveConditionError("condition update rules must be Boolean")
        _digest("condition source receipt", self.source_receipt_sha256)


@dataclass(frozen=True, slots=True)
class MassiveConditionAuthority:
    rules: tuple[MassiveTradeConditionRule, ...]
    source_object_receipt_sha256: str
    unknown_condition_invalidates_symbol_day: bool
    receipt_sha256: str
    schema: str = MASSIVE_CONDITION_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rules": [asdict(rule) for rule in self.rules],
            "source_object_receipt_sha256": self.source_object_receipt_sha256,
            "unknown_condition_invalidates_symbol_day": self.unknown_condition_invalidates_symbol_day,
        }

    def validate(self) -> None:
        if self.schema != MASSIVE_CONDITION_AUTHORITY_SCHEMA:
            raise MassiveConditionError("condition authority schema drifted")
        if not self.rules:
            raise MassiveConditionError("condition authority has no trade rules")
        ids = tuple(rule.condition_id for rule in self.rules)
        if ids != tuple(sorted(set(ids))):
            raise MassiveConditionError("condition rules must be sorted and unique")
        for rule in self.rules:
            rule.validate()
            if rule.source_receipt_sha256 != self.source_object_receipt_sha256:
                raise MassiveConditionError("condition source identities differ")
        _digest("condition source object receipt", self.source_object_receipt_sha256)
        if self.unknown_condition_invalidates_symbol_day is not True:
            raise MassiveConditionError("unknown conditions must fail closed")
        _digest("condition authority receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveConditionError("condition authority receipt differs")

    def resolve(self, condition_ids: Sequence[int]) -> tuple[bool, bool]:
        """Return price-forming and volume-forming flags for one trade."""

        self.validate()
        by_id = {rule.condition_id: rule for rule in self.rules}
        normalized = tuple(sorted(set(condition_ids)))
        if len(normalized) != len(tuple(condition_ids)):
            raise MassiveConditionError("trade condition IDs contain duplicates")
        unknown = tuple(value for value in normalized if value not in by_id)
        if unknown:
            raise MassiveConditionError(f"unknown trade conditions: {unknown}")
        if not normalized:
            return True, True
        rules = tuple(by_id[value] for value in normalized)
        price_forming = all(
            rule.updates_high_low or rule.updates_open_close for rule in rules
        )
        volume_forming = all(rule.updates_volume for rule in rules)
        return price_forming, volume_forming


def build_massive_condition_authority(
    records: Sequence[Mapping[str, object]], *, source_object_receipt_sha256: str
) -> MassiveConditionAuthority:
    """Build a frozen condition map from the Massive reference response."""

    source = _digest("condition source object receipt", source_object_receipt_sha256)
    rules: list[MassiveTradeConditionRule] = []
    for record in records:
        data_types = tuple(
            sorted(set(_string_sequence("condition data types", record.get("data_types"))))
        )
        if "trade" not in data_types:
            continue
        update_rules = record.get("update_rules")
        if not isinstance(update_rules, Mapping):
            raise MassiveConditionError("condition update_rules are absent")
        consolidated = update_rules.get("consolidated")
        if not isinstance(consolidated, Mapping):
            raise MassiveConditionError("consolidated condition rules are absent")
        rules.append(
            MassiveTradeConditionRule(
                condition_id=_integer("condition ID", record["id"]),
                name=str(record["name"]),
                data_types=data_types,
                updates_high_low=consolidated.get("updates_high_low") is True,
                updates_open_close=consolidated.get("updates_open_close") is True,
                updates_volume=consolidated.get("updates_volume") is True,
                source_receipt_sha256=source,
            )
        )
    ordered = tuple(sorted(rules, key=lambda row: row.condition_id))
    body = {
        "schema": MASSIVE_CONDITION_AUTHORITY_SCHEMA,
        "rules": [asdict(rule) for rule in ordered],
        "source_object_receipt_sha256": source,
        "unknown_condition_invalidates_symbol_day": True,
    }
    authority = MassiveConditionAuthority(
        rules=ordered,
        source_object_receipt_sha256=source,
        unknown_condition_invalidates_symbol_day=True,
        receipt_sha256=semantic_sha256(body),
    )
    authority.validate()
    return authority


__all__ = [
    "MASSIVE_CONDITION_AUTHORITY_SCHEMA",
    "MassiveConditionAuthority",
    "MassiveConditionError",
    "MassiveTradeConditionRule",
    "build_massive_condition_authority",
]
