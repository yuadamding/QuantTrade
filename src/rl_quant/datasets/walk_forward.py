"""Decision-level purged and embargoed walk-forward folds.

``rl_quant.protocol.partition`` owns validation and splitting of on-disk
partition windows.  This module operates one layer later, after data has been
ordered into decision positions.  Keeping the two concerns separate avoids
reimplementing partition-label chronology while providing the finer-grained
folds needed by offline and walk-forward learning.

Integer position is the canonical identity of a decision.  Decision dates are
metadata and may repeat (for example, several intraday decisions can share a
session date); they must only be non-decreasing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib


def _require_integer(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")


def _require_decision_axis_id(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "decision_axis_id must be a non-empty immutable dataset snapshot identifier "
            "without surrounding whitespace."
        )


@dataclass(frozen=True, order=True)
class DecisionPosition:
    """One canonical integer position and its point-in-time decision date.

    Dates may repeat.  ``position`` is what disambiguates multiple decisions
    made on the same date and is what all split boundaries use.
    """

    position: int
    decision_date: date

    def __post_init__(self) -> None:
        _require_integer("position", self.position, minimum=0)
        if type(self.decision_date) is not date:
            raise TypeError("decision_date must be a datetime.date (not datetime).")


@dataclass(frozen=True)
class DecisionBlock:
    """An immutable contiguous half-open block of decision positions."""

    start_position: int
    stop_position: int
    decisions: tuple[DecisionPosition, ...]

    def __post_init__(self) -> None:
        _require_integer("start_position", self.start_position, minimum=0)
        _require_integer("stop_position", self.stop_position, minimum=0)
        if self.stop_position < self.start_position:
            raise ValueError("DecisionBlock stop_position must be >= start_position.")
        if not isinstance(self.decisions, tuple):
            raise TypeError("DecisionBlock decisions must be an immutable tuple.")
        if any(not isinstance(item, DecisionPosition) for item in self.decisions):
            raise TypeError("DecisionBlock decisions must contain only DecisionPosition values.")
        expected = tuple(range(self.start_position, self.stop_position))
        actual = tuple(item.position for item in self.decisions)
        if actual != expected:
            raise ValueError(
                "DecisionBlock decisions must exactly cover its contiguous half-open position range."
            )
        dates = self.dates
        if any(later < earlier for earlier, later in zip(dates, dates[1:])):
            raise ValueError("DecisionBlock dates must be chronological and non-decreasing.")

    @property
    def size(self) -> int:
        return self.stop_position - self.start_position

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(item.position for item in self.decisions)

    @property
    def dates(self) -> tuple[date, ...]:
        return tuple(item.decision_date for item in self.decisions)

    @property
    def first_date(self) -> date | None:
        return self.decisions[0].decision_date if self.decisions else None

    @property
    def last_date(self) -> date | None:
        return self.decisions[-1].decision_date if self.decisions else None


@dataclass(frozen=True)
class WalkForwardConfig:
    """Explicit decision-count geometry for purged walk-forward folds.

    ``label_horizon`` is the *effective* maximum future decision offset touched
    by any target, auxiliary label, selection statistic, or reported outcome.
    It must already include execution/settlement delay (for example, execution
    delay + return horizon), not merely the nominal return horizon.  Both
    train/validation and validation/test gaps must therefore contain at least
    that many decisions.  ``max_train_size=None`` produces expanding training;
    a positive cap produces a rolling window after the initial fold.

    ``decision_axis_id`` must identify the exact point-in-time dataset snapshot,
    not merely its calendar.  It must change when constituents, data revisions,
    or upstream feature inputs change; a content-addressed ID is preferred.
    """

    decision_axis_id: str
    initial_train_size: int
    validation_size: int
    test_size: int
    label_horizon: int
    purge_size: int
    embargo_size: int
    max_train_size: int | None = None
    fold_count: int | None = None

    def __post_init__(self) -> None:
        _require_decision_axis_id(self.decision_axis_id)
        _require_integer("initial_train_size", self.initial_train_size, minimum=1)
        _require_integer("validation_size", self.validation_size, minimum=1)
        _require_integer("test_size", self.test_size, minimum=1)
        _require_integer("label_horizon", self.label_horizon, minimum=0)
        _require_integer("purge_size", self.purge_size, minimum=0)
        _require_integer("embargo_size", self.embargo_size, minimum=0)
        if self.purge_size < self.label_horizon:
            raise ValueError(
                "purge_size must be >= label_horizon so labels cannot cross a split boundary."
            )
        if self.max_train_size is not None:
            _require_integer("max_train_size", self.max_train_size, minimum=1)
            if self.max_train_size < self.initial_train_size:
                raise ValueError("max_train_size cannot be smaller than initial_train_size.")
        if self.fold_count is not None:
            _require_integer("fold_count", self.fold_count, minimum=1)

    @property
    def minimum_decision_count(self) -> int:
        """Decisions needed for one evaluable fold, excluding trailing embargo.

        A final fold needs its complete test-label support, but no full embargo
        is required when no later fold reuses those observations.
        """

        return (
            self.initial_train_size
            + self.purge_size
            + self.validation_size
            + self.purge_size
            + self.test_size
            + self.label_horizon
        )


@dataclass(frozen=True, order=True)
class WalkForwardFoldIdentity:
    """Stable immutable identity derived from geometry and its decision axis."""

    decision_axis_id: str
    ordinal: int
    train_start_position: int
    train_stop_position: int
    validation_start_position: int
    validation_stop_position: int
    test_start_position: int
    test_stop_position: int
    train_first_date: date
    train_last_date: date
    validation_first_date: date
    validation_last_date: date
    test_first_date: date
    test_last_date: date
    label_horizon: int
    purge_size: int
    embargo_size: int
    decision_digest: str

    def __post_init__(self) -> None:
        _require_decision_axis_id(self.decision_axis_id)
        _require_integer("ordinal", self.ordinal, minimum=0)
        for name in (
            "train_start_position",
            "train_stop_position",
            "validation_start_position",
            "validation_stop_position",
            "test_start_position",
            "test_stop_position",
            "label_horizon",
            "purge_size",
            "embargo_size",
        ):
            _require_integer(name, getattr(self, name), minimum=0)
        for name in (
            "train_first_date",
            "train_last_date",
            "validation_first_date",
            "validation_last_date",
            "test_first_date",
            "test_last_date",
        ):
            if type(getattr(self, name)) is not date:
                raise TypeError(f"{name} must be a datetime.date.")
        if (
            not isinstance(self.decision_digest, str)
            or len(self.decision_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.decision_digest)
        ):
            raise ValueError("decision_digest must be a lowercase SHA-256 hex digest.")
        if not self.train_start_position < self.train_stop_position:
            raise ValueError("Fold identity training positions must be non-empty and ordered.")
        if not self.validation_start_position < self.validation_stop_position:
            raise ValueError("Fold identity validation positions must be non-empty and ordered.")
        if not self.test_start_position < self.test_stop_position:
            raise ValueError("Fold identity test positions must be non-empty and ordered.")
        if self.validation_start_position - self.train_stop_position != self.purge_size:
            raise ValueError("Fold identity train/validation positions do not match purge_size.")
        if self.test_start_position - self.validation_stop_position != self.purge_size:
            raise ValueError("Fold identity validation/test positions do not match purge_size.")
        for first_name, last_name in (
            ("train_first_date", "train_last_date"),
            ("validation_first_date", "validation_last_date"),
            ("test_first_date", "test_last_date"),
        ):
            if getattr(self, last_name) < getattr(self, first_name):
                raise ValueError(f"{first_name}/{last_name} must be chronological.")
        if self.validation_first_date < self.train_last_date:
            raise ValueError("Fold identity validation dates cannot precede training dates.")
        if self.test_first_date < self.validation_last_date:
            raise ValueError("Fold identity test dates cannot precede validation dates.")

    @property
    def value(self) -> str:
        """Human-readable stable identity with half-open position ranges."""

        return (
            f"wf-{self.ordinal:04d}"
            f"-tr-{self.train_start_position:08d}-{self.train_stop_position:08d}"
            f"-va-{self.validation_start_position:08d}-{self.validation_stop_position:08d}"
            f"-te-{self.test_start_position:08d}-{self.test_stop_position:08d}"
            f"-h{self.label_horizon}-p{self.purge_size}-e{self.embargo_size}"
            f"-{self.decision_digest}"
        )


def _decision_digest(
    decision_axis_id: str,
    blocks: tuple[tuple[str, DecisionBlock], ...],
) -> str:
    digest = hashlib.sha256()
    axis_id_bytes = decision_axis_id.encode("utf-8")
    digest.update(len(axis_id_bytes).to_bytes(8, byteorder="big"))
    digest.update(axis_id_bytes)
    for name, block in blocks:
        digest.update(f"{name}:{block.start_position}:{block.stop_position}\n".encode())
        for decision in block.decisions:
            digest.update(f"{decision.position}:{decision.decision_date.isoformat()}\n".encode())
    return digest.hexdigest()


def _fold_identity(
    ordinal: int,
    config: WalkForwardConfig,
    train: DecisionBlock,
    train_validation_purge: DecisionBlock,
    validation: DecisionBlock,
    validation_test_purge: DecisionBlock,
    test: DecisionBlock,
    test_label_tail: DecisionBlock,
) -> WalkForwardFoldIdentity:
    if (
        train.first_date is None
        or train.last_date is None
        or validation.first_date is None
        or validation.last_date is None
        or test.first_date is None
        or test.last_date is None
    ):
        raise ValueError("Train, validation, and test blocks must be non-empty.")
    return WalkForwardFoldIdentity(
        decision_axis_id=config.decision_axis_id,
        ordinal=ordinal,
        train_start_position=train.start_position,
        train_stop_position=train.stop_position,
        validation_start_position=validation.start_position,
        validation_stop_position=validation.stop_position,
        test_start_position=test.start_position,
        test_stop_position=test.stop_position,
        train_first_date=train.first_date,
        train_last_date=train.last_date,
        validation_first_date=validation.first_date,
        validation_last_date=validation.last_date,
        test_first_date=test.first_date,
        test_last_date=test.last_date,
        label_horizon=config.label_horizon,
        purge_size=config.purge_size,
        embargo_size=config.embargo_size,
        decision_digest=_decision_digest(
            config.decision_axis_id,
            (
                ("train", train),
                ("train_validation_purge", train_validation_purge),
                ("validation", validation),
                ("validation_test_purge", validation_test_purge),
                ("test", test),
                ("test_label_tail", test_label_tail),
            )
        ),
    )


@dataclass(frozen=True)
class WalkForwardFold:
    """One evaluable purged fold, including label-support and observed embargo."""

    identity: WalkForwardFoldIdentity
    config: WalkForwardConfig
    train: DecisionBlock
    train_validation_purge: DecisionBlock
    validation: DecisionBlock
    validation_test_purge: DecisionBlock
    test: DecisionBlock
    test_label_tail: DecisionBlock
    embargo: DecisionBlock

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WalkForwardFoldIdentity):
            raise TypeError("identity must be a WalkForwardFoldIdentity.")
        if not isinstance(self.config, WalkForwardConfig):
            raise TypeError("config must be a WalkForwardConfig.")
        chain = (
            self.train,
            self.train_validation_purge,
            self.validation,
            self.validation_test_purge,
            self.test,
            self.test_label_tail,
            self.embargo,
        )
        if any(not isinstance(block, DecisionBlock) for block in chain):
            raise TypeError("Walk-forward fold blocks must be DecisionBlock values.")
        for left, right in zip(chain, chain[1:]):
            if left.stop_position != right.start_position:
                raise ValueError("Walk-forward fold blocks must be contiguous and ordered.")
        dates = tuple(
            decision.decision_date
            for block in chain
            for decision in block.decisions
        )
        if any(later < earlier for earlier, later in zip(dates, dates[1:])):
            raise ValueError("Walk-forward fold dates must be chronological across block boundaries.")
        expected_required_sizes = (
            self.config.purge_size,
            self.config.validation_size,
            self.config.purge_size,
            self.config.test_size,
            self.config.label_horizon,
        )
        actual_required_sizes = tuple(block.size for block in chain[1:-1])
        if actual_required_sizes != expected_required_sizes:
            raise ValueError(
                "Walk-forward fold required block sizes "
                f"{actual_required_sizes} do not match config {expected_required_sizes}."
            )
        if self.embargo.size > self.config.embargo_size:
            raise ValueError("Observed trailing embargo cannot exceed configured embargo_size.")
        if self.train.size < self.config.initial_train_size:
            raise ValueError("Every fold must retain at least initial_train_size training decisions.")
        if self.config.max_train_size is not None and self.train.size > self.config.max_train_size:
            raise ValueError("Fold training size exceeds max_train_size.")
        expected_identity = _fold_identity(
            self.identity.ordinal,
            self.config,
            self.train,
            self.train_validation_purge,
            self.validation,
            self.validation_test_purge,
            self.test,
            self.test_label_tail,
        )
        if self.identity != expected_identity:
            raise ValueError("Walk-forward fold identity does not match its decision blocks/config.")

    @property
    def fold_id(self) -> str:
        return self.identity.value

    @property
    def reuse_not_before_position(self) -> int:
        """Planned cutoff before which this fold's observations cannot be reused."""

        return self.test_label_tail.stop_position + self.config.embargo_size

    @property
    def embargo_complete(self) -> bool:
        """Whether the available axis contains the full configured trailing embargo."""

        return self.embargo.size == self.config.embargo_size


def _normalize_decision_dates(values: Sequence[date | str]) -> tuple[DecisionPosition, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("decision_dates must be a deterministic sequence of date objects or YYYY-MM-DD strings.")
    if not values:
        raise ValueError("decision_dates cannot be empty.")
    normalized: list[date] = []
    for index, value in enumerate(values):
        if isinstance(value, datetime):
            raise TypeError(
                f"decision_dates[{index}] is datetime; pass an explicit date so truncation is not implicit."
            )
        if type(value) is date:
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(f"decision_dates[{index}] is not a valid ISO date: {value!r}.") from error
            if parsed.isoformat() != value:
                raise ValueError(
                    f"decision_dates[{index}] must use canonical YYYY-MM-DD form; got {value!r}."
                )
        else:
            raise TypeError(f"decision_dates[{index}] must be a date or YYYY-MM-DD string; got {type(value).__name__}.")
        normalized.append(parsed)
    if any(later < earlier for earlier, later in zip(normalized, normalized[1:])):
        raise ValueError("decision_dates must be chronological and non-decreasing.")
    return tuple(DecisionPosition(position, value) for position, value in enumerate(normalized))


def _block(
    decisions: tuple[DecisionPosition, ...],
    start: int,
    stop: int,
) -> DecisionBlock:
    return DecisionBlock(start, stop, decisions[start:stop])


def generate_walk_forward_folds(
    decision_dates: Sequence[date | str],
    config: WalkForwardConfig,
) -> tuple[WalkForwardFold, ...]:
    """Generate maximal or exact-count leak-resistant chronological folds.

    A later fold's training cutoff begins only after the preceding test's label
    tail and embargo have elapsed.  Previous validation/test observations may
    then become historical training data; they are never reused early.  A final
    evaluable fold may have a partial trailing embargo because no later fold can
    reuse it.  An exact ``fold_count`` fails rather than returning a silently
    partial plan.  Equal dates are permitted; integer positions are canonical.
    """

    if not isinstance(config, WalkForwardConfig):
        raise TypeError("config must be a WalkForwardConfig.")
    decisions = _normalize_decision_dates(decision_dates)
    count = len(decisions)
    folds: list[WalkForwardFold] = []
    train_stop = config.initial_train_size

    while config.fold_count is None or len(folds) < config.fold_count:
        train_start = (
            0
            if config.max_train_size is None
            else max(0, train_stop - config.max_train_size)
        )
        train_validation_purge_stop = train_stop + config.purge_size
        validation_stop = train_validation_purge_stop + config.validation_size
        validation_test_purge_stop = validation_stop + config.purge_size
        test_stop = validation_test_purge_stop + config.test_size
        test_label_tail_stop = test_stop + config.label_horizon
        if test_label_tail_stop > count:
            break
        planned_embargo_stop = test_label_tail_stop + config.embargo_size
        observed_embargo_stop = min(planned_embargo_stop, count)

        train = _block(decisions, train_start, train_stop)
        train_validation_purge = _block(decisions, train_stop, train_validation_purge_stop)
        validation = _block(decisions, train_validation_purge_stop, validation_stop)
        validation_test_purge = _block(decisions, validation_stop, validation_test_purge_stop)
        test = _block(decisions, validation_test_purge_stop, test_stop)
        test_label_tail = _block(decisions, test_stop, test_label_tail_stop)
        embargo = _block(decisions, test_label_tail_stop, observed_embargo_stop)

        ordinal = len(folds)
        identity = _fold_identity(
            ordinal,
            config,
            train,
            train_validation_purge,
            validation,
            validation_test_purge,
            test,
            test_label_tail,
        )
        folds.append(
            WalkForwardFold(
                identity=identity,
                config=config,
                train=train,
                train_validation_purge=train_validation_purge,
                validation=validation,
                validation_test_purge=validation_test_purge,
                test=test,
                test_label_tail=test_label_tail,
                embargo=embargo,
            )
        )
        if not folds[-1].embargo_complete:
            break
        train_stop = planned_embargo_stop

    if not folds:
        raise ValueError(
            f"need at least {config.minimum_decision_count} chronological decisions for one evaluable fold; "
            f"got {count}."
        )
    if config.fold_count is not None and len(folds) != config.fold_count:
        raise ValueError(
            f"requested {config.fold_count} complete folds but only {len(folds)} fit in {count} decisions."
        )
    return tuple(folds)


__all__ = [
    "DecisionBlock",
    "DecisionPosition",
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardFoldIdentity",
    "generate_walk_forward_folds",
]
