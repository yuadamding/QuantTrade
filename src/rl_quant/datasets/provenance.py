"""Point-in-time dataset provenance and dynamic-universe support.

Reportability is deliberately fail-closed: a static symbol list is useful for
development, but it is not evidence that the list was knowable at the start of
the sample. A reportable dataset records the selection timestamp/method and,
for a changing universe, an event-time membership table.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


@dataclass(frozen=True)
class DatasetProvenance:
    reportable: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    universe_selection_date: str | None
    coverage_start: str | None
    membership_mode: str | None
    manifest_hash: str

    def require_reportable(self) -> None:
        if not self.reportable:
            raise ValueError("dataset is not point-in-time reportable: " + "; ".join(self.errors))


def _date(value: object | None) -> dt.date | None:
    if not value:
        return None
    normalized = str(value).strip()
    try:
        return dt.date.fromisoformat(normalized)
    except ValueError:
        try:
            return dt.datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
        except ValueError:
            return None


def _coverage_start(manifest: dict) -> str | None:
    direct = manifest.get("first_timestamp") or manifest.get("coverage_start")
    if direct:
        return str(direct)
    first_window = manifest.get("first_window")
    if first_window:
        return str(first_window).split("_to_")[0]
    coverage = manifest.get("coverage")
    if isinstance(coverage, str) and "->" in coverage:
        return coverage.split("->", 1)[0].strip()
    return None


def _bar_file_start(path: Path) -> dt.date:
    """Read the earliest actual exchange date, preferring Parquet statistics."""

    parquet = pq.ParquetFile(path)
    try:
        column_index = parquet.schema_arrow.names.index("date_exchange")
    except ValueError as exc:
        raise ValueError(f"{path} is missing date_exchange") from exc
    candidates: list[dt.date] = []
    for row_group in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(row_group).column(column_index).statistics
        if statistics is None or not statistics.has_min_max:
            continue
        value = statistics.min
        if isinstance(value, bytes):
            value = value.decode()
        parsed = _date(str(value))
        if parsed is not None:
            candidates.append(parsed)
    if not candidates:
        values = pq.read_table(path, columns=["date_exchange"]).column("date_exchange").to_pylist()
        for value in values:
            parsed = _date(None if value is None else str(value))
            if parsed is not None:
                candidates.append(parsed)
    if not candidates:
        raise ValueError(f"{path} has no valid date_exchange values")
    return min(candidates)


def _actual_coverage_start(root: Path) -> str | None:
    """Return the earliest bars-content date that can enter training."""

    partitions = root / "partitions"
    if not partitions.exists():
        return None
    starts: list[dt.date] = []
    for partition in partitions.iterdir():
        bars_path = partition / "bars.parquet"
        if not bars_path.exists():
            continue
        starts.append(_bar_file_start(bars_path))
    return None if not starts else min(starts).isoformat()


def _membership_rows(path: Path) -> list[tuple[str, str, bool, int]]:
    required = {"symbol", "effective_date", "is_member", "available_timestamp_ms"}
    schema = pq.read_schema(path)
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"{path} missing membership columns: {sorted(missing)}")
    if not pa.types.is_boolean(schema.field("is_member").type):
        raise ValueError(f"{path} is_member must be a boolean column")
    if not pa.types.is_integer(schema.field("available_timestamp_ms").type):
        raise ValueError(f"{path} available_timestamp_ms must be an integer column")
    if not (
        pa.types.is_string(schema.field("symbol").type)
        or pa.types.is_large_string(schema.field("symbol").type)
    ):
        raise ValueError(f"{path} symbol must be a string column")
    effective_type = schema.field("effective_date").type
    if not (
        pa.types.is_string(effective_type)
        or pa.types.is_large_string(effective_type)
        or pa.types.is_date(effective_type)
    ):
        raise ValueError(f"{path} effective_date must be a string/date column")

    table = pq.read_table(path, columns=sorted(required)).to_pydict()
    rows: list[tuple[str, str, bool, int]] = []
    seen: dict[tuple[str, str, int], bool] = {}
    for row_index, (symbol, effective, member, available) in enumerate(zip(
        table["symbol"],
        table["effective_date"],
        table["is_member"],
        table["available_timestamp_ms"],
    )):
        if symbol is None or effective is None or member is None or available is None:
            raise ValueError(f"{path} membership row {row_index} contains null required values")
        if not str(symbol).strip():
            raise ValueError(f"{path} membership row {row_index} has an empty symbol")
        parsed_effective = _date(str(effective))
        if parsed_effective is None:
            raise ValueError(f"{path} membership row {row_index} has an invalid effective_date")
        if not isinstance(member, bool):
            raise ValueError(f"{path} membership row {row_index} is_member is not boolean")
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or not 0 <= available <= np.iinfo(np.int64).max
        ):
            raise ValueError(f"{path} membership row {row_index} has an invalid availability timestamp")
        normalized = (str(symbol), parsed_effective.isoformat(), member, available)
        key = (normalized[0], normalized[1], normalized[3])
        previous = seen.get(key)
        if previous is not None and previous != member:
            raise ValueError(f"{path} has conflicting duplicate membership event {key}")
        seen[key] = member
        rows.append(normalized)
    if not rows:
        raise ValueError(f"{path} contains no membership events")
    return rows


def _declared_universe(root: str | Path) -> tuple[list[str], dict[str, int]]:
    root = Path(root)
    path = root / "universe.json"
    if not path.exists():
        raise ValueError("universe.json is missing")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ValueError(f"universe.json is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("universe.json must contain an object")
    actions = payload.get("actions")
    cash_index = payload.get("cash_index")
    if (
        not isinstance(actions, list)
        or not actions
        or any(
            not isinstance(action, str) or not action or action != action.strip()
            for action in actions
        )
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("universe.json actions must be non-empty, unique strings")
    if isinstance(cash_index, bool) or not isinstance(cash_index, int) or cash_index != 0:
        raise ValueError("universe.json cash_index must be integer 0")
    if actions[0] != "CASH":
        raise ValueError("universe.json actions[0] must be CASH")
    action_count = payload.get("action_count")
    if action_count is not None and (
        isinstance(action_count, bool)
        or not isinstance(action_count, int)
        or action_count != len(actions)
    ):
        raise ValueError("universe.json action_count does not match actions")
    aliases = payload.get("source_symbol_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(action, str)
        or not isinstance(source, str)
        or not action
        or not source
        or action != action.strip()
        or source != source.strip()
        for action, source in aliases.items()
    ):
        raise ValueError("universe.json source_symbol_aliases must map canonical strings to strings")
    unknown_aliases = sorted(set(aliases) - set(actions[1:]))
    if unknown_aliases:
        raise ValueError(
            "universe.json source_symbol_aliases contains undeclared/non-tradable actions: "
            + ", ".join(unknown_aliases)
        )
    source_to_index: dict[str, int] = {}
    for index, action in enumerate(actions[1:], start=1):
        source = aliases.get(action, action)
        if source in source_to_index:
            raise ValueError(f"universe.json maps multiple non-CASH actions to source symbol {source!r}")
        source_to_index[source] = index
    return actions, source_to_index


def declared_universe_actions(root: str | Path) -> list[str]:
    """Return unique policy action IDs, with synthetic CASH fixed at index 0."""

    actions, _ = _declared_universe(root)
    return actions


def source_symbol_to_action_index(root: str | Path) -> dict[str, int]:
    """Map raw market symbols to non-CASH policy action indices.

    ``source_symbol_aliases`` in ``universe.json`` resolves reserved-name
    collisions without changing raw bars, covariates, news, or membership
    events. Synthetic CASH deliberately has no source-market symbol.
    """

    _, source_to_index = _declared_universe(root)
    return source_to_index


def inspect_dataset_provenance(root: str | Path) -> DatasetProvenance:
    """Inspect manifest claims against the actual partition coverage.

    Universe metadata can be top-level or nested under ``universe``. A
    point-in-time/rolling universe additionally requires the event table used
    by :func:`point_in_time_membership`.
    """
    root = Path(root)
    path = root / "manifest.json"
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return DatasetProvenance(False, ("manifest.json is missing",), (), None, None, None, "")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        return DatasetProvenance(False, (f"manifest.json is invalid: {exc}",), (), None, None, None, digest)
    if not isinstance(manifest, dict):
        return DatasetProvenance(
            False,
            ("manifest.json must contain an object",),
            (),
            None,
            None,
            None,
            digest,
        )

    try:
        _, declared_source_index = _declared_universe(root)
    except (OSError, ValueError) as exc:
        declared_source_index = None
        errors.append(f"invalid universe declaration: {exc}")

    universe_value = manifest.get("universe")
    universe: dict[str, object] = universe_value if isinstance(universe_value, dict) else {}
    selection = manifest.get("universe_selection_date") or universe.get("selection_date")
    mode = manifest.get("membership_mode") or universe.get("membership_mode") or "static"
    method = manifest.get("universe_selection_method") or universe.get("selection_method")
    start = _coverage_start(manifest)
    coverage_error: str | None = None
    try:
        actual_start = _actual_coverage_start(root)
    except (OSError, ValueError) as exc:
        actual_start = None
        coverage_error = str(exc)
        errors.append(f"cannot verify actual bars coverage: {coverage_error}")
    selection_date, start_date = _date(selection), _date(start)
    if selection_date is None:
        errors.append("universe_selection_date is missing or invalid")
    if start_date is None:
        errors.append("coverage start is missing or invalid")
    if actual_start is None and coverage_error is None:
        errors.append("no dated bars partitions are available to verify coverage start")
    elif start_date is not None and start_date != _date(actual_start):
        errors.append(
            f"manifest coverage start {start_date.isoformat()} does not match earliest actual bars date "
            f"{actual_start}"
        )
    if selection_date is not None and start_date is not None and selection_date > start_date:
        errors.append(
            f"universe was selected on {selection_date.isoformat()} after sample start {start_date.isoformat()}"
        )
    if not method:
        errors.append("universe selection method is missing")
    if mode not in {"static", "point_in_time", "rolling"}:
        errors.append(f"unsupported membership_mode {mode!r}")
    membership_path = root / "universe_membership.parquet"
    if membership_path.exists():
        try:
            membership_rows = _membership_rows(membership_path)
        except (OSError, ValueError) as exc:
            membership_rows = None
            errors.append(f"invalid universe membership table: {exc}")
        if membership_rows is not None and declared_source_index is not None:
            declared = set(declared_source_index)
            event_symbols = {symbol for symbol, _, _, _ in membership_rows}
            active_symbols = {symbol for symbol, _, member, _ in membership_rows if member}
            missing = sorted(declared - active_symbols)
            unexpected = sorted(event_symbols - declared)
            if missing:
                errors.append(
                    "universe membership has no positive event for declared actions: "
                    + ", ".join(missing)
                )
            if unexpected:
                errors.append(
                    "universe membership contains undeclared actions: " + ", ".join(unexpected)
                )
        if mode == "static":
            errors.append(
                "membership_mode=static conflicts with universe_membership.parquet used by the dataset builder"
            )
    elif mode in {"point_in_time", "rolling"}:
        errors.append(f"membership_mode={mode} requires universe_membership.parquet")
    if mode == "static":
        warnings.append("static membership can omit later delistings; prefer point_in_time membership events")
    if manifest.get("dataset_reportable") is False:
        declared = manifest.get("reportability_errors", ["manifest marks dataset unreportable"])
        errors.extend(str(item) for item in declared)

    errors = list(dict.fromkeys(errors))
    return DatasetProvenance(
        not errors,
        tuple(errors),
        tuple(warnings),
        str(selection) if selection is not None else None,
        start,
        str(mode),
        digest,
    )


def point_in_time_membership(
    root: str | Path,
    dates: list[str],
    actions: list[str],
    decision_timestamp_ms: np.ndarray,
) -> np.ndarray:
    """Return a ``[day, action]`` membership mask known at decision time.

    The event table schema is ``symbol``, ``effective_date``, ``is_member`` and
    ``available_timestamp_ms``. Events apply only when both their effective
    date and availability timestamp are no later than the decision. Absence of
    a table retains the development-only static universe; the provenance gate
    prevents that fallback from being represented as reportable.
    """
    root = Path(root)
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("actions must be non-empty and unique")
    if not all(isinstance(date, str) and _date(date) is not None for date in dates):
        raise ValueError("membership dates must be valid ISO dates")
    if decision_timestamp_ms.shape != (len(dates),):
        raise ValueError("decision_timestamp_ms must be a vector aligned with dates")
    if not np.issubdtype(decision_timestamp_ms.dtype, np.integer):
        raise ValueError("decision_timestamp_ms must use an integer dtype")
    if bool(np.any(decision_timestamp_ms < 0)) or bool(
        np.any(decision_timestamp_ms > np.iinfo(np.int64).max)
    ):
        raise ValueError("decision_timestamp_ms must lie in nonnegative int64 range")
    if dates != sorted(dates) or bool(np.any(np.diff(decision_timestamp_ms.astype(np.int64)) < 0)):
        raise ValueError("membership decisions must be chronological")
    path = root / "universe_membership.parquet"
    out = np.ones((len(dates), len(actions)), dtype=bool)
    out[:, 0] = True
    if not path.exists():
        return out
    if (root / "universe.json").exists():
        declared_actions, source_action_index = _declared_universe(root)
        if actions != declared_actions:
            raise ValueError("membership actions do not match the ordered universe declaration")
    else:
        source_action_index = {symbol: i for i, symbol in enumerate(actions[1:], start=1)}
    events: dict[int, list[tuple[str, int, bool]]] = {}
    membership_rows = _membership_rows(path)
    positive_indices = {
        source_action_index[symbol]
        for symbol, _, member, _ in membership_rows
        if member and symbol in source_action_index
    }
    missing_actions = [actions[index] for index in range(1, len(actions)) if index not in positive_indices]
    if missing_actions:
        raise ValueError(
            "membership table has no positive event for requested actions: "
            + ", ".join(missing_actions)
        )
    for symbol, effective, member, available in membership_rows:
        ai = source_action_index.get(symbol)
        if ai is None or ai == 0:
            continue
        events.setdefault(ai, []).append((effective, available, member))
    out[:, 1:] = False
    for ai, rows in events.items():
        rows.sort(key=lambda row: (row[1], row[0]))
        known: dict[str, tuple[int, bool]] = {}
        effective_dates: list[str] = []
        cursor = 0
        for di, (date, decision_ms) in enumerate(zip(dates, decision_timestamp_ms)):
            while cursor < len(rows) and rows[cursor][1] <= int(decision_ms):
                effective, available, member = rows[cursor]
                previous = known.get(effective)
                if previous is None:
                    bisect.insort(effective_dates, effective)
                if previous is None or available >= previous[0]:
                    known[effective] = (available, member)
                cursor += 1
            position = bisect.bisect_right(effective_dates, date) - 1
            if position >= 0:
                # Effective date defines the state timeline. Availability only
                # decides when an event becomes knowable; a late-arriving old
                # event cannot block/override a newer effective event already known.
                out[di, ai] = known[effective_dates[position]][1]
    return out
