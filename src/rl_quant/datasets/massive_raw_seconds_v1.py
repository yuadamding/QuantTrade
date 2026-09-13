"""Lossless REST-second references; no market feature or scaler surface.

Capture integrity is not historical-vintage or security-identity qualification.
This schema is deliberately disjoint from native V5 and the older raw loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from urllib.parse import parse_qs, quote, urlsplit

import torch

SCHEMA = "rl-quant.massive-raw-second-window-v1"
FIELDS = ("open", "high", "low", "close", "volume")
PROVIDER_FIELDS = ("o", "h", "l", "c", "v")


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(body: bytes) -> dict:
    result = json.loads(body, object_pairs_hook=_pairs)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _read(path: Path, expected: str) -> bytes:
    # The deployment owns the parent tree. Do not follow a substituted leaf or
    # accept a mutation during reading, including a same-size rewrite.
    if path.absolute().resolve() != path.absolute():
        raise ValueError("Symlinked source path rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024**2:
            raise ValueError("Invalid/beyond-bound capture file")
        body = stream.read()
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in keys) or sha256(body).hexdigest() != expected:
        raise ValueError("Raw source content changed")
    return body


def _write(path: Path, body: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class RawSecondContract:
    """Fixed input semantics; forbidden options fail, never fall back."""

    availability_assumption: str
    availability_delay_ms: int
    provider: str = "massive"
    source: str = "rest_second_aggregates"
    timespan: str = "second"
    multiplier: int = 1
    adjusted: bool = False
    market_fields: tuple[str, ...] = FIELDS
    resampling: None = None
    raw_input_normalization: None = None
    feature_engineering: bool = False
    forward_fill_market_inputs: bool = False
    covariates: bool = False
    news: bool = False
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        fixed = dict(provider="massive", source="rest_second_aggregates", timespan="second",
                     multiplier=1, adjusted=False, market_fields=FIELDS, resampling=None,
                     raw_input_normalization=None, feature_engineering=False,
                     forward_fill_market_inputs=False, covariates=False, news=False, schema=SCHEMA)
        for key, expected in fixed.items():
            actual = getattr(self, key)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"Forbidden raw-second input setting: {key}")
        _integer(self.availability_delay_ms, "availability_delay_ms")
        if self.availability_assumption not in (
            "historical-finalized-assumed-delay", "developer-delayed-assumption", "captured-receipt-time"
        ):
            raise ValueError("An explicit supported availability assumption is required")
        if self.availability_assumption == "developer-delayed-assumption" and self.availability_delay_ms < 900_000:
            raise ValueError("Developer-faithful delay must be at least 15 minutes")
        if self.availability_assumption == "captured-receipt-time" and self.availability_delay_ms:
            raise ValueError("Receipt-time mode uses actual capture times, not an assumed delay")


@dataclass(frozen=True)
class SecondQuery:
    ticker: str
    start_ms: int
    end_ms: int  # inclusive bar-start timestamp

    def __post_init__(self) -> None:
        if not self.ticker or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for c in self.ticker):
            raise ValueError("Noncanonical ticker; preserve BRK.B")
        _integer(self.start_ms, "start_ms")
        _integer(self.end_ms, "end_ms")
        if self.start_ms % 1000 or self.end_ms % 1000 or self.end_ms < self.start_ms:
            raise ValueError("Query must specify exact chronological one-second intervals")

    @property
    def url(self) -> str:
        return (f"https://api.massive.com/v2/aggs/ticker/{quote(self.ticker)}/range/1/second/"
                f"{self.start_ms}/{self.end_ms}?adjusted=false&sort=asc&limit=50000")


@dataclass(frozen=True)
class CapturedSecondPage:
    request_url: str
    received_at_ms: int
    body: bytes


def _safe_url(url: str, query: SecondQuery) -> None:
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ValueError("Noncanonical pagination URL")
    parsed = urlsplit(url)
    expected = urlsplit(query.url)
    args = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    # Provider cursors may advance the start timestamp in the path. They may
    # not change the ticker, second multiplier, original end, or query scope.
    prefix = expected.path.rsplit("/", 2)[0] + "/"
    bounds = parsed.path[len(prefix):].split("/") if parsed.path.startswith(prefix) else []
    bounded_path = (len(bounds) == 2 and bounds[0].isascii() and bounds[0].isdigit()
                    and str(int(bounds[0])) == bounds[0]
                    and query.start_ms <= int(bounds[0]) <= query.end_ms
                    and bounds[1] == str(query.end_ms))
    if (parsed.scheme != "https" or parsed.netloc != "api.massive.com" or not bounded_path
            or parsed.fragment or not set(args).issubset({"adjusted", "sort", "limit", "cursor"})
            or any(len(v) != 1 or not v[0] for v in args.values())
            or (url != query.url and "cursor" not in args)
            or args.get("adjusted", ["false"]) != ["false"]
            or args.get("sort", ["asc"]) != ["asc"]
            or args.get("limit", ["50000"]) != ["50000"]):
        raise ValueError("Unexpected/credential-bearing pagination URL")


def _rows(query: SecondQuery, pages: tuple[CapturedSecondPage, ...]) -> list[tuple[int, tuple[float, ...], int]]:
    if not pages:
        raise ValueError("Unknown coverage: no completed response")
    expected: str | None = query.url
    result = []
    prior_stamp = -1
    prior_received = -1
    seen = set()
    for page in pages:
        if expected is None or page.request_url != expected or page.request_url in seen:
            raise ValueError("Incomplete, cyclic or mismatched pagination")
        _safe_url(page.request_url, query)
        seen.add(page.request_url)
        _integer(page.received_at_ms, "capture timestamp")
        if page.received_at_ms < prior_received:
            raise ValueError("Capture chronology changed")
        prior_received = page.received_at_ms
        payload = _json(page.body)
        rows = payload.get("results", [])
        if (payload.get("status") != "OK" or payload.get("ticker") != query.ticker
                or payload.get("adjusted") is not False or not isinstance(rows, list)
                or type(payload.get("resultsCount")) is not int or payload["resultsCount"] != len(rows)):
            raise ValueError("Wrong product, adjusted data, or failed response")
        for row in rows:
            if not isinstance(row, dict) or not {"t", *PROVIDER_FIELDS}.issubset(row):
                raise ValueError("Missing raw OHLCV field")
            stamp = _integer(row["t"], "bar timestamp")
            if stamp % 1000 or not query.start_ms <= stamp <= query.end_ms or stamp <= prior_stamp:
                raise ValueError("Duplicate/out-of-order/nonsecond/out-of-query bar")
            values = []
            for key in PROVIDER_FIELDS:
                v = row[key]
                if type(v) not in (int, float) or not math.isfinite(v) or v < 0 or (key != "v" and v == 0):
                    raise ValueError("Invalid raw field; no clipping or repair allowed")
                values.append(float(v))
            o, h, low, c, _ = values
            if not low <= min(o, c) <= max(o, c) <= h:
                raise ValueError("Invalid provider OHLC ordering")
            result.append((stamp, tuple(values), page.received_at_ms))
            prior_stamp = stamp
        expected = payload.get("next_url")
        if expected is None and (len(rows) >= 50_000 or payload.get("queryCount", 0) >= 50_000):
            raise ValueError("Unknown coverage at the query limit; split the request")
        if expected is not None:
            _safe_url(expected, query)
    if expected is not None:
        raise ValueError("Unknown coverage: pagination incomplete")
    return result


@dataclass(frozen=True)
class SecondCaptureRef:
    path: str
    manifest_sha256: str

    def _manifest(self) -> dict:
        root = Path(self.path)
        manifest = _json(_read(root / "capture.json", self.manifest_sha256))
        if set(manifest) != {"schema", "query", "pages", "interpretation"} or manifest["schema"] != SCHEMA:
            raise ValueError("Legacy/engineered source schema rejected")
        if manifest["interpretation"] != "provider-finalized-unadjusted-second-aggregates":
            raise ValueError("Unsupported source interpretation")
        names = {"capture.json", *(f"page-{i:06d}.json" for i in range(len(manifest["pages"])))}
        if {p.name for p in root.iterdir()} != names:
            raise ValueError("Unexpected source artifact (scalers/features are forbidden)")
        return manifest

    def query(self) -> SecondQuery:
        """Hash-bound interval metadata; does not open unrelated page bodies."""
        return SecondQuery(**self._manifest()["query"])

    def load(self) -> tuple[SecondQuery, tuple[CapturedSecondPage, ...]]:
        root = Path(self.path)
        manifest = self._manifest()
        query = SecondQuery(**manifest["query"])
        pages = tuple(CapturedSecondPage(row["request_url"], row["received_at_ms"],
                      _read(root / f"page-{index:06d}.json", row["sha256"]))
                      for index, row in enumerate(manifest["pages"]))
        _rows(query, pages)
        return query, pages


def publish_second_capture(root: Path, query: SecondQuery, pages: tuple[CapturedSecondPage, ...]) -> SecondCaptureRef:
    """Archive exact response bytes, commit last; never repair an existing path.

    This is an offline transport handoff, not entitlement, PIT or V5 authority.
    It cannot issue trading/training authorization, even for a complete query.
    """
    _rows(query, pages)
    root.mkdir(parents=True, exist_ok=False)
    inventory = []
    for index, page in enumerate(pages):
        _write(root / f"page-{index:06d}.json", page.body)
        inventory.append(dict(request_url=page.request_url, received_at_ms=page.received_at_ms,
                              sha256=sha256(page.body).hexdigest()))
    manifest = dict(schema=SCHEMA, query=asdict(query), pages=inventory,
                    interpretation="provider-finalized-unadjusted-second-aggregates")
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _write(root / "capture.json", body)
    return SecondCaptureRef(str(root), sha256(body).hexdigest())


def _is_second_capture(value: object) -> bool:
    if type(value) is SecondCaptureRef:
        return True
    from rl_quant.datasets.massive_raw_second_packed_v1 import PackedSecondCaptureRef
    return type(value) is PackedSecondCaptureRef


def second_capture_ref_from_dict(value: dict) -> SecondCaptureRef:
    """Restore explicit storage representations, never a permissive fallback."""
    if type(value) is dict and set(value) == {"path", "manifest_sha256"}:
        return SecondCaptureRef(**value)
    from rl_quant.datasets.massive_raw_second_packed_v1 import PackedSecondCaptureRef
    return PackedSecondCaptureRef.from_dict(value)


@dataclass(frozen=True)
class SecondPartitionSet:
    """One instrument's immutable source partitions, not a merged provider response.

    Ticker, overlap and physical completeness are verified by the reader. Merely
    constructing this metadata must not open a later experiment role's sources.
    """

    partitions: tuple[SecondCaptureRef, ...]

    def __post_init__(self) -> None:
        if (type(self.partitions) is not tuple or not self.partitions
                or any(not _is_second_capture(p) for p in self.partitions)
                or len({p.manifest_sha256 for p in self.partitions}) != len(self.partitions)):
            raise ValueError("Unique immutable second partitions are required")

    @property
    def identity(self) -> str:
        return digest(dict(schema="rl-quant.second-partition-set-v1",
                           partitions=[p.manifest_sha256 for p in self.partitions]))


def second_partitions(source: SecondCaptureRef | SecondPartitionSet) -> tuple[SecondCaptureRef, ...]:
    if _is_second_capture(source):
        return (source,)
    if type(source) is SecondPartitionSet:
        return source.partitions
    raise ValueError("Only immutable raw second capture partitions are allowed")


@dataclass(frozen=True)
class ResolvedSecondInterval:
    """Lossless observed rows plus verified half-open clock coverage.

    Gaps are UNKNOWN, not zero volume. Coverage receipt times also determine
    when an absent observation may be called known-empty in receipt-time mode.
    """

    rows: tuple
    coverage: tuple[tuple[int, int, int], ...]
    source_hashes: tuple[str, ...]

    def coverage_receipt(self, stamp: int) -> int | None:
        receipts = [receipt for start, end, receipt in self.coverage if start <= stamp < end]
        return min(receipts) if receipts else None


def resolve_second_interval(source: SecondCaptureRef | SecondPartitionSet, start_ms: int, end_ms: int,
                            *, require_complete: bool = True) -> ResolvedSecondInterval:
    """Resolve [start, end) without resampling, filling, scaling or revision guesses.

    Equal overlapping observations coalesce, retaining all source identities and
    the earliest actual receipt. A changed value OR observed/empty disagreement
    is an ambiguous vintage and fails closed. No learned representations cache.
    """
    _integer(start_ms, "interval start")
    _integer(end_ms, "interval end")
    if start_ms % 1000 or end_ms % 1000 or end_ms < start_ms:
        raise ValueError("Expected an aligned half-open second interval")
    population, coverage, hashes = {}, [], []
    ticker = None
    previous = []
    for capture in second_partitions(source):
        # Both storage representations authenticate their query metadata before
        # opening/decompressing prices. Whole-source replay is a separate gate.
        query = capture.query()
        if ticker is not None and ticker != query.ticker:
            raise ValueError("Mixed ticker partitions cannot establish an issue identity")
        ticker = query.ticker
        left, right = max(start_ms, query.start_ms), min(end_ms, query.end_ms + 1000)
        if left >= right:
            continue
        loaded_query, pages = capture.load()
        if loaded_query != query:
            raise ValueError("Source query changed between metadata and page reads")
        rows = {t: (values, receipt) for t, values, receipt in _rows(query, pages) if left <= t < right}
        for old_left, old_right, old_rows in previous:
            overlap_left, overlap_right = max(left, old_left), min(right, old_right)
            if overlap_left < overlap_right:
                old_values = {t: v[0] for t, v in old_rows.items() if overlap_left <= t < overlap_right}
                new_values = {t: v[0] for t, v in rows.items() if overlap_left <= t < overlap_right}
                if old_values != new_values:
                    raise ValueError("Conflicting duplicate/empty second across source partitions")
        for stamp, (values, receipt) in rows.items():
            if stamp in population:
                receipt = min(receipt, population[stamp][1])
            population[stamp] = values, receipt
        previous.append((left, right, rows))
        coverage.append((left, right, pages[-1].received_at_ms))
        hashes.append(capture.manifest_sha256)
    coverage.sort()
    cursor = start_ms
    for left, right, _ in coverage:
        if require_complete and left > cursor:
            raise ValueError("Unknown source coverage between second partitions")
        cursor = max(cursor, right)
    if require_complete and cursor < end_ms:
        raise ValueError("Unknown source coverage after second partitions")
    return ResolvedSecondInterval(tuple((t, *population[t]) for t in sorted(population)), tuple(coverage), tuple(hashes))


@dataclass(frozen=True)
class RawSecondWindowRef:
    captures: tuple[SecondCaptureRef | SecondPartitionSet, ...]
    asset_ids: tuple[str, ...]  # stable issue IDs, not ticker aliases
    start_ms: int
    seconds: int
    decision_ms: int
    contract: RawSecondContract

    def __post_init__(self) -> None:
        if (type(self.captures) is not tuple or type(self.asset_ids) is not tuple
                or not self.asset_ids or len(self.captures) != len(self.asset_ids)
                or len(set(self.asset_ids)) != len(self.asset_ids)
                or any(not isinstance(s, str) or not s or s == "CASH" for s in self.asset_ids)):
            raise ValueError("One partition source per unique stable equity identity is required")
        for source in self.captures:
            second_partitions(source)
        _integer(self.start_ms, "window start")
        _integer(self.seconds, "window seconds", 1)
        _integer(self.decision_ms, "decision timestamp")
        if self.seconds > 23_400 or self.start_ms % 1000 or self.decision_ms <= self.start_ms:
            raise ValueError("Invalid/beyond-bound raw context")

    @property
    def identity(self) -> str:
        return digest(dict(schema=SCHEMA, captures=[c.manifest_sha256 if _is_second_capture(c) else c.identity for c in self.captures],
                           asset_ids=self.asset_ids, start_ms=self.start_ms, seconds=self.seconds,
                           decision_ms=self.decision_ms, contract=asdict(self.contract)))


@dataclass(frozen=True)
class RawSecondObservation:
    raw_ohlcv: torch.Tensor  # [batch, equity, clock second, 5]; float32, no scaler
    observed_mask: torch.Tensor
    known_mask: torch.Tensor
    padding_mask: torch.Tensor  # includes data not yet available at this decision
    second_timestamp_ms: torch.Tensor  # [batch, second]
    available_at_ms: torch.Tensor  # [batch, equity, second]
    decision_timestamp_ms: torch.Tensor  # [batch]
    asset_ids: tuple[str, ...]

    def validate(self) -> None:
        x = self.raw_ohlcv
        if x.ndim != 4 or x.shape[-1] != 5 or x.dtype != torch.float32 or min(x.shape) < 1:
            raise ValueError("Raw input must be FP32 [batch, equity, second, 5]")
        b, a, s, _ = x.shape
        if len(self.asset_ids) != a or len(set(self.asset_ids)) != a:
            raise ValueError("Instrument axis differs")
        for mask in (self.observed_mask, self.known_mask, self.padding_mask):
            if mask.shape != (b, a, s) or mask.dtype != torch.bool or mask.device != x.device:
                raise ValueError("Explicit bool observed/known/padding masks required")
        for clock, shape in ((self.second_timestamp_ms, (b, s)), (self.available_at_ms, (b, a, s)),
                             (self.decision_timestamp_ms, (b,))):
            if clock.shape != shape or clock.dtype != torch.int64 or clock.device != x.device:
                raise ValueError("Timestamp identity must remain int64")
        if (bool((self.second_timestamp_ms % 1000 != 0).any())
                or bool((self.second_timestamp_ms[:, 1:] - self.second_timestamp_ms[:, :-1] != 1000).any())):
            raise ValueError("Nonsecond/resampled clock rejected")
        active = ~self.padding_mask
        if bool((active & ~self.known_mask).any()):
            raise ValueError("Unknown coverage invalidates the sample")
        if bool((self.observed_mask & (~self.known_mask | self.padding_mask)).any()):
            raise ValueError("Observed, empty and padding states conflict")
        if bool((active & (self.available_at_ms > self.decision_timestamp_ms[:, None, None])).any()):
            raise ValueError("Unavailable observation at decision")
        ends = self.second_timestamp_ms[:, None, :] + 1000
        if bool((active & ((ends > self.decision_timestamp_ms[:, None, None]) | (self.available_at_ms < ends))).any()):
            raise ValueError("Incomplete bar or availability before bar end")
        values = x[self.observed_mask]
        if not bool(torch.isfinite(values).all()) or bool((values[:, :4] <= 0).any()) or bool((values[:, 4] < 0).any()):
            raise ValueError("Invalid raw numeric cast; no scaling/clipping fallback")
        if not bool(active.any(dim=-1).all()):
            raise ValueError("Each instrument needs available context")


def load_raw_second_window(ref: RawSecondWindowRef, *, device: torch.device | str) -> RawSecondObservation:
    """Every load verifies original bytes; no learned or transformed cache."""
    stamps = [ref.start_ms + 1000 * i for i in range(ref.seconds)]
    raw = torch.zeros((1, len(ref.asset_ids), ref.seconds, 5), dtype=torch.float32)
    known = torch.zeros(raw.shape[:-1], dtype=torch.bool)
    observed = torch.zeros_like(known)
    available = torch.zeros(raw.shape[:-1], dtype=torch.int64)
    for asset, source in enumerate(ref.captures):
        resolved = resolve_second_interval(source, ref.start_ms, ref.start_ms + ref.seconds * 1000, require_complete=False)
        values = {t: (x, received) for t, x, received in resolved.rows}
        for index, stamp in enumerate(stamps):
            receipt_mode = ref.contract.availability_assumption == "captured-receipt-time"
            receipt = resolved.coverage_receipt(stamp)
            known[0, asset, index] = receipt is not None
            row = values.get(stamp)
            arrival = (row[1] if row else (receipt or stamp + 1000)) if receipt_mode else stamp + 1000 + ref.contract.availability_delay_ms
            available[0, asset, index] = max(arrival, stamp + 1000)
            if row is not None and arrival <= ref.decision_ms and stamp + 1000 <= ref.decision_ms:
                raw[0, asset, index] = torch.tensor(row[0], dtype=torch.float32)
                observed[0, asset, index] = True
    padding = (available > ref.decision_ms) | (torch.tensor(stamps)[None, None, :] + 1000 > ref.decision_ms)
    result = RawSecondObservation(raw.to(device), observed.to(device), known.to(device), padding.to(device),
                                  torch.tensor([stamps], dtype=torch.int64, device=device), available.to(device),
                                  torch.tensor([ref.decision_ms], dtype=torch.int64, device=device), ref.asset_ids)
    result.validate()
    return result


class RawSecondCatalog:
    """Frozen reference table. PPO stores indices AND this table's digest."""

    def __init__(self, windows: tuple[RawSecondWindowRef, ...], *, execution_sources: tuple[SecondPartitionSet, ...] | None = None):
        if type(windows) is not tuple or not windows:
            raise ValueError("An immutable raw-window inventory is required")
        if any(w.asset_ids != windows[0].asset_ids or w.contract != windows[0].contract for w in windows):
            raise ValueError("Mixed universe/input contracts rejected")
        self.windows = windows
        self.asset_ids = windows[0].asset_ids
        if execution_sources is not None and (type(execution_sources) is not tuple
                or len(execution_sources) != len(self.asset_ids)
                or any(type(s) is not SecondPartitionSet for s in execution_sources)):
            raise ValueError("Execution sources must bind each instrument's immutable partitions")
        # Execution is indexed independently of the *next* observation window.
        # A compact observation context need not carry all intervening sources.
        self.execution_sources = execution_sources
        self.sources = tuple(SecondPartitionSet(tuple({c.manifest_sha256: c
            for source in (*(w.captures[i] for w in windows), *((execution_sources[i],) if execution_sources else ()))
            for c in second_partitions(source)}.values())) for i in range(len(self.asset_ids)))
        self.identity = self._identity()

    def _identity(self) -> str:
        return digest(dict(schema="rl-quant.raw-second-catalog-v2", windows=[w.identity for w in self.windows],
                           source_sets=[s.identity for s in self.sources]))

    def resolve(self, asset: str, start_ms: int, end_ms: int, *, require_complete: bool = True) -> ResolvedSecondInterval:
        self.validate_identity()
        return resolve_second_interval(self.sources[self.asset_ids.index(asset)], start_ms, end_ms,
                                       require_complete=require_complete)

    def validate_identity(self) -> None:
        if self._identity() != self.identity:
            raise ValueError("Raw reference catalog changed")

    def load(self, index: int, *, device: torch.device | str) -> RawSecondObservation:
        if type(index) is not int or not 0 <= index < len(self.windows):
            raise ValueError("Invalid raw reference index")
        self.validate_identity()
        return load_raw_second_window(self.windows[index], device=device)

    def identity_tensor(self, *, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(list(bytes.fromhex(self.identity)), dtype=torch.uint8, device=device)
