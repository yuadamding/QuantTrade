"""Bounded, lossless packed storage for original Massive second responses.

One committed batch has an authenticated index and a framed gzip payload file.
Pages retain their original JSON bytes (and supplied gzip/receipt bytes). No
native per-query directory or expanded JSON copy is required. An interrupted
writer is never resumable or complete; its framed originals remain on disk.

Selected reads authenticate only the requested frames. Full-store verification
additionally hashes every packed byte. There is deliberately no numeric,
embedding, or process-global cache. Storage completion is not data eligibility.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import struct

from rl_quant.datasets.massive_raw_seconds_v1 import (
    SCHEMA, CapturedSecondPage, SecondCaptureRef, SecondQuery, _integer, _json,
    _rows, _safe_url, _write,
)

PACKED_SCHEMA = "rl-quant.massive-raw-second-packed-v1"
INTERPRETATION = "provider-finalized-unadjusted-second-aggregates"
MAX_CAPTURES = 24
MAX_PAGES_PER_CAPTURE = 8
MAX_PAGE_BYTES = 32 * 1024**2
MAX_RAW_BYTES = 128_000_000
MAX_PACK_BYTES = 160_000_000
MAX_INDEX_BYTES = 4 * 1024**2
MAX_RECEIPT_BYTES = 64 * 1024
MAX_HEADER_BYTES = 128 * 1024
MAGIC = b"QT1SGZ1\n"
PACK_NAME = "pages.gzpack"
INDEX_NAME = "index.json"
_STAT_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
_PAGE_KEYS = {"frame_offset", "header_bytes", "header_sha256", "offset",
              "gzip_bytes", "gzip_sha256", "raw_bytes", "raw_sha256",
              "request_url", "received_at_ms"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(body: bytes) -> str:
    return sha256(body).hexdigest()


def _digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256 identity")


def _path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("Packed source requires a canonical absolute nonsymlink path")
    return path


def _snapshot(stream, maximum: int):
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
        raise ValueError("Unsafe or beyond-bound packed file")
    return info


def _unchanged(stream, before) -> None:
    after = os.fstat(stream.fileno())
    if any(getattr(before, key) != getattr(after, key) for key in _STAT_KEYS):
        raise ValueError("Packed source changed while reading")


def _read(path: Path, maximum: int, expected: str) -> bytes:
    _digest(expected)
    with os.fdopen(os.open(_path(path), os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = _snapshot(stream, maximum)
        body = stream.read(maximum + 1)
        _unchanged(stream, before)
    if len(body) != before.st_size or _hash(body) != expected:
        raise ValueError("Packed source content changed")
    return body


def _sync(root: Path) -> None:
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _inflate(body: bytes, raw_bytes: int, expected: str) -> bytes:
    _integer(raw_bytes, "raw page bytes")
    _digest(expected)
    if raw_bytes > MAX_PAGE_BYTES or len(body) > MAX_PAGE_BYTES + 1024**2:
        raise ValueError("Packed page exceeds bounded decompression limits")
    try:
        with gzip.GzipFile(fileobj=BytesIO(body)) as stream:
            raw = stream.read(raw_bytes + 1)
            if len(raw) != raw_bytes or stream.read(1):
                raise ValueError("Packed page decoded size differs")
    except (OSError, EOFError) as error:
        raise ValueError("Invalid packed gzip page") from error
    if _hash(raw) != expected:
        raise ValueError("Packed page decoded hash differs")
    return raw


def _manifest(query: SecondQuery, pages: list[dict]) -> dict:
    return dict(schema=SCHEMA, query=asdict(query), interpretation=INTERPRETATION,
                pages=[dict(request_url=p["request_url"], received_at_ms=p["received_at_ms"],
                            sha256=p["raw_sha256"]) for p in pages])


def _index(root: Path, expected: str) -> dict:
    root = _path(root)
    body = _json(_read(root / INDEX_NAME, MAX_INDEX_BYTES, expected))
    if (set(body) != {"schema", "interpretation", "captures", "pack", "provenance",
                      "raw_values_transformed", "training_ready", "point_in_time_qualified"}
            or body["schema"] != PACKED_SCHEMA or body["interpretation"] != INTERPRETATION
            or any(body[k] is not False for k in ("raw_values_transformed", "training_ready", "point_in_time_qualified"))
            or type(body["provenance"]) is not dict
            or len(_canonical(body["provenance"])) > MAX_RECEIPT_BYTES):
        raise ValueError("Packed index schema or qualification differs")
    if {p.name for p in root.iterdir()} != {INDEX_NAME, PACK_NAME}:
        raise ValueError("Unexpected packed source artifact")
    pack = body["pack"]
    if type(pack) is not dict or set(pack) != {"name", "bytes", "sha256"} or pack["name"] != PACK_NAME:
        raise ValueError("Packed file identity differs")
    _integer(pack["bytes"], "packed bytes", len(MAGIC))
    _digest(pack["sha256"])
    if pack["bytes"] > MAX_PACK_BYTES:
        raise ValueError("Packed file exceeds bound")
    captures = body["captures"]
    if type(captures) is not list or not 1 <= len(captures) <= MAX_CAPTURES:
        raise ValueError("Packed capture population exceeds bound")
    offset, raw_total, seen, queries = len(MAGIC), 0, set(), []
    for capture in captures:
        if type(capture) is not dict or set(capture) != {"manifest", "manifest_sha256", "pages"}:
            raise ValueError("Invalid packed capture member")
        manifest = capture["manifest"]
        if (type(manifest) is not dict or set(manifest) != {"schema", "query", "pages", "interpretation"}
                or manifest["schema"] != SCHEMA or manifest["interpretation"] != INTERPRETATION
                or _hash(_canonical(manifest)) != capture["manifest_sha256"]
                or capture["manifest_sha256"] in seen):
            raise ValueError("Packed capture manifest differs or is duplicated")
        seen.add(capture["manifest_sha256"])
        query = SecondQuery(**manifest["query"])
        _check_query(query, queries)
        queries.append(query)
        pages = capture["pages"]
        if type(pages) is not list or not 1 <= len(pages) <= MAX_PAGES_PER_CAPTURE:
            raise ValueError("Packed page population exceeds bound")
        for page in pages:
            if type(page) is not dict or set(page) != _PAGE_KEYS:
                raise ValueError("Invalid packed page index")
            for field in ("frame_offset", "header_bytes", "offset", "gzip_bytes", "raw_bytes", "received_at_ms"):
                _integer(page[field], field)
            for field in ("header_sha256", "gzip_sha256", "raw_sha256"):
                _digest(page[field])
            _safe_url(page["request_url"], query)
            if (page["frame_offset"] != offset or not 1 <= page["header_bytes"] <= MAX_HEADER_BYTES
                    or page["offset"] != offset + 4 + page["header_bytes"]
                    or not 1 <= page["gzip_bytes"] <= MAX_PAGE_BYTES + 1024**2
                    or page["raw_bytes"] > MAX_PAGE_BYTES):
                raise ValueError("Packed frame offsets or page bounds differ")
            offset = page["offset"] + page["gzip_bytes"]
            raw_total += page["raw_bytes"]
        if manifest != _manifest(query, pages):
            raise ValueError("Packed index altered native capture meaning")
    if offset != pack["bytes"] or raw_total > MAX_RAW_BYTES:
        raise ValueError("Packed byte population or decoded batch bound differs")
    return body


def _check_query(query: SecondQuery, previous: list[SecondQuery]) -> None:
    if type(query) is not SecondQuery or query.end_ms - query.start_ms >= 3_600_000:
        raise ValueError("Packed queries must be bounded to one hour")
    if any(query.ticker == old.ticker and max(query.start_ms, old.start_ms) <= min(query.end_ms, old.end_ms)
           for old in previous):
        raise ValueError("Overlapping queries inside one packed batch")


def _member(index: dict, manifest_sha256: str) -> dict:
    _digest(manifest_sha256)
    for capture in index["captures"]:
        if capture["manifest_sha256"] == manifest_sha256:
            return capture
    raise ValueError("Capture member absent from packed index")


def _load_member(root: Path, index: dict, member: dict) -> tuple[SecondQuery, tuple[CapturedSecondPage, ...]]:
    query = SecondQuery(**member["manifest"]["query"])
    pages = []
    with os.fdopen(os.open(_path(root / PACK_NAME), os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = _snapshot(stream, MAX_PACK_BYTES)
        if before.st_size != index["pack"]["bytes"] or stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("Packed payload length or format differs")
        for page in member["pages"]:
            stream.seek(page["frame_offset"])
            if stream.read(4) != struct.pack(">I", page["header_bytes"]):
                raise ValueError("Packed frame header length differs")
            header_raw = stream.read(page["header_bytes"])
            if _hash(header_raw) != page["header_sha256"]:
                raise ValueError("Packed frame header changed")
            header = _json(header_raw)
            expected = {k: page[k] for k in ("request_url", "received_at_ms", "gzip_bytes", "gzip_sha256", "raw_bytes", "raw_sha256")}
            if (set(header) != {"query", "page", "source_receipt_base64"} or header["query"] != asdict(query)
                    or header["page"] != expected):
                raise ValueError("Packed frame provenance differs")
            if header["source_receipt_base64"] is not None:
                try:
                    receipt = base64.b64decode(header["source_receipt_base64"], validate=True)
                except (ValueError, TypeError) as error:
                    raise ValueError("Invalid packed source receipt") from error
                if len(receipt) > MAX_RECEIPT_BYTES:
                    raise ValueError("Packed source receipt exceeds bound")
                _json(receipt)
            packed = stream.read(page["gzip_bytes"])
            if len(packed) != page["gzip_bytes"] or _hash(packed) != page["gzip_sha256"]:
                raise ValueError("Packed gzip page changed")
            raw = _inflate(packed, page["raw_bytes"], page["raw_sha256"])
            pages.append(CapturedSecondPage(page["request_url"], page["received_at_ms"], raw))
        _unchanged(stream, before)
    result = tuple(pages)
    _rows(query, result)
    return query, result


@dataclass(frozen=True)
class PackedSecondCaptureRef(SecondCaptureRef):
    """Portable reference: original capture identity plus physical index hash."""

    index_sha256: str

    def query(self) -> SecondQuery:
        """Authenticate the interval without decompressing any response page."""
        return SecondQuery(**_member(_index(Path(self.path), self.index_sha256),
                                     self.manifest_sha256)["manifest"]["query"])

    def load(self) -> tuple[SecondQuery, tuple[CapturedSecondPage, ...]]:
        index = _index(Path(self.path), self.index_sha256)
        return _load_member(Path(self.path), index, _member(index, self.manifest_sha256))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PackedSecondCaptureRef:
        if type(value) is not dict or set(value) != {"path", "manifest_sha256", "index_sha256"}:
            raise ValueError("Invalid packed capture reference fields")
        _path(Path(value["path"]))
        _digest(value["manifest_sha256"])
        _digest(value["index_sha256"])
        return cls(**value)


class SecondPackWriter:
    """Exclusive streaming writer; frames survive failures, no implicit resume.

    Call append_page before interpreting provider status. A failed/error page is
    preserved but cannot pass finish_query or acquire a committed index. Caller
    owns credentials, HTTP semantics, quotas and a separate failure receipt.
    """

    def __init__(self, root: Path, *, provenance: dict | None = None):
        self.root = _path(root)
        self.provenance = {} if provenance is None else _json(_canonical(provenance))
        if len(_canonical(self.provenance)) > MAX_RECEIPT_BYTES:
            raise ValueError("Packed provenance exceeds bound")
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._stream = os.fdopen(os.open(self.root / PACK_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "wb")
        self._stream.write(MAGIC)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        _sync(self.root)
        self._hash = sha256(MAGIC)
        self._offset = len(MAGIC)
        self._raw_total = 0
        self._captures: list[dict] = []
        self._query: SecondQuery | None = None
        self._pages: list[dict] = []
        self._poisoned = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def _active(self) -> None:
        if self._stream.closed or self._poisoned:
            raise ValueError("Closed or failed packed writer cannot be reused")

    def begin_query(self, query: SecondQuery) -> None:
        self._active()
        if self._query is not None or len(self._captures) >= MAX_CAPTURES:
            raise ValueError("Finish the current bounded query first")
        _check_query(query, [SecondQuery(**r["manifest"]["query"]) for r in self._captures])
        self._query, self._pages = query, []

    def append_page(self, page: CapturedSecondPage, *, compressed_body: bytes | None = None,
                    source_receipt: bytes | None = None) -> None:
        try:
            self._append_page(page, compressed_body=compressed_body, source_receipt=source_receipt)
        except BaseException:
            self._poisoned = True
            raise

    def _append_page(self, page: CapturedSecondPage, *, compressed_body: bytes | None,
                     source_receipt: bytes | None) -> None:
        self._active()
        if self._query is None or len(self._pages) >= MAX_PAGES_PER_CAPTURE or type(page) is not CapturedSecondPage:
            raise ValueError("Expected a page in an active bounded query")
        _safe_url(page.request_url, self._query)
        _integer(page.received_at_ms, "capture timestamp")
        if type(page.body) is not bytes or len(page.body) > MAX_PAGE_BYTES or self._raw_total + len(page.body) > MAX_RAW_BYTES:
            raise ValueError("Raw page/batch bound exhausted")
        if source_receipt is not None:
            if type(source_receipt) is not bytes or len(source_receipt) > MAX_RECEIPT_BYTES:
                raise ValueError("Source receipt exceeds bound")
            _json(source_receipt)
        compressed = gzip.compress(page.body, compresslevel=9, mtime=0) if compressed_body is None else compressed_body
        if type(compressed) is not bytes or _inflate(compressed, len(page.body), _hash(page.body)) != page.body:
            raise ValueError("Supplied gzip does not preserve the original response")
        fields = dict(request_url=page.request_url, received_at_ms=page.received_at_ms,
                      gzip_bytes=len(compressed), gzip_sha256=_hash(compressed),
                      raw_bytes=len(page.body), raw_sha256=_hash(page.body))
        header = _canonical(dict(query=asdict(self._query), page=fields,
            source_receipt_base64=None if source_receipt is None else base64.b64encode(source_receipt).decode("ascii")))
        if len(header) > MAX_HEADER_BYTES or self._offset + 4 + len(header) + len(compressed) > MAX_PACK_BYTES:
            raise ValueError("Packed frame/batch bound exhausted")
        descriptor = dict(fields, frame_offset=self._offset, header_bytes=len(header),
                          header_sha256=_hash(header), offset=self._offset + 4 + len(header))
        try:
            for body in (struct.pack(">I", len(header)), header, compressed):
                self._stream.write(body)
                self._hash.update(body)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self._poisoned = True
            raise
        self._offset += 4 + len(header) + len(compressed)
        self._raw_total += len(page.body)
        self._pages.append(descriptor)

    def finish_query(self) -> str:
        self._active()
        if self._query is None or not self._pages:
            raise ValueError("No complete query pages supplied")
        manifest = _manifest(self._query, self._pages)
        member = dict(manifest=manifest, manifest_sha256=_hash(_canonical(manifest)), pages=list(self._pages))
        try:
            _load_member(self.root, {"pack": {"bytes": self._offset}}, member)
        except BaseException:
            self._poisoned = True
            raise
        self._captures.append(member)
        self._query, self._pages = None, []
        return member["manifest_sha256"]

    def finalize(self) -> dict:
        self._active()
        if self._query is not None or not self._captures:
            raise ValueError("Cannot commit an incomplete packed batch")
        index = dict(schema=PACKED_SCHEMA, interpretation=INTERPRETATION,
            captures=self._captures, pack=dict(name=PACK_NAME, bytes=self._offset, sha256=self._hash.hexdigest()),
            provenance=self.provenance, raw_values_transformed=False, training_ready=False, point_in_time_qualified=False)
        body = _canonical(index)
        if len(body) > MAX_INDEX_BYTES:
            raise ValueError("Packed index exceeds bound")
        self.close()
        _write(self.root / INDEX_NAME, body)
        _sync(self.root)
        return verify_packed_second_store(root=self.root, index_sha256=_hash(body))


def verify_packed_second_store(*, root: Path, index_sha256: str) -> dict:
    """Fresh full-byte hash and native replay; read-only, bounded one page/query."""
    root = _path(root)
    index = _index(root, index_sha256)
    state = sha256()
    with os.fdopen(os.open(root / PACK_NAME, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = _snapshot(stream, MAX_PACK_BYTES)
        while block := stream.read(1024**2):
            state.update(block)
        _unchanged(stream, before)
    if before.st_size != index["pack"]["bytes"] or state.hexdigest() != index["pack"]["sha256"]:
        raise ValueError("Whole packed payload identity differs")
    captures = []
    for member in index["captures"]:
        _load_member(root, index, member)
        captures.append(PackedSecondCaptureRef(str(root), member["manifest_sha256"], index_sha256).to_dict())
    if _index(root, index_sha256) != index:
        raise ValueError("Packed index changed during verification")
    return dict(schema=PACKED_SCHEMA + "-verified", path=str(root), index_sha256=index_sha256,
        pack_sha256=index["pack"]["sha256"], pack_bytes=index["pack"]["bytes"], index_bytes=len(_canonical(index)),
        capture_count=len(captures), page_count=sum(len(r["pages"]) for r in index["captures"]), captures=captures,
        raw_response_bytes=sum(p["raw_bytes"] for r in index["captures"] for p in r["pages"]),
        source_provenance=index["provenance"], raw_values_transformed=False, training_ready=False,
        point_in_time_qualified=False, original_inputs_deleted=False)


def pack_verified_second_capture(*, root: Path, plan_sha256: str, completion_sha256: str, output: Path) -> dict:
    """Repack retained acquisition originals; never rewrite or remove them."""
    from rl_quant.data_sources.massive import raw_second_capture_v1 as acquisition
    from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport

    root = _path(root)
    before = acquisition.replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256)
    queries = acquisition._plan(root, plan_sha256, current=False)
    complete = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1024**2))
    plan = transport.parse_json(transport.read_regular(root / "plan.json", 1024**2))
    provenance = dict(capture_root=str(root), plan_sha256=plan_sha256,
                      capture_complete_sha256=completion_sha256, capture_schema=plan["schema"],
                      alias_routes=plan.get("alias_routes"), continuous_identity_qualified=False,
                      provider_tickers_rewritten=False)
    with SecondPackWriter(output, provenance=provenance) as writer:
        for query, entry in zip(queries, complete["queries"], strict=True):
            writer.begin_query(SecondQuery(**asdict(query)))
            pages = acquisition._pages(root, query, entry)
            for number, page in enumerate(pages):
                base = root / query.name
                receipt = transport.read_regular(base / f"page-{number:04d}.receipt.json", MAX_RECEIPT_BYTES)
                packed = transport.read_regular(base / f"page-{number:04d}.json.gz", MAX_PAGE_BYTES + 1024**2)
                writer.append_page(page, compressed_body=packed, source_receipt=receipt)
            writer.finish_query()
        if acquisition.replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256) != before:
            raise ValueError("Retained capture changed while packing")
        result = writer.finalize()
    if acquisition.replay_second_capture(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256) != before:
        raise ValueError("Retained capture changed after packing")
    return result
