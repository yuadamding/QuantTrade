"""Bounded, resumable *preparation* of a hash-bound raw-second registry.

No credentials are opened here, no experiment roles are assigned, and no
training/evaluation code is called. A consumed nonterminal batch is never
retried. An operator may advance disjoint <=64-batch tranches only from the
preceding immutable completion. Registry reuse hints alone are not evidence.

The opt-in slot-accounting generation accepts explicit, reviewed prelisting
dispositions without creating bars or dropping fixed-universe coordinates.
An absent reference is not known-unavailable coverage. Missing or unresolved
routes still block the whole affected batch before its first write. This
bounded controller alone cannot unlock full-history acquisition or training.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date as Date, datetime, timezone
import fcntl
from itertools import islice
import os
from pathlib import Path
import time

from rl_quant.data_sources.massive import raw_second_capture_v1 as capture
from rl_quant.data_sources.massive import qt200_research_capture_v1 as io
from rl_quant.data_sources.massive.qt200_daily_capture_v1 import SYMBOLS
from rl_quant.data_sources.massive.raw_second_alias_v1 import (
    AliasIdentityRef, SecondAliasRoute, validate_fixed_slot_identity,
)
from rl_quant.data_sources.massive.raw_second_successor_v1 import SecondSuccessorRoute
from rl_quant.data_sources.massive.raw_second_documented_alias_v1 import SecondDocumentedAliasRoute
from rl_quant.data_sources.massive.raw_second_reviewed_class_v1 import SecondReviewedClassRoute
from rl_quant.data_sources.massive import raw_second_evidence_v1 as evidence
from rl_quant.datasets.massive_raw_second_packed_v1 import (
    pack_verified_second_capture, verify_packed_second_store,
)
from rl_quant.datasets import massive_raw_seconds_v1 as raw_seconds
from rl_quant.datasets.massive_raw_seconds_v1 import RawSecondContract, SecondQuery


SCHEMA = "rl-quant.raw-second-corpus-v1"
DIRECT_SCHEMA = "rl-quant.raw-second-direct-packed-corpus-v1"
LEGACY_STORAGE = "loose-and-packed-v1"
DIRECT_STORAGE = "direct-packed-v1"
SLOT_ACCOUNTING = "explicit-prelisting-v1"
DOCUMENTED_ALIAS_ROUTING = "documented-rename-v1"
REVIEWED_CLASS_ROUTING = "reviewed-common-class-v1"
MAX_METADATA = 16 * 1024 * 1024
MAX_TRANCHE_BATCHES = 64
# Worst-case capture + packed duplicate, incompressible gzip overhead and
# <=192 bounded receipts/manifests. This is a conservative logical-byte
# reservation, not a GPFS allocation/quota guarantee; fresh operator-owned
# allocated-byte/fileset/inode admission is independently mandatory.
# Original captures are never deleted.
RETAINED_RESERVATION = 2 * capture.MAX_BYTES + 32 * 1024 * 1024
INODE_RESERVATION = 1024
DIRECT_RETAINED_RESERVATION = 160_000_000 + 8 * 1024 * 1024
DIRECT_INODE_RESERVATION = 32


def _require(condition, message):
    if not condition:
        raise io.ResearchCaptureError(message)


def _json(path, expected=None):
    raw = io.read_regular(path, MAX_METADATA)
    _require(expected is None or io.digest(raw) == expected, "Bound metadata changed")
    return io.parse_json(raw)


def _write(path, value):
    return io.write_once(path, io.canonical(value))["sha256"]


def _storage_schema(storage_mode):
    _require(storage_mode in (LEGACY_STORAGE, DIRECT_STORAGE), "Unsupported explicit corpus storage mode")
    return DIRECT_SCHEMA if storage_mode == DIRECT_STORAGE else SCHEMA


def _corpus_schema(storage_mode, slot_accounting=None):
    _require(slot_accounting in (None, SLOT_ACCOUNTING), "Unsupported explicit slot accounting")
    return _storage_schema(storage_mode) + ("-slots-v1" if slot_accounting is not None else "")


def _storage(plan):
    mode = plan.get("storage_mode", LEGACY_STORAGE)
    _require(plan["schema"] == _corpus_schema(mode, plan.get("slot_accounting")) + "-plan"
             and ("slot_accounting" not in plan or plan["slot_accounting"] == SLOT_ACCOUNTING)
             and ("alias_routing" not in plan or plan["alias_routing"] == DOCUMENTED_ALIAS_ROUTING)
             and ("class_routing" not in plan or plan["class_routing"] == REVIEWED_CLASS_ROUTING)
             and (mode != LEGACY_STORAGE or "storage_mode" not in plan),
             "Corpus storage generation differs")
    return mode


def _schema(plan):
    return _corpus_schema(_storage(plan), plan.get("slot_accounting"))


def _prelisting_type():
    from rl_quant.data_sources.massive.raw_second_prelisting_v1 import SecondPrelistingDisposition
    return SecondPrelistingDisposition


def _implementation(storage_mode=LEGACY_STORAGE, slot_accounting=None, alias_routing=None, class_routing=None):
    _corpus_schema(storage_mode, slot_accounting)
    _require(alias_routing in (None, DOCUMENTED_ALIAS_ROUTING), "Unsupported explicit alias routing")
    _require(class_routing in (None, REVIEWED_CLASS_ROUTING), "Unsupported explicit class routing")
    paths = (Path(__file__), Path(capture.__file__), Path(io.__file__),
             Path(raw_seconds.__file__), Path(evidence.__file__),
             Path(validate_fixed_slot_identity.__code__.co_filename),
             Path(SecondSuccessorRoute.validate.__code__.co_filename),
             Path(pack_verified_second_capture.__code__.co_filename))
    if storage_mode == DIRECT_STORAGE:
        from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
        paths += (Path(direct.__file__),)
    if slot_accounting is not None:
        from rl_quant.data_sources.massive import raw_second_prelisting_v1 as prelisting
        paths += (Path(prelisting.__file__),)
    if alias_routing is not None:
        paths += (Path(SecondDocumentedAliasRoute.validate.__code__.co_filename),)
    if class_routing is not None:
        paths += (Path(SecondReviewedClassRoute.validate.__code__.co_filename),)
    return {p.name: io.digest(io.read_regular(p.resolve(), MAX_METADATA)) for p in paths}


@dataclass(frozen=True)
class CorpusRegistry:
    root: Path
    completion_sha256: str

    def read(self):
        terminal = _json(self.root / "COMPLETE.json", self.completion_sha256)
        _require(terminal.get("metadata_complete") is True, "Registry is not complete metadata")
        files = {row["path"]: row for row in terminal["files"]}
        _require(set(files) == {"summary.json", "universe.json", "sessions.jsonl", "reuse-queries.jsonl"}
                 and len(terminal["files"]) == 4, "Unexpected registry inventory")
        bodies = {}
        for name, row in files.items():
            raw = io.read_regular(self.root / name, MAX_METADATA)
            _require(io.digest(raw) == row["sha256"] and len(raw) == row["bytes"], "Registry file changed")
            bodies[name] = raw
        summary = io.parse_json(bodies["summary.json"])
        universe = io.parse_json(bodies["universe.json"])["tickers"]
        _require(tuple(universe) == SYMBOLS and len(universe) == 200
                 and summary["ordered_universe_sha256"] == io.digest(io.canonical(universe))
                 and summary["start_date"] == "2017-01-03" and summary["end_date"] == "2026-08-31"
                 and summary["calendar_versions"]["exchange-calendars"] == "4.13.2"
                 and summary["calendar_is_planning_only"] is True, "Fixed corpus scope differs")
        fields = summary["market_contract"]
        _require(type(fields["market_fields"]) is list, "Raw market fields must be a JSON list")
        contract = RawSecondContract(**{**fields, "market_fields": tuple(fields["market_fields"])})
        _require(contract.availability_assumption == "developer-delayed-assumption"
                 and contract.availability_delay_ms == 900000, "Raw market availability contract differs")
        return summary, universe, bodies["sessions.jsonl"]

    def batches(self, start=0):
        """Stable IDs before reuse; at most one 24-slot descriptor in memory."""
        _require(type(start) is int and start >= 0, "Invalid deterministic batch offset")
        _, universe, sessions = self.read()
        ordinal, previous_date = 0, ""
        for raw in sessions.splitlines():
            row = io.parse_json(raw)
            date, left, close = row["session_date"], row["open_ms"], row["close_ms"]
            _require(type(date) is str and Date.fromisoformat(date).isoformat() == date
                     and type(left) is int and type(close) is int and 0 <= left < close
                     and left % 1000 == close % 1000 == 0
                     and datetime.fromtimestamp(left // 1000, timezone.utc).date().isoformat() == date
                     and datetime.fromtimestamp(close // 1000, timezone.utc).date().isoformat() == date
                     and previous_date < date and "2017-01-03" <= date <= "2026-08-31"
                     and close - left in (23400000, 12600000), "Planning session differs")
            previous_date = date
            expected = [(a, min(close, a + 3600000) - 1000) for a in range(left, close, 3600000)]
            _require(all(type(p[k]) is int and p[k] % 1000 == 0
                         for p in row["partitions"] for k in ("start_ms", "end_ms"))
                     and [(p["start_ms"], p["end_ms"]) for p in row["partitions"]] == expected,
                     "Planning partitions are not exact seconds")
            for slot, (begin, end) in enumerate(expected):
                for offset in range(0, 200, capture.MAX_QUERIES):
                    if ordinal >= start:
                        yield {"ordinal": ordinal, "batch_id": f"batch-{ordinal:06d}",
                               "session_date": date, "slot": slot, "start_ms": begin,
                               "end_ms": end, "fixed_slot_tickers": universe[offset:offset + 24]}
                    ordinal += 1


@dataclass(frozen=True)
class CorpusLimits:
    maximum_http_requests: int
    maximum_raw_response_bytes: int
    maximum_retained_bytes: int
    project_ceiling_bytes: int
    tranche_batches: int = 64
    capacity_maximum_age_seconds: int = 300

    def validate(self):
        _require(all(type(v) is int and v > 0 for v in asdict(self).values())
                 and self.tranche_batches <= MAX_TRANCHE_BATCHES
                 and self.capacity_maximum_age_seconds <= 3600, "Invalid explicit corpus limits")


@dataclass(frozen=True)
class CorpusIdentity:
    fixed_slot_ticker: str
    provider_ticker: str
    identity: AliasIdentityRef
    alias: SecondAliasRoute | SecondDocumentedAliasRoute | None = None
    successor: SecondSuccessorRoute | None = None
    reviewed_class: SecondReviewedClassRoute | None = None

    def to_dict(self):
        """Keep old STARTED payloads byte-compatible; tag only the new route."""
        result = asdict(self)
        if self.alias is not None:
            result["alias"] = capture._alias_route_dict(self.alias)
        if self.reviewed_class is None:
            del result["reviewed_class"]
        else:
            result["reviewed_class"] = self.reviewed_class.to_dict()
        return result


@dataclass(frozen=True)
class CaptureReuse:
    root: Path
    plan_sha256: str
    completion_sha256: str
    storage_mode: str = LEGACY_STORAGE

    def verify(self):
        _storage_schema(self.storage_mode)
        replay = capture.replay_second_capture
        if self.storage_mode == DIRECT_STORAGE:
            from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
            replay = direct.replay_direct_second_capture
        proof = replay(root=self.root, plan_sha256=self.plan_sha256,
                       completion_sha256=self.completion_sha256)
        plan = _json(self.root / "plan.json", self.plan_sha256)
        return {(q["ticker"], q["start_ms"], q["end_ms"]) for q in plan["queries"]}, proof

    def to_dict(self):
        result = {**asdict(self), "root": str(self.root)}
        if self.storage_mode == LEGACY_STORAGE:
            del result["storage_mode"]
        return result


def _corpus_reuse(ref, plan):
    """Bind semantics for the entire replay, including off-batch source rows."""
    source = _json(ref.root / "plan.json", ref.plan_sha256)
    routes = tuple(capture._alias_route_from_dict(row) for row in source.get("alias_routes", []))
    documented = (capture._documented_aliases(routes)
                  or "documented_alias_schema" in source
                  or "documented_alias_implementation_sha256" in source)
    _require(not documented or plan.get("alias_routing") == DOCUMENTED_ALIAS_ROUTING,
             "Replaying documented alias reuse requires explicit corpus routing opt-in")
    return ref.verify()


def publish_corpus_plan(*, root: Path, registry: CorpusRegistry, limits: CorpusLimits,
                        intended_issue_source: Path, owner: Path,
                        previous: CaptureReuse | None = None,
                        storage_mode: str = LEGACY_STORAGE, slot_accounting: str | None = None,
                        alias_routing: str | None = None, class_routing: str | None = None):
    """Create a bounded tranche, not an authority to bypass operator admission.

    ``previous`` identifies a corpus root/plan/COMPLETE (not a provider capture).
    Its terminal is bookkeeping lineage only, never reusable market evidence.
    """
    limits.validate()
    _require(alias_routing in (None, DOCUMENTED_ALIAS_ROUTING), "Unsupported explicit alias routing")
    _require(class_routing in (None, REVIEWED_CLASS_ROUTING), "Unsupported explicit class routing")
    schema = _corpus_schema(storage_mode, slot_accounting)
    summary, _, _ = registry.read()
    identity_sha = summary["identity_evidence"]["sha256"]
    # The identity verifier opens gzip evidence; bind its original bytes here.
    _require(io.digest(io.read_regular(intended_issue_source, MAX_METADATA)) == identity_sha,
             "Intended issue evidence changed")
    start, consumed = 0, {"http_requests": 0, "raw_response_bytes": 0, "retained_bytes": 0}
    if previous is not None:
        prior_plan = _json(previous.root / "plan.json", previous.plan_sha256)
        prior = _json(previous.root / "COMPLETE.json", previous.completion_sha256)
        _reserved_plan(previous.root, prior_plan, previous.plan_sha256)
        _require(_storage(prior_plan) == storage_mode and prior.get("schema") == schema + "-complete"
                 and prior_plan.get("slot_accounting") == slot_accounting
                 and prior_plan.get("alias_routing") == alias_routing
                 and prior_plan.get("class_routing") == class_routing
                 and prior.get("range_complete") is True
                 and prior["plan_sha256"] == previous.plan_sha256
                 and prior_plan["registry_sha256"] == registry.completion_sha256
                 and prior_plan["limits"] == asdict(limits), "Predecessor is not a completed matching tranche")
        _require(prior_plan["owner"] == str(owner), "Corpus acquisition owner changed")
        _verify_tranche(previous.root, prior_plan, prior)
        start, consumed = prior["next_batch"], prior["consumed"]
    descriptors = list(islice(registry.batches(start), limits.tranche_batches))
    _require(descriptors, "Corpus planning range exhausted")
    plan = {"schema": schema + "-plan", "registry_root": str(registry.root),
            "registry_sha256": registry.completion_sha256, "limits": asdict(limits),
            "intended_issue_source": str(intended_issue_source), "intended_issue_sha256": identity_sha,
            "owner": str(owner),
            "start_batch": start, "batch_count": len(descriptors), "consumed_before": consumed,
            "previous": None if previous is None else previous.to_dict(),
            "implementation_sha256": _implementation(storage_mode, slot_accounting, alias_routing, class_routing),
            "training_ready": False, "training_authorized": False, "experiment_roles_changed": False}
    if storage_mode == DIRECT_STORAGE:
        plan["storage_mode"] = storage_mode
    if slot_accounting is not None:
        plan["slot_accounting"] = slot_accounting
    if alias_routing is not None:
        plan["alias_routing"] = alias_routing
    if class_routing is not None:
        plan["class_routing"] = class_routing
    if not os.path.lexists(owner):
        owner.mkdir(mode=0o700, parents=True, exist_ok=False)
        io.write_once(owner / "owner.lock", b"")
        _write(owner / "registry.json", {"registry_sha256": registry.completion_sha256})
    _require(_json(owner / "registry.json") == {"registry_sha256": registry.completion_sha256},
             "Shared acquisition owner registry differs")
    with _owner(owner):
        predecessor = "initial" if previous is None else previous.completion_sha256
        # A crash after this reservation cannot create an ambiguous second successor.
        _write(owner / (predecessor + ".json"), {"root": str(root), "plan_sha256": io.digest(io.canonical(plan))})
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (root / "batches").mkdir(mode=0o700)
        return _write(root / "plan.json", plan)


@contextmanager
def _owner(root):
    io.read_regular(root / "owner.lock", 0)
    fd = os.open(root / "owner.lock", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _reserved_plan(root, plan, plan_sha256):
    """A newly supplied hash cannot replace the owner's immutable reservation."""
    _storage(plan)
    _require(plan["training_ready"] is False
             and plan["training_authorized"] is False and plan["experiment_roles_changed"] is False,
             "Corpus plan claims differ")
    owner = Path(plan["owner"])
    _require(_json(owner / "registry.json") == {"registry_sha256": plan["registry_sha256"]},
             "Corpus owner registry differs")
    previous = plan["previous"]
    key = "initial" if previous is None else previous["completion_sha256"]
    _require(key == "initial" or (type(key) is str and len(key) == 64
             and all(c in "0123456789abcdef" for c in key)), "Invalid predecessor identity")
    _require(_json(owner / (key + ".json")) == {"root": str(root), "plan_sha256": plan_sha256},
             "Corpus owner publication reservation differs")


def _admit_plan(root, plan, plan_sha256):
    _reserved_plan(root, plan, plan_sha256)
    limits = CorpusLimits(**plan["limits"])
    limits.validate()
    registry = CorpusRegistry(Path(plan["registry_root"]), plan["registry_sha256"])
    start, consumed = 0, {"http_requests": 0, "raw_response_bytes": 0, "retained_bytes": 0}
    if plan["previous"] is not None:
        ref = plan["previous"]
        prior_root = Path(ref["root"])
        prior_plan = _json(prior_root / "plan.json", ref["plan_sha256"])
        previous = _json(prior_root / "COMPLETE.json", ref["completion_sha256"])
        _reserved_plan(prior_root, prior_plan, ref["plan_sha256"])
        _require(_storage(prior_plan) == _storage(plan)
                 and prior_plan.get("slot_accounting") == plan.get("slot_accounting")
                 and prior_plan.get("alias_routing") == plan.get("alias_routing")
                 and prior_plan.get("class_routing") == plan.get("class_routing")
                 and previous["schema"] == _schema(plan) + "-complete"
                 and previous["range_complete"] is True
                 and previous["plan_sha256"] == ref["plan_sha256"]
                 and all(prior_plan[k] == plan[k] for k in ("owner", "registry_root", "registry_sha256",
                     "limits", "intended_issue_source", "intended_issue_sha256")),
                 "Corpus predecessor contract differs")
        _verify_tranche(prior_root, prior_plan, previous)
        start, consumed = previous["next_batch"], previous["consumed"]
    expected_count = len(list(islice(registry.batches(start), limits.tranche_batches)))
    _require(type(plan["start_batch"]) is int and plan["start_batch"] == start
             and type(plan["batch_count"]) is int and plan["batch_count"] == expected_count
             and expected_count > 0 and plan["consumed_before"] == consumed
             and all(type(value) is int and value >= 0 for value in consumed.values()),
             "Corpus range or predecessor counters changed")


def _capacity(snapshot, limits, reservation, inode_reservation=INODE_RESERVATION):
    """Operator-owned, fresh quota/space admission; not a fabricated receipt."""
    unsigned = {key: value for key, value in snapshot.items() if key != "receipt_sha256"}
    _require(snapshot["receipt_sha256"] == io.digest(io.canonical(unsigned)), "Capacity receipt changed")
    age = time.time_ns() - snapshot["observed_at_ns"]
    _require(0 <= age <= limits.capacity_maximum_age_seconds * 1000000000
             and snapshot["project_ceiling_bytes"] == limits.project_ceiling_bytes
             and 0 <= snapshot["project_allocated_bytes"] <= limits.project_ceiling_bytes - reservation
             and snapshot["filesystem_free_bytes"] >= reservation
             and snapshot["fileset_available_bytes"] >= reservation
             and snapshot["filesystem_free_inodes"] >= inode_reservation
             and snapshot["fileset_available_inodes"] >= inode_reservation
             and len(snapshot["receipt_sha256"]) == 64, "Fresh project/fileset capacity admission failed")


def _retained_bytes(root):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=lambda exc: (_ for _ in ()).throw(exc)):
        _require(all(not (Path(directory) / name).is_symlink() for name in dirs), "Linked output directory")
        for name in files:
            if Path(directory) == root and name == "COMPLETE.json":
                continue
            total += len(io.read_regular(Path(directory) / name, RETAINED_RESERVATION))
    return total


def _identity_queries(plan, batch, identities):
    _require(tuple(i.fixed_slot_ticker for i in identities) == tuple(batch["fixed_slot_tickers"]),
             "Missing, reordered, or duplicate exact-date identity routes")
    queries, aliases, proofs = [], [], []
    for identity in identities:
        query = SecondQuery(identity.provider_ticker, batch["start_ms"], batch["end_ms"])
        if identity.reviewed_class is not None:
            route = identity.reviewed_class
            _require(plan.get("class_routing") == REVIEWED_CLASS_ROUTING,
                     "Reviewed class requires explicit corpus routing opt-in")
            _require(type(route) is SecondReviewedClassRoute
                     and identity.alias is None and identity.successor is None
                     and route.fixed_slot_ticker == identity.fixed_slot_ticker
                     and route.provider_ticker == identity.provider_ticker
                     and route.session_date == batch["session_date"] and route.identity == identity.identity
                     and route.intended_issue_source == plan["intended_issue_source"]
                     and route.intended_issue_sha256 == plan["intended_issue_sha256"],
                     "Reviewed class is not bound to this exact corpus issue/date")
            proof = route.validate(query)
        elif identity.successor is not None:
            route = identity.successor
            _require(type(route) is SecondSuccessorRoute and identity.alias is None
                     and route.fixed_slot_ticker == identity.fixed_slot_ticker
                     and route.provider_ticker == identity.provider_ticker
                     and route.session_date == batch["session_date"]
                     and route.identity == identity.identity
                     and str(route.intended_issue_source) == plan["intended_issue_source"]
                     and route.intended_issue_sha256 == plan["intended_issue_sha256"],
                     "Successor route is not bound to this exact corpus issue/date")
            proof = route.validate(query)
        elif identity.provider_ticker != identity.fixed_slot_ticker:
            _require(type(identity.alias) in (SecondAliasRoute, SecondDocumentedAliasRoute),
                     "Historical alias lacks an exact supported typed route")
            if type(identity.alias) is SecondDocumentedAliasRoute:
                _require(plan.get("alias_routing") == DOCUMENTED_ALIAS_ROUTING,
                         "Documented alias requires explicit corpus routing opt-in")
                _require(str(identity.alias.intended_issue_source) == plan["intended_issue_source"]
                         and identity.alias.intended_issue_sha256 == plan["intended_issue_sha256"],
                         "Documented alias is not bound to the corpus intended issue evidence")
            _require(identity.alias.fixed_slot_ticker == identity.fixed_slot_ticker
                     and identity.alias.provider_ticker == identity.provider_ticker
                     and identity.alias.session_date == batch["session_date"]
                     and identity.alias.identity == identity.identity, "Alias exact-day source differs")
            proof = identity.alias.validate(query)
            aliases.append(identity.alias)
        else:
            _require(identity.alias is None, "Unexpected alias for literal query")
            proof = validate_fixed_slot_identity(fixed_slot_ticker=identity.fixed_slot_ticker,
                provider_ticker=identity.provider_ticker, session_date=batch["session_date"],
                identity=identity.identity, intended_issue_source=Path(plan["intended_issue_source"]),
                intended_issue_sha256=plan["intended_issue_sha256"])
        queries.append(query)
        proofs.append(proof)
    return queries, aliases, proofs


def _slot_queries(plan, batch, identities, unavailable):
    """Validate a complete fixed-slot cover, without querying unavailable issues."""
    if plan.get("slot_accounting") is None:
        _require(not unavailable, "Prelisting requires explicit slot accounting")
        return (*_identity_queries(plan, batch, identities), None)
    kind = _prelisting_type()
    _require(type(identities) is tuple and type(unavailable) is tuple
             and all(type(i) is CorpusIdentity for i in identities)
             and all(type(d) is kind for d in unavailable), "Untyped corpus slot disposition")
    names = batch["fixed_slot_tickers"]
    query_names = [i.fixed_slot_ticker for i in identities]
    absent_names = [d.fixed_slot_ticker for d in unavailable]
    _require(len(set(query_names + absent_names)) == len(names)
             and set(query_names + absent_names) == set(names)
             and query_names == [name for name in names if name in query_names]
             and absent_names == [name for name in names if name in absent_names],
             "Missing, reordered, duplicate, or overlapping corpus slot dispositions")
    queries, aliases, proofs = _identity_queries(plan, {**batch, "fixed_slot_tickers": query_names}, identities)
    slots = {identity.fixed_slot_ticker: {"fixed_slot_ticker": identity.fixed_slot_ticker,
        "disposition": "acquisition-required", "query": asdict(query), "proof": proof}
        for identity, query, proof in zip(identities, queries, proofs, strict=True)}
    for disposition in unavailable:
        _require(disposition.session_date == batch["session_date"]
                 and str(disposition.intended_issue_source) == plan["intended_issue_source"]
                 and disposition.intended_issue_sha256 == plan["intended_issue_sha256"],
                 "Prelisting route is not bound to this exact corpus issue/date")
        # This query is an interval-validation value only, not an HTTP request.
        query = SecondQuery(disposition.fixed_slot_ticker, batch["start_ms"], batch["end_ms"])
        proof = disposition.validate(query)
        _require(proof["acquisition_required"] is False
                 and proof["market_observation_status"] == "known-unavailable"
                 and proof["reason"] == "intended-class-before-public-trading",
                 "Prelisting disposition claims differ")
        slots[disposition.fixed_slot_ticker] = {"fixed_slot_ticker": disposition.fixed_slot_ticker,
            "disposition": "known-unavailable", "query": None, "proof": proof}
    return queries, aliases, proofs, [slots[name] for name in names]


def _slot_coverage(slots, reused):
    if slots is None:
        return None
    rows = []
    for slot in slots:
        query = slot["query"]
        disposition = "known-unavailable" if query is None else (
            "reused" if (query["ticker"], query["start_ms"], query["end_ms"]) in reused else "acquired")
        rows.append({"fixed_slot_ticker": slot["fixed_slot_ticker"],
            "fixed_slot_index": SYMBOLS.index(slot["fixed_slot_ticker"]),
            "disposition": disposition, "query": query})
    return {"schema": SLOT_ACCOUNTING, "fixed_universe_size": 200, "slot_count": len(rows),
        "acquired_queries": sum(r["disposition"] == "acquired" for r in rows),
        "reused_queries": sum(r["disposition"] == "reused" for r in rows),
        "known_unavailable_slots": sum(r["disposition"] == "known-unavailable" for r in rows),
        "slots": rows, "training_ready": False}


def _slot_totals(completions):
    return {key: sum(row["slot_accounting"][key] for row in completions)
            for key in ("slot_count", "acquired_queries", "reused_queries", "known_unavailable_slots")}


def _capture_api(storage_mode):
    _storage_schema(storage_mode)
    if storage_mode == DIRECT_STORAGE:
        from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
        return (direct.publish_direct_second_capture_plan, direct.capture_direct_seconds,
                direct.replay_direct_second_capture)
    return (capture.publish_second_capture_plan, capture.capture_seconds,
            capture.replay_second_capture)


def _verify_batch(root, terminal, consumed, plan):
    started = _json(root / "STARTED.json", terminal["started_sha256"])
    _require(started["consumed_before"] == consumed and started["batch"] == terminal["batch"]
             and started["plan_sha256"] == terminal["plan_sha256"], "Batch counter lineage changed")
    identities = tuple(CorpusIdentity(row["fixed_slot_ticker"], row["provider_ticker"],
        AliasIdentityRef(**row["identity"]), None if row["alias"] is None else capture._alias_route_from_dict(row["alias"]),
        None if row.get("successor") is None else SecondSuccessorRoute.from_dict(row["successor"]),
        None if row.get("reviewed_class") is None else SecondReviewedClassRoute.from_dict(row["reviewed_class"]))
        for row in started["identity_routes"])
    unavailable = () if plan.get("slot_accounting") is None else tuple(
        _prelisting_type().from_dict(row) for row in started["unavailable_routes"])
    queries, _, proofs, slots = _slot_queries(plan, terminal["batch"], identities, unavailable)
    _require(proofs == started["identity_proofs"], "Exact-date identity evidence changed")
    if slots is not None:
        _require(io.canonical(slots) == io.canonical(started["slot_proofs"]),
                 "Prelisting slot evidence changed")
    expected = {(q.ticker, q.start_ms, q.end_ms) for q in queries}
    reused = set()
    for item in started["reused"]:
        keys, _ = _corpus_reuse(CaptureReuse(**{**item, "root": Path(item["root"])}), plan)
        selected = keys & expected
        _require(selected and not selected & reused, "Completed reuse scope differs")
        reused.update(selected)
    fresh = [asdict(q) for q in queries if (q.ticker, q.start_ms, q.end_ms) not in reused]
    _require(bool(fresh) == (terminal["capture"] is not None), "Completed query scope differs")
    if slots is not None:
        _require(io.canonical(terminal["slot_accounting"]) == io.canonical(_slot_coverage(slots, reused)),
                 "Completed fixed-slot accounting differs")
        if not fresh:
            _require(terminal["packed_index_sha256"] is None
                     and not os.path.lexists(root / "capture") and not os.path.lexists(root / "packed"),
                     "Unavailable/reused slots cannot create capture artifacts")
    actual = {"http_requests": 0, "raw_response_bytes": 0}
    if terminal["capture"] is not None:
        ref = terminal["capture"]
        _require(_json(root / "capture" / "plan.json", ref["plan_sha256"])["queries"] == fresh,
                 "Completed capture query population changed")
        _, _, replay_capture = _capture_api(_storage(plan))
        replay = replay_capture(root=root / "capture", plan_sha256=ref["plan_sha256"],
                                completion_sha256=ref["completion_sha256"])
        actual = {"http_requests": replay["page_count"], "raw_response_bytes": replay["raw_response_bytes"]}
        if _storage(plan) == DIRECT_STORAGE:
            _require(replay["packed_index_sha256"] == terminal["packed_index_sha256"]
                     and not os.path.lexists(root / "packed"), "Direct packed capture identity differs")
        else:
            packed = verify_packed_second_store(root=root / "packed", index_sha256=terminal["packed_index_sha256"])
            provenance = packed["source_provenance"]
            _require(provenance["capture_root"] == str(root / "capture")
                     and provenance["plan_sha256"] == ref["plan_sha256"]
                     and provenance["capture_complete_sha256"] == ref["completion_sha256"]
                     and packed["capture_count"] == replay["query_count"]
                     and packed["page_count"] == replay["page_count"]
                     and packed["raw_response_bytes"] == replay["raw_response_bytes"],
                     "Packed store belongs to a different capture")
    actual["retained_bytes"] = _retained_bytes(root) + 65536
    _require(terminal["schema"] == _schema(plan) + "-batch-complete"
             and terminal["training_ready"] is False and terminal["batch_complete"] is True,
             "Batch is not completed preparation")
    _require(terminal["actual"] == actual and all(terminal["consumed_after"][key] == consumed[key] + actual[key]
                 and 0 <= terminal["actual"][key] <= started["reservation"][key] for key in consumed),
             "Batch accounting differs")


def _verify_tranche(root, plan, terminal):
    _require(terminal["schema"] == _schema(plan) + "-complete"
             and terminal["range_complete"] is True and terminal["training_ready"] is False
             and terminal["training_authorized"] is False, "Tranche storage or completion claims differ")
    registry = CorpusRegistry(Path(plan["registry_root"]), plan["registry_sha256"])
    batches = list(islice(registry.batches(plan["start_batch"]), plan["batch_count"]))
    consumed = dict(plan["consumed_before"])
    hashes, completions = {}, []
    for batch in batches:
        target = root / "batches" / batch["batch_id"]
        hashes[batch["batch_id"]] = io.digest(io.read_regular(target / "COMPLETE.json", MAX_METADATA))
        completed = _json(target / "COMPLETE.json")
        _require(completed["batch"] == batch and completed["plan_sha256"] == terminal["plan_sha256"],
                 "Predecessor batch range changed")
        _verify_batch(target, completed, consumed, plan)
        completions.append(completed)
        consumed = completed["consumed_after"]
    _require(hashes == terminal["batch_completion_sha256"] and consumed == terminal["consumed"]
             and terminal["next_batch"] == batches[-1]["ordinal"] + 1,
             "Predecessor global accounting changed")
    if plan.get("slot_accounting") is not None:
        _require(io.canonical(terminal["slot_totals"]) == io.canonical(_slot_totals(completions)),
                 "Tranche fixed-slot accounting differs")


def run_corpus_batch(*, root: Path, plan_sha256: str, identities: tuple[CorpusIdentity, ...],
                     api_key: str, capacity_snapshot: dict, reuse: tuple[CaptureReuse, ...] = (),
                     unavailable: tuple = ()):
    """Advance exactly one batch. No retry, even after a completed HTTP phase.

    Reuse proofs and exact-day identity routes must be supplied for this batch.
    An existing completed prefix is reopened before any new acquisition. The
    API key only reaches the existing bounded provider capture implementation.
    """
    plan = _json(root / "plan.json", plan_sha256)
    with _owner(Path(plan["owner"])):
        _admit_plan(root, plan, plan_sha256)
        storage_mode = _storage(plan)
        schema = _schema(plan)
        _require(plan["implementation_sha256"] == _implementation(
                     storage_mode, plan.get("slot_accounting"), plan.get("alias_routing"), plan.get("class_routing")),
                 "Corpus code changed")
        registry = CorpusRegistry(Path(plan["registry_root"]), plan["registry_sha256"])
        limits = CorpusLimits(**plan["limits"])
        limits.validate()
        consumed = dict(plan["consumed_before"])
        descriptors = list(islice(registry.batches(plan["start_batch"]), plan["batch_count"]))
        _require({p.name for p in (root / "batches").iterdir()} <= {d["batch_id"] for d in descriptors},
                 "Unexpected tranche batch population")
        for batch in descriptors:
            target = root / "batches" / batch["batch_id"]
            if os.path.lexists(target):
                _require((target / "COMPLETE.json").is_file(), "Consumed batch is ambiguous; never resubmit")
                terminal = _json(target / "COMPLETE.json")
                _require(terminal["plan_sha256"] == plan_sha256 and terminal["batch"] == batch,
                         "Completed batch differs")
                _verify_batch(target, terminal, consumed, plan)
                consumed = terminal["consumed_after"]
                continue
            break
        else:
            _require((root / "COMPLETE.json").is_file(), "Tranche terminal missing; no automatic repair")
            complete = _json(root / "COMPLETE.json")
            _verify_tranche(root, plan, complete)
            return complete
        queries, aliases, identity_proofs, slots = _slot_queries(plan, batch, identities, unavailable)
        reused_keys = set()
        for ref in reuse:
            keys, _ = _corpus_reuse(ref, plan)
            selected = keys & {(q.ticker, q.start_ms, q.end_ms) for q in queries}
            _require(selected and not selected & reused_keys, "Conflicting or irrelevant reuse")
            reused_keys.update(selected)
        fresh = tuple(q for q in queries if (q.ticker, q.start_ms, q.end_ms) not in reused_keys)
        reserve = {"http_requests": len(fresh) * capture.MAX_PAGES,
                   # The legacy transport checks its cumulative limit after
                   # receiving one bounded page. Charge that failed-page
                   # overshoot too; it is not permission to preserve >128 MB.
                   "raw_response_bytes": (capture.MAX_BYTES +
                       (0 if storage_mode == DIRECT_STORAGE else io.MAX_PAGE_BYTES)) if fresh else 0,
                   "retained_bytes": (DIRECT_RETAINED_RESERVATION if storage_mode == DIRECT_STORAGE
                       else RETAINED_RESERVATION) if fresh else 1024 * 1024}
        for key, value in reserve.items():
            _require(consumed[key] + value <= getattr(limits, "maximum_" + key), "Global corpus bound exhausted")
        _capacity(capacity_snapshot, limits, reserve["retained_bytes"],
                  DIRECT_INODE_RESERVATION if storage_mode == DIRECT_STORAGE else INODE_RESERVATION)
        started = {"plan_sha256": plan_sha256, "batch": batch,
            "reservation": reserve, "consumed_before": consumed, "capacity": capacity_snapshot,
            "identity_proofs": identity_proofs, "identity_routes": [i.to_dict() for i in identities],
            "reused": [r.to_dict() for r in reuse],
            "started_at_ns": time.time_ns(), "retry_allowed": False, "training_ready": False}
        if slots is not None:
            started.update(unavailable_routes=[d.to_dict() for d in unavailable], slot_proofs=slots)
            _require(len(io.canonical(started)) + 65536 <= reserve["retained_bytes"],
                     "Slot metadata exceeds retained reservation")
        target.mkdir(mode=0o700)
        started_sha = _write(target / "STARTED.json", started)
        acquired, packed, actual = None, None, {"http_requests": 0, "raw_response_bytes": 0}
        if fresh:
            selected_aliases = tuple(a for a in aliases if a.provider_ticker in {q.ticker for q in fresh})
            publish, acquire, replay_capture = _capture_api(storage_mode)
            capture_sha = publish(root=target / "capture", queries=fresh, alias_routes=selected_aliases)
            acquire(root=target / "capture", plan_sha256=capture_sha, api_key=api_key)
            complete_sha = io.digest(io.read_regular(target / "capture" / "COMPLETE.json", MAX_METADATA))
            replay = replay_capture(root=target / "capture", plan_sha256=capture_sha,
                                    completion_sha256=complete_sha)
            if storage_mode == DIRECT_STORAGE:
                packed = {"index_sha256": replay["packed_index_sha256"]}
            else:
                packed = pack_verified_second_capture(root=target / "capture", plan_sha256=capture_sha,
                                                       completion_sha256=complete_sha, output=target / "packed")
                verify_packed_second_store(root=target / "packed", index_sha256=packed["index_sha256"])
            acquired = {"plan_sha256": capture_sha, "completion_sha256": complete_sha}
            actual = {"http_requests": replay["page_count"], "raw_response_bytes": replay["raw_response_bytes"]}
        actual["retained_bytes"] = _retained_bytes(target) + 65536
        _require(all(actual[k] <= reserve[k] for k in actual), "Completed batch exceeded reserved budget")
        consumed = {k: consumed[k] + actual[k] for k in consumed}
        terminal = {"schema": schema + "-batch-complete", "plan_sha256": plan_sha256,
            "batch": batch, "started_sha256": started_sha, "capture": acquired,
            "packed_index_sha256": None if packed is None else packed["index_sha256"],
            "actual": actual, "consumed_after": consumed, "batch_complete": True, "training_ready": False}
        if slots is not None:
            terminal["slot_accounting"] = _slot_coverage(slots, reused_keys)
        _write(target / "COMPLETE.json", terminal)
        if batch == descriptors[-1]:
            complete = {"schema": schema + "-complete", "plan_sha256": plan_sha256,
                "next_batch": batch["ordinal"] + 1, "consumed": consumed, "range_complete": True,
                "batch_completion_sha256": {d["batch_id"]: io.digest(io.read_regular(
                    root / "batches" / d["batch_id"] / "COMPLETE.json", MAX_METADATA)) for d in descriptors},
                "training_ready": False, "training_authorized": False}
            if slots is not None:
                complete["slot_totals"] = _slot_totals([
                    _json(root / "batches" / d["batch_id"] / "COMPLETE.json") for d in descriptors])
            _write(root / "COMPLETE.json", complete)
        return terminal
