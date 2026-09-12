"""Replay actual reference supplements and retain source-bound research inputs.

No issuer/ticker continuity, feature eligibility, native source authority or
historical availability is invented. This is an explicit pretraining boundary.
"""

from __future__ import annotations

import gzip
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from rl_quant.data_sources.massive import qt200_reference_supplement_v1 as supplement
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.qt200_aggregate_pilot_v1 import _query_rows, normalize_bar


def compare_issue(row: dict, reference: dict) -> dict:
    """A matching CIK/name never substitutes for a stable share-class ID."""
    matching, conflicting, missing = [], [], []
    for field in ("composite_figi", "share_class_figi"):
        observed, expected = row.get(field), reference.get("current_" + field)
        if not isinstance(observed, str) or not observed or not isinstance(expected, str) or not expected:
            missing.append(field)
        elif observed == expected:
            matching.append(field)
        else:
            conflicting.append(field)
    classification = ("conflicting_issue_identifiers" if conflicting else
                      "exact_date_issue_identifiers_match_only" if matching else "unknown_issue_identity")
    return dict(classification=classification, matching=matching, conflicting=conflicting,
                unavailable=missing, issuer_used_as_issue_identity=False,
                continuous_interval_inferred=False, training_eligible=False)


def economic_problems(product: str, row: dict) -> list[str]:
    problems = []
    fields = ("split_from", "split_to") if product == "splits" else ("cash_amount",)
    for field in fields:
        try:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            number = Decimal(str(value))
            if not number.is_finite() or number <= 0:
                raise ValueError
        except (InvalidOperation, ValueError):
            problems.append("invalid_" + field)
    if not isinstance(row.get("ticker"), str) or not row["ticker"]:
        problems.append("missing_ticker")
    if product == "dividends":
        currency = row.get("currency")
        if not isinstance(currency, str) or currency.upper() != "USD":
            problems.append("non_usd_or_absent_currency")
        for field in ("pay_date", "record_date", "declaration_date"):
            value = row.get(field)
            try:
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError:
                problems.append("missing_or_unclassified_" + field)
    return problems


def reconcile_supplement(*, root: Path, plan_sha256: str, completion_sha256: str,
                         evidence: Path, evidence_sha256: str, output: Path) -> dict:
    args = dict(root=root, plan_sha256=plan_sha256, completion_sha256=completion_sha256,
                evidence=evidence, evidence_sha256=evidence_sha256)
    before = supplement.replay_supplement(**args)
    document = supplement.load_evidence(evidence, evidence_sha256)
    queries = supplement.queries_from_evidence(document)
    entities = {r["qt200_ticker"]: r["current_reference_evidence"] for r in document["security_resolutions"]}
    completed = transport.parse_json(transport.read_regular(root / "COMPLETE.json", 1048576))
    bars, references, events, summaries = [], [], [], []
    aliases, empty, invalid = [], [], []
    target_symbols = set(supplement.SYMBOLS) | {r[1] for r in supplement.ALIASES}
    for query, entry in zip(queries, completed["queries"], strict=True):
        seen_days, count = set(), 0
        for row, source in _query_rows(root, query, entry):
            count += 1
            provenance = dict(query=query.name, query_completion_sha256=entry["completion"]["sha256"],
                              raw_row_sha256=transport.digest(transport.canonical(row)), **source)
            if query.product == "day":
                try:
                    bar = normalize_bar(row, query, received_at_ns=source["received_at_ns"])
                    day = bar["session_date_et"]
                    if day in seen_days or (seen_days and day < max(seen_days)):
                        raise transport.ResearchCaptureError("Duplicate/unordered alias interval")
                    seen_days.add(day)
                    bars.append(dict(**bar, qt200_candidate=query.target, source=provenance,
                                     automatic_stitching=False, training_eligible=False))
                except (ValueError, OverflowError) as exc:
                    invalid.append(dict(source=provenance, error_type=type(exc).__name__))
            elif query.product.startswith("reference-"):
                facts = compare_issue(row, entities[query.target])
                references.append(dict(qt200_candidate=query.target, query_date=query.start,
                    provider_row=row, source=provenance, **facts))
                if (facts["classification"] == "exact_date_issue_identifiers_match_only"
                        and row.get("ticker") != query.target):
                    aliases.append(dict(qt200_candidate=query.target, observed_ticker=row["ticker"],
                        observed_date=query.start, source=provenance, continuous_interval_inferred=False,
                        bar_query_or_stitching_authorized=False))
            else:
                # Preserve the complete scoped response; panel filtering is an
                # annotation, never silent deletion of economic observations.
                events.append(dict(product=query.product, provider_row=row, source=provenance,
                    panel_ticker_candidate=row.get("ticker") in target_symbols,
                    problems=economic_problems(query.product, row), accounting_authorized=False))
        if count != entry["result_count"]:
            raise transport.ResearchCaptureError("Source result population differs")
        summaries.append(dict(query=query.name, product=query.product, target=query.target,
            returned_rows=count, completion_sha256=entry["completion"]["sha256"]))
        if count == 0:
            empty.append(dict(query=query.name, target=query.target, product=query.product,
                              absence_is_tradability_or_event_authority=False))
    after = supplement.replay_supplement(**args)
    if before != after or document != supplement.load_evidence(evidence, evidence_sha256):
        raise transport.ResearchCaptureError("Source evidence changed during reconciliation")
    output.mkdir(mode=0o700)
    payloads = {
        "alias-daily-bars.jsonl.gz": b"".join(transport.canonical(r) for r in bars),
        "dated-reference-observations.jsonl.gz": b"".join(transport.canonical(r) for r in references),
        "economic-tail-observations.jsonl.gz": b"".join(transport.canonical(r) for r in events),
        "coverage-and-issues.json.gz": transport.canonical(dict(queries=summaries,
            empty_queries=empty, invalid_alias_rows=invalid, candidate_alias_observations=aliases)),
    }
    files = {name: transport.write_once(output / name, gzip.compress(body, compresslevel=9, mtime=0))
             for name, body in payloads.items()}
    counts = dict(Counter(r["classification"] for r in references))
    result = dict(schema="rl-quant.qt200-supplement-reconciliation-v1",
        plan_sha256=plan_sha256, capture_sha256=completion_sha256, identity_evidence_sha256=evidence_sha256,
        capture_replayed_without_writes=True, queries=len(summaries), files=files,
        alias_bar_rows=len(bars), invalid_alias_rows=len(invalid), dated_reference_rows=len(references),
        reference_classification_counts=counts, economic_rows=len(events),
        panel_candidate_economic_rows=sum(r["panel_ticker_candidate"] for r in events),
        economic_records_with_problems=sum(bool(r["problems"]) for r in events),
        candidate_alias_observations=len(aliases), empty_queries=len(empty),
        observation_schema_accepted=bool(bars or references or events) and not invalid, source_unchanged=True,
        complete_issue_history_qualified=False, corporate_action_accounting_qualified=False,
        point_in_time_qualified=False, native_v5_qualified=False, training_ready=False)
    transport.write_once(output / "COMPLETE.json", transport.canonical(result))
    return result
