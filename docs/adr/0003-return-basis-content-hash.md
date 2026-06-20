# ADR-0003: A canonical `ReturnBasis` + content hash is the return-economics contract

**Status:** Accepted

## Context

The economic meaning of a dataset's `action_returns` — the weight semantics, return formula, clip bounds, fill
convention, and (v2) the precise entry/exit fill rules, execution latency, source bar interval, and price
source — was previously implicit, spread across builders and loaders. Two datasets with identical inputs but
different return economics could be indistinguishable, and a reportable result could fail to declare how its
returns were formed.

## Decision

`rl_quant.protocol.action_return_basis.ReturnBasis` is the single canonical object for the action-return basis.
It wraps the loose `action_return_*` fields (additive — it does not replace them), supports v1 (flat) and v2
(structured) shapes with version-aware completeness, validates declared values (finite/ordered clips,
non-negative integer latency, non-blank strings, recognized weight semantics / basis version), and computes a
stable `content_hash()` over the declared fields.

Each builder declares a complete, **truthful** basis (the recorded fields must describe what the code actually
computes), writes it into both the `.pt` payload and `dataset_manifest.json` from one source (so the two copies
cannot drift within a build), folds the *basis* (not its hash) into `source_manifest_hash` (so the provenance
hash captures the economics without depending on the hash algorithm), and persists `return_basis_content_hash`
as a first-class `DatasetManifest` field.

`DatasetManifest.validate()` requires a recorded `return_basis_content_hash` to match a complete, valid declared
basis — catching a stale or hand-edited hash. The reportability agreement check (ADR-0002) compares the eval-side
basis against the manifest-side basis.

## Consequences

- A change to the return economics changes the content hash and `source_manifest_hash`; the hashing algorithm
  is pinned by `tests/test_golden_artifacts.py` so an accidental change to it (which would silently stale every
  persisted hash) fails CI.
- The fill convention is honestly disclosed even when optimistic: the direct-hourly builder records a
  zero-latency decision-bar-close fill (`execution_latency_ms=0`) plus a known-limitation, rather than implying
  a more conservative fill than it performs. A latency-aware/executable basis variant is a future, separately
  versioned option (see ADR-0004 / the return-basis backlog).
