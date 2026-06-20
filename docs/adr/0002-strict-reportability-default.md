# ADR-0002: `reportable: true` requires the strict tier

**Status:** Accepted

## Context

A research result must not be labeled "reportable" unless causality, split containment, schemas, masks, costs,
baselines/stress coverage, manifests, invalid-return handling, AND a complete declared return basis all hold.
Earlier, strict return-basis validation was opt-in (`--strict-return-basis`), so a run with a missing or
partially-declared basis could still be labeled `reportable: true`. That made `reportable: true` mean two
different things — a real risk of treating a diagnostic run as a research result.

## Decision

`reportable: true` means exactly one thing: a **strict** artifact. `decision_framework.classify_reportability(summary)`
tiers every run:

- **`strict`** — passes the base contract AND a complete, agreeing, value-valid return basis on both the
  evaluation and dataset-manifest sides. The ONLY tier with `reportable == true`.
- **`legacy_diagnostic`** — base-reportable but the basis is not strictly complete. `reportable: false`, but
  `base_reportable: true` — still classified and readable, not broken.
- **`non_reportable`** — fails the base contract (basis contradiction/invalid value, missing baselines/stress,
  below-cash, over-concentration, or an upstream reportability-flags failure).

The verdict records `reportable`, `reportable_tier`, `base_reportable`, `strict_return_basis`,
`return_basis_policy`, and `reasons`. `--strict-return-basis` now only gates the **fail-fast preflight** (abort a
serious run early if a split lacks a complete basis); the final tiering is always strict-aware.

## Consequences

- Default-preserving via the tier split: an incomplete-basis run that previously got `reportable: true` is now
  `legacy_diagnostic` (downgraded a tier, still classified) — not deleted or errored.
- A freshly-built dataset (the builders emit a complete v2 basis on both the `.pt` and the manifest) classifies
  `strict`, so a correct current run is unaffected.
- `validate_reportable_summary` is unchanged (the base/strict checks it already did); `classify_reportability`
  composes it. Enforced by `tests/test_decision_framework.py::...classify_reportability_tiers`.
- The flag channel is permissive only on the flag itself (an absent reportability section defaults to "no flag
  failure"); the base and strict contracts still gate everything.
