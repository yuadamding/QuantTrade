# ADR-0005: `scripts/` are thin wrappers; the package owns implementation

**Status:** Accepted (migration in progress, CI-tracked)

## Context

Historically many `scripts/*.py` held substantial implementation (parsers, dataset semantics, builders), and the
`qt` CLI dispatched into them via `runpy`. That gave two competing centers of gravity: a future contributor
could not tell whether the source of truth was the script, a package module, a manifest validator, or a loader.

## Decision

A script is a **thin wrapper** — it bootstraps `src/` onto `sys.path`, imports `main` from a
`rl_quant.workflows.commands.<name>` (or other package) module, and exits via `raise SystemExit(main())`. It
holds **no top-level function or class definitions** (business logic lives in the package). Command logic moves
into the package, with an argv-parameterizable `main(argv)` so it is testable and CLI-callable; path/default
logic uses the canonical `rl_quant.paths` helpers rather than per-script `Path(__file__).parents[...]` (which is
position-dependent and duplicative).

Migration is **phased and default-preserving, not a big-bang**: each script is migrated one at a time, every
existing path is preserved (the wrapper keeps `qt`/`python scripts/...` working), and the move is
behavior-preserving.

## Consequences

- Enforced by `tests/test_scripts_are_wrappers.py`: every script not in `LEGACY_NON_WRAPPER_SCRIPTS` must be a
  wrapper; the allowlist is the migration backlog and is **shrink-only** (a migrated script must be removed from
  it, and the test fails if a backlog entry has already become a wrapper or if a new non-wrapper script appears).
- Exemplars done: `validate_research_protocol`, `build_stock_covariate_silver_features`,
  `build_stock_second_silver_features` → `rl_quant.workflows.commands.*`.
- Follow-up (deferred): once enough commands live in the package, the CLI can dispatch directly to package
  command functions instead of `runpy`-ing the wrapper scripts (kept uniform via `runpy` meanwhile).
- Live/broker integration stays OUT of the core research package (a separate, guarded `rl_quant_live` package);
  the core never requires broker credentials or order-routing dependencies.
