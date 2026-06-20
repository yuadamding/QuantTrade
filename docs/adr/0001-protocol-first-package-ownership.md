# ADR-0001: Protocol-first package ownership with an enforced layer DAG

**Status:** Accepted

## Context

QuantTrade is a research framework, not live trading. Its correctness depends on semantics (causality, masks,
costs, return basis, reportability) being defined in *one* place rather than re-derived across scripts, loaders,
manifests, validators, envs, and docs. Duplicated semantics is the project's main long-term risk: a future
contributor changes one path and silently leaves another stale.

## Decision

Implementation lives in the `src/rl_quant` package, organized into layers that form a strict dependency DAG. A
package may import only *lower* (more foundational) layers, never a higher one. The canonical order:

```
protocol < data_sources < execution < features < models < reportability < datasets < evaluation < envs < training < workflows
```

`protocol/` is the lowest layer and is stdlib-only (no torch): it owns reusable contracts (decision-tensor
masks, the action-return basis, the reportability grid). A new subpackage must be placed in the order.

`TYPE_CHECKING`-only imports are exempt (they create no runtime coupling).

## Consequences

- The layering is **enforced in CI** by `tests/test_import_boundaries.py` (AST scan; a lower→higher runtime
  import fails the build; a genuinely necessary upward edge must be added to an explicit, justified allow-list,
  currently empty).
- Shared utilities that several layers need are pushed *down* to the layer that owns them (e.g. the
  action-return basis moved into `protocol/`), not duplicated upward.
- Backward-compatibility shims (e.g. `rl_quant.decision_framework` re-exporting
  `rl_quant.evaluation.decision_framework`) keep old import paths working during migration; new code should use
  the canonical path.
