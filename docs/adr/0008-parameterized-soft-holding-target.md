# ADR-0008: Parameterized soft holding target with a generic three-session default

**Status:** Accepted framework contract; legacy Hold-30 protocols remain explicit

**Date:** 2026-08-15

## Context

QuantTrade historically encoded an approximately 30-session soft holding prior
in Hold-30-specific model, action, accounting, and protocol surfaces. A generic
framework needs a configurable target and a shorter default, without silently
changing immutable Hold-30 experiments or conflating holding duration with a
return-prediction horizon.

Simply replacing 30 with 3 in the legacy age logit does not produce a neutral
expected duration of three sessions. It also risks changing label support,
purge geometry, age-state capacity, or cohort evaluation tails that happen to
use the number 30 for different reasons.

## Decision

New generic holding APIs consume an immutable `HoldTargetSpec` and default to:

```text
target_sessions = 3
age_cap_sessions = 60
prior_family = calibrated-logistic-v1
hard_minimum_hold = false
```

The generic logistic hazard location is deterministically calibrated so the
finite-age survival sum equals `target_sessions`. The target remains a soft
prior: early release is legal and there is no minimum-hold action mask.

Existing Hold-30 APIs remain compatibility wrappers around:

```text
target_sessions = 30
age_cap_sessions = 60
prior_family = legacy-hold30-v1
```

Holding target, prediction horizon, label support, purge, age cap, and cohort
evaluation horizon are independent configuration concepts. No subsystem may
derive one from another without an explicit protocol rule.

New artifacts bind the resolved holding specification and reject target
mismatch even when parameter shapes match. Old Hold-30 schemas without the new
field resolve to legacy target 30; new schemas fail closed when the field is
absent.

## Consequences

- `HoldTargetSpec()` means a neutral expected three earned sessions.
- Historical `hold30_*`, `Hold30*`, and immutable M03R contracts retain target
  30 unless a new scientific generation explicitly changes them.
- M03R-v16 binds legacy target 30, 30-session common label support, and a
  truncated 30-return survival-value diagnostic. The generic default does not
  reinterpret it.
- New generic policy/environment/replay surfaces must carry the same resolved
  specification receipt end to end before multi-target replay is permitted.
- A fixed target-3 cohort earns one decision/fill return plus two tail returns;
  this rule does not change prediction or label horizons.

## Enforced by

- `tests/test_hold_target_protocol.py`
- `tests/test_top2000_m03r_v16_initial_state.py`
- `tests/test_top2000_m03r_v16_checkpoint.py`
- `tests/test_hold30_alpha_m03r_v16_top2000_dev_protocol.py`
