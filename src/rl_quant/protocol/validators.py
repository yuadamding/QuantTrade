"""Protocol layer: enforce the decision-tensor contract independently of any builder/trainer.

The core invariant (docs/decision_tensor_protocol.md): realized-outcome / future fields are LABELS and must
NEVER be model inputs. The builder already guards this inline; this lifts it into a reusable, slightly
stronger validator so (a) trainers can require a VALIDATED key split rather than trusting an arbitrary dict,
and (b) architecture tests can assert no workflow leaks a label/future field into the model. Pure, stdlib
only; changes no data, number, or training behavior."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def validate_model_input_label_split(
    *,
    model_input_keys: Sequence[str],
    label_keys: Sequence[str],
    forbidden_model_input_keys: Sequence[str],
) -> tuple[bool, tuple[str, ...]]:
    """Validate the model-input / label / forbidden-key split. Returns (ok, issues). Hard rules: model_input
    is non-empty and duplicate-free; forbidden is non-empty; NO model input is a forbidden key (the
    anti-leakage rule -- a realized/future field must never feed the model); and every forbidden key is
    declared as a label (so the forbidden set is a coherent subset of the labels)."""
    mi = list(model_input_keys)
    mi_set = set(mi)
    labels = set(label_keys)
    forbidden = set(forbidden_model_input_keys)
    issues: list[str] = []
    if not mi:
        issues.append("model_input_keys is empty")
    if len(mi) != len(mi_set):
        issues.append(f"model_input_keys has duplicates: {sorted({k for k in mi if mi.count(k) > 1})}")
    if not forbidden:
        issues.append("forbidden_model_input_keys is empty")
    leaked = sorted(mi_set & forbidden)
    if leaked:
        issues.append(f"model_input_keys leak forbidden (label/future) keys: {leaked}")
    orphan_forbidden = sorted(forbidden - labels)
    if orphan_forbidden:
        issues.append(f"forbidden_model_input_keys not declared as label_keys: {orphan_forbidden}")
    return (not issues, tuple(issues))


def assert_no_model_input_leakage(
    *,
    model_input_keys: Sequence[str],
    label_keys: Sequence[str],
    forbidden_model_input_keys: Sequence[str],
) -> None:
    """Fail closed: raise ValueError if the split violates the contract (for builders/trainers)."""
    ok, issues = validate_model_input_label_split(
        model_input_keys=model_input_keys,
        label_keys=label_keys,
        forbidden_model_input_keys=forbidden_model_input_keys,
    )
    if not ok:
        raise ValueError("decision-tensor contract violation: " + "; ".join(issues))


def _to_rows(value: object) -> list[list[object]]:
    """Coerce a 2-D tensor (anything with .tolist()) or a nested sequence to a list-of-lists, so the
    validators stay torch-free + work on either representation."""
    listed = value.tolist() if hasattr(value, "tolist") else value
    return [list(row) for row in listed]  # type: ignore[union-attr]


def validate_invalid_returns_are_nan(
    action_returns: object, valid_mask: object, *, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """The decision-tensor honesty rule (docs/decision_tensor_protocol.md): an action's outcome is observable
    -- a FINITE return -- IF AND ONLY IF the action is marked VALID. An INVALID action must carry a non-finite
    (NaN) return, never a silent 0/finite value (which would let a missing outcome masquerade as a flat
    result); and a VALID action must carry a finite return (a 'valid' outcome cannot be NaN). The builders
    enforce this inline at build time; this lifts it into a reusable validator (mirroring
    ``validate_model_input_label_split``) so any payload can be checked uniformly. ``action_returns`` and
    ``valid_mask`` are [rows][actions] -- tensors (``.tolist()``) or nested sequences. Returns (ok, issues),
    issues capped at ``max_issues``. Pure; stdlib only."""
    returns = _to_rows(action_returns)
    mask = _to_rows(valid_mask)
    if len(returns) != len(mask):
        return (False, (f"action_returns has {len(returns)} rows but valid_mask has {len(mask)}",))
    issues: list[str] = []
    for t, (rrow, mrow) in enumerate(zip(returns, mask)):
        if len(rrow) != len(mrow):
            issues.append(f"row {t}: action_returns width {len(rrow)} != valid_mask width {len(mrow)}")
        else:
            for a, (ret, valid) in enumerate(zip(rrow, mrow)):
                finite = isinstance(ret, (int, float)) and not isinstance(ret, bool) and math.isfinite(float(ret))
                if bool(valid) and not finite:
                    issues.append(f"row {t} action {a}: marked VALID but return is not finite ({ret!r})")
                elif not bool(valid) and finite:
                    issues.append(f"row {t} action {a}: marked INVALID but return is finite ({ret!r}); must be NaN")
                if len(issues) > max_issues:
                    break  # bound a single wide row (e.g. ~2000 actions) to the cap, not 2000 issues
        if len(issues) > max_issues:
            issues = issues[:max_issues]
            issues.append("... (further issues truncated)")
            break
    return (not issues, tuple(issues))


def assert_invalid_returns_are_nan(action_returns: object, valid_mask: object) -> None:
    """Fail closed: raise ValueError if the valid-mask / finite-return honesty contract is violated."""
    ok, issues = validate_invalid_returns_are_nan(action_returns, valid_mask)
    if not ok:
        raise ValueError("decision-tensor contract violation (invalid-returns-must-be-NaN): " + "; ".join(issues))


def _to_list(value: object) -> list[object]:
    """Coerce a 1-D tensor (anything with .tolist()) or a sequence to a list, keeping the layer torch-free."""
    return list(value.tolist() if hasattr(value, "tolist") else value)  # type: ignore[union-attr]


def validate_causal_timestamp_chain(
    chain: Sequence[object], *, names: Sequence[str] | None = None, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """Point-in-time causality (docs/decision_tensor_protocol.md): a sequence of equal-length per-row
    timestamp arrays, given in CAUSAL ORDER (e.g. context_available_until, decision_ts, entry_execution_ts,
    reward_end_ts, exit_execution_ts), must be NON-DECREASING within every row -- ``chain[i][row] <=
    chain[i+1][row]``. A decreasing step is LOOK-AHEAD: a later stage is timestamped before an earlier one,
    so the model could have seen the future. Each array is a tensor (``.tolist()``) or sequence; all must be
    the same length, and every timestamp must be finite. Returns (ok, issues), capped at ``max_issues``.
    Pure; stdlib only -- the reusable, vectorized-over-rows counterpart of the decision-log timestamp chain."""
    arrays = [_to_list(a) for a in chain]
    stage = list(names) if names is not None else [f"stage[{i}]" for i in range(len(arrays))]
    if len(arrays) < 2:
        return (True, ())  # zero/one stage: nothing to order
    n_rows = len(arrays[0])
    for i, a in enumerate(arrays):
        if len(a) != n_rows:
            return (False, (f"{stage[i]} has {len(a)} rows but {stage[0]} has {n_rows}",))

    def _finite(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    issues: list[str] = []
    for row in range(n_rows):
        for i in range(len(arrays) - 1):
            lo, hi = arrays[i][row], arrays[i + 1][row]
            if not _finite(lo) or not _finite(hi):
                issues.append(f"row {row}: non-finite timestamp ({stage[i]}={lo!r}, {stage[i + 1]}={hi!r})")
            elif float(lo) > float(hi):
                issues.append(f"row {row}: {stage[i]} ({lo!r}) > {stage[i + 1]} ({hi!r}) -- look-ahead")
        if len(issues) > max_issues:
            issues = issues[:max_issues]
            issues.append("... (further issues truncated)")
            break
    return (not issues, tuple(issues))


def assert_causal_timestamp_chain(chain: Sequence[object], *, names: Sequence[str] | None = None) -> None:
    """Fail closed: raise ValueError if the per-row timestamp chain is not non-decreasing (look-ahead)."""
    ok, issues = validate_causal_timestamp_chain(chain, names=names)
    if not ok:
        raise ValueError("decision-tensor contract violation (causal-timestamp-ordering): " + "; ".join(issues))


def validate_cash_contract(
    action_returns: object, valid_mask: object, *, cash_index: int = 0, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """The CASH-fallback contract (docs/decision_tensor_protocol.md): de-risking to the zero-notional CASH
    position is ALWAYS available and its outcome is ALWAYS observable, so the CASH action must be SELECTABLE
    (valid_mask True) AND carry a FINITE return on EVERY decision row. A row where CASH is masked off or has a
    non-finite return is a broken contract (the policy could be trapped with no safe fallback, or CASH's
    outcome is unscorable). ``action_returns`` and ``valid_mask`` are [rows][actions] (tensors or nested
    sequences). Returns (ok, issues); raises ValueError for a malformed ``cash_index``. Pure; stdlib only."""
    if isinstance(cash_index, bool) or not isinstance(cash_index, int) or cash_index < 0:
        raise ValueError(f"cash_index must be a non-negative integer; got {cash_index!r}.")
    returns = _to_rows(action_returns)
    mask = _to_rows(valid_mask)
    if len(returns) != len(mask):
        return (False, (f"action_returns has {len(returns)} rows but valid_mask has {len(mask)}",))
    issues: list[str] = []
    for t, (rrow, mrow) in enumerate(zip(returns, mask)):
        if cash_index >= len(rrow) or cash_index >= len(mrow):
            issues.append(f"row {t}: cash_index {cash_index} out of range (returns width {len(rrow)}, mask width {len(mrow)})")
        else:
            ret = rrow[cash_index]
            finite = isinstance(ret, (int, float)) and not isinstance(ret, bool) and math.isfinite(float(ret))
            if not bool(mrow[cash_index]):
                issues.append(f"row {t}: CASH (index {cash_index}) is not selectable (valid_mask is False)")
            if not finite:
                issues.append(f"row {t}: CASH (index {cash_index}) return is not finite ({ret!r})")
        if len(issues) > max_issues:
            issues = issues[:max_issues]
            issues.append("... (further issues truncated)")
            break
    return (not issues, tuple(issues))


def assert_cash_contract(action_returns: object, valid_mask: object, *, cash_index: int = 0) -> None:
    """Fail closed: raise ValueError if CASH is not selectable-and-finite on every decision row."""
    ok, issues = validate_cash_contract(action_returns, valid_mask, cash_index=cash_index)
    if not ok:
        raise ValueError("decision-tensor contract violation (cash-fallback): " + "; ".join(issues))


def validate_label_mask_subset_of_decision_mask(
    label_mask: object, decision_mask: object, *, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """The ex-post label-valid mask must never mark an action scorable if the ex-ante decision mask says the
    policy could not select that action. Such labels are future observations for non-decision-valid actions and
    must be dropped before training/evaluation."""
    labels = _to_rows(label_mask)
    decisions = _to_rows(decision_mask)
    if len(labels) != len(decisions):
        return (False, (f"label_mask has {len(labels)} rows but decision_mask has {len(decisions)}",))
    issues: list[str] = []
    for t, (label_row, decision_row) in enumerate(zip(labels, decisions)):
        if len(label_row) != len(decision_row):
            issues.append(f"row {t}: label_mask width {len(label_row)} != decision_mask width {len(decision_row)}")
        else:
            for a, (label_valid, decision_valid) in enumerate(zip(label_row, decision_row)):
                if bool(label_valid) and not bool(decision_valid):
                    issues.append(f"row {t} action {a}: label_valid_mask is true while decision mask is false")
                if len(issues) > max_issues:
                    break  # bound a single wide row to the cap, not its full width
        if len(issues) > max_issues:
            issues = issues[:max_issues]
            issues.append("... (further issues truncated)")
            break
    return (not issues, tuple(issues))


def validate_decision_tensor_shapes(
    named_arrays: Mapping[str, object], *, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """All named decision-tensor arrays must agree on the ROW count (axis 0), and every 2-D array must be
    rectangular and agree on the ACTION count (axis 1). A mismatch means a row/action in one tensor has no
    counterpart in another (e.g. returns for T rows but a mask for T-1), which silently misaligns labels with
    inputs. ``named_arrays`` maps a name to a tensor (``.tolist()``) or nested sequence. Returns (ok, issues).
    Pure; stdlib only."""
    row_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    issues: list[str] = []
    for name, arr in named_arrays.items():
        rows = list(arr.tolist() if hasattr(arr, "tolist") else arr)  # type: ignore[union-attr]
        row_counts[name] = len(rows)
        widths = {len(r) for r in rows if isinstance(r, (list, tuple))}
        if len(widths) > 1:
            issues.append(f"{name}: ragged 2-D array (row widths {sorted(widths)})")
        elif widths:
            action_counts[name] = next(iter(widths))
    if len(set(row_counts.values())) > 1:
        issues.append(f"inconsistent row counts across arrays: {row_counts}")
    if len(set(action_counts.values())) > 1:
        issues.append(f"inconsistent action counts across 2-D arrays: {action_counts}")
    if len(issues) > max_issues:
        issues = [*issues[:max_issues], "... (further issues truncated)"]
    return (not issues, tuple(issues))


def assert_decision_tensor_shapes(named_arrays: Mapping[str, object]) -> None:
    """Fail closed: raise ValueError if the decision-tensor arrays disagree on row or action count."""
    ok, issues = validate_decision_tensor_shapes(named_arrays)
    if not ok:
        raise ValueError("decision-tensor contract violation (shape-consistency): " + "; ".join(issues))


def validate_action_mask(
    mask: object, *, require_row_selectable: bool = True, max_issues: int = 20
) -> tuple[bool, tuple[str, ...]]:
    """An action mask must be a rectangular 2-D [rows][actions] array of BOOLEAN entries (True/False or 0/1 --
    a fractional/other numeric entry is malformed), and -- when ``require_row_selectable`` (the contract for an
    ex-ante DECISION mask) -- every row must have at least one selectable action, or the policy is trapped
    with no legal move. (Pass ``require_row_selectable=False`` for an ex-post LABEL mask, where a row may have
    no scorable action.) Tensor (``.tolist()``) or nested sequence. Returns (ok, issues). Pure; stdlib only."""
    rows = [list(r) if isinstance(r, (list, tuple)) else r for r in (mask.tolist() if hasattr(mask, "tolist") else mask)]  # type: ignore[union-attr]
    issues: list[str] = []
    widths = {len(r) for r in rows if isinstance(r, list)}
    if len(widths) > 1:
        return (False, (f"action mask is ragged (row widths {sorted(widths)})",))
    for t, row in enumerate(rows):
        if not isinstance(row, list):
            issues.append(f"row {t}: mask row is not a sequence ({row!r}); expected [actions] booleans")
            continue
        # bool first (bool subclasses int); accept only bool or int 0/1. This rejects float 1.0/0.0/0.5
        # and int 2 — `value not in (0, 1)` would NOT, since 1.0 in (0, 1) is True.
        if any(not (isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))) for value in row):
            issues.append(f"row {t}: mask has non-boolean entries ({row!r})")
        if require_row_selectable and not any(bool(value) for value in row):
            issues.append(f"row {t}: no selectable action (every entry is False) -- the policy has no legal move")
        if len(issues) > max_issues:
            issues = [*issues[:max_issues], "... (further issues truncated)"]
            break
    return (not issues, tuple(issues))


def assert_action_mask(mask: object, *, require_row_selectable: bool = True) -> None:
    """Fail closed: raise ValueError if the action mask is non-rectangular, non-boolean, or (for a decision
    mask) leaves a row with no selectable action."""
    ok, issues = validate_action_mask(mask, require_row_selectable=require_row_selectable)
    if not ok:
        raise ValueError("decision-tensor contract violation (action-mask): " + "; ".join(issues))


def validate_decision_tensor_payload(
    payload: Mapping, manifest: Mapping | None = None, *,
    require_full_contract: bool = False, require_timestamps: bool = False, cash_index: int = 0,
) -> tuple[bool, tuple[str, ...]]:
    """Trainer-facing entry: enforce the decision-tensor contract on a payload, aggregating all violations.

    ALWAYS validates the model-input / label / forbidden-key SPLIT (the anti-leakage rule; pulls the three key
    lists from the payload, falling back to ``manifest``). ADDITIONALLY runs each tensor contract validator
    for whatever decision-tensor TENSORS are present in the payload:
      * invalid-returns-are-NaN  (action_returns vs the label-valid mask),
      * action-mask validity     (the ex-ante decision mask must leave a selectable action per row; the
                                   ex-post label mask need not),
      * label subset             (label_valid_mask must be a subset of decision_action_valid_mask),
      * the CASH fallback        (CASH selectable + finite on every row),
      * shape consistency        (all present [rows][actions] tensors agree),
      * the per-row causal chain  (decision_timestamps_ms <= next_timestamps_ms), when numeric ``*_ms``
                                   anchors are present.
    Tensor checks for ABSENT keys are SKIPPED, so a key-only payload (the legacy use) behaves exactly as
    before -- this is backward-compatible. With ``require_full_contract=True`` the core tensors
    (action_returns + a decision mask + a label mask) MUST be present, for reportable training/eval that must
    not silently skip the contract. With ``require_timestamps=True`` the causal anchors
    (decision_timestamps_ms AND next_timestamps_ms) MUST be present -- otherwise the causal-chain check
    silently does not run, so a reportable run could omit the timestamps and still pass the "full" contract;
    this flag closes that gap (it is independent of ``require_full_contract``, which governs only the core
    tensors). Returns (ok, issues). Mask aliases are resolved
    (decision_action_valid_mask | action_valid_mask; label_valid_mask | action_label_valid_mask)."""
    fallback = manifest or {}

    def _keys(name: str) -> Sequence[str]:
        return payload.get(name, fallback.get(name, []))

    ok, issues = validate_model_input_label_split(
        model_input_keys=_keys("model_input_keys"),
        label_keys=_keys("label_keys"),
        forbidden_model_input_keys=_keys("forbidden_model_input_keys"),
    )
    all_issues: list[str] = list(issues)

    def _present(*names: str) -> object | None:
        for n in names:
            value = payload.get(n)
            if value is not None:
                return value
        return None

    action_returns = _present("action_returns")
    decision_mask = _present("decision_action_valid_mask", "action_valid_mask")
    label_mask = _present("label_valid_mask", "action_label_valid_mask")

    if require_full_contract:
        for label, value in (("action_returns", action_returns), ("decision mask", decision_mask),
                             ("label mask", label_mask)):
            if value is None:
                all_issues.append(f"require_full_contract: payload is missing {label}")

    def _add(prefix: str, result: tuple[bool, tuple[str, ...]]) -> None:
        all_issues.extend(f"{prefix}: {m}" for m in result[1])

    if action_returns is not None and label_mask is not None:
        _add("invalid_returns_are_nan", validate_invalid_returns_are_nan(action_returns, label_mask))
    if decision_mask is not None:
        _add("decision_mask", validate_action_mask(decision_mask, require_row_selectable=True))
    if label_mask is not None:
        _add("label_mask", validate_action_mask(label_mask, require_row_selectable=False))
    if decision_mask is not None and label_mask is not None:
        _add("label_mask_subset", validate_label_mask_subset_of_decision_mask(label_mask, decision_mask))
    if action_returns is not None and decision_mask is not None:
        _add("cash_contract", validate_cash_contract(action_returns, decision_mask, cash_index=cash_index))
    # Shape-check the RESOLVED tensors (alias-aware), not hardcoded canonical keys -- otherwise a payload
    # carrying only the ``action_label_valid_mask`` alias (no ``label_valid_mask``) would skip its shape check.
    shape_arrays = {
        name: value
        for name, value in (("action_returns", action_returns), ("decision_mask", decision_mask),
                            ("label_mask", label_mask))
        if value is not None
    }
    if len(shape_arrays) >= 2:
        _add("shapes", validate_decision_tensor_shapes(shape_arrays))
    decision_ts = _present("decision_timestamps_ms")
    next_ts = _present("next_timestamps_ms")
    if require_timestamps:
        for label, value in (("decision_timestamps_ms", decision_ts), ("next_timestamps_ms", next_ts)):
            if value is None:
                all_issues.append(f"require_timestamps: payload is missing {label}")
    if decision_ts is not None and next_ts is not None:
        _add("causal_chain", validate_causal_timestamp_chain([decision_ts, next_ts],
                                                              names=["decision_ts", "next_ts"]))

    return (not all_issues, tuple(all_issues))
