"""Protocol layer: enforce the decision-tensor contract independently of any builder/trainer.

The core invariant (docs/decision_tensor_protocol.md): realized-outcome / future fields are LABELS and must
NEVER be model inputs. The builder already guards this inline; this lifts it into a reusable, slightly
stronger validator so (a) trainers can require a VALIDATED key split rather than trusting an arbitrary dict,
and (b) architecture tests can assert no workflow leaks a label/future field into the model. Pure, stdlib
only; changes no data, number, or training behavior."""

from __future__ import annotations

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


def validate_decision_tensor_payload(
    payload: Mapping, manifest: Mapping | None = None
) -> tuple[bool, tuple[str, ...]]:
    """Trainer-facing entry: pull the three key lists from a payload (falling back to a manifest, mirroring
    the builder's guard) and validate the split. Require this before training/eval on a payload so a leaking
    or malformed split fails closed rather than silently feeding a label into the model."""
    fallback = manifest or {}

    def _keys(name: str) -> Sequence[str]:
        return payload.get(name, fallback.get(name, []))

    return validate_model_input_label_split(
        model_input_keys=_keys("model_input_keys"),
        label_keys=_keys("label_keys"),
        forbidden_model_input_keys=_keys("forbidden_model_input_keys"),
    )
