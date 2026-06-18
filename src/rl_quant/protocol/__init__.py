"""Protocol layer: the decision-tensor contract (mask / model-input-vs-label semantics) enforced as reusable
code, independent of any builder or trainer. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Phase 2 ships the model-input/label/forbidden-key anti-leakage
validators; the full DecisionTensorPayload loader and the trading_constraints contract re-export follow."""

from rl_quant.protocol.validators import (
    assert_cash_contract,
    assert_causal_timestamp_chain,
    assert_invalid_returns_are_nan,
    assert_no_model_input_leakage,
    validate_cash_contract,
    validate_causal_timestamp_chain,
    validate_decision_tensor_payload,
    validate_invalid_returns_are_nan,
    validate_model_input_label_split,
)

__all__ = [
    "assert_cash_contract",
    "assert_causal_timestamp_chain",
    "assert_invalid_returns_are_nan",
    "assert_no_model_input_leakage",
    "validate_cash_contract",
    "validate_causal_timestamp_chain",
    "validate_decision_tensor_payload",
    "validate_invalid_returns_are_nan",
    "validate_model_input_label_split",
]
