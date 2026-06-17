"""Protocol layer: the decision-tensor contract (mask / model-input-vs-label semantics) enforced as reusable
code, independent of any builder or trainer. Part of the protocol-first layered architecture
(see architecture_migration_plan.md). Phase 2 ships the model-input/label/forbidden-key anti-leakage
validators; the full DecisionTensorPayload loader and the trading_constraints contract re-export follow."""

from rl_quant.protocol.validators import (
    assert_no_model_input_leakage,
    validate_decision_tensor_payload,
    validate_model_input_label_split,
)

__all__ = [
    "assert_no_model_input_leakage",
    "validate_decision_tensor_payload",
    "validate_model_input_label_split",
]
