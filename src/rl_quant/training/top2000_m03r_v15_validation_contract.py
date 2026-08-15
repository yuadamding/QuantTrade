"""Dependency-light checkpoint-selection receipts for M03R-v15.

This module is imported by the Seadragon host-side lifecycle. Keep it free
of torch, pyarrow, market-data adapters, and model/runtime imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_SHA256,
)

M03R_V15_CHECKPOINT_SELECTION_SCHEMA = (
    "rl-quant.top2000-dev.m03r-v15-checkpoint-selection-v1"
)


class M03RV15ValidationContractError(ValueError):
    """A dependency-light validation receipt drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV15ValidationContractError(
            f"{name} is not a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class M03RV15CheckpointSelectionReceipt:
    setting_index: int
    fold_index: int
    selected_epoch_index: int
    selected_model_state_sha256: str
    selected_validation_receipt_sha256: str
    candidate_validation_receipt_sha256: tuple[str, ...]
    selection_rule: str = M03R_V15_PREDICTIVE_SPEC.checkpoint_selection_rule
    qualification_tail_accessed: bool = False
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    schema: str = M03R_V15_CHECKPOINT_SELECTION_SCHEMA

    def validate(self) -> None:
        if (
            self.setting_index not in range(2)
            or self.fold_index not in range(6)
            or self.selected_epoch_index
            not in range(M03R_V15_PREDICTIVE_SPEC.training_epochs)
            or len(self.candidate_validation_receipt_sha256)
            != M03R_V15_PREDICTIVE_SPEC.training_epochs
            or self.selected_validation_receipt_sha256
            != self.candidate_validation_receipt_sha256[
                self.selected_epoch_index
            ]
            or self.selection_rule
            != M03R_V15_PREDICTIVE_SPEC.checkpoint_selection_rule
            or self.qualification_tail_accessed
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or self.schema != M03R_V15_CHECKPOINT_SELECTION_SCHEMA
        ):
            raise M03RV15ValidationContractError(
                "v15 checkpoint selection drifted"
            )
        _digest("selected_model_state_sha256", self.selected_model_state_sha256)
        _digest(
            "selected_validation_receipt_sha256",
            self.selected_validation_receipt_sha256,
        )
        for value in self.candidate_validation_receipt_sha256:
            _digest("candidate_validation_receipt_sha256", value)

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


__all__ = [
    "M03R_V15_CHECKPOINT_SELECTION_SCHEMA",
    "M03RV15CheckpointSelectionReceipt",
    "M03RV15ValidationContractError",
]
