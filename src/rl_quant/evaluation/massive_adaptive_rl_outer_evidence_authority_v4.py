"""Create-only replay authority for hard-gated dual-ladder outer evidence V4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
import json
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.evaluation.massive_adaptive_outer_access_commitment_v1 import (
    MassiveAdaptiveOuterAccessCommitmentV1,
)
from rl_quant.evaluation.massive_adaptive_rl_fixed_control_outer_cost_ladder_v1 import (
    MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_cost_ladder_v1 import (
    MassiveAdaptiveRLOuterCostLadderAuthorityV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_authority_v3 import (
    MassiveAdaptiveRLOuterEvidenceAuthorityV3,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_evidence_v4 import (
    MassiveAdaptiveAuthenticatedRLOuterFoldV4,
    MassiveAdaptiveRLOuterEvidenceV4,
    build_massive_adaptive_authenticated_rl_outer_fold_v4,
    build_massive_adaptive_rl_outer_evidence_v4,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_forecast_archive_v1 import (
    MassiveAdaptiveRLOuterForecastArchiveV1,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v2 import (
    MassiveAdaptiveRLOuterPlanV2,
)
from rl_quant.evaluation.massive_adaptive_rl_outer_plan_v3 import (
    MassiveAdaptiveRLOuterPlanV3,
    MassiveAdaptiveRLOuterPlanV3Error,
    build_massive_adaptive_rl_outer_plan_v3,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    file_sha256,
    semantic_sha256,
)
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)


MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SCHEMA = (
    "rl-quant.massive-adaptive-rl-outer-evidence-authority-v4"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_DATASET = (
    "massive-adaptive-rl-outer-evidence-authority-v4"
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SCHEMA,
        "publication": "create-only-source-transaction",
        "generic_reload": "nonauthorizing",
        "replay": "hard-access-and-both-outer-cost-ladders",
    }
)
MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SPEC_SHA256 = semantic_sha256(
    {
        "primary": "replayed-outer-evidence-authority-v3",
        "outer_plan": "rebuilt-from-v2-commitment-and-gated-forecast",
        "ppo_cost_ladder": "replayed-frozen-target-authority",
        "static_cost_ladder": "replayed-frozen-target-authority",
        "additional_gate": "40bp-ppo-minus-static-nonnegative",
        "profitability_reporting": False,
        "lockbox": False,
    }
)


class MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(ValueError):
    """Committed V4 evidence failed hard-access or cost-ladder replay."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "adaptive RL V4 outer evidence artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceRecordV4:
    evidence_receipt_sha256: str
    evidence_v3_authority_receipt_sha256: str
    authenticated_fold_inventory_sha256: str
    authenticated_fold_receipts: tuple[str, ...]
    outer_plan_v2_receipts: tuple[str, ...]
    outer_access_commitment_receipts: tuple[str, ...]
    gated_outer_forecast_archive_receipts: tuple[str, ...]
    outer_plan_v3_receipts: tuple[str, ...]
    ppo_cost_ladder_authority_receipts: tuple[str, ...]
    fixed_control_cost_ladder_authority_receipts: tuple[str, ...]
    mean_high_cost_ppo_minus_fixed_control_log_return: float
    passed_gate_names: tuple[str, ...]
    failed_gate_names: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        count = len(self.authenticated_fold_receipts)
        if (
            not count
            or any(
                len(values) != count
                for values in (
                    self.outer_plan_v2_receipts,
                    self.outer_access_commitment_receipts,
                    self.gated_outer_forecast_archive_receipts,
                    self.outer_plan_v3_receipts,
                    self.ppo_cost_ladder_authority_receipts,
                    self.fixed_control_cost_ladder_authority_receipts,
                )
            )
            or any(
                len(set(values)) != count
                for values in (
                    self.authenticated_fold_receipts,
                    self.outer_plan_v2_receipts,
                    self.outer_access_commitment_receipts,
                    self.gated_outer_forecast_archive_receipts,
                    self.outer_plan_v3_receipts,
                    self.ppo_cost_ladder_authority_receipts,
                    self.fixed_control_cost_ladder_authority_receipts,
                )
            )
            or set(self.passed_gate_names) & set(self.failed_gate_names)
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
                "adaptive RL V4 outer evidence record differs"
            )
        for value in (
            self.evidence_receipt_sha256,
            self.evidence_v3_authority_receipt_sha256,
            self.authenticated_fold_inventory_sha256,
            *self.authenticated_fold_receipts,
            *self.outer_plan_v2_receipts,
            *self.outer_access_commitment_receipts,
            *self.gated_outer_forecast_archive_receipts,
            *self.outer_plan_v3_receipts,
            *self.ppo_cost_ladder_authority_receipts,
            *self.fixed_control_cost_ladder_authority_receipts,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL V4 outer evidence record", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLOuterEvidenceAuthorityV4:
    record: MassiveAdaptiveRLOuterEvidenceRecordV4
    evidence_source_receipt_sha256: str
    source_data_qualified: bool
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_evidence: MassiveAdaptiveRLOuterEvidenceV4 | None
    runtime_folds: tuple[MassiveAdaptiveAuthenticatedRLOuterFoldV4, ...] | None
    runtime_evidence_replayed: bool
    outer_development_conclusion_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "record_receipt_sha256": self.record.semantic_receipt_sha256,
            "evidence_source_receipt_sha256": self.evidence_source_receipt_sha256,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
        }

    def validate(self) -> None:
        self.record.validate()
        self.loaded_source.validate()
        runtime_present = (
            self.runtime_evidence is not None and self.runtime_folds is not None
        )
        expected_authorized = bool(
            runtime_present
            and self.source_data_qualified
            and not self.record.failed_gate_names
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SCHEMA
            or self.evidence_source_receipt_sha256
            != self.loaded_source.receipt.receipt_sha256
            or self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.record.evidence_receipt_sha256
            or self.source_data_qualified != self.record.source_data_qualified
            or self.runtime_evidence_replayed != runtime_present
            or self.outer_development_conclusion_authorized != expected_authorized
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
                "adaptive RL V4 outer evidence authority differs"
            )
        if self.runtime_evidence is not None and self.runtime_folds is not None:
            self.runtime_evidence.validate()
            rebuilt = build_massive_adaptive_rl_outer_evidence_v4(self.runtime_folds)
            if (
                rebuilt.semantic_receipt_sha256 != self.record.evidence_receipt_sha256
                or rebuilt.semantic_unsigned()
                != self.runtime_evidence.semantic_unsigned()
                or _record(
                    evidence=rebuilt,
                    evidence_v3_authority_receipt_sha256=(
                        self.record.evidence_v3_authority_receipt_sha256
                    ),
                    outer_plan_v2_receipts=self.record.outer_plan_v2_receipts,
                    outer_access_commitment_receipts=(
                        self.record.outer_access_commitment_receipts
                    ),
                    gated_outer_forecast_archive_receipts=(
                        self.record.gated_outer_forecast_archive_receipts
                    ),
                    outer_plan_v3_receipts=self.record.outer_plan_v3_receipts,
                    ppo_cost_ladder_authority_receipts=(
                        self.record.ppo_cost_ladder_authority_receipts
                    ),
                    fixed_control_cost_ladder_authority_receipts=(
                        self.record.fixed_control_cost_ladder_authority_receipts
                    ),
                ).semantic_unsigned()
                != self.record.semantic_unsigned()
            ):
                raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
                    "adaptive RL V4 evidence does not replay from authenticated folds"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _record(
    *,
    evidence: MassiveAdaptiveRLOuterEvidenceV4,
    evidence_v3_authority_receipt_sha256: str,
    outer_plan_v2_receipts: tuple[str, ...],
    outer_access_commitment_receipts: tuple[str, ...],
    gated_outer_forecast_archive_receipts: tuple[str, ...],
    outer_plan_v3_receipts: tuple[str, ...],
    ppo_cost_ladder_authority_receipts: tuple[str, ...],
    fixed_control_cost_ladder_authority_receipts: tuple[str, ...],
) -> MassiveAdaptiveRLOuterEvidenceRecordV4:
    body = {
        "evidence_receipt_sha256": evidence.semantic_receipt_sha256,
        "evidence_v3_authority_receipt_sha256": (evidence_v3_authority_receipt_sha256),
        "authenticated_fold_inventory_sha256": (
            evidence.authenticated_fold_inventory_sha256
        ),
        "authenticated_fold_receipts": tuple(
            fold.semantic_receipt_sha256 for fold in evidence.authenticated_folds
        ),
        "outer_plan_v2_receipts": outer_plan_v2_receipts,
        "outer_access_commitment_receipts": outer_access_commitment_receipts,
        "gated_outer_forecast_archive_receipts": (
            gated_outer_forecast_archive_receipts
        ),
        "outer_plan_v3_receipts": outer_plan_v3_receipts,
        "ppo_cost_ladder_authority_receipts": ppo_cost_ladder_authority_receipts,
        "fixed_control_cost_ladder_authority_receipts": (
            fixed_control_cost_ladder_authority_receipts
        ),
        "mean_high_cost_ppo_minus_fixed_control_log_return": (
            evidence.mean_high_cost_ppo_minus_fixed_control_log_return
        ),
        "passed_gate_names": evidence.passed_gate_names,
        "failed_gate_names": evidence.failed_gate_names,
        "source_data_qualified": evidence.source_data_qualified,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    }
    provisional = MassiveAdaptiveRLOuterEvidenceRecordV4(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


def _payload(record: MassiveAdaptiveRLOuterEvidenceRecordV4) -> dict[str, object]:
    record.validate()
    return {
        **record.semantic_unsigned(),
        "semantic_receipt_sha256": record.semantic_receipt_sha256,
    }


def _load_payload(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> dict[str, object]:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "adaptive RL V4 evidence is not canonical JSON"
        )
    return dict(value)


def _tuple_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "adaptive RL V4 evidence inventory differs"
        )
    return tuple(value)


def parse_massive_adaptive_rl_outer_evidence_authority_v4(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV4:
    payload = _load_payload(root=root, loaded_source=loaded_source)
    record_body = {
        "evidence_receipt_sha256": str(payload["evidence_receipt_sha256"]),
        "evidence_v3_authority_receipt_sha256": str(
            payload["evidence_v3_authority_receipt_sha256"]
        ),
        "authenticated_fold_inventory_sha256": str(
            payload["authenticated_fold_inventory_sha256"]
        ),
        "authenticated_fold_receipts": _tuple_strings(
            payload["authenticated_fold_receipts"]
        ),
        "outer_plan_v2_receipts": _tuple_strings(payload["outer_plan_v2_receipts"]),
        "outer_access_commitment_receipts": _tuple_strings(
            payload["outer_access_commitment_receipts"]
        ),
        "gated_outer_forecast_archive_receipts": _tuple_strings(
            payload["gated_outer_forecast_archive_receipts"]
        ),
        "outer_plan_v3_receipts": _tuple_strings(payload["outer_plan_v3_receipts"]),
        "ppo_cost_ladder_authority_receipts": _tuple_strings(
            payload["ppo_cost_ladder_authority_receipts"]
        ),
        "fixed_control_cost_ladder_authority_receipts": _tuple_strings(
            payload["fixed_control_cost_ladder_authority_receipts"]
        ),
        "mean_high_cost_ppo_minus_fixed_control_log_return": float(
            payload["mean_high_cost_ppo_minus_fixed_control_log_return"]  # type: ignore[arg-type]
        ),
        "passed_gate_names": _tuple_strings(payload["passed_gate_names"]),
        "failed_gate_names": _tuple_strings(payload["failed_gate_names"]),
        "source_data_qualified": bool(payload["source_data_qualified"]),
        "protocol_receipt_sha256": str(payload["protocol_receipt_sha256"]),
    }
    record = MassiveAdaptiveRLOuterEvidenceRecordV4(
        **record_body,  # type: ignore[arg-type]
        semantic_receipt_sha256=str(payload["semantic_receipt_sha256"]),
    )
    record.validate()
    authority_body = {
        "schema": MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SCHEMA,
        "record": record,
        "evidence_source_receipt_sha256": loaded_source.receipt.receipt_sha256,
        "source_data_qualified": record.source_data_qualified,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLOuterEvidenceAuthorityV4(
        **authority_body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
        loaded_source=loaded_source,
        runtime_evidence=None,
        runtime_folds=None,
        runtime_evidence_replayed=False,
        outer_development_conclusion_authorized=False,
    )
    result = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
    )
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class _ReplayedOuterPlanInputsV4:
    outer_plan_v2: MassiveAdaptiveRLOuterPlanV2
    outer_access_commitment: MassiveAdaptiveOuterAccessCommitmentV1
    gated_outer_forecast_archive: MassiveAdaptiveRLOuterForecastArchiveV1
    outer_plan_v3: MassiveAdaptiveRLOuterPlanV3
    ppo_cost_ladder_authority_receipt_sha256: str
    fixed_control_cost_ladder_authority_receipt_sha256: str


def _build(
    *,
    evidence_v3_authority: MassiveAdaptiveRLOuterEvidenceAuthorityV3,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    outer_access_commitments: Sequence[MassiveAdaptiveOuterAccessCommitmentV1],
    gated_outer_forecast_archives: Sequence[MassiveAdaptiveRLOuterForecastArchiveV1],
    outer_plans_v3: Sequence[MassiveAdaptiveRLOuterPlanV3],
    ppo_cost_ladder_authorities: Sequence[MassiveAdaptiveRLOuterCostLadderAuthorityV1],
    fixed_control_cost_ladder_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1
    ],
) -> tuple[
    MassiveAdaptiveRLOuterEvidenceV4,
    tuple[MassiveAdaptiveAuthenticatedRLOuterFoldV4, ...],
    tuple[_ReplayedOuterPlanInputsV4, ...],
]:
    evidence_v3_authority.validate()
    evidence_v3 = evidence_v3_authority.runtime_evidence
    folds_v3 = evidence_v3_authority.runtime_folds
    if (
        evidence_v3 is None
        or folds_v3 is None
        or not evidence_v3_authority.runtime_evidence_replayed
        or not (
            len(folds_v3)
            == len(outer_plans_v2)
            == len(outer_access_commitments)
            == len(gated_outer_forecast_archives)
            == len(outer_plans_v3)
            == len(ppo_cost_ladder_authorities)
            == len(fixed_control_cost_ladder_authorities)
        )
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "V3 evidence or V4 fold inputs are not replay authorized"
        )
    tuples = sorted(
        zip(
            folds_v3,
            outer_plans_v2,
            outer_access_commitments,
            gated_outer_forecast_archives,
            outer_plans_v3,
            ppo_cost_ladder_authorities,
            fixed_control_cost_ladder_authorities,
            strict=True,
        ),
        key=lambda values: values[0].fold_index,
    )
    replayed_inputs: list[_ReplayedOuterPlanInputsV4] = []
    fold_inputs = []
    for (
        fold,
        outer_plan_v2,
        commitment,
        gated_archive,
        supplied_plan_v3,
        ppo_ladder,
        fixed_ladder,
    ) in tuples:
        outer_plan_v2.validate()
        commitment.validate()
        gated_archive.validate()
        supplied_plan_v3.validate()
        try:
            rebuilt_plan_v3 = build_massive_adaptive_rl_outer_plan_v3(
                outer_plan_v2=outer_plan_v2,
                outer_access_commitment=commitment,
                gated_outer_forecast_archive=gated_archive,
            )
        except MassiveAdaptiveRLOuterPlanV3Error as error:
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
                "outer plan V3 did not replay from its access commitment"
            ) from error
        if (
            supplied_plan_v3.semantic_receipt_sha256
            != rebuilt_plan_v3.semantic_receipt_sha256
            or supplied_plan_v3.semantic_unsigned()
            != rebuilt_plan_v3.semantic_unsigned()
        ):
            raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
                "supplied outer plan V3 differs from the committed access path"
            )
        replayed = _ReplayedOuterPlanInputsV4(
            outer_plan_v2=outer_plan_v2,
            outer_access_commitment=commitment,
            gated_outer_forecast_archive=gated_archive,
            outer_plan_v3=rebuilt_plan_v3,
            ppo_cost_ladder_authority_receipt_sha256=(
                ppo_ladder.semantic_receipt_sha256
            ),
            fixed_control_cost_ladder_authority_receipt_sha256=(
                fixed_ladder.semantic_receipt_sha256
            ),
        )
        replayed_inputs.append(replayed)
        fold_inputs.append((fold, rebuilt_plan_v3, ppo_ladder, fixed_ladder))
    folds = tuple(
        build_massive_adaptive_authenticated_rl_outer_fold_v4(
            authenticated_fold_v3=fold,
            outer_plan_v3=plan,
            ppo_cost_ladder_authority=ppo_ladder,
            fixed_control_cost_ladder_authority=fixed_ladder,
        )
        for fold, plan, ppo_ladder, fixed_ladder in fold_inputs
    )
    result = build_massive_adaptive_rl_outer_evidence_v4(folds)
    if (
        result.evidence_v3.semantic_receipt_sha256
        != evidence_v3.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "V4 did not preserve its replay-authorized V3 evidence"
        )
    return result, folds, tuple(replayed_inputs)


def authorize_massive_adaptive_rl_outer_evidence_authority_v4(
    *,
    root: str | Path,
    authority: MassiveAdaptiveRLOuterEvidenceAuthorityV4,
    evidence_v3_authority: MassiveAdaptiveRLOuterEvidenceAuthorityV3,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    outer_access_commitments: Sequence[MassiveAdaptiveOuterAccessCommitmentV1],
    gated_outer_forecast_archives: Sequence[MassiveAdaptiveRLOuterForecastArchiveV1],
    outer_plans_v3: Sequence[MassiveAdaptiveRLOuterPlanV3],
    ppo_cost_ladder_authorities: Sequence[MassiveAdaptiveRLOuterCostLadderAuthorityV1],
    fixed_control_cost_ladder_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1
    ],
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV4:
    parsed = parse_massive_adaptive_rl_outer_evidence_authority_v4(
        root=root, loaded_source=authority.loaded_source
    )
    evidence, folds, replayed_inputs = _build(
        evidence_v3_authority=evidence_v3_authority,
        outer_plans_v2=outer_plans_v2,
        outer_access_commitments=outer_access_commitments,
        gated_outer_forecast_archives=gated_outer_forecast_archives,
        outer_plans_v3=outer_plans_v3,
        ppo_cost_ladder_authorities=ppo_cost_ladder_authorities,
        fixed_control_cost_ladder_authorities=fixed_control_cost_ladder_authorities,
    )
    record = _record(
        evidence=evidence,
        evidence_v3_authority_receipt_sha256=(
            evidence_v3_authority.semantic_receipt_sha256
        ),
        outer_plan_v2_receipts=tuple(
            value.outer_plan_v2.semantic_receipt_sha256 for value in replayed_inputs
        ),
        outer_access_commitment_receipts=tuple(
            value.outer_access_commitment.semantic_receipt_sha256
            for value in replayed_inputs
        ),
        gated_outer_forecast_archive_receipts=tuple(
            value.gated_outer_forecast_archive.semantic_receipt_sha256
            for value in replayed_inputs
        ),
        outer_plan_v3_receipts=tuple(
            value.outer_plan_v3.semantic_receipt_sha256 for value in replayed_inputs
        ),
        ppo_cost_ladder_authority_receipts=tuple(
            value.ppo_cost_ladder_authority_receipt_sha256 for value in replayed_inputs
        ),
        fixed_control_cost_ladder_authority_receipts=tuple(
            value.fixed_control_cost_ladder_authority_receipt_sha256
            for value in replayed_inputs
        ),
    )
    if record.semantic_unsigned() != parsed.record.semantic_unsigned():
        raise MassiveAdaptiveRLOuterEvidenceAuthorityV4Error(
            "adaptive RL V4 outer evidence did not replay"
        )
    result = replace(
        parsed,
        runtime_evidence=evidence,
        runtime_folds=folds,
        runtime_evidence_replayed=True,
        outer_development_conclusion_authorized=bool(
            parsed.source_data_qualified and not parsed.record.failed_gate_names
        ),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_outer_evidence_authority_v4(
    *,
    root: str | Path,
    artifact_id: str,
    evidence_v3_authority: MassiveAdaptiveRLOuterEvidenceAuthorityV3,
    outer_plans_v2: Sequence[MassiveAdaptiveRLOuterPlanV2],
    outer_access_commitments: Sequence[MassiveAdaptiveOuterAccessCommitmentV1],
    gated_outer_forecast_archives: Sequence[MassiveAdaptiveRLOuterForecastArchiveV1],
    outer_plans_v3: Sequence[MassiveAdaptiveRLOuterPlanV3],
    ppo_cost_ladder_authorities: Sequence[MassiveAdaptiveRLOuterCostLadderAuthorityV1],
    fixed_control_cost_ladder_authorities: Sequence[
        MassiveAdaptiveRLFixedControlOuterCostLadderAuthorityV1
    ],
    committed_at_ms: int,
) -> MassiveAdaptiveRLOuterEvidenceAuthorityV4:
    artifact = _artifact_id(artifact_id)
    evidence, _, replayed_inputs = _build(
        evidence_v3_authority=evidence_v3_authority,
        outer_plans_v2=outer_plans_v2,
        outer_access_commitments=outer_access_commitments,
        gated_outer_forecast_archives=gated_outer_forecast_archives,
        outer_plans_v3=outer_plans_v3,
        ppo_cost_ladder_authorities=ppo_cost_ladder_authorities,
        fixed_control_cost_ladder_authorities=fixed_control_cost_ladder_authorities,
    )
    record = _record(
        evidence=evidence,
        evidence_v3_authority_receipt_sha256=(
            evidence_v3_authority.semantic_receipt_sha256
        ),
        outer_plan_v2_receipts=tuple(
            value.outer_plan_v2.semantic_receipt_sha256 for value in replayed_inputs
        ),
        outer_access_commitment_receipts=tuple(
            value.outer_access_commitment.semantic_receipt_sha256
            for value in replayed_inputs
        ),
        gated_outer_forecast_archive_receipts=tuple(
            value.gated_outer_forecast_archive.semantic_receipt_sha256
            for value in replayed_inputs
        ),
        outer_plan_v3_receipts=tuple(
            value.outer_plan_v3.semantic_receipt_sha256 for value in replayed_inputs
        ),
        ppo_cost_ladder_authority_receipts=tuple(
            value.ppo_cost_ladder_authority_receipt_sha256 for value in replayed_inputs
        ),
        fixed_control_cost_ladder_authority_receipts=tuple(
            value.fixed_control_cost_ladder_authority_receipt_sha256
            for value in replayed_inputs
        ),
    )
    relative = f"massive-adaptive/rl-outer-evidence-authority-v4/{artifact}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(_payload(record))),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=(
            MASSIVE_ADAPTIVE_RL_OUTER_EVIDENCE_AUTHORITY_V4_SOURCE_SCHEMA_SHA256
        ),
        entitlement_receipt_sha256=evidence.semantic_receipt_sha256,
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-RL-OUTER-EVIDENCE-AUTHORITY-V4-{artifact}",
    )
    loaded = load_massive_source_bundle(
        root=root, relative_payload_path=relative, verified_at_ms=committed_at_ms
    )
    return authorize_massive_adaptive_rl_outer_evidence_authority_v4(
        root=root,
        authority=parse_massive_adaptive_rl_outer_evidence_authority_v4(
            root=root, loaded_source=loaded
        ),
        evidence_v3_authority=evidence_v3_authority,
        outer_plans_v2=outer_plans_v2,
        outer_access_commitments=outer_access_commitments,
        gated_outer_forecast_archives=gated_outer_forecast_archives,
        outer_plans_v3=outer_plans_v3,
        ppo_cost_ladder_authorities=ppo_cost_ladder_authorities,
        fixed_control_cost_ladder_authorities=(fixed_control_cost_ladder_authorities),
    )


__all__ = [
    "MassiveAdaptiveRLOuterEvidenceAuthorityV4",
    "MassiveAdaptiveRLOuterEvidenceAuthorityV4Error",
    "MassiveAdaptiveRLOuterEvidenceRecordV4",
    "authorize_massive_adaptive_rl_outer_evidence_authority_v4",
    "materialize_massive_adaptive_rl_outer_evidence_authority_v4",
    "parse_massive_adaptive_rl_outer_evidence_authority_v4",
]
