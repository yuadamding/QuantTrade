"""Create-only archive of source-replayed adaptive target roots.

The canonical file stores only root and experiment inventories.  Generic
reload cannot train.  Runtime source-target objects become available only
after the package rebuilds each target root from its live Massive authorities;
an explicitly named canary path remains permanently unqualified.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.features.massive_adaptive_target_root_v1 import (
    MassiveAdaptiveTargetRootV1,
    MassiveAdaptiveTargetSourceRuntimeV1,
    build_massive_adaptive_target_root_canary_v1,
    build_massive_adaptive_target_root_v1,
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


MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA = (
    "rl-quant.massive-adaptive-target-archive-v1"
)
MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_DATASET = "massive-adaptive-target-archive-v1"
MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SCHEMA_SHA256 = semantic_sha256(
    {
        "schema": MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA,
        "encoding": "canonical-json-root-and-experiment-inventory",
        "publication": "create-only-source-transaction",
    }
)
MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SHA256 = file_sha256(Path(__file__))
MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SPEC_SHA256 = semantic_sha256(
    {
        "protocol": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "rows": "one-target-root-per-eligible-window-origin",
        "generic_reload": "nonauthorizing-without-runtime-targets",
        "promotion": "rebuild-every-target-from-live-source-roots",
        "canary": "runtime-replay-permitted-but-always-unqualified",
        "profitability_reporting": False,
        "lockbox": False,
        "rl": False,
    }
)


class MassiveAdaptiveTargetArchiveV1Error(ValueError):
    """Adaptive target archive contents or source replay differ."""


def _artifact_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target archive artifact ID is not path safe"
        )
    return value


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveTargetArchiveV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveTargetArchiveV1:
    decision_session_dates: tuple[str, ...]
    origin_decision_root_receipts: tuple[str, ...]
    target_root_receipts: tuple[str, ...]
    source_target_receipts: tuple[str, ...]
    experiment_source_receipts: tuple[str, ...]
    origin_decision_root_inventory_sha256: str
    target_root_inventory_sha256: str
    source_target_inventory_sha256: str
    experiment_inventory_sha256: str
    committed_source_data_qualified: bool
    protocol_receipt_sha256: str
    specification_sha256: str
    implementation_source_sha256: str
    semantic_receipt_sha256: str
    loaded_source: LoadedMassiveSourceObject
    runtime_target_roots: tuple[MassiveAdaptiveTargetRootV1, ...] | None
    runtime_source_targets: tuple[MassiveAdaptiveSourceTargetsV1, ...] | None
    runtime_roots_replayed: bool
    development_training_authorized: bool
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    reinforcement_learning_authorized: bool = False
    schema: str = MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "decision_session_dates": self.decision_session_dates,
            "origin_decision_root_receipts": self.origin_decision_root_receipts,
            "target_root_receipts": self.target_root_receipts,
            "source_target_receipts": self.source_target_receipts,
            "experiment_source_receipts": self.experiment_source_receipts,
            "origin_decision_root_inventory_sha256": (
                self.origin_decision_root_inventory_sha256
            ),
            "target_root_inventory_sha256": self.target_root_inventory_sha256,
            "source_target_inventory_sha256": self.source_target_inventory_sha256,
            "experiment_inventory_sha256": self.experiment_inventory_sha256,
            "committed_source_data_qualified": self.committed_source_data_qualified,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
            "implementation_source_sha256": self.implementation_source_sha256,
            "profitability_reporting_authorized": False,
            "lockbox_access_authorized": False,
            "reinforcement_learning_authorized": False,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            **self.semantic_unsigned(),
            "semantic_receipt_sha256": self.semantic_receipt_sha256,
        }

    def validate(self) -> None:
        runtime_present = (
            self.runtime_target_roots is not None
            and self.runtime_source_targets is not None
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA
            or not self.decision_session_dates
            or self.decision_session_dates
            != tuple(sorted(set(self.decision_session_dates)))
            or any(
                len(values) != len(self.decision_session_dates)
                for values in (
                    self.origin_decision_root_receipts,
                    self.target_root_receipts,
                    self.source_target_receipts,
                    self.experiment_source_receipts,
                )
            )
            or self.origin_decision_root_inventory_sha256
            != semantic_sha256(self.origin_decision_root_receipts)
            or self.target_root_inventory_sha256
            != semantic_sha256(self.target_root_receipts)
            or self.source_target_inventory_sha256
            != semantic_sha256(self.source_target_receipts)
            or self.experiment_inventory_sha256
            != semantic_sha256(self.experiment_source_receipts)
            or not isinstance(self.committed_source_data_qualified, bool)
            or self.protocol_receipt_sha256
            != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256
            != semantic_sha256(self.semantic_unsigned())
            or self.runtime_roots_replayed != runtime_present
            or self.development_training_authorized
            != (runtime_present and self.committed_source_data_qualified)
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.reinforcement_learning_authorized
        ):
            raise MassiveAdaptiveTargetArchiveV1Error(
                "adaptive target archive identity or authorization differs"
            )
        for values in (
            self.origin_decision_root_receipts,
            self.target_root_receipts,
            self.source_target_receipts,
            self.experiment_source_receipts,
        ):
            for value in values:
                _digest("adaptive target archive row", value)
        for name in (
            "origin_decision_root_inventory_sha256",
            "target_root_inventory_sha256",
            "source_target_inventory_sha256",
            "experiment_inventory_sha256",
            "protocol_receipt_sha256",
            "specification_sha256",
            "implementation_source_sha256",
            "semantic_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if runtime_present:
            assert self.runtime_target_roots is not None
            assert self.runtime_source_targets is not None
            for target_root in self.runtime_target_roots:
                target_root.validate()
            for source_target in self.runtime_source_targets:
                source_target.validate()
            if (
                tuple(
                    row.decision_session_date for row in self.runtime_target_roots
                )
                != self.decision_session_dates
                or tuple(
                    row.decision_root_receipt_sha256
                    for row in self.runtime_target_roots
                )
                != self.origin_decision_root_receipts
                or tuple(
                    row.semantic_receipt_sha256 for row in self.runtime_target_roots
                )
                != self.target_root_receipts
                or tuple(
                    row.semantic_receipt_sha256 for row in self.runtime_source_targets
                )
                != self.source_target_receipts
            ):
                raise MassiveAdaptiveTargetArchiveV1Error(
                    "adaptive runtime target inventory differs"
                )
        self.loaded_source.validate()
        if (
            self.loaded_source.receipt.dataset_id
            != MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_DATASET
            or self.loaded_source.receipt.schema_sha256
            != MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SCHEMA_SHA256
            or self.loaded_source.receipt.entitlement_receipt_sha256
            != self.origin_decision_root_inventory_sha256
        ):
            raise MassiveAdaptiveTargetArchiveV1Error(
                "adaptive target archive source transaction differs"
            )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _ordered_roots(
    values: Sequence[MassiveAdaptiveDecisionRootV1],
) -> tuple[MassiveAdaptiveDecisionRootV1, ...]:
    result = tuple(sorted(values, key=lambda row: row.decision_session_date))
    if not result or len({row.decision_session_date for row in result}) != len(result):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive decision-root inventory is empty or duplicated"
        )
    for row in result:
        row.validate()
    return result


def _ordered_targets(
    values: Sequence[MassiveAdaptiveSourceTargetsV1],
) -> tuple[MassiveAdaptiveSourceTargetsV1, ...]:
    result = tuple(sorted(values, key=lambda row: row.decision_session_date))
    if not result or len({row.decision_session_date for row in result}) != len(result):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive source-target inventory is empty or duplicated"
        )
    for row in result:
        row.validate()
    return result


def _body(
    *,
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...],
    target_roots: tuple[MassiveAdaptiveTargetRootV1, ...],
    source_targets: tuple[MassiveAdaptiveSourceTargetsV1, ...],
) -> dict[str, object]:
    dates = tuple(row.decision_session_date for row in decision_roots)
    if (
        dates != tuple(row.decision_session_date for row in target_roots)
        or dates != tuple(row.decision_session_date for row in source_targets)
    ):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target archive date inventories differ"
        )
    decision_receipts = tuple(row.semantic_receipt_sha256 for row in decision_roots)
    target_receipts = tuple(row.semantic_receipt_sha256 for row in target_roots)
    source_receipts = tuple(row.semantic_receipt_sha256 for row in source_targets)
    experiment_receipts = tuple(
        row.experiment_source_receipt_sha256 for row in target_roots
    )
    return {
        "schema": MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA,
        "decision_session_dates": dates,
        "origin_decision_root_receipts": decision_receipts,
        "target_root_receipts": target_receipts,
        "source_target_receipts": source_receipts,
        "experiment_source_receipts": experiment_receipts,
        "origin_decision_root_inventory_sha256": semantic_sha256(
            decision_receipts
        ),
        "target_root_inventory_sha256": semantic_sha256(target_receipts),
        "source_target_inventory_sha256": semantic_sha256(source_receipts),
        "experiment_inventory_sha256": semantic_sha256(experiment_receipts),
        "committed_source_data_qualified": all(
            row.source_data_qualified for row in target_roots
        ),
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SPEC_SHA256,
        "implementation_source_sha256": MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SHA256,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "reinforcement_learning_authorized": False,
    }


def _publish(
    *,
    root: str | Path,
    artifact_id: str,
    body: dict[str, object],
    committed_at_ms: int,
) -> MassiveAdaptiveTargetArchiveV1:
    identifier = _artifact_id(artifact_id)
    receipt = semantic_sha256(body)
    payload = {**body, "semantic_receipt_sha256": receipt}
    relative = f"massive-adaptive/target-archive-v1/{identifier}.json"
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_DATASET,
        source_object_key=relative,
        requested_at_ms=committed_at_ms,
        downloaded_at_ms=committed_at_ms,
        schema_sha256=MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=str(
            body["origin_decision_root_inventory_sha256"]
        ),
        committed_at_ms=committed_at_ms,
        request_id=f"ADAPTIVE-TARGET-ARCHIVE-V1-{identifier}",
    )
    loaded = load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=committed_at_ms,
    )
    return parse_massive_adaptive_target_archive_v1(root=root, loaded_source=loaded)


def materialize_massive_adaptive_target_archive_v1(
    *,
    root: str | Path,
    artifact_id: str,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    source_runtimes: Sequence[MassiveAdaptiveTargetSourceRuntimeV1],
    committed_at_ms: int,
) -> MassiveAdaptiveTargetArchiveV1:
    """Publish target roots only after full live-source reconstruction."""

    ordered_decisions = _ordered_roots(decision_roots)
    ordered_targets = _ordered_targets(source_targets)
    runtime_by_date = {
        row.decision_clock.session_date: row for row in source_runtimes
    }
    if (
        len(runtime_by_date) != len(source_runtimes)
        or tuple(runtime_by_date) != tuple(
            row.decision_session_date for row in ordered_decisions
        )
    ):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target runtime inventory differs"
        )
    targets_by_date = {row.decision_session_date: row for row in ordered_targets}
    target_roots = tuple(
        build_massive_adaptive_target_root_v1(
            decision_root=decision_root,
            source_target=targets_by_date[decision_root.decision_session_date],
            source_runtime=runtime_by_date[decision_root.decision_session_date],
        )
        for decision_root in ordered_decisions
    )
    generic = _publish(
        root=root,
        artifact_id=artifact_id,
        body=_body(
            decision_roots=ordered_decisions,
            target_roots=target_roots,
            source_targets=ordered_targets,
        ),
        committed_at_ms=committed_at_ms,
    )
    return authorize_massive_adaptive_target_archive_v1(
        root=root,
        archive=generic,
        decision_roots=ordered_decisions,
        source_targets=ordered_targets,
        source_runtimes=source_runtimes,
    )


def materialize_massive_adaptive_target_archive_canary_v1(
    *,
    root: str | Path,
    artifact_id: str,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    committed_at_ms: int,
) -> MassiveAdaptiveTargetArchiveV1:
    """Publish synthetic target bindings that can never authorize training."""

    ordered_decisions = _ordered_roots(decision_roots)
    ordered_targets = _ordered_targets(source_targets)
    targets_by_date = {row.decision_session_date: row for row in ordered_targets}
    target_roots = tuple(
        build_massive_adaptive_target_root_canary_v1(
            decision_root=decision_root,
            source_target=targets_by_date[decision_root.decision_session_date],
        )
        for decision_root in ordered_decisions
    )
    generic = _publish(
        root=root,
        artifact_id=artifact_id,
        body=_body(
            decision_roots=ordered_decisions,
            target_roots=target_roots,
            source_targets=ordered_targets,
        ),
        committed_at_ms=committed_at_ms,
    )
    parsed = parse_massive_adaptive_target_archive_v1(
        root=root, loaded_source=generic.loaded_source
    )
    result = replace(
        parsed,
        runtime_target_roots=target_roots,
        runtime_source_targets=ordered_targets,
        runtime_roots_replayed=True,
        development_training_authorized=False,
    )
    result.validate()
    return result


def parse_massive_adaptive_target_archive_v1(
    *, root: str | Path, loaded_source: LoadedMassiveSourceObject
) -> MassiveAdaptiveTargetArchiveV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded_source)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or raw != canonical_json_file_bytes(payload):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target archive is not canonical JSON"
        )
    for name in (
        "decision_session_dates",
        "origin_decision_root_receipts",
        "target_root_receipts",
        "source_target_receipts",
        "experiment_source_receipts",
    ):
        payload[name] = tuple(payload[name])
    result = MassiveAdaptiveTargetArchiveV1(
        **payload,
        loaded_source=loaded_source,
        runtime_target_roots=None,
        runtime_source_targets=None,
        runtime_roots_replayed=False,
        development_training_authorized=False,
    )
    result.validate()
    if canonical_json_file_bytes(result.canonical_payload()) != raw:
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target archive canonical bytes differ"
        )
    return result


def authorize_massive_adaptive_target_archive_v1(
    *,
    root: str | Path,
    archive: MassiveAdaptiveTargetArchiveV1,
    decision_roots: Sequence[MassiveAdaptiveDecisionRootV1],
    source_targets: Sequence[MassiveAdaptiveSourceTargetsV1],
    source_runtimes: Sequence[MassiveAdaptiveTargetSourceRuntimeV1],
) -> MassiveAdaptiveTargetArchiveV1:
    """Reopen the archive and repeat every live target reconstruction."""

    parsed = parse_massive_adaptive_target_archive_v1(
        root=root, loaded_source=archive.loaded_source
    )
    ordered_decisions = _ordered_roots(decision_roots)
    ordered_targets = _ordered_targets(source_targets)
    targets_by_date = {row.decision_session_date: row for row in ordered_targets}
    runtime_by_date = {
        row.decision_clock.session_date: row for row in source_runtimes
    }
    if len(runtime_by_date) != len(source_runtimes):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target runtime inventory is duplicated"
        )
    rebuilt_roots = tuple(
        build_massive_adaptive_target_root_v1(
            decision_root=decision_root,
            source_target=targets_by_date[decision_root.decision_session_date],
            source_runtime=runtime_by_date[decision_root.decision_session_date],
        )
        for decision_root in ordered_decisions
    )
    expected = _body(
        decision_roots=ordered_decisions,
        target_roots=rebuilt_roots,
        source_targets=ordered_targets,
    )
    if (
        parsed.semantic_receipt_sha256 != archive.semantic_receipt_sha256
        or parsed.semantic_unsigned() != expected
    ):
        raise MassiveAdaptiveTargetArchiveV1Error(
            "adaptive target archive does not replay from its source roots"
        )
    result = replace(
        parsed,
        runtime_target_roots=rebuilt_roots,
        runtime_source_targets=ordered_targets,
        runtime_roots_replayed=True,
        development_training_authorized=(
            parsed.committed_source_data_qualified
        ),
    )
    result.validate()
    return result


__all__ = [
    "MASSIVE_ADAPTIVE_TARGET_ARCHIVE_V1_SCHEMA",
    "MassiveAdaptiveTargetArchiveV1",
    "MassiveAdaptiveTargetArchiveV1Error",
    "authorize_massive_adaptive_target_archive_v1",
    "materialize_massive_adaptive_target_archive_canary_v1",
    "materialize_massive_adaptive_target_archive_v1",
    "parse_massive_adaptive_target_archive_v1",
]
