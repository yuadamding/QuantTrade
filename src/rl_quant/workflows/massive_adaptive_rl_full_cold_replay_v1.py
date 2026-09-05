"""Create-only proof that the complete Manifest-V5 report cold-replays.

The authoritative runner reconstructs the complete report boundary with every
materializer disabled, snapshots the protected experiment evidence before and
after that reconstruction, and only then asks this module to persist the
completion proof.  A generic reload is integrity-valid but cannot authorize
the final prequential state; exact replay must supply the reconstructed run and
the unchanged evidence inventories again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import time
from typing import cast

from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    canonical_json_file_bytes,
    load_massive_source_bundle,
    publish_massive_source_object,
    read_loaded_massive_source_bytes,
)
from rl_quant.protocol.canonical_artifact import file_sha256, semantic_sha256
from rl_quant.protocol.massive_adaptive_alpha_v1 import (
    MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
    assert_no_adaptive_hold_semantics,
)
from rl_quant.workflows.massive_adaptive_rl_execution_implementation_registration_v1 import (
    MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    massive_adaptive_rl_experiment_materialization_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5 import (
    MassiveAdaptiveRLExperimentManifestV5,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v5_registration import (
    MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1,
)
from rl_quant.workflows.massive_adaptive_rl_writer_guard_v5 import (
    massive_adaptive_rl_manifest_v5_writer_scope_v1,
)


MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-full-cold-replay-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_V1_SPEC_SHA256 = semantic_sha256(
    {
        "manifest_stage": "full-cold-replay-verified",
        "input": "complete-replay-authorized-profitability-report-prefix",
        "reconstruction": "package-owned-full-v5-report-boundary",
        "publication_during_reconstruction": False,
        "protected_evidence_inventory": ("identical-before-and-after-reconstruction"),
        "proof_issuance": "package-verifier-owned-opaque-replay-evidence",
        "caller_supplied_replay_evidence": False,
        "completion_proof": "distinct-create-only-implementation-authority",
        "generic_reload": "integrity-only-nonauthorizing",
        "execution_complete_is_not_profitability": True,
    }
)
MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_DATASET = (
    "massive-adaptive-rl-full-cold-replay-authority-v1"
)
MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256 = (
    semantic_sha256(
        {
            "schema": MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA,
            "payload": "complete-v5-nonmaterializing-reconstruction-proof",
            "generic_reload": "nonauthorizing",
        }
    )
)


class MassiveAdaptiveRLFullColdReplayV1Error(ValueError):
    """The replayed report, evidence inventory, or completion proof differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _required_digest(name: str, value: object) -> str:
    return _digest(name, value)


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MassiveAdaptiveRLFullColdReplayV1Error(f"{name} is absent or invalid")
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "cold-replay experiment ID is not path safe"
        )
    return value


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLProtectedEvidenceFileV1:
    root_role: str
    relative_path: str
    size_bytes: int
    content_sha256: str

    def validate(self) -> None:
        relative = Path(self.relative_path)
        if (
            self.root_role not in {"artifact", "source", "artifact-and-source"}
            or not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in {"adaptive-rl", "massive-adaptive"}
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "protected evidence inventory row differs"
            )
        _digest("protected evidence content", self.content_sha256)


def _completion_evidence_excluded(*, relative: Path, experiment_id: str) -> bool:
    parts = relative.parts
    if parts[:2] != ("adaptive-rl", experiment_id):
        return False
    if len(parts) >= 3 and parts[2] in {
        "orchestration-lease-v1",
        "full-cold-replay-authority-v1",
    }:
        return True
    return bool(
        len(parts) >= 4
        and parts[2] == "prequential-experiment-state-v1"
        and parts[3].startswith("014-full-cold-replay-verified.json")
    )


def _path_belongs_to_experiment(
    *,
    relative: Path,
    experiment_id: str,
    base_manifest_v4_receipt_sha256: str,
) -> bool:
    parts = relative.parts
    rendered = relative.as_posix()
    if parts[:2] == ("adaptive-rl", experiment_id):
        return not _completion_evidence_excluded(
            relative=relative, experiment_id=experiment_id
        )
    return bool(
        parts
        and parts[0] in {"adaptive-rl", "massive-adaptive"}
        and (
            any(
                part == experiment_id or part.startswith(f"{experiment_id}-")
                for part in parts[1:]
            )
            or base_manifest_v4_receipt_sha256 in rendered
        )
    )


def _root_inventory(
    *,
    root: Path,
    root_role: str,
    experiment_id: str,
    base_manifest_v4_receipt_sha256: str,
) -> tuple[MassiveAdaptiveRLProtectedEvidenceFileV1, ...]:
    rows: list[MassiveAdaptiveRLProtectedEvidenceFileV1] = []
    for top_name in ("adaptive-rl", "massive-adaptive"):
        top = root / top_name
        if not top.exists():
            continue
        if top.is_symlink() or not top.is_dir():
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "protected evidence root is not a no-follow directory"
            )
        for directory, directory_names, file_names in os.walk(top, followlinks=False):
            current = Path(directory)
            retained_directories: list[str] = []
            for name in directory_names:
                candidate = current / name
                relative = candidate.relative_to(root)
                if candidate.is_symlink() and _path_belongs_to_experiment(
                    relative=relative,
                    experiment_id=experiment_id,
                    base_manifest_v4_receipt_sha256=(base_manifest_v4_receipt_sha256),
                ):
                    raise MassiveAdaptiveRLFullColdReplayV1Error(
                        "protected evidence contains a directory symlink"
                    )
                if not _completion_evidence_excluded(
                    relative=relative, experiment_id=experiment_id
                ):
                    retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in file_names:
                path = current / name
                relative = path.relative_to(root)
                if not _path_belongs_to_experiment(
                    relative=relative,
                    experiment_id=experiment_id,
                    base_manifest_v4_receipt_sha256=(base_manifest_v4_receipt_sha256),
                ):
                    continue
                details = path.lstat()
                if not stat.S_ISREG(details.st_mode):
                    raise MassiveAdaptiveRLFullColdReplayV1Error(
                        "protected evidence contains a non-regular file"
                    )
                row = MassiveAdaptiveRLProtectedEvidenceFileV1(
                    root_role=root_role,
                    relative_path=relative.as_posix(),
                    size_bytes=details.st_size,
                    content_sha256=file_sha256(path),
                )
                row.validate()
                rows.append(row)
    return tuple(sorted(rows, key=lambda row: (row.root_role, row.relative_path)))


def massive_adaptive_rl_protected_evidence_inventory_v1(
    *,
    artifact_root: str | Path,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
) -> tuple[MassiveAdaptiveRLProtectedEvidenceFileV1, ...]:
    """Hash the experiment-derived evidence protected by cold replay.

    Operational locks and the completion proof/final state are deliberately
    excluded.  Raw provider data are not scanned; every committed source used
    by the replay remains authenticated by its package source transaction.
    """

    if type(manifest) is not MassiveAdaptiveRLExperimentManifestV5:
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "protected inventory requires exact Manifest V5"
        )
    manifest.validate()
    artifact_input = Path(artifact_root)
    source_input = Path(source_root)
    if (
        artifact_input.is_symlink()
        or not artifact_input.is_dir()
        or source_input.is_symlink()
        or not source_input.is_dir()
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "protected evidence roots must be existing no-follow directories"
        )
    artifact = artifact_input.resolve(strict=True)
    source = source_input.resolve(strict=True)
    roots = (
        ((artifact, "artifact-and-source"),)
        if artifact == source
        else ((artifact, "artifact"), (source, "source"))
    )
    rows = tuple(
        row
        for root, role in roots
        for row in _root_inventory(
            root=root,
            root_role=role,
            experiment_id=manifest.experiment_id,
            base_manifest_v4_receipt_sha256=(
                manifest.base_manifest.semantic_receipt_sha256
            ),
        )
    )
    if not rows or len({(row.root_role, row.relative_path) for row in rows}) != len(
        rows
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "protected evidence inventory is empty or duplicated"
        )
    return tuple(sorted(rows, key=lambda row: (row.root_role, row.relative_path)))


def _inventory_receipt(
    rows: Sequence[MassiveAdaptiveRLProtectedEvidenceFileV1],
) -> str:
    ordered = tuple(rows)
    for row in ordered:
        if type(row) is not MassiveAdaptiveRLProtectedEvidenceFileV1:
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "protected evidence row generation differs"
            )
        row.validate()
    return semantic_sha256(tuple(asdict(row) for row in ordered))


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFullColdReplayAuthorityV1:
    experiment_id: str
    manifest_v5_receipt_sha256: str
    manifest_v5_registration_receipt_sha256: str
    execution_implementation_registration_receipt_sha256: str
    replayed_run_receipt_sha256: str
    profitability_report_authority_receipt_sha256: str
    profitability_report_source_receipt_sha256: str
    profitability_report_commit_receipt_sha256: str
    profitability_report_committed_at_ms: int
    profitability_report_state_receipt_sha256: str
    profitability_report_state_source_receipt_sha256: str
    profitability_report_state_commit_receipt_sha256: str
    profitability_report_state_committed_at_ms: int
    outer_fold_seal_receipts: tuple[str, ...]
    protected_evidence_inventory: tuple[MassiveAdaptiveRLProtectedEvidenceFileV1, ...]
    protected_evidence_inventory_sha256: str
    protected_evidence_file_count: int
    before_replay_inventory_sha256: str
    after_replay_inventory_sha256: str
    nonmaterializing_replay: bool
    policy_schedule_disposition: str
    policy_schedule_qualified: bool
    profitability_gates_passed: bool
    source_data_qualified: bool
    semantic_receipt_sha256: str
    runtime_cold_replay_replayed: bool = False
    development_full_cold_replay_verified: bool = False
    end_to_end_profitability_execution_complete: bool = False
    positive_profitability_authorization_eligible: bool = False
    lockbox_access_authorized: bool = False
    live_trading_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_V1_SPEC_SHA256
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA
    _runtime_replayed_run: object | None = field(
        default=None, compare=False, repr=False
    )
    _loaded_source: LoadedMassiveSourceObject | None = field(
        default=None, compare=False, repr=False
    )

    def semantic_unsigned(self) -> dict[str, object]:
        result = {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
            if not descriptor.name.startswith("_")
            and descriptor.name
            not in {
                "semantic_receipt_sha256",
                "runtime_cold_replay_replayed",
                "development_full_cold_replay_verified",
                "end_to_end_profitability_execution_complete",
                "positive_profitability_authorization_eligible",
            }
        }
        result["protected_evidence_inventory"] = tuple(
            asdict(row) for row in self.protected_evidence_inventory
        )
        return result

    @property
    def source_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.receipt.receipt_sha256
        )

    @property
    def source_transaction_receipt_sha256(self) -> str | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.receipt_sha256
        )

    @property
    def source_transaction_committed_at_ms(self) -> int | None:
        return (
            None
            if self._loaded_source is None
            else self._loaded_source.commit.committed_at_ms
        )

    def validate(self) -> None:
        runtime = self._runtime_replayed_run is not None
        expected_qualified = self.policy_schedule_disposition == (
            "policy-prefix-qualified"
        )
        inventory_receipt = _inventory_receipt(self.protected_evidence_inventory)
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA
            or _identifier(self.experiment_id) != self.experiment_id
            or len(self.outer_fold_seal_receipts) != 4
            or len(set(self.outer_fold_seal_receipts)) != 4
            or self.protected_evidence_file_count
            != len(self.protected_evidence_inventory)
            or self.protected_evidence_file_count <= 0
            or self.protected_evidence_inventory_sha256 != inventory_receipt
            or self.before_replay_inventory_sha256 != inventory_receipt
            or self.after_replay_inventory_sha256 != inventory_receipt
            or not self.nonmaterializing_replay
            or self.policy_schedule_disposition
            not in {"policy-prefix-qualified", "policy-prefix-diagnostic-only"}
            or self.policy_schedule_qualified != expected_qualified
            or not isinstance(self.profitability_gates_passed, bool)
            or not self.source_data_qualified
            or self.runtime_cold_replay_replayed != runtime
            or self.development_full_cold_replay_verified
            != bool(runtime and self.source_data_qualified)
            or self.end_to_end_profitability_execution_complete
            or self.positive_profitability_authorization_eligible
            or self.lockbox_access_authorized
            or self.live_trading_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "full cold-replay authority differs"
            )
        for value in (
            self.manifest_v5_receipt_sha256,
            self.manifest_v5_registration_receipt_sha256,
            self.execution_implementation_registration_receipt_sha256,
            self.replayed_run_receipt_sha256,
            self.profitability_report_authority_receipt_sha256,
            self.profitability_report_source_receipt_sha256,
            self.profitability_report_commit_receipt_sha256,
            self.profitability_report_state_receipt_sha256,
            self.profitability_report_state_source_receipt_sha256,
            self.profitability_report_state_commit_receipt_sha256,
            self.protected_evidence_inventory_sha256,
            self.before_replay_inventory_sha256,
            self.after_replay_inventory_sha256,
            self.semantic_receipt_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            *self.outer_fold_seal_receipts,
        ):
            _digest("full cold-replay receipt", value)
        _timestamp(
            "profitability report timestamp", self.profitability_report_committed_at_ms
        )
        _timestamp(
            "profitability report state timestamp",
            self.profitability_report_state_committed_at_ms,
        )
        if (
            self.profitability_report_state_committed_at_ms
            <= self.profitability_report_committed_at_ms
        ):
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "full cold-replay report chronology differs"
            )
        if self._loaded_source is not None:
            self._loaded_source.validate()
            if (
                self._loaded_source.receipt.dataset_id
                != MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_DATASET
                or self._loaded_source.receipt.schema_sha256
                != MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                or self._loaded_source.receipt.entitlement_receipt_sha256
                != self.semantic_receipt_sha256
                or self._loaded_source.commit.committed_at_ms
                <= self.profitability_report_state_committed_at_ms
            ):
                raise MassiveAdaptiveRLFullColdReplayV1Error(
                    "full cold-replay source transaction differs"
                )
        if runtime:
            _validate_replayed_run(
                run=self._runtime_replayed_run,
                experiment_id=self.experiment_id,
                manifest_receipt=self.manifest_v5_receipt_sha256,
                execution_registration_receipt=(
                    self.execution_implementation_registration_receipt_sha256
                ),
            )
            run = self._runtime_replayed_run
            if (
                getattr(run, "semantic_receipt_sha256", None)
                != self.replayed_run_receipt_sha256
                or getattr(run, "profitability_report_authority_receipt_sha256", None)
                != self.profitability_report_authority_receipt_sha256
                or getattr(run, "profitability_report_source_receipt_sha256", None)
                != self.profitability_report_source_receipt_sha256
                or getattr(run, "profitability_report_commit_receipt_sha256", None)
                != self.profitability_report_commit_receipt_sha256
                or getattr(run, "profitability_report_committed_at_ms", None)
                != self.profitability_report_committed_at_ms
                or getattr(run, "prequential_state_head_receipt_sha256", None)
                != self.profitability_report_state_receipt_sha256
                or getattr(run, "prequential_state_head_source_receipt_sha256", None)
                != self.profitability_report_state_source_receipt_sha256
                or getattr(run, "prequential_state_head_commit_receipt_sha256", None)
                != self.profitability_report_state_commit_receipt_sha256
                or getattr(run, "prequential_state_head_committed_at_ms", None)
                != self.profitability_report_state_committed_at_ms
                or tuple(getattr(run, "outer_fold_seal_authority_receipts", ()))
                != self.outer_fold_seal_receipts
                or getattr(run, "policy_schedule_disposition", None)
                != self.policy_schedule_disposition
                or getattr(run, "profitability_gates_passed", None)
                != self.profitability_gates_passed
            ):
                raise MassiveAdaptiveRLFullColdReplayV1Error(
                    "full cold-replay runtime result differs"
                )
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def _validate_replayed_run(
    *,
    run: object,
    experiment_id: str,
    manifest_receipt: str,
    execution_registration_receipt: str,
) -> None:
    # Local import avoids a module cycle: the authoritative runner owns replay.
    from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
        MassiveAdaptiveRLPrequentialRunV5,
    )

    if type(run) is not MassiveAdaptiveRLPrequentialRunV5:
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "full cold replay requires the exact V5 report-stage result"
        )
    run.validate()
    if (
        run.experiment_id != experiment_id
        or run.manifest_v5_receipt_sha256 != manifest_receipt
        or run.execution_implementation_registration_authority_receipt_sha256
        != execution_registration_receipt
        or run.prequential_state_head_stage != "profitability-report-published"
        or run.sealed_outer_fold_indices != (0, 1, 2, 3)
        or not run.profitability_reporting_authorized
        or run.end_to_end_profitability_execution_complete
        or run.next_required_stage != "full-cold-replay-verification"
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "V5 report boundary is not ready for cold replay completion"
        )


_VERIFIED_REPLAY_EVIDENCE_SEAL_V1 = object()


@dataclass(frozen=True, slots=True)
class _MassiveAdaptiveRLVerifiedReplayEvidenceV1:
    """Opaque evidence issued only after the package verifier reconstructs V5."""

    expected_run_receipt_sha256: str
    replayed_run: object
    evidence_inventory_before: tuple[MassiveAdaptiveRLProtectedEvidenceFileV1, ...]
    evidence_inventory_after: tuple[MassiveAdaptiveRLProtectedEvidenceFileV1, ...]
    _seal: object = field(compare=False, repr=False)

    def validate(self) -> None:
        if self._seal is not _VERIFIED_REPLAY_EVIDENCE_SEAL_V1:
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "cold-replay evidence was not issued by the package verifier"
            )
        _digest("expected report-boundary receipt", self.expected_run_receipt_sha256)
        replayed_receipt = getattr(self.replayed_run, "semantic_receipt_sha256", None)
        if replayed_receipt != self.expected_run_receipt_sha256:
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "cold replay did not reproduce the expected report boundary"
            )
        before_receipt = _inventory_receipt(self.evidence_inventory_before)
        after_receipt = _inventory_receipt(self.evidence_inventory_after)
        if (
            not self.evidence_inventory_before
            or self.evidence_inventory_before != self.evidence_inventory_after
            or before_receipt != after_receipt
        ):
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "cold replay changed protected experiment evidence"
            )


def _issue_massive_adaptive_rl_verified_replay_evidence_v1(
    *,
    expected_run: object,
    replayed_run: object,
    evidence_inventory_before: Sequence[MassiveAdaptiveRLProtectedEvidenceFileV1],
    evidence_inventory_after: Sequence[MassiveAdaptiveRLProtectedEvidenceFileV1],
) -> _MassiveAdaptiveRLVerifiedReplayEvidenceV1:
    """Bind a verifier-owned reconstruction to its immutable evidence snapshots."""

    # Local import avoids the module cycle: the authoritative runner owns both
    # the expected report boundary and the reconstruction that must reproduce it.
    from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v5 import (
        MassiveAdaptiveRLPrequentialRunV5,
    )

    if (
        type(expected_run) is not MassiveAdaptiveRLPrequentialRunV5
        or type(replayed_run) is not MassiveAdaptiveRLPrequentialRunV5
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "cold replay requires exact V5 report-boundary results"
        )
    expected_run.validate()
    replayed_run.validate()
    expected_receipt = _required_digest(
        "expected report-boundary receipt",
        expected_run.semantic_receipt_sha256,
    )
    if (
        expected_run.semantic_unsigned() != replayed_run.semantic_unsigned()
        or expected_receipt != replayed_run.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "cold replay did not reproduce the expected report boundary"
        )
    evidence = _MassiveAdaptiveRLVerifiedReplayEvidenceV1(
        expected_run_receipt_sha256=expected_receipt,
        replayed_run=replayed_run,
        evidence_inventory_before=tuple(evidence_inventory_before),
        evidence_inventory_after=tuple(evidence_inventory_after),
        _seal=_VERIFIED_REPLAY_EVIDENCE_SEAL_V1,
    )
    evidence.validate()
    return evidence


def _build_body(
    *,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    verified_replay: _MassiveAdaptiveRLVerifiedReplayEvidenceV1,
) -> dict[str, object]:
    manifest.validate()
    manifest_registration.validate()
    execution_registration.validate()
    if (
        not manifest_registration.development_protocol_registered
        or not execution_registration.development_execution_registered
        or manifest_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_receipt_sha256
        != manifest.semantic_receipt_sha256
        or execution_registration.manifest_v5_registration_authority_receipt_sha256
        != manifest_registration.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "full cold-replay registration lineage differs"
        )
    if type(verified_replay) is not _MassiveAdaptiveRLVerifiedReplayEvidenceV1:
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "full cold replay requires verifier-issued evidence"
        )
    verified_replay.validate()
    replayed_run = verified_replay.replayed_run
    _validate_replayed_run(
        run=replayed_run,
        experiment_id=manifest.experiment_id,
        manifest_receipt=manifest.semantic_receipt_sha256,
        execution_registration_receipt=execution_registration.semantic_receipt_sha256,
    )
    before = verified_replay.evidence_inventory_before
    after = verified_replay.evidence_inventory_after
    before_receipt = _inventory_receipt(before)
    after_receipt = _inventory_receipt(after)
    if not before or before != after or before_receipt != after_receipt:
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "cold replay changed protected experiment evidence"
        )
    disposition = str(getattr(replayed_run, "policy_schedule_disposition"))
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_v5_receipt_sha256": manifest.semantic_receipt_sha256,
        "manifest_v5_registration_receipt_sha256": (
            manifest_registration.semantic_receipt_sha256
        ),
        "execution_implementation_registration_receipt_sha256": (
            execution_registration.semantic_receipt_sha256
        ),
        "replayed_run_receipt_sha256": getattr(replayed_run, "semantic_receipt_sha256"),
        "profitability_report_authority_receipt_sha256": getattr(
            replayed_run, "profitability_report_authority_receipt_sha256"
        ),
        "profitability_report_source_receipt_sha256": getattr(
            replayed_run, "profitability_report_source_receipt_sha256"
        ),
        "profitability_report_commit_receipt_sha256": getattr(
            replayed_run, "profitability_report_commit_receipt_sha256"
        ),
        "profitability_report_committed_at_ms": getattr(
            replayed_run, "profitability_report_committed_at_ms"
        ),
        "profitability_report_state_receipt_sha256": getattr(
            replayed_run, "prequential_state_head_receipt_sha256"
        ),
        "profitability_report_state_source_receipt_sha256": getattr(
            replayed_run, "prequential_state_head_source_receipt_sha256"
        ),
        "profitability_report_state_commit_receipt_sha256": getattr(
            replayed_run, "prequential_state_head_commit_receipt_sha256"
        ),
        "profitability_report_state_committed_at_ms": getattr(
            replayed_run, "prequential_state_head_committed_at_ms"
        ),
        "outer_fold_seal_receipts": tuple(
            getattr(replayed_run, "outer_fold_seal_authority_receipts")
        ),
        "protected_evidence_inventory": before,
        "protected_evidence_inventory_sha256": before_receipt,
        "protected_evidence_file_count": len(before),
        "before_replay_inventory_sha256": before_receipt,
        "after_replay_inventory_sha256": after_receipt,
        "nonmaterializing_replay": True,
        "policy_schedule_disposition": disposition,
        "policy_schedule_qualified": disposition == "policy-prefix-qualified",
        "profitability_gates_passed": bool(
            getattr(replayed_run, "profitability_gates_passed")
        ),
        "source_data_qualified": True,
        "lockbox_access_authorized": False,
        "live_trading_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_V1_SPEC_SHA256,
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA,
    }


def full_cold_replay_authority_relative_path_v1(
    *, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> str:
    manifest.validate()
    return (
        f"adaptive-rl/{manifest.experiment_id}/"
        "full-cold-replay-authority-v1/completion.json"
    )


def _parse_inventory_row(value: object) -> MassiveAdaptiveRLProtectedEvidenceFileV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "protected evidence inventory payload differs"
        )
    row = MassiveAdaptiveRLProtectedEvidenceFileV1(**dict(value))  # type: ignore[arg-type]
    row.validate()
    return row


def _parse(
    *, root: str | Path, loaded: LoadedMassiveSourceObject
) -> MassiveAdaptiveRLFullColdReplayAuthorityV1:
    raw = read_loaded_massive_source_bytes(root=root, loaded_source=loaded)
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "full cold-replay payload is not canonical JSON"
        )
    semantic_receipt = semantic_sha256(value)
    body = dict(value)
    body["outer_fold_seal_receipts"] = tuple(
        cast(Sequence[str], body["outer_fold_seal_receipts"])
    )
    body["protected_evidence_inventory"] = tuple(
        _parse_inventory_row(row)
        for row in cast(Sequence[object], body["protected_evidence_inventory"])
    )
    result = MassiveAdaptiveRLFullColdReplayAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256=semantic_receipt,
        _loaded_source=loaded,
    )
    result.validate()
    return result


def load_massive_adaptive_rl_full_cold_replay_authority_v1(
    *, root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV5
) -> MassiveAdaptiveRLFullColdReplayAuthorityV1:
    """Load the completion proof without granting runtime authorization."""

    relative = full_cold_replay_authority_relative_path_v1(manifest=manifest)
    return _parse(
        root=root,
        loaded=load_massive_source_bundle(
            root=root,
            relative_payload_path=relative,
            verified_at_ms=time.time_ns() // 1_000_000,
        ),
    )


def _persist_or_replay_massive_adaptive_rl_full_cold_replay_v1(
    *,
    root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV5,
    manifest_registration: MassiveAdaptiveRLManifestV5RegistrationAuthorityV1,
    execution_registration: MassiveAdaptiveRLExecutionImplementationRegistrationAuthorityV1,
    verified_replay: _MassiveAdaptiveRLVerifiedReplayEvidenceV1,
    allow_materialize: bool = True,
) -> MassiveAdaptiveRLFullColdReplayAuthorityV1:
    """Internal persistence for proof issued by the package-owned verifier."""

    if type(allow_materialize) is not bool:
        raise MassiveAdaptiveRLFullColdReplayV1Error(
            "full cold-replay materialization choice differs"
        )
    body = _build_body(
        manifest=manifest,
        manifest_registration=manifest_registration,
        execution_registration=execution_registration,
        verified_replay=verified_replay,
    )
    replayed_run = verified_replay.replayed_run
    provisional = MassiveAdaptiveRLFullColdReplayAuthorityV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    expected = replace(
        provisional,
        semantic_receipt_sha256=semantic_sha256(provisional.semantic_unsigned()),
        runtime_cold_replay_replayed=True,
        development_full_cold_replay_verified=True,
        _runtime_replayed_run=replayed_run,
    )
    expected.validate()
    relative = full_cold_replay_authority_relative_path_v1(manifest=manifest)
    with massive_adaptive_rl_experiment_materialization_lock_v1(
        artifact_root=root, experiment_id=manifest.experiment_id
    ):
        payload = Path(root) / relative
        paths = (
            payload,
            payload.with_name(payload.name + ".receipt.json"),
            payload.with_name(payload.name + ".commit.json"),
        )
        present = tuple(path.exists() or path.is_symlink() for path in paths)
        if any(present) and not all(present):
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "full cold-replay transaction is incomplete"
            )
        if not all(present):
            if not allow_materialize:
                raise MassiveAdaptiveRLFullColdReplayV1Error(
                    "full cold-replay completion proof is absent"
                )
            committed_at_ms = (
                max(
                    time.time_ns() // 1_000_000,
                    expected.profitability_report_state_committed_at_ms,
                )
                + 1
            )
            capability = issue_massive_adaptive_rl_manifest_v5_prequential_outer_execution_capability_v1(
                root=root, authority=manifest_registration
            )
            with massive_adaptive_rl_manifest_v5_writer_scope_v1(
                root=root, capability=capability
            ):
                publish_massive_source_object(
                    stream=BytesIO(
                        canonical_json_file_bytes(expected.semantic_unsigned())
                    ),
                    root=root,
                    relative_payload_path=relative,
                    dataset_id=MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_DATASET,
                    source_object_key=relative,
                    requested_at_ms=committed_at_ms,
                    downloaded_at_ms=committed_at_ms,
                    schema_sha256=(
                        MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256
                    ),
                    entitlement_receipt_sha256=expected.semantic_receipt_sha256,
                    committed_at_ms=committed_at_ms,
                    request_id=(
                        f"ADAPTIVE-RL-FULL-COLD-REPLAY-V1-{manifest.experiment_id}"
                    ),
                )
        parsed = load_massive_adaptive_rl_full_cold_replay_authority_v1(
            root=root, manifest=manifest
        )
        if parsed.semantic_unsigned() != expected.semantic_unsigned():
            raise MassiveAdaptiveRLFullColdReplayV1Error(
                "full cold-replay completion proof does not replay"
            )
        result = replace(
            parsed,
            runtime_cold_replay_replayed=True,
            development_full_cold_replay_verified=True,
            _runtime_replayed_run=replayed_run,
        )
        result.validate()
        return result


__all__ = [
    "MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_DATASET",
    "MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SCHEMA_SHA256",
    "MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_AUTHORITY_V1_SOURCE_SHA256",
    "MASSIVE_ADAPTIVE_RL_FULL_COLD_REPLAY_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFullColdReplayAuthorityV1",
    "MassiveAdaptiveRLFullColdReplayV1Error",
    "MassiveAdaptiveRLProtectedEvidenceFileV1",
    "full_cold_replay_authority_relative_path_v1",
    "load_massive_adaptive_rl_full_cold_replay_authority_v1",
    "massive_adaptive_rl_protected_evidence_inventory_v1",
]
