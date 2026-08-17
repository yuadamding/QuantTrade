"""Controller-side atomic Pod-attestation publication for M03R-v16."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)
from rl_quant.training.top2000_m03r_v16_activation import (
    M03RV16AdmittedJobAuthority,
    M03RV16PhaseLaunchAuthority,
    _issue_m03r_v16_pod_runtime_attestation,
    pod_runtime_attestation_file_sha256,
    write_m03r_v16_pod_runtime_attestation,
)
from rl_quant.training.top2000_m03r_v16_package import (
    M03RV16ExecutionAuthorization,
    M03RV16PackagePlan,
)

_PATH_ANNOTATION = "rl-quant/pod-runtime-attestation-path"
_FILE_ANNOTATION = "rl-quant/pod-runtime-attestation-file-sha256"
_RECEIPT_ANNOTATION = "rl-quant/pod-runtime-attestation-receipt-sha256"
_STORAGE_ISSUER = object()
_MAX_STORAGE_EVIDENCE_BYTES = 1024 * 1024


class M03RV16LifecycleControllerError(RuntimeError):
    """A lifecycle-controller observation or publication step drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16PodObservation:
    completion_index: int
    pod_uid: str
    pod_name: str
    node_name: str
    observed_owner_job_uid: str
    observed_owner_job_name: str
    observed_completion_index: int
    observed_pod_resource_version: str
    attested_container_name: str
    attested_container_kind: Literal["init", "app"]
    observed_spec_image: str
    observed_status_image: str
    observed_status_image_id: str


@dataclass(frozen=True, slots=True)
class M03RV16PublishedPodAttestation:
    final_path: Path
    file_sha256: str
    receipt_sha256: str
    relative_path: str
    patched_annotations: Mapping[str, str]
    controller_transaction_receipt_sha256: str


@dataclass(frozen=True, slots=True)
class M03RV16PodPatchPrecondition:
    pod_name: str
    pod_uid: str
    pod_resource_version: str


@dataclass(frozen=True, slots=True)
class M03RV16PodAnnotationReadback:
    pod_uid: str
    pod_resource_version: str
    annotations: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class M03RV16StorageSemanticsEvidence:
    authority_root_sha256: str
    observer_root_sha256: str
    payload_sha256: str
    hard_link_supported: bool
    directory_fsync_supported: bool
    observer_read_matched: bool
    observer_same_file: bool
    duplicate_publication_rejected: bool
    distinct_observer_mount: bool
    _issuer: object = field(repr=False)
    schema: str = (
        "rl-quant.top2000-dev.m03r-v16-storage-semantics-evidence-v3"
    )

    @property
    def receipt_sha256(self) -> str:
        payload = asdict(self)
        payload.pop("_issuer")
        return semantic_sha256(payload)

    def validate_for(
        self,
        authority_root: str | Path,
        observer_root: str | Path,
    ) -> None:
        expected_authority = semantic_sha256(
            {"resolved_root": str(Path(authority_root).resolve())}
        )
        expected_observer = semantic_sha256(
            {"resolved_root": str(Path(observer_root).resolve())}
        )
        valid_payload_digest = len(self.payload_sha256) == 64 and all(
            character in "0123456789abcdef"
            for character in self.payload_sha256
        )
        if (
            self._issuer is not _STORAGE_ISSUER
            or self.schema
            != "rl-quant.top2000-dev.m03r-v16-storage-semantics-evidence-v3"
            or self.authority_root_sha256 != expected_authority
            or self.observer_root_sha256 != expected_observer
            or self.authority_root_sha256 == self.observer_root_sha256
            or not self.distinct_observer_mount
            or not valid_payload_digest
            or not self.hard_link_supported
            or not self.directory_fsync_supported
            or not self.observer_read_matched
            or not self.observer_same_file
            or not self.duplicate_publication_rejected
        ):
            raise M03RV16LifecycleControllerError(
                "V16 append-only storage semantics are unqualified"
            )


def qualify_m03r_v16_append_only_storage(
    authority_root: str | Path,
    *,
    observer_root: str | Path,
    publish_observer_view: Callable[[Path, Path], None] | None = None,
) -> M03RV16StorageSemanticsEvidence:
    """Qualify link/fsync/create-only behavior in one disposable namespace."""

    root = Path(authority_root).resolve()
    observer = Path(observer_root).resolve()
    if root == observer:
        raise M03RV16LifecycleControllerError(
            "V16 storage qualification requires a distinct observer mount"
        )
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    token = secrets.token_hex(12)
    relative_directory = Path("storage-probes") / token
    probe = root / relative_directory
    probe.mkdir(mode=0o700, parents=True, exist_ok=False)
    temporary = probe / "payload.tmp"
    final = probe / "payload.final"
    observed_path = observer / relative_directory / "payload.final"
    payload = f"m03r-v16-storage-probe:{token}\n".encode("ascii")
    payload_sha = hashlib.sha256(payload).hexdigest()
    directory_fsync_supported = False
    observer_read_matched = False
    observer_same_file = False
    duplicate_rejected = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, final)
        directory_descriptor = os.open(probe, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
            directory_fsync_supported = True
        finally:
            os.close(directory_descriptor)
        if publish_observer_view is not None:
            publish_observer_view(final, observed_path)
        observed = observed_path.read_bytes()
        observer_read_matched = (
            observed == payload and hashlib.sha256(observed).hexdigest() == payload_sha
        )
        published_stat = final.stat()
        observed_stat = observed_path.stat()
        observer_same_file = (
            published_stat.st_dev == observed_stat.st_dev
            and published_stat.st_ino == observed_stat.st_ino
        )
        try:
            os.link(temporary, final)
        except FileExistsError:
            duplicate_rejected = True
        else:
            raise M03RV16LifecycleControllerError(
                "V16 storage allowed duplicate immutable publication"
            )
    except OSError as exc:
        raise M03RV16LifecycleControllerError(
            "V16 append-only storage semantics are unsupported"
        ) from exc
    finally:
        final.unlink(missing_ok=True)
        try:
            observed_path.unlink(missing_ok=True)
        except OSError:
            # A real observer is mounted read-only.  Removing the authoritative
            # directory entry above makes its observer alias disappear.
            if observed_path.exists():
                raise
        temporary.unlink(missing_ok=True)
        try:
            probe.rmdir()
            probe.parent.rmdir()
        except OSError:
            pass
    evidence = M03RV16StorageSemanticsEvidence(
        authority_root_sha256=semantic_sha256(
            {"resolved_root": str(root)}
        ),
        observer_root_sha256=semantic_sha256(
            {"resolved_root": str(observer)}
        ),
        payload_sha256=payload_sha,
        hard_link_supported=True,
        directory_fsync_supported=directory_fsync_supported,
        observer_read_matched=observer_read_matched,
        observer_same_file=observer_same_file,
        duplicate_publication_rejected=duplicate_rejected,
        distinct_observer_mount=True,
        _issuer=_STORAGE_ISSUER,
    )
    evidence.validate_for(root, observer)
    return evidence


def write_m03r_v16_storage_semantics_evidence(
    path: str | Path,
    evidence: M03RV16StorageSemanticsEvidence,
    *,
    authority_root: str | Path,
    observer_root: str | Path,
) -> str:
    """Publish one create-only storage qualification receipt."""

    evidence.validate_for(authority_root, observer_root)
    payload = asdict(evidence)
    payload.pop("_issuer")
    value = {
        "evidence": payload,
        "receipt_sha256": evidence.receipt_sha256,
    }
    raw = canonical_json_file_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o440,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def load_m03r_v16_storage_semantics_evidence(
    path: str | Path,
    *,
    expected_file_sha256: str,
    authority_root: str | Path,
    observer_root: str | Path,
) -> M03RV16StorageSemanticsEvidence:
    """Reissue storage evidence only from the exact immutable file."""

    candidate = Path(path)
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise M03RV16LifecycleControllerError(
            "V16 storage evidence is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_STORAGE_EVIDENCE_BYTES
        ):
            raise M03RV16LifecycleControllerError(
                "V16 storage evidence type or size drifted"
            )
        raw = os.read(descriptor, before.st_size + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
        or hashlib.sha256(raw).hexdigest() != expected_file_sha256
    ):
        raise M03RV16LifecycleControllerError(
            "V16 storage evidence changed or its hash drifted"
        )
    try:
        outer = json.loads(raw)
        row = dict(outer["evidence"])
        evidence = M03RV16StorageSemanticsEvidence(
            **row,
            _issuer=_STORAGE_ISSUER,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise M03RV16LifecycleControllerError(
            "V16 storage evidence is malformed"
        ) from exc
    if (
        not isinstance(outer, dict)
        or raw != canonical_json_file_bytes(outer)
        or outer.get("receipt_sha256") != evidence.receipt_sha256
    ):
        raise M03RV16LifecycleControllerError(
            "V16 storage evidence receipt drifted"
        )
    evidence.validate_for(authority_root, observer_root)
    return evidence


def publish_m03r_v16_pod_runtime_attestation_after_annotation_patch(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: M03RV16PhaseLaunchAuthority,
    observation: M03RV16PodObservation,
    output_root_sha256: str,
    authority_root: str | Path,
    observer_root: str | Path,
    storage_evidence: M03RV16StorageSemanticsEvidence,
    storage_authority_identity_root: str | Path | None = None,
    storage_observer_identity_root: str | Path | None = None,
    patch_annotations: Callable[
        [M03RV16PodPatchPrecondition, Mapping[str, str]], None
    ],
    read_annotations: Callable[
        [M03RV16PodPatchPrecondition], M03RV16PodAnnotationReadback
    ],
) -> M03RV16PublishedPodAttestation:
    """Patch and observe annotations before the atomic final link appears."""

    storage_evidence.validate_for(
        authority_root
        if storage_authority_identity_root is None
        else storage_authority_identity_root,
        observer_root
        if storage_observer_identity_root is None
        else storage_observer_identity_root,
    )
    if (
        launch.storage_semantics_receipt_sha256
        != storage_evidence.receipt_sha256
        or launch.storage_authority_root_sha256
        != storage_evidence.authority_root_sha256
        or launch.storage_observer_root_sha256
        != storage_evidence.observer_root_sha256
    ):
        raise M03RV16LifecycleControllerError(
            "V16 launch authority is not bound to storage qualification"
        )
    if (
        observation.observed_owner_job_uid != admission.job_uid
        or not observation.observed_owner_job_name
        or observation.observed_completion_index != observation.completion_index
        or not observation.observed_pod_resource_version
    ):
        raise M03RV16LifecycleControllerError(
            "V16 observed Pod ownership or completion identity drifted"
        )
    relative_path = launch.pod_runtime_attestation_relative_path(
        observation.completion_index
    )
    try:
        attestation = _issue_m03r_v16_pod_runtime_attestation(
            package=package,
            authorization=authorization,
            admission=admission,
            launch=launch,
            completion_index=observation.completion_index,
            pod_uid=observation.pod_uid,
            pod_name=observation.pod_name,
            node_name=observation.node_name,
            observed_owner_job_uid=observation.observed_owner_job_uid,
            observed_owner_job_name=observation.observed_owner_job_name,
            observed_completion_index=observation.observed_completion_index,
            observed_pod_resource_version=(
                observation.observed_pod_resource_version
            ),
            relative_path=relative_path,
            attested_container_name=observation.attested_container_name,
            attested_container_kind=observation.attested_container_kind,
            observed_spec_image=observation.observed_spec_image,
            observed_status_image=observation.observed_status_image,
            observed_status_image_id=observation.observed_status_image_id,
            output_root_sha256=output_root_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV16LifecycleControllerError(
            "V16 Pod observation cannot issue a runtime attestation"
        ) from exc
    final_path = Path(authority_root) / relative_path
    if final_path.is_symlink():
        raise M03RV16LifecycleControllerError(
            "V16 Pod attestation final path is a symlink"
        )
    annotations = {
        _PATH_ANNOTATION: relative_path,
        _FILE_ANNOTATION: pod_runtime_attestation_file_sha256(attestation),
        _RECEIPT_ANNOTATION: attestation.receipt_sha256,
    }
    precondition = M03RV16PodPatchPrecondition(
        pod_name=observation.pod_name,
        pod_uid=observation.pod_uid,
        pod_resource_version=observation.observed_pod_resource_version,
    )
    patch_annotations(precondition, annotations)
    observed = read_annotations(precondition)
    if not isinstance(observed, M03RV16PodAnnotationReadback):
        raise M03RV16LifecycleControllerError(
            "V16 Pod annotation readback lacks UID/resource-version evidence"
        )
    if (
        observed.pod_uid != observation.pod_uid
        or not observed.pod_resource_version
        or observed.pod_resource_version
        == observation.observed_pod_resource_version
        or any(
            observed.annotations.get(key) != value
            for key, value in annotations.items()
        )
    ):
        raise M03RV16LifecycleControllerError(
            "V16 Pod attestation annotations were not observed exactly"
        )
    if not final_path.exists():
        for stale in final_path.parent.glob(
            f".{final_path.name}.{attestation.receipt_sha256}.*.tmp"
        ):
            if stale.is_symlink() or not stale.is_file():
                raise M03RV16LifecycleControllerError(
                    "V16 stale attestation temporary path is unsafe"
                )
            stale.unlink()
    observed_file_sha = write_m03r_v16_pod_runtime_attestation(
        final_path, attestation
    )
    if observed_file_sha != annotations[_FILE_ANNOTATION]:
        raise M03RV16LifecycleControllerError(
            "V16 published Pod attestation bytes drifted"
        )
    transaction_receipt = semantic_sha256(
        {
            "schema": (
                "rl-quant.top2000-dev."
                "m03r-v16-pod-attestation-publication-v2"
            ),
            "pod_uid": observation.pod_uid,
            "pre_patch_resource_version": (
                observation.observed_pod_resource_version
            ),
            "post_patch_resource_version": observed.pod_resource_version,
            "relative_path": relative_path,
            "file_sha256": observed_file_sha,
            "attestation_receipt_sha256": attestation.receipt_sha256,
            "storage_semantics_receipt_sha256": storage_evidence.receipt_sha256,
            "annotations": dict(annotations),
        }
    )
    return M03RV16PublishedPodAttestation(
        final_path=final_path,
        file_sha256=observed_file_sha,
        receipt_sha256=attestation.receipt_sha256,
        relative_path=relative_path,
        patched_annotations=annotations,
        controller_transaction_receipt_sha256=transaction_receipt,
    )


__all__ = [
    "M03RV16LifecycleControllerError",
    "M03RV16PodAnnotationReadback",
    "M03RV16PodObservation",
    "M03RV16PodPatchPrecondition",
    "M03RV16PublishedPodAttestation",
    "M03RV16StorageSemanticsEvidence",
    "load_m03r_v16_storage_semantics_evidence",
    "publish_m03r_v16_pod_runtime_attestation_after_annotation_patch",
    "qualify_m03r_v16_append_only_storage",
    "write_m03r_v16_storage_semantics_evidence",
]
