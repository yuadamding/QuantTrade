"""Controller-side atomic Pod-attestation publication for M03R-v16."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


class M03RV16LifecycleControllerError(RuntimeError):
    """A lifecycle-controller observation or publication step drifted."""


@dataclass(frozen=True, slots=True)
class M03RV16PodObservation:
    completion_index: int
    pod_uid: str
    pod_name: str
    node_name: str
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


def publish_m03r_v16_pod_runtime_attestation_after_annotation_patch(
    *,
    package: M03RV16PackagePlan,
    authorization: M03RV16ExecutionAuthorization,
    admission: M03RV16AdmittedJobAuthority,
    launch: M03RV16PhaseLaunchAuthority,
    observation: M03RV16PodObservation,
    output_root_sha256: str,
    authority_root: str | Path,
    patch_annotations: Callable[[str, Mapping[str, str]], None],
    read_annotations: Callable[[str], Mapping[str, str]],
) -> M03RV16PublishedPodAttestation:
    """Patch and observe annotations before the atomic final link appears."""

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
    if final_path.exists() or final_path.is_symlink():
        raise M03RV16LifecycleControllerError(
            "V16 Pod attestation final path already exists"
        )
    annotations = {
        _PATH_ANNOTATION: relative_path,
        _FILE_ANNOTATION: pod_runtime_attestation_file_sha256(attestation),
        _RECEIPT_ANNOTATION: attestation.receipt_sha256,
    }
    patch_annotations(observation.pod_name, annotations)
    observed = read_annotations(observation.pod_name)
    if any(observed.get(key) != value for key, value in annotations.items()):
        raise M03RV16LifecycleControllerError(
            "V16 Pod attestation annotations were not observed exactly"
        )
    if final_path.exists() or final_path.is_symlink():
        raise M03RV16LifecycleControllerError(
            "V16 Pod attestation became visible before controller publication"
        )
    observed_file_sha = write_m03r_v16_pod_runtime_attestation(
        final_path, attestation
    )
    if observed_file_sha != annotations[_FILE_ANNOTATION]:
        raise M03RV16LifecycleControllerError(
            "V16 published Pod attestation bytes drifted"
        )
    return M03RV16PublishedPodAttestation(
        final_path=final_path,
        file_sha256=observed_file_sha,
        receipt_sha256=attestation.receipt_sha256,
        relative_path=relative_path,
        patched_annotations=annotations,
    )


__all__ = [
    "M03RV16LifecycleControllerError",
    "M03RV16PodObservation",
    "M03RV16PublishedPodAttestation",
    "publish_m03r_v16_pod_runtime_attestation_after_annotation_patch",
]
