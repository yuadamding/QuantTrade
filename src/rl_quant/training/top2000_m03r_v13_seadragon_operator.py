"""One-attempt suspended-Job preparation for M03R-v13 on Seadragon.

This operator is the sole v13 create surface.  It performs one server dry-run,
proves exact-name absence twice, issues exactly one create request, reconciles
that request for the full bounded uncertainty window, binds two stable
suspended reads with zero UID-owned Pods, and publishes the exact activation
request consumed by an attach-only supervisor.  It never activates a Job and
never retries create.

All work is development-only, non-PHI research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from rl_quant.training import top2000_m03r_v7_seadragon_lifecycle as common
from rl_quant.training import top2000_m03r_v13_seadragon_lifecycle as lifecycle
from rl_quant.training.hold30_alpha_m03r_v7_kubernetes import (
    M03R_TOP2000_KUBERNETES_CONTEXT,
    M03R_TOP2000_KUBERNETES_NAMESPACE,
    M03RV7ExactJobCleanupRequest,
    build_m03r_v7_exact_job_activation_request,
)
from rl_quant.training.top2000_m03r_v13_kubernetes import (
    M03R_V13_RENDERED_JOB_SCHEMA,
    M03RV13RenderedJob,
    bind_m03r_v13_admitted_suspended_job,
)

SEADRAGON_KUBECTL: Final = "/risapps/noarch/kubectl/1.28.4/bin/kubectl"
SEADRAGON_KUBECONFIG: Final = "/rsrch8/home/bcb/yding4/.kube/config"
SEADRAGON_QUANTTRADE_ROOT: Final = "/rsrch8/home/bcb/yding4/quant/training"
CREATE_CONFIG_SCHEMA: Final = "rl-quant.top2000-dev.m03r-v13-create-config-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class M03RV13SeadragonOperatorError(RuntimeError):
    """The one-attempt suspended preparation failed closed."""


class M03RV13CreateAttachRequired(M03RV13SeadragonOperatorError):
    """Create may have been accepted; the exact identity must not be retried."""


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise M03RV13SeadragonOperatorError(f"{name} must be a lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M03RV13SeadragonOperatorError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _project_path(path: str, label: str) -> Path:
    value = Path(path)
    root = Path(SEADRAGON_QUANTTRADE_ROOT)
    if not value.is_absolute():
        raise M03RV13SeadragonOperatorError(f"{label} must be absolute")
    try:
        value.relative_to(root)
    except ValueError as exc:
        raise M03RV13SeadragonOperatorError(
            f"{label} must stay under the approved QuantTrade root"
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class M03RV13CreateOperatorConfig:
    mode: Literal["static", "capacity", "predictive"]
    job_name: str
    run_id: str
    rendered_path: str
    rendered_file_sha256: str
    manifest_path: str
    manifest_file_sha256: str
    evidence_root: str
    binding_output_path: str
    activation_output_path: str
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    source_archive_sha256: str
    capacity_receipt_sha256: str
    operator_source_sha256: str
    completions: int
    parallelism: int
    request_timeout_seconds: int = 30
    reconciliation_poll_seconds: float = 1.0
    kubectl_path: str = SEADRAGON_KUBECTL
    kubeconfig_path: str = SEADRAGON_KUBECONFIG
    context: str = M03R_TOP2000_KUBERNETES_CONTEXT
    namespace: str = M03R_TOP2000_KUBERNETES_NAMESPACE
    schema: str = CREATE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "rendered_file_sha256",
            "manifest_file_sha256",
            "package_plan_sha256",
            "execution_authorization_receipt_sha256",
            "source_archive_sha256",
            "operator_source_sha256",
        ):
            _require_sha256(name, cast(str, getattr(self, name)))
        expected_capacity = (
            "not-yet-created"
            if self.mode in {"static", "capacity"}
            else self.capacity_receipt_sha256
        )
        if self.mode == "predictive":
            _require_sha256("capacity_receipt_sha256", self.capacity_receipt_sha256)
        if (
            self.schema != CREATE_CONFIG_SCHEMA
            or self.mode not in {"static", "capacity", "predictive"}
            or not self.job_name
            or not self.run_id
            or self.context != M03R_TOP2000_KUBERNETES_CONTEXT
            or self.namespace != M03R_TOP2000_KUBERNETES_NAMESPACE
            or self.kubectl_path != SEADRAGON_KUBECTL
            or self.kubeconfig_path != SEADRAGON_KUBECONFIG
            or (self.mode, self.completions)
            not in {
                ("static", 1),
                ("capacity", 1),
                ("predictive", 2),
            }
            or not 1 <= self.parallelism <= self.completions
            or (self.mode != "static" and self.parallelism * 2 > 4)
            or self.request_timeout_seconds < 5
            or not 0.1 <= self.reconciliation_poll_seconds <= 5.0
            or self.capacity_receipt_sha256 != expected_capacity
        ):
            raise M03RV13SeadragonOperatorError("v13 create config identity drifted")
        for name in (
            "rendered_path",
            "manifest_path",
            "evidence_root",
            "binding_output_path",
            "activation_output_path",
        ):
            _project_path(cast(str, getattr(self, name)), name)


@dataclass(frozen=True, slots=True)
class M03RV13CreateAttempt:
    returncode: int
    stdout: bytes
    stderr: bytes


class CreateTransport(Protocol):
    def server_dry_run(self, manifest_path: Path) -> Mapping[str, Any]: ...

    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None: ...

    def get_pods_by_job_name(self) -> tuple[Mapping[str, Any], ...]: ...

    def get_owned_pods(self, job_uid: str) -> tuple[Mapping[str, Any], ...]: ...

    def create_once(self, manifest_path: Path) -> M03RV13CreateAttempt: ...

    def delete(
        self, request: M03RV7ExactJobCleanupRequest, options_path: Path
    ) -> None: ...


class OneCreateKubectl:
    """Narrow kubectl transport containing one explicit create method."""

    def __init__(
        self,
        *,
        kubectl_path: str,
        kubeconfig_path: str,
        context: str,
        namespace: str,
        job_name: str,
        request_timeout_seconds: int,
    ) -> None:
        self.kubectl_path = kubectl_path
        self.kubeconfig_path = kubeconfig_path
        self.context = context
        self.namespace = namespace
        self.job_name = job_name
        self.request_timeout_seconds = request_timeout_seconds
        self._create_attempted = False

    def _run(
        self, arguments: Sequence[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        if not arguments or arguments[0] not in {"create", "get", "delete"}:
            raise M03RV13SeadragonOperatorError(
                "create transport rejected kubectl verb"
            )
        command = [
            self.kubectl_path,
            "--context",
            self.context,
            "--namespace",
            self.namespace,
            f"--request-timeout={self.request_timeout_seconds}s",
            *arguments,
        ]
        environment = {
            "KUBECONFIG": self.kubeconfig_path,
            "PATH": str(Path(self.kubectl_path).parent),
            "LANG": "C",
            "LC_ALL": "C",
        }
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=self.request_timeout_seconds + 5,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise M03RV13SeadragonOperatorError(
                "bounded kubectl create/get invocation failed"
            ) from exc
        if check and completed.returncode != 0:
            raise M03RV13SeadragonOperatorError(
                completed.stderr.decode("utf-8", errors="replace")[-2000:]
            )
        return completed

    def server_dry_run(self, manifest_path: Path) -> Mapping[str, Any]:
        payload = self._run(
            ("create", "--dry-run=server", "-f", str(manifest_path), "-o", "json")
        ).stdout
        return _mapping(json.loads(payload), "server dry-run Job")

    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None:
        arguments = ["get", "job", self.job_name]
        if allow_absent:
            arguments.append("--ignore-not-found")
        arguments.extend(("-o", "json"))
        payload = self._run(arguments).stdout
        if not payload.strip():
            return None
        return _mapping(json.loads(payload), "observed Job")

    def _pods(self, selector: str) -> tuple[Mapping[str, Any], ...]:
        payload = _mapping(
            json.loads(self._run(("get", "pods", "-l", selector, "-o", "json")).stdout),
            "observed PodList",
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise M03RV13SeadragonOperatorError("PodList items are invalid")
        return tuple(_mapping(item, "observed Pod") for item in items)

    def get_pods_by_job_name(self) -> tuple[Mapping[str, Any], ...]:
        return self._pods(f"job-name={self.job_name}")

    def get_owned_pods(self, job_uid: str) -> tuple[Mapping[str, Any], ...]:
        pods = self._pods(f"job-name={self.job_name},controller-uid={job_uid}")
        common._validate_owned_pods(pods, expected_uid=job_uid)
        return pods

    def create_once(self, manifest_path: Path) -> M03RV13CreateAttempt:
        if self._create_attempted:
            raise M03RV13SeadragonOperatorError("create request must never be retried")
        self._create_attempted = True
        completed = self._run(
            ("create", "-f", str(manifest_path), "-o", "json"), check=False
        )
        return M03RV13CreateAttempt(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def delete(self, request: M03RV7ExactJobCleanupRequest, options_path: Path) -> None:
        if request.job_name != self.job_name or request.namespace != self.namespace:
            raise M03RV13SeadragonOperatorError("cleanup target is not this Job")
        raw_uri = f"/apis/batch/v1/namespaces/{self.namespace}/jobs/{self.job_name}"
        self._run(("delete", "--raw", raw_uri, "-f", str(options_path)))


class _CleanupAdapter:
    """Present the accepted UID as the attach lifecycle's narrow transport."""

    def __init__(self, transport: CreateTransport, job_uid: str) -> None:
        self.transport = transport
        self.job_uid = job_uid

    def get_job(self, *, allow_absent: bool = False) -> Mapping[str, Any] | None:
        return self.transport.get_job(allow_absent=allow_absent)

    def get_owned_pods(self) -> tuple[Mapping[str, Any], ...]:
        return self.transport.get_owned_pods(self.job_uid)

    def get_pod_log(self, pod_name: str, *, limit_bytes: int) -> bytes:
        del pod_name, limit_bytes
        raise M03RV13SeadragonOperatorError("preactivation cleanup has no Pod logs")

    def activate(self, request: Any) -> Mapping[str, Any]:
        del request
        raise M03RV13SeadragonOperatorError("create operator cannot activate")

    def delete(self, request: M03RV7ExactJobCleanupRequest, options_path: Path) -> None:
        self.transport.delete(request, options_path)


@dataclass(frozen=True, slots=True)
class _AcceptedConfig:
    job_name: str
    run_id: str
    job_uid: str
    package_plan_sha256: str
    execution_authorization_receipt_sha256: str
    source_archive_sha256: str
    capacity_receipt_sha256: str
    request_timeout_seconds: int


def _load_config(path: Path, expected_sha256: str) -> M03RV13CreateOperatorConfig:
    _require_sha256("create config file", expected_sha256)
    payload = _mapping(
        common._read_json_file(path, expected_sha256=expected_sha256),
        "v13 create config",
    )
    try:
        return M03RV13CreateOperatorConfig(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise M03RV13SeadragonOperatorError("v13 create config is invalid") from exc


def _load_rendered(
    config: M03RV13CreateOperatorConfig,
) -> M03RV13RenderedJob:
    rendered_path = common._regular_no_symlink(
        Path(config.rendered_path), label="rendered Job"
    )
    manifest_path = common._regular_no_symlink(
        Path(config.manifest_path), label="Job manifest"
    )
    if (
        _file_sha256(rendered_path) != config.rendered_file_sha256
        or _file_sha256(manifest_path) != config.manifest_file_sha256
    ):
        raise M03RV13SeadragonOperatorError("rendered or manifest file hash drifted")
    rendered_payload = dict(
        _mapping(common._read_json_file(rendered_path), "rendered Job")
    )
    manifest = dict(_mapping(common._read_json_file(manifest_path), "Job manifest"))
    try:
        rendered = M03RV13RenderedJob(**rendered_payload)
    except (TypeError, ValueError) as exc:
        raise M03RV13SeadragonOperatorError("rendered Job receipt is invalid") from exc
    metadata = _mapping(manifest.get("metadata"), "manifest metadata")
    annotations = _mapping(metadata.get("annotations"), "manifest annotations")
    if (
        rendered.schema != M03R_V13_RENDERED_JOB_SCHEMA
        or rendered.manifest != manifest
        or rendered.manifest_sha256 != _compact_sha256(manifest)
        or rendered.pod_template_sha256
        != _compact_sha256(
            _mapping(
                _mapping(
                    _mapping(manifest.get("spec"), "manifest spec").get("template"),
                    "manifest template",
                ).get("spec"),
                "manifest Pod spec",
            )
        )
        or rendered.mode != config.mode
        or rendered.completions != config.completions
        or rendered.parallelism != config.parallelism
        or rendered.package_plan_sha256 != config.package_plan_sha256
        or rendered.execution_authorization_receipt_sha256
        != config.execution_authorization_receipt_sha256
        or metadata.get("name") != config.job_name
        or metadata.get("namespace") != config.namespace
        or annotations.get("rl-quant/run-id") != config.run_id
        or annotations.get("rl-quant/package-plan-sha256") != config.package_plan_sha256
        or annotations.get("rl-quant/execution-authorization-sha256")
        != config.execution_authorization_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
        or annotations.get("rl-quant/capacity-receipt-sha256")
        != config.capacity_receipt_sha256
        or annotations.get("rl-quant/economic-panel-authorized") != "false"
        or rendered.economic_panel_authorized
    ):
        raise M03RV13SeadragonOperatorError(
            "rendered Job and create config identity drifted"
        )
    return rendered


def _bind_rendered(
    *,
    rendered: M03RV13RenderedJob,
    first_read: dict[str, Any],
    second_read: dict[str, Any],
    attached_owned_pod_uids: tuple[str, ...],
) -> Any:
    return bind_m03r_v13_admitted_suspended_job(
        rendered=rendered,
        first_read=first_read,
        second_read=second_read,
        attached_owned_pod_uids=attached_owned_pod_uids,
    )


def _synthetic_dry_read(
    dry: Mapping[str, Any], config: M03RV13CreateOperatorConfig
) -> dict[str, Any]:
    value = json.loads(json.dumps(dry))
    spec = _mapping(value.get("spec"), "server dry-run spec")
    selector = _mapping(spec.get("selector"), "server dry-run selector")
    match_labels = _mapping(selector.get("matchLabels"), "server dry-run matchLabels")
    uid = match_labels.get("batch.kubernetes.io/controller-uid")
    if not isinstance(uid, str) or not uid:
        raise M03RV13SeadragonOperatorError(
            "server dry-run did not generate the controller UID selector"
        )
    metadata = cast(dict[str, Any], value["metadata"])
    metadata["uid"] = uid
    metadata["resourceVersion"] = "server-dry-run"
    metadata["namespace"] = config.namespace
    return cast(dict[str, Any], value)


def _validate_created_identity(
    job: Mapping[str, Any],
    config: M03RV13CreateOperatorConfig,
    *,
    expected_uid: str | None = None,
) -> str:
    metadata = _mapping(job.get("metadata"), "created Job metadata")
    annotations = _mapping(metadata.get("annotations"), "created Job annotations")
    spec = _mapping(job.get("spec"), "created Job spec")
    uid = metadata.get("uid")
    if (
        metadata.get("name") != config.job_name
        or metadata.get("namespace") != config.namespace
        or not isinstance(uid, str)
        or not uid
        or (expected_uid is not None and uid != expected_uid)
        or annotations.get("rl-quant/run-id") != config.run_id
        or annotations.get("rl-quant/package-plan-sha256") != config.package_plan_sha256
        or annotations.get("rl-quant/execution-authorization-sha256")
        != config.execution_authorization_receipt_sha256
        or annotations.get("rl-quant/source-archive-sha256")
        != config.source_archive_sha256
        or annotations.get("rl-quant/capacity-receipt-sha256")
        != config.capacity_receipt_sha256
        or spec.get("suspend") is not True
        or spec.get("completions") != config.completions
        or spec.get("parallelism") != config.parallelism
    ):
        raise M03RV13SeadragonOperatorError("created Job identity drifted")
    return uid


def _publish_attach(
    root: Path,
    config: M03RV13CreateOperatorConfig,
    *,
    phase: str,
    error: Exception,
    create_attempted: bool,
    observed: Mapping[str, Any] | None,
) -> None:
    common._exclusive_json(
        root / f"{phase}-attach-required.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v13-create-attach-required-v1",
            "phase": phase,
            "job_name": config.job_name,
            "run_id": config.run_id,
            "observed_job_sha256": (
                None if observed is None else common._content_sha256(observed)
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "create_attempted": create_attempted,
            "create_retried": False,
            "cleanup_performed": False,
            "attach_required": True,
        },
    )


def _cleanup_safe_binding(
    *,
    root: Path,
    config: M03RV13CreateOperatorConfig,
    binding: Any,
    job_uid: str,
    live: CreateTransport,
    sleep: Any,
) -> None:
    accepted = _AcceptedConfig(
        job_name=config.job_name,
        run_id=config.run_id,
        job_uid=job_uid,
        package_plan_sha256=config.package_plan_sha256,
        execution_authorization_receipt_sha256=(
            config.execution_authorization_receipt_sha256
        ),
        source_archive_sha256=config.source_archive_sha256,
        capacity_receipt_sha256=config.capacity_receipt_sha256,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    lifecycle._cleanup_preactivation_exact(
        root=root,
        config=cast(Any, accepted),
        binding=binding,
        transport=cast(Any, _CleanupAdapter(live, job_uid)),
        sleep=sleep,
    )


def prepare_suspended_job_once(
    config_path: str | Path,
    expected_config_sha256: str,
    *,
    transport: CreateTransport | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> None:
    config = _load_config(Path(config_path), expected_config_sha256)
    root = common._directory_no_symlink(
        Path(config.evidence_root), label="create evidence root"
    )
    if any(root.iterdir()):
        raise M03RV13SeadragonOperatorError(
            "create evidence root must be a fresh empty directory"
        )
    if (
        Path(config.binding_output_path).exists()
        or Path(config.activation_output_path).exists()
    ):
        raise M03RV13SeadragonOperatorError(
            "binding and activation outputs must be absent before create"
        )
    source = common._regular_no_symlink(Path(__file__), label="operator source")
    if _file_sha256(source) != config.operator_source_sha256:
        raise M03RV13SeadragonOperatorError("operator source hash drifted")
    rendered = _load_rendered(config)
    live = transport or OneCreateKubectl(
        kubectl_path=config.kubectl_path,
        kubeconfig_path=config.kubeconfig_path,
        context=config.context,
        namespace=config.namespace,
        job_name=config.job_name,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    dry = live.server_dry_run(Path(config.manifest_path))
    synthetic = _synthetic_dry_read(dry, config)
    try:
        _bind_rendered(
            rendered=rendered,
            first_read=synthetic,
            second_read=synthetic,
            attached_owned_pod_uids=(),
        )
    except Exception as exc:
        common._exclusive_json(root / "server-dry-run-rejected.json", dry)
        raise M03RV13SeadragonOperatorError(
            "server dry-run failed the exact admitted-surface allowlist"
        ) from exc
    common._exclusive_json(root / "server-dry-run.json", dry)
    for ordinal in ("first", "second"):
        job = live.get_job(allow_absent=True)
        pods = live.get_pods_by_job_name()
        if job is not None or pods:
            raise M03RV13SeadragonOperatorError(
                "precreate exact Job name or Pod label is already occupied"
            )
        common._exclusive_json(
            root / f"precreate-absence-{ordinal}.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v13-precreate-absence-v1",
                "job_name": config.job_name,
                "job_absent": True,
                "name_scoped_pods": [],
            },
        )
        if ordinal == "first":
            sleep(0.1)
    attempt: M03RV13CreateAttempt | None = None
    transport_error: Exception | None = None
    try:
        attempt = live.create_once(Path(config.manifest_path))
    except Exception as exc:  # noqa: BLE001 - reconcile sole uncertain create
        transport_error = exc
    common._exclusive_json(
        root / "create-outcome.json",
        {
            "schema": "rl-quant.top2000-dev.m03r-v13-create-outcome-v1",
            "returncode": None if attempt is None else attempt.returncode,
            "stderr_tail": (
                None
                if attempt is None
                else attempt.stderr.decode("utf-8", errors="replace")[-2000:]
            ),
            "transport_error": (
                None
                if transport_error is None
                else f"{type(transport_error).__name__}: {transport_error}"
            ),
            "create_attempt_count": 1,
            "create_retried": False,
        },
    )
    response: Mapping[str, Any] | None = None
    response_uid: str | None = None
    if attempt is not None and attempt.stdout.strip():
        try:
            response = _mapping(json.loads(attempt.stdout), "create response")
            common._exclusive_json(root / "create-response.json", response)
            if attempt.returncode == 0:
                response_uid = _validate_created_identity(response, config)
        except Exception as exc:
            _publish_attach(
                root,
                config,
                phase="create-response",
                error=exc,
                create_attempted=True,
                observed=response,
            )
            raise M03RV13CreateAttachRequired(
                "create response identity is ambiguous; never retry this name"
            ) from exc
    if attempt is not None and attempt.returncode == 0 and response_uid is None:
        error = M03RV13SeadragonOperatorError(
            "successful create did not return one exact Job UID"
        )
        _publish_attach(
            root,
            config,
            phase="create-response",
            error=error,
            create_attempted=True,
            observed=response,
        )
        raise M03RV13CreateAttachRequired(
            "successful create response is ambiguous; never retry this name"
        )
    started = monotonic()
    deadline = started + config.request_timeout_seconds
    observed: Mapping[str, Any] | None = None
    read_errors: list[str] = []
    absence_reads = 0
    last_state = "none"
    while True:
        try:
            candidate = live.get_job(allow_absent=True)
        except Exception as exc:  # noqa: BLE001 - preserve reconciliation ambiguity
            last_state = "read-error"
            read_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            if candidate is None:
                last_state = "absent"
                absence_reads += 1
            else:
                last_state = "present"
                try:
                    _validate_created_identity(
                        candidate, config, expected_uid=response_uid
                    )
                except Exception as exc:
                    _publish_attach(
                        root,
                        config,
                        phase="create-reconciliation-identity",
                        error=exc,
                        create_attempted=True,
                        observed=candidate,
                    )
                    raise M03RV13CreateAttachRequired(
                        "reconciled Job identity drifted; never retry this name"
                    ) from exc
                observed = candidate
                break
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(config.reconciliation_poll_seconds, remaining))
    if observed is None:
        elapsed = monotonic() - started
        create_reported_success = attempt is not None and attempt.returncode == 0
        stable_absence = (
            elapsed >= config.request_timeout_seconds
            and absence_reads >= 2
            and last_state == "absent"
            and not read_errors
            and not create_reported_success
        )
        common._exclusive_json(
            root / "create-reconciliation.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v13-create-reconciliation-v1",
                "outcome": (
                    "stable-absence" if stable_absence else "unknown-attach-required"
                ),
                "job_absent": True if stable_absence else None,
                "absence_read_count": absence_reads,
                "read_errors": read_errors,
                "last_state": last_state,
                "create_retried": False,
                "attach_required": not stable_absence,
            },
        )
        if stable_absence:
            raise M03RV13SeadragonOperatorError(
                "sole create attempt reconciled to stable absence; use a fresh identity"
            )
        error = M03RV13SeadragonOperatorError(
            "create result remains ambiguous after the full request-timeout window"
        )
        _publish_attach(
            root,
            config,
            phase="create-reconciliation",
            error=error,
            create_attempted=True,
            observed=None,
        )
        raise M03RV13CreateAttachRequired(
            "create result remains ambiguous; never retry this name"
        )
    uid = _validate_created_identity(observed, config, expected_uid=response_uid)
    try:
        sleep(0.1)
        second = live.get_job()
        if second is None:
            raise M03RV13SeadragonOperatorError(
                "accepted suspended Job disappeared before its second read"
            )
        _validate_created_identity(second, config, expected_uid=uid)
        pods = live.get_owned_pods(uid)
        if pods:
            raise M03RV13SeadragonOperatorError(
                "accepted suspended Job unexpectedly created Pods"
            )
        binding = _bind_rendered(
            rendered=rendered,
            first_read=dict(observed),
            second_read=dict(second),
            attached_owned_pod_uids=(),
        )
    except Exception as exc:
        _publish_attach(
            root,
            config,
            phase="postaccept-binding",
            error=exc,
            create_attempted=True,
            observed=observed,
        )
        raise M03RV13CreateAttachRequired(
            "accepted Job lacks a safe binding; create must not be retried"
        ) from exc
    try:
        sleep(0.1)
        fresh = live.get_job()
        if fresh is None or live.get_owned_pods(uid):
            raise M03RV13SeadragonOperatorError(
                "accepted Job lost the fresh suspended zero-Pod activation boundary"
            )
        activation = build_m03r_v7_exact_job_activation_request(binding, fresh)
    except Exception as exc:
        try:
            _cleanup_safe_binding(
                root=root,
                config=config,
                binding=binding,
                job_uid=uid,
                live=live,
                sleep=sleep,
            )
        except Exception as cleanup_error:
            _publish_attach(
                root,
                config,
                phase="postaccept-cleanup",
                error=cleanup_error,
                create_attempted=True,
                observed=observed,
            )
            raise M03RV13CreateAttachRequired(
                "accepted Job cleanup is ambiguous; create must not be retried"
            ) from cleanup_error
        common._exclusive_json(
            root / "postaccept-cleanup-error.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v13-postaccept-cleanup-error-v1",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exact_cleanup_completed": True,
                "create_retried": False,
            },
        )
        raise M03RV13SeadragonOperatorError(
            "activation handoff failed after safe exact cleanup; use a fresh identity"
        ) from exc
    try:
        first_sha = common._exclusive_json(root / "first-read.json", observed)
        second_sha = common._exclusive_json(root / "second-read.json", second)
        pods_sha = common._exclusive_json(
            root / "zero-pods.json",
            {"apiVersion": "v1", "kind": "PodList", "items": []},
        )
        binding_sha = common._exclusive_json(
            Path(config.binding_output_path), asdict(binding)
        )
        activation_sha = common._exclusive_json(
            Path(config.activation_output_path), asdict(activation)
        )
        common._exclusive_json(
            root / "prepare-success.json",
            {
                "schema": "rl-quant.top2000-dev.m03r-v13-prepare-success-v1",
                "job_name": config.job_name,
                "job_uid": uid,
                "run_id": config.run_id,
                "manifest_file_sha256": config.manifest_file_sha256,
                "rendered_file_sha256": config.rendered_file_sha256,
                "first_read_file_sha256": first_sha,
                "second_read_file_sha256": second_sha,
                "zero_pods_file_sha256": pods_sha,
                "binding_file_sha256": binding_sha,
                "activation_request_file_sha256": activation_sha,
                "create_attempt_count": 1,
                "create_retried": False,
                "suspended": True,
                "zero_owned_pods": True,
                "economic_panel_authorized": False,
            },
        )
    except Exception as exc:
        try:
            _cleanup_safe_binding(
                root=root,
                config=config,
                binding=binding,
                job_uid=uid,
                live=live,
                sleep=sleep,
            )
        except Exception as cleanup_error:
            _publish_attach(
                root,
                config,
                phase="prepared-publication-cleanup",
                error=cleanup_error,
                create_attempted=True,
                observed=observed,
            )
            raise M03RV13CreateAttachRequired(
                "prepared publication cleanup is ambiguous"
            ) from cleanup_error
        raise M03RV13SeadragonOperatorError(
            "prepared publication failed after safe exact cleanup"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepare_suspended_job_once(args.config, args.config_sha256)
    except M03RV13CreateAttachRequired as exc:
        print(
            json.dumps(
                {"status": "attach_required", "error": str(exc)},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CREATE_CONFIG_SCHEMA",
    "M03RV13CreateAttachRequired",
    "M03RV13CreateAttempt",
    "M03RV13CreateOperatorConfig",
    "M03RV13SeadragonOperatorError",
    "OneCreateKubectl",
    "prepare_suspended_job_once",
]
