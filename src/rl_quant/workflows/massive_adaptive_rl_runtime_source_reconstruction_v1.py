"""Package-owned reconstruction of adaptive-RL runtime source authorities.

The byte-level source bundle and runtime-source graph deliberately do not
serialize live Python witnesses.  This module closes that restart boundary
with a create-only, dependency-complete object index.  Source preparation may
publish the index only from an already witnessed graph; later root workflows
reconstruct the exact role map from ``source_root`` without accepting dates,
roles, runtime objects, or qualification flags from their callers.

The object snapshots are a transport for already-authorized source inputs,
not profitability evidence.  Load-bearing calibration and forecast archives
are replayed again through their package authorizers during reconstruction.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import importlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, cast

import numpy as np
import torch

from rl_quant.alpha.pit_universe import PITSecurityUniverseAuthority
from rl_quant.data_sources.massive.conditions import MassiveConditionAuthority
from rl_quant.data_sources.massive.decision_clock import MassiveDecisionClockAuthority
from rl_quant.data_sources.massive.finalized_persisted_partitions import (
    MassivePersistedPartitionManifestV1,
)
from rl_quant.data_sources.massive.session_calendar import MassiveSessionAuthority
from rl_quant.evaluation.massive_adaptive_forecast_archive_v1 import (
    MassiveAdaptiveForecastArchiveV1,
    authorize_massive_adaptive_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_forecast_calibration_v2 import (
    MassiveAdaptiveForecastCalibrationV2,
    authorize_massive_adaptive_forecast_calibration_v2,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_forecast_archive_v1 import (
    MassiveAdaptiveRLFitForecastArchiveV1,
    authorize_massive_adaptive_rl_fit_forecast_archive_v1,
)
from rl_quant.evaluation.massive_adaptive_rl_fit_inference_plan_v1 import (
    MassiveAdaptiveRLFitInferencePlanV1,
)
from rl_quant.features.massive_adaptive_context_origin_authority_v1 import (
    MassiveAdaptiveContextOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_decision_root_v1 import (
    MassiveAdaptiveDecisionRootV1,
)
from rl_quant.features.massive_adaptive_decision_tensor_v1 import (
    MassiveAdaptiveDecisionTensorV1,
    authorize_massive_adaptive_decision_tensor_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import (
    MassiveAdaptiveFillSourceV1,
)
from rl_quant.features.massive_adaptive_origin_authority_v1 import (
    MassiveAdaptiveOriginAuthorityV1,
)
from rl_quant.features.massive_adaptive_source_targets_v1 import (
    MassiveAdaptiveSourceTargetsV1,
)
from rl_quant.features.massive_adaptive_target_archive_v1 import (
    MassiveAdaptiveTargetArchiveV1,
)
from rl_quant.features.massive_economic_authority_v6 import (
    MassiveProviderEconomicArchiveAuthorityV6,
)
from rl_quant.features.massive_profitability_daily_input_authority_v1 import (
    MassiveProfitabilityDailyInputAuthorityV1,
)
from rl_quant.features.massive_profitability_origin_features_v3 import (
    MassiveProfitabilityOriginFeaturesV3,
)
from rl_quant.models.adaptive_alpha_term_structure_v1 import (
    MassiveAdaptiveAlphaModelSpecV1,
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
from rl_quant.training.massive_adaptive_checkpoint_v1 import (
    MassiveAdaptiveCheckpointV1,
    authorize_massive_adaptive_checkpoint_v1,
)
from rl_quant.training.massive_adaptive_rl_training_forecast_authority_v1 import (
    MassiveAdaptiveCausalCheckpointChoiceV1,
    build_massive_adaptive_causal_checkpoint_choice_v1,
)
from rl_quant.training.massive_adaptive_split_plan_v1 import (
    MassiveAdaptiveSplitPlanV1,
)
from rl_quant.training.massive_adaptive_supervised_trainer_v1 import (
    MassiveAdaptiveSupervisedTrainingConfigV1,
)
from rl_quant.training.massive_adaptive_training_authority_v1 import (
    MassiveAdaptiveTrainingAuthorityV1,
    build_massive_adaptive_training_authority_v1,
)
from rl_quant.training.massive_adaptive_window_plan_v1 import (
    MassiveAdaptiveWindowPlanV1,
)
from rl_quant.workflows.massive_adaptive_rl_manifest_v3 import (
    MassiveAdaptiveRLExperimentManifestV3,
)
from rl_quant.workflows.massive_adaptive_rl_runtime_source_graph_authority_v1 import (
    _DOMAIN_INVENTORY_ITEM_TYPES,
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
    MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error,
    MassiveAdaptiveRLTypedAuthorityInventoryV1,
    authorize_massive_adaptive_rl_runtime_source_graph_authority_v1,
    build_massive_adaptive_rl_typed_authority_inventory_v1,
    load_massive_adaptive_rl_runtime_source_graph_authority_v1,
)
from rl_quant.workflows.massive_adaptive_rl_source_bundle_v1 import (
    MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1,
    MassiveAdaptiveRLRoleBoundSourceAuthorityV1,
    MassiveAdaptiveRLSourceAuthorityProtocol,
    bind_massive_adaptive_rl_source_authority_v1,
    load_massive_adaptive_rl_source_bundle_v1,
)


MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-replay-dependency-v1"
)
MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-replay-dependency-index-v1"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_OBJECT_SNAPSHOT_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-object-snapshot-v1"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V1_SCHEMA = (
    "rl-quant.massive-adaptive-rl-runtime-sources-v1"
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256 = file_sha256(
    Path(__file__)
)
MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256 = semantic_sha256(
    {
        "input": "manifest-v3-source-bundle-v1-witnessed-runtime-graph-v1",
        "index": "create-only-path-hash-schema-specification-and-dependency-closed",
        "codec": "restricted-rl-quant-dataclass-enum-and-tensor-json-v2",
        "mapping_order": "canonical-encoded-key-order",
        "caller_runtime_mapping": False,
        "qualification_flags": "package-derived-only",
        "checkpoint_dependencies": "training-authority-tensor-window-model-config",
        "calibration_dependencies": "checkpoint-training-forecast-target-window",
        "fit_forecast_dependencies": (
            "checkpoint-training-and-inference-tensors-plan-roots-model"
        ),
        "native_replay": "tensor-checkpoint-training-forecast-calibration-fit-forecast",
        "reconstruction_dependency_closure": "independently-recomputed",
        "temporary_source_unavailability": "retryable-blocker",
        "publication": "fsync-atomic-content-addressed-and-create-only-index",
        "profitability_reporting": False,
        "lockbox_access": False,
    }
)

_DEPENDENCY_ROLE = "replay-dependency"
_ROOT_LOGICAL_KEY = "root"
_CODEC_ID = "restricted-rl-quant-dataclass-enum-and-tensor-json-v2"


class MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(ValueError):
    """The persisted runtime-source reconstruction graph is incomplete."""


class MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
    MassiveAdaptiveRLRuntimeSourceReconstructionV1Error
):
    """A replay dependency cannot currently be reached and may be retried."""


class MassiveAdaptiveRLRuntimeSourceIntegrityError(
    MassiveAdaptiveRLRuntimeSourceReconstructionV1Error
):
    """Persisted runtime-source bytes or semantic lineage are invalid."""


class MassiveAdaptiveRLRuntimeSourceImplementationMismatch(
    MassiveAdaptiveRLRuntimeSourceIntegrityError
):
    """Persisted runtime-source code identity differs from this package."""


class MassiveAdaptiveRLRuntimeSourceDependencyMismatch(
    MassiveAdaptiveRLRuntimeSourceIntegrityError
):
    """The persisted runtime-source dependency closure differs."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _receipt(value: object) -> str:
    for name in ("semantic_receipt_sha256", "receipt_sha256"):
        observed = getattr(value, name, None)
        if observed is not None:
            return _digest("runtime-source dependency receipt", observed)
    raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
        "runtime-source dependency exposes no receipt"
    )


def _type_name(value: type[object] | object) -> str:
    runtime_type = value if isinstance(value, type) else type(value)
    return f"{runtime_type.__module__}.{runtime_type.__qualname__}"


def _implementation_source_sha256(value: type[object]) -> str:
    source = inspect.getsourcefile(value)
    if source is None:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source dependency implementation source is absent"
        )
    return file_sha256(source)


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source reconstruction path must be a string"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source reconstruction path is not normalized"
        )
    return path.as_posix()


def _resolve_regular_file(*, root: Path, relative_path: str) -> Path:
    relative = _safe_relative_path(relative_path)
    candidate = root / relative
    cursor = candidate
    while cursor != root:
        if root not in cursor.parents:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source reconstruction path escapes the source root"
            )
        if cursor.is_symlink():
            raise MassiveAdaptiveRLRuntimeSourceIntegrityError(
                "runtime-source reconstruction path contains a symlink"
            )
        cursor = cursor.parent
    if not candidate.exists():
        raise MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
            "runtime-source reconstruction file is temporarily absent"
        )
    resolved = candidate.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise MassiveAdaptiveRLRuntimeSourceIntegrityError(
            "runtime-source reconstruction file is not regular"
        )
    return resolved


def _create_only_output_path(*, root: Path, relative_path: str) -> Path:
    relative = _safe_relative_path(relative_path)
    candidate = root / relative
    cursor = candidate
    while cursor != root:
        if root not in cursor.parents:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source reconstruction output escapes the source root"
            )
        if cursor.is_symlink():
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source reconstruction output contains a symlink"
            )
        cursor = cursor.parent
    return candidate


def _atomic_install(
    *, output: Path, payload: Mapping[str, object], allow_exact_existing: bool
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = canonical_json_file_bytes(payload)
    if output.exists():
        if (
            allow_exact_existing
            and not output.is_symlink()
            and output.is_file()
            and output.read_bytes() == expected
        ):
            return
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source reconstruction artifact is create-only"
        )
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source reconstruction artifact is create-only"
            ) from error
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_TORCH_DTYPES: dict[str, torch.dtype] = {
    str(value): value
    for value in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
}


def _encode_value(value: object) -> object:
    if isinstance(value, Enum):
        enum_type = type(value)
        if not enum_type.__module__.startswith("rl_quant."):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot enum is outside rl_quant"
            )
        return {
            "kind": "enum",
            "module": enum_type.__module__,
            "qualname": enum_type.__qualname__,
            "implementation_source_sha256": (_implementation_source_sha256(enum_type)),
            "member_name": value.name,
            "member_value": _encode_value(value.value),
        }
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot contains a non-finite float"
            )
        return value
    if isinstance(value, complex):
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "kind": "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": value.as_posix()}
    if isinstance(value, torch.device):
        return {"kind": "torch-device", "value": str(value)}
    if isinstance(value, torch.dtype):
        return {"kind": "torch-dtype", "value": str(value)}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if (
            tensor.layout is not torch.strided
            or tensor.dtype not in _TORCH_DTYPES.values()
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot tensor layout or dtype is unsupported"
            )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "torch-tensor",
            "dtype": str(tensor.dtype),
            "shape": tuple(int(part) for part in tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot object array is unsupported"
            )
        return {
            "kind": "numpy-array",
            "dtype": array.dtype.str,
            "shape": tuple(int(part) for part in array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        dataclass_type = type(value)
        if not dataclass_type.__module__.startswith("rl_quant."):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot dataclass is outside rl_quant"
            )
        encoded_fields = []
        for current in fields(value):
            if not current.init or current.name == "_issuer":
                continue
            encoded_fields.append(
                (current.name, _encode_value(getattr(value, current.name)))
            )
        return {
            "kind": "dataclass",
            "module": dataclass_type.__module__,
            "qualname": dataclass_type.__qualname__,
            "implementation_source_sha256": (
                _implementation_source_sha256(dataclass_type)
            ),
            "fields": tuple(encoded_fields),
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": tuple(_encode_value(row) for row in value)}
    if isinstance(value, list):
        return {"kind": "list", "items": tuple(_encode_value(row) for row in value)}
    if isinstance(value, Mapping):
        encoded_items = tuple(
            (_encode_value(key), _encode_value(current))
            for key, current in value.items()
        )
        return {
            "kind": "mapping",
            "items": tuple(
                sorted(
                    encoded_items,
                    key=lambda row: json.dumps(
                        row[0],
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                )
            ),
        }
    if isinstance(value, (set, frozenset)):
        encoded = tuple(_encode_value(row) for row in value)
        ordered = tuple(
            sorted(
                encoded,
                key=lambda row: json.dumps(
                    row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ),
            )
        )
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": ordered,
        }
    raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
        f"runtime-source snapshot value type is unsupported: {_type_name(value)}"
    )


def _resolve_dataclass_type(
    *, module_name: object, qualname: object, expected_source_sha256: object
) -> type[object]:
    if (
        not isinstance(module_name, str)
        or not module_name.startswith("rl_quant.")
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source snapshot type is not package-owned"
        )
    module = importlib.import_module(module_name)
    value: object = module
    for part in qualname.split("."):
        value = getattr(value, part, None)
        if value is None:
            break
    if (
        not isinstance(value, type)
        or not is_dataclass(value)
        or _implementation_source_sha256(value)
        != _digest("snapshot implementation", expected_source_sha256)
    ):
        raise MassiveAdaptiveRLRuntimeSourceImplementationMismatch(
            "runtime-source snapshot implementation differs"
        )
    return value


def _resolve_enum_type(
    *, module_name: object, qualname: object, expected_source_sha256: object
) -> type[Enum]:
    if (
        not isinstance(module_name, str)
        or not module_name.startswith("rl_quant.")
        or not isinstance(qualname, str)
        or not qualname
        or "<locals>" in qualname
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source snapshot enum type is not package-owned"
        )
    module = importlib.import_module(module_name)
    value: object = module
    for part in qualname.split("."):
        value = getattr(value, part, None)
        if value is None:
            break
    if (
        not isinstance(value, type)
        or not issubclass(value, Enum)
        or _implementation_source_sha256(value)
        != _digest("snapshot enum implementation", expected_source_sha256)
    ):
        raise MassiveAdaptiveRLRuntimeSourceImplementationMismatch(
            "runtime-source snapshot enum implementation differs"
        )
    return value


def _decode_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source snapshot shape is malformed"
        )
    result: list[int] = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, int) or part < 0:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot shape is malformed"
            )
        result.append(part)
    return tuple(result)


def _decode_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source snapshot value is malformed"
        )
    kind = value.get("kind")
    if kind == "enum":
        enum_type = _resolve_enum_type(
            module_name=value.get("module"),
            qualname=value.get("qualname"),
            expected_source_sha256=value.get("implementation_source_sha256"),
        )
        member_name = value.get("member_name")
        if not isinstance(member_name, str):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot enum member is malformed"
            )
        try:
            member = enum_type.__members__[member_name]
            member_value = _decode_value(value.get("member_value"))
        except (KeyError, TypeError, ValueError) as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot enum member is unsupported"
            ) from error
        if type(member.value) is not type(member_value) or member.value != member_value:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot enum name and value differ"
            )
        return member
    if kind == "complex":
        return complex(float(value["real"]), float(value["imag"]))
    if kind == "bytes":
        return base64.b64decode(str(value["data"]), validate=True)
    if kind == "path":
        return Path(str(value["value"]))
    if kind == "torch-device":
        return torch.device(str(value["value"]))
    if kind == "torch-dtype":
        try:
            return _TORCH_DTYPES[str(value["value"])]
        except KeyError as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot torch dtype is unsupported"
            ) from error
    if kind == "torch-tensor":
        try:
            torch_dtype = _TORCH_DTYPES[str(value["dtype"])]
            shape = _decode_shape(value["shape"])
            raw = base64.b64decode(str(value["data"]), validate=True)
            tensor_result = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
            item_size = torch.empty((), dtype=torch_dtype).element_size()
            expected = math.prod(shape) * item_size
            if len(raw) != expected:
                raise ValueError("tensor byte count differs")
            return tensor_result.view(torch_dtype).reshape(shape)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot tensor is malformed"
            ) from error
    if kind == "numpy-array":
        try:
            numpy_dtype = np.dtype(str(value["dtype"]))
            shape = _decode_shape(value["shape"])
            raw = base64.b64decode(str(value["data"]), validate=True)
            numpy_result = np.frombuffer(raw, dtype=numpy_dtype).copy().reshape(shape)
        except (KeyError, TypeError, ValueError) as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot NumPy array is malformed"
            ) from error
        return numpy_result
    if kind in {"tuple", "list", "set", "frozenset"}:
        items = tuple(
            _decode_value(row) for row in cast(Sequence[object], value["items"])
        )
        if kind == "tuple":
            return items
        if kind == "list":
            return list(items)
        if kind == "set":
            return set(items)
        return frozenset(items)
    if kind == "mapping":
        mapping_result: dict[object, object] = {}
        for row in cast(Sequence[object], value["items"]):
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
            ):
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source snapshot mapping row is malformed"
                )
            row_values = cast(Sequence[object], row)
            key = _decode_value(row_values[0])
            if key in mapping_result:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source snapshot mapping key is duplicated"
                )
            mapping_result[key] = _decode_value(row_values[1])
        return mapping_result
    if kind == "dataclass":
        dataclass_type = _resolve_dataclass_type(
            module_name=value.get("module"),
            qualname=value.get("qualname"),
            expected_source_sha256=value.get("implementation_source_sha256"),
        )
        raw_fields = value.get("fields")
        if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot dataclass fields are malformed"
            )
        kwargs: dict[str, object] = {}
        for row in raw_fields:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != 2
            ):
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source snapshot dataclass field is malformed"
                )
            name = row[0]
            if not isinstance(name, str) or name in kwargs:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source snapshot dataclass field is duplicated"
                )
            kwargs[name] = _decode_value(row[1])
        field_names = {
            current.name
            for current in fields(cast(Any, dataclass_type))
            if current.init
        }
        if not set(kwargs) <= field_names:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot dataclass field inventory differs"
            )
        if "_issuer" in field_names:
            module = sys.modules[dataclass_type.__module__]
            issuer = getattr(module, "_ISSUER", None)
            if issuer is None:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source snapshot issuer is absent"
                )
            kwargs["_issuer"] = issuer
        try:
            return dataclass_type(**kwargs)
        except (TypeError, ValueError) as error:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot dataclass cannot be reconstructed"
            ) from error
    raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
        "runtime-source snapshot kind is unsupported"
    )


def _snapshot_payload(value: object) -> dict[str, object]:
    runtime_type = type(value)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_RUNTIME_OBJECT_SNAPSHOT_V1_SCHEMA,
        "codec_id": _CODEC_ID,
        "domain_type_name": _type_name(runtime_type),
        "implementation_source_sha256": (_implementation_source_sha256(runtime_type)),
        "semantic_receipt_sha256": _receipt(value),
        "encoded_value": _encode_value(value),
    }
    return {**body, "snapshot_receipt_sha256": semantic_sha256(body)}


def _parse_snapshot(payload: Mapping[str, object]) -> object:
    body = {
        key: value for key, value in payload.items() if key != "snapshot_receipt_sha256"
    }
    if (
        payload.get("schema") != MASSIVE_ADAPTIVE_RL_RUNTIME_OBJECT_SNAPSHOT_V1_SCHEMA
        or payload.get("codec_id") != _CODEC_ID
        or payload.get("snapshot_receipt_sha256") != semantic_sha256(body)
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source object snapshot identity differs"
        )
    result = _decode_value(payload.get("encoded_value"))
    if (
        _type_name(result) != payload.get("domain_type_name")
        or _implementation_source_sha256(type(result))
        != payload.get("implementation_source_sha256")
        or _receipt(result) != payload.get("semantic_receipt_sha256")
        or not hasattr(result, "validate")
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source object snapshot domain identity differs"
        )
    cast(Any, result).validate()
    return result


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLReplayDependencyV1:
    role: str
    fold_index: int | None
    logical_key: str
    role_authority_receipt_sha256: str
    relative_path: str
    file_sha256: str
    semantic_receipt_sha256: str
    domain_type_name: str
    domain_schema: str | None
    domain_specification_sha256: str | None
    implementation_source_sha256: str
    parser_id: str
    dependency_receipts: tuple[str, ...]
    receipt_sha256: str
    schema: str = MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_V1_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value for key, value in asdict(self).items() if key != "receipt_sha256"
        }

    def validate(self) -> None:
        expected_relative = (
            "adaptive-rl/runtime-source-reconstruction-v1/"
            f"objects/{self.semantic_receipt_sha256}.json"
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_V1_SCHEMA
            or not self.role
            or self.fold_index is not None
            and (isinstance(self.fold_index, bool) or self.fold_index not in range(4))
            or not self.logical_key
            or self.relative_path != expected_relative
            or _safe_relative_path(self.relative_path) != self.relative_path
            or not self.domain_type_name.startswith("rl_quant.")
            or not self.domain_schema
            or self.parser_id != _CODEC_ID
            or self.dependency_receipts != tuple(sorted(set(self.dependency_receipts)))
            or self.semantic_receipt_sha256 in self.dependency_receipts
            or self.receipt_sha256 != semantic_sha256(self.unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "adaptive RL replay-dependency row differs"
            )
        for value in (
            self.role_authority_receipt_sha256,
            self.file_sha256,
            self.semantic_receipt_sha256,
            self.implementation_source_sha256,
            *self.dependency_receipts,
            self.receipt_sha256,
        ):
            _digest("adaptive RL replay dependency", value)
        if self.domain_specification_sha256 is not None:
            _digest(
                "adaptive RL replay dependency specification",
                self.domain_specification_sha256,
            )


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLReplayDependencyIndexV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    base_manifest_receipt_sha256: str
    source_bundle_receipt_sha256: str
    persisted_runtime_source_graph_receipt_sha256: str
    runtime_source_graph_witness_receipt_sha256: str
    rows: tuple[MassiveAdaptiveRLReplayDependencyV1, ...]
    row_inventory_sha256: str
    object_inventory_sha256: str
    dependency_edge_inventory_sha256: str
    committed_source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
    )
    implementation_source_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "semantic_receipt_sha256"
        }

    def validate(self) -> None:
        keys = tuple((row.role, row.fold_index, row.logical_key) for row in self.rows)
        object_rows = tuple(
            sorted(
                {
                    (
                        row.semantic_receipt_sha256,
                        row.relative_path,
                        row.file_sha256,
                        row.domain_type_name,
                    )
                    for row in self.rows
                }
            )
        )
        edges = tuple(
            sorted(
                (
                    row.semantic_receipt_sha256,
                    dependency,
                )
                for row in self.rows
                for dependency in row.dependency_receipts
            )
        )
        receipts = {row.semantic_receipt_sha256 for row in self.rows}
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V1_SCHEMA
            or not self.experiment_id
            or not self.rows
            or keys
            != tuple(
                sorted(
                    keys,
                    key=lambda row: (
                        row[0] == _DEPENDENCY_ROLE,
                        row[1] or -1,
                        row[0],
                        row[2],
                    ),
                )
            )
            or len(set(keys)) != len(keys)
            or any(
                dependency not in receipts
                for row in self.rows
                for dependency in row.dependency_receipts
            )
            or self.row_inventory_sha256
            != semantic_sha256(tuple(row.receipt_sha256 for row in self.rows))
            or self.object_inventory_sha256 != semantic_sha256(object_rows)
            or self.dependency_edge_inventory_sha256 != semantic_sha256(edges)
            or self.committed_source_data_qualified is not True
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
            or self.implementation_source_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "adaptive RL replay-dependency index differs"
            )
        for row in self.rows:
            row.validate()
        for value in (
            self.manifest_v3_receipt_sha256,
            self.base_manifest_receipt_sha256,
            self.source_bundle_receipt_sha256,
            self.persisted_runtime_source_graph_receipt_sha256,
            self.runtime_source_graph_witness_receipt_sha256,
            self.row_inventory_sha256,
            self.object_inventory_sha256,
            self.dependency_edge_inventory_sha256,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.implementation_source_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL replay-dependency index", value)
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLFoldRuntimeSourcesV1:
    outer_fold_index: int
    training_windows: tuple[MassiveAdaptiveWindowPlanV1, ...]
    checkpoint_choices: tuple[MassiveAdaptiveCausalCheckpointChoiceV1, ...]
    calibrations: tuple[MassiveAdaptiveForecastCalibrationV2, ...]
    fit_forecast_archives: tuple[MassiveAdaptiveRLFitForecastArchiveV1, ...]
    decision_roots: tuple[MassiveAdaptiveDecisionRootV1, ...]
    context_origins: tuple[MassiveAdaptiveContextOriginAuthorityV1, ...]

    def validate(self) -> None:
        source_folds = tuple(range(self.outer_fold_index + 1))
        if (
            self.outer_fold_index not in range(4)
            or tuple(row.fold_index for row in self.training_windows) != source_folds
            or tuple(row.fold_index for row in self.checkpoint_choices) != source_folds
            or tuple(row.fold_index for row in self.calibrations) != source_folds
            or tuple(row.block_index for row in self.fit_forecast_archives)
            != tuple(range(len(self.fit_forecast_archives)))
            or tuple(row.decision_session_date for row in self.decision_roots)
            != tuple(row.decision_session_date for row in self.context_origins)
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "adaptive RL fold runtime-source inventory differs"
            )
        for inventory in (
            self.training_windows,
            self.checkpoint_choices,
            self.calibrations,
            self.fit_forecast_archives,
            self.decision_roots,
            self.context_origins,
        ):
            if not inventory:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "adaptive RL fold runtime-source inventory is empty"
                )
            for row in inventory:
                row.validate()


@dataclass(frozen=True, slots=True)
class MassiveAdaptiveRLRuntimeSourcesV1:
    experiment_id: str
    manifest_v3_receipt_sha256: str
    source_bundle_receipt_sha256: str
    replay_dependency_index_receipt_sha256: str
    runtime_source_graph_authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1
    session_authority: MassiveSessionAuthority
    condition_authority: MassiveConditionAuthority
    persisted_partition_manifests: tuple[MassivePersistedPartitionManifestV1, ...]
    identity_authority: PITSecurityUniverseAuthority
    economic_event_archive: MassiveProviderEconomicArchiveAuthorityV6
    daily_input_authority: MassiveProfitabilityDailyInputAuthorityV1
    fill_source: MassiveAdaptiveFillSourceV1
    split_plan: MassiveAdaptiveSplitPlanV1
    folds: tuple[MassiveAdaptiveRLFoldRuntimeSourcesV1, ...]
    replay_dependency_receipts: tuple[str, ...]
    source_data_qualified: bool
    semantic_receipt_sha256: str
    profitability_reporting_authorized: bool = False
    lockbox_access_authorized: bool = False
    protocol_receipt_sha256: str = MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
    specification_sha256: str = (
        MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
    )
    schema: str = MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V1_SCHEMA

    def semantic_unsigned(self) -> dict[str, object]:
        runtime_receipt = (
            self.runtime_source_graph_authority.runtime_authority_receipt_sha256
        )
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "manifest_v3_receipt_sha256": self.manifest_v3_receipt_sha256,
            "source_bundle_receipt_sha256": self.source_bundle_receipt_sha256,
            "replay_dependency_index_receipt_sha256": (
                self.replay_dependency_index_receipt_sha256
            ),
            "runtime_source_graph_witness_receipt_sha256": runtime_receipt,
            "session_authority_receipt_sha256": self.session_authority.receipt_sha256,
            "condition_authority_receipt_sha256": self.condition_authority.receipt_sha256,
            "persisted_partition_receipts": tuple(
                row.receipt_sha256 for row in self.persisted_partition_manifests
            ),
            "identity_authority_receipt_sha256": self.identity_authority.receipt_sha256,
            "economic_event_archive_receipt_sha256": self.economic_event_archive.receipt_sha256,
            "daily_input_authority_receipt_sha256": (
                self.daily_input_authority.semantic_receipt_sha256
            ),
            "fill_source_receipt_sha256": self.fill_source.semantic_receipt_sha256,
            "split_plan_receipt_sha256": self.split_plan.semantic_receipt_sha256,
            "fold_receipt_inventories": tuple(
                tuple(
                    _receipt(row)
                    for inventory in (
                        fold.training_windows,
                        fold.checkpoint_choices,
                        fold.calibrations,
                        fold.fit_forecast_archives,
                        fold.decision_roots,
                        fold.context_origins,
                    )
                    for row in inventory
                )
                for fold in self.folds
            ),
            "replay_dependency_receipts": self.replay_dependency_receipts,
            "source_data_qualified": self.source_data_qualified,
            "profitability_reporting_authorized": (
                self.profitability_reporting_authorized
            ),
            "lockbox_access_authorized": self.lockbox_access_authorized,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "specification_sha256": self.specification_sha256,
        }

    def validate(self) -> None:
        self.runtime_source_graph_authority.validate()
        runtime_receipt = (
            self.runtime_source_graph_authority.runtime_authority_receipt_sha256
        )
        if (
            self.schema != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V1_SCHEMA
            or not self.experiment_id
            or runtime_receipt is None
            or not self.runtime_source_graph_authority.source_data_qualified
            or self.runtime_source_graph_authority.experiment_id != self.experiment_id
            or self.runtime_source_graph_authority.manifest_v3_receipt_sha256
            != self.manifest_v3_receipt_sha256
            or self.runtime_source_graph_authority.source_bundle_receipt_sha256
            != self.source_bundle_receipt_sha256
            or tuple(fold.outer_fold_index for fold in self.folds) != tuple(range(4))
            or not self.persisted_partition_manifests
            or self.replay_dependency_receipts
            != tuple(sorted(set(self.replay_dependency_receipts)))
            or self.source_data_qualified is not True
            or self.profitability_reporting_authorized
            or self.lockbox_access_authorized
            or self.protocol_receipt_sha256 != MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256
            or self.specification_sha256
            != MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
            or self.semantic_receipt_sha256 != semantic_sha256(self.semantic_unsigned())
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "adaptive RL reconstructed runtime sources differ"
            )
        for value in (
            self.manifest_v3_receipt_sha256,
            self.source_bundle_receipt_sha256,
            self.replay_dependency_index_receipt_sha256,
            *self.replay_dependency_receipts,
            self.protocol_receipt_sha256,
            self.specification_sha256,
            self.semantic_receipt_sha256,
        ):
            _digest("adaptive RL reconstructed runtime sources", value)
        for authority in (
            self.session_authority,
            self.condition_authority,
            self.identity_authority,
            self.economic_event_archive,
            self.daily_input_authority,
            self.fill_source,
            self.split_plan,
            *self.persisted_partition_manifests,
        ):
            cast(Any, authority).validate()
        for fold in self.folds:
            fold.validate()
        assert_no_adaptive_hold_semantics(self.semantic_unsigned())


def replay_dependency_index_path_v1(
    *, source_root: str | Path, experiment_id: str
) -> Path:
    return (
        Path(source_root)
        / "adaptive-rl"
        / "runtime-source-reconstruction-v1"
        / f"{experiment_id}.json"
    )


def _snapshot_relative_path(receipt: str) -> str:
    return (
        "adaptive-rl/runtime-source-reconstruction-v1/"
        f"objects/{_digest('snapshot object receipt', receipt)}.json"
    )


def _domain_specification(value: object) -> str | None:
    result = getattr(value, "specification_sha256", None)
    if result is None:
        result = getattr(value, "partition_spec_sha256", None)
    return None if result is None else _digest("dependency specification", result)


def _domain_schema(*, role: str, value: object) -> str:
    observed = getattr(value, "schema", None)
    if isinstance(observed, str) and observed:
        return observed
    if role != _DEPENDENCY_ROLE:
        spec = MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.get(role)
        if spec is not None:
            return spec.runtime_schema
    return f"python-dataclass:{_type_name(value)}"


def _primary_bindings(
    graph: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
) -> tuple[tuple[str, int | None, str, str, object], ...]:
    result: list[tuple[str, int | None, str, str, object]] = []
    for role, spec in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.items():
        for fold_index in range(4) if spec.fold_scoped else (None,):
            authority = graph.runtime_authority(role=role, fold_index=fold_index)
            if isinstance(authority, MassiveAdaptiveRLTypedAuthorityInventoryV1):
                if authority.runtime_items is None:
                    raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                        "runtime-source inventory witness is absent"
                    )
                for item in authority.runtime_items:
                    from rl_quant.workflows import (  # local import avoids public coupling
                        massive_adaptive_rl_runtime_source_graph_authority_v1 as graph_module,
                    )

                    result.append(
                        (
                            role,
                            fold_index,
                            graph_module._item_logical_key(role=role, item=item),
                            authority.semantic_receipt_sha256,
                            item,
                        )
                    )
            else:
                result.append(
                    (
                        role,
                        fold_index,
                        _ROOT_LOGICAL_KEY,
                        _receipt(authority),
                        authority,
                    )
                )
    return tuple(result)


def _add_expected(
    target: dict[str, type[object]], *, receipt: object, expected_type: type[object]
) -> None:
    digest = _digest("replay dependency", receipt)
    previous = target.setdefault(digest, expected_type)
    if previous is not expected_type:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "replay dependency receipt is claimed by different domain types"
        )


def _add_expected_pairs(
    target: dict[str, type[object]],
    pairs: Sequence[tuple[object, type[object]]],
) -> None:
    for dependency_receipt, expected_type in pairs:
        _add_expected(
            target,
            receipt=dependency_receipt,
            expected_type=expected_type,
        )


def _expected_dependencies(
    *,
    value: object,
    objects_by_receipt: Mapping[str, object],
    primary_decisions_by_date: Mapping[str, MassiveAdaptiveDecisionRootV1],
) -> dict[str, type[object]]:
    result: dict[str, type[object]] = {}
    if isinstance(value, MassiveAdaptiveWindowPlanV1):
        _add_expected(
            result,
            receipt=value.decision_tensor_receipt_sha256,
            expected_type=MassiveAdaptiveDecisionTensorV1,
        )
        _add_expected(
            result,
            receipt=value.split_plan_receipt_sha256,
            expected_type=MassiveAdaptiveSplitPlanV1,
        )
    elif isinstance(value, MassiveAdaptiveCausalCheckpointChoiceV1):
        for receipt in value.candidate_checkpoint_receipts:
            _add_expected(
                result,
                receipt=receipt,
                expected_type=MassiveAdaptiveCheckpointV1,
            )
        _add_expected(
            result,
            receipt=value.training_window_plan_receipt_sha256,
            expected_type=MassiveAdaptiveWindowPlanV1,
        )
        _add_expected(
            result,
            receipt=value.training_config_receipt_sha256,
            expected_type=MassiveAdaptiveSupervisedTrainingConfigV1,
        )
    elif isinstance(value, MassiveAdaptiveForecastCalibrationV2):
        _add_expected_pairs(
            result,
            (
                (value.checkpoint_receipt_sha256, MassiveAdaptiveCheckpointV1),
                (
                    value.training_forecast_archive_receipt_sha256,
                    MassiveAdaptiveForecastArchiveV1,
                ),
                (
                    value.training_target_archive_receipt_sha256,
                    MassiveAdaptiveTargetArchiveV1,
                ),
                (
                    value.training_window_plan_receipt_sha256,
                    MassiveAdaptiveWindowPlanV1,
                ),
            ),
        )
    elif isinstance(value, MassiveAdaptiveRLFitForecastArchiveV1):
        _add_expected_pairs(
            result,
            (
                (value.checkpoint_receipt_sha256, MassiveAdaptiveCheckpointV1),
                (
                    value.training_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (
                    value.training_window_plan_receipt_sha256,
                    MassiveAdaptiveWindowPlanV1,
                ),
                (
                    value.inference_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (
                    value.inference_plan_receipt_sha256,
                    MassiveAdaptiveRLFitInferencePlanV1,
                ),
                (value.split_plan_receipt_sha256, MassiveAdaptiveSplitPlanV1),
                (
                    value.model_spec_receipt_sha256,
                    MassiveAdaptiveAlphaModelSpecV1,
                ),
            ),
        )
        tensor = objects_by_receipt.get(value.inference_tensor_receipt_sha256)
        if type(tensor) is not MassiveAdaptiveDecisionTensorV1:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "RL-fit forecast inference tensor dependency is absent"
            )
        for date in tensor.decision_session_dates:
            decision = primary_decisions_by_date.get(date)
            if decision is None:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "RL-fit forecast dependency decision root is absent"
                )
            _add_expected(
                result,
                receipt=decision.semantic_receipt_sha256,
                expected_type=MassiveAdaptiveDecisionRootV1,
            )
    elif isinstance(value, MassiveAdaptiveDecisionRootV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.context_origin_receipt_sha256,
                    MassiveAdaptiveContextOriginAuthorityV1,
                ),
                (
                    value.action_origin_receipt_sha256,
                    MassiveAdaptiveOriginAuthorityV1,
                ),
                (
                    value.feature_semantic_receipt_sha256,
                    MassiveProfitabilityOriginFeaturesV3,
                ),
                (
                    value.decision_clock_receipt_sha256,
                    MassiveDecisionClockAuthority,
                ),
                (value.session_authority_receipt_sha256, MassiveSessionAuthority),
            ),
        )
    elif isinstance(value, MassiveAdaptiveContextOriginAuthorityV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.decision_clock_receipt_sha256,
                    MassiveDecisionClockAuthority,
                ),
                (value.session_authority_receipt_sha256, MassiveSessionAuthority),
                (
                    value.identity_authority_receipt_sha256,
                    PITSecurityUniverseAuthority,
                ),
                (
                    value.feature_semantic_receipt_sha256,
                    MassiveProfitabilityOriginFeaturesV3,
                ),
            ),
        )
    elif isinstance(value, MassiveAdaptiveDecisionTensorV1):
        for receipt in value.feature_semantic_receipts:
            _add_expected(
                result,
                receipt=receipt,
                expected_type=MassiveProfitabilityOriginFeaturesV3,
            )
        for receipt in value.action_origin_receipts:
            _add_expected(
                result,
                receipt=receipt,
                expected_type=MassiveAdaptiveOriginAuthorityV1,
            )
    elif isinstance(value, MassiveAdaptiveCheckpointV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.training_authority_receipt_sha256,
                    MassiveAdaptiveTrainingAuthorityV1,
                ),
                (
                    value.decision_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (value.split_plan_receipt_sha256, MassiveAdaptiveSplitPlanV1),
                (value.window_plan_receipt_sha256, MassiveAdaptiveWindowPlanV1),
                (
                    value.model_spec_receipt_sha256,
                    MassiveAdaptiveAlphaModelSpecV1,
                ),
                (
                    value.training_config_receipt_sha256,
                    MassiveAdaptiveSupervisedTrainingConfigV1,
                ),
            ),
        )
    elif isinstance(value, MassiveAdaptiveForecastArchiveV1):
        _add_expected_pairs(
            result,
            (
                (value.checkpoint_receipt_sha256, MassiveAdaptiveCheckpointV1),
                (
                    value.decision_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (value.window_plan_receipt_sha256, MassiveAdaptiveWindowPlanV1),
                (
                    value.model_spec_receipt_sha256,
                    MassiveAdaptiveAlphaModelSpecV1,
                ),
            ),
        )
        tensor = objects_by_receipt.get(value.decision_tensor_receipt_sha256)
        if type(tensor) is not MassiveAdaptiveDecisionTensorV1:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "training forecast decision tensor dependency is absent"
            )
        for date in tensor.decision_session_dates:
            training_decision = primary_decisions_by_date.get(date)
            if training_decision is None:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "training forecast dependency decision root is absent"
                )
            _add_expected(
                result,
                receipt=training_decision.semantic_receipt_sha256,
                expected_type=MassiveAdaptiveDecisionRootV1,
            )
    elif isinstance(value, MassiveAdaptiveTargetArchiveV1):
        for receipt in value.origin_decision_root_receipts:
            _add_expected(
                result, receipt=receipt, expected_type=MassiveAdaptiveDecisionRootV1
            )
        for receipt in value.source_target_receipts:
            _add_expected(
                result, receipt=receipt, expected_type=MassiveAdaptiveSourceTargetsV1
            )
    elif isinstance(value, MassiveAdaptiveTrainingAuthorityV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.decision_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (
                    value.target_archive_receipt_sha256,
                    MassiveAdaptiveTargetArchiveV1,
                ),
                (value.split_plan_receipt_sha256, MassiveAdaptiveSplitPlanV1),
                (value.window_plan_receipt_sha256, MassiveAdaptiveWindowPlanV1),
            ),
        )
        tensor = objects_by_receipt.get(value.decision_tensor_receipt_sha256)
        if type(tensor) is not MassiveAdaptiveDecisionTensorV1:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "training authority decision tensor dependency is absent"
            )
        for date in tensor.decision_session_dates:
            decision = primary_decisions_by_date.get(date)
            if decision is None:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "training authority decision root dependency is absent"
                )
            _add_expected(
                result,
                receipt=decision.semantic_receipt_sha256,
                expected_type=MassiveAdaptiveDecisionRootV1,
            )
    elif isinstance(value, MassiveAdaptiveRLFitInferencePlanV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.decision_tensor_receipt_sha256,
                    MassiveAdaptiveDecisionTensorV1,
                ),
                (value.split_plan_receipt_sha256, MassiveAdaptiveSplitPlanV1),
                (value.session_authority_receipt_sha256, MassiveSessionAuthority),
                (
                    value.model_spec_receipt_sha256,
                    MassiveAdaptiveAlphaModelSpecV1,
                ),
            ),
        )
    elif isinstance(value, MassiveAdaptiveSourceTargetsV1):
        _add_expected_pairs(
            result,
            (
                (
                    value.origin_authority_receipt_sha256,
                    MassiveAdaptiveOriginAuthorityV1,
                ),
                (
                    value.decision_clock_receipt_sha256,
                    MassiveDecisionClockAuthority,
                ),
                (value.session_authority_receipt_sha256, MassiveSessionAuthority),
                (
                    value.identity_authority_receipt_sha256,
                    PITSecurityUniverseAuthority,
                ),
                (
                    value.daily_input_authority_receipt_sha256,
                    MassiveProfitabilityDailyInputAuthorityV1,
                ),
                (value.fill_source_receipt_sha256, MassiveAdaptiveFillSourceV1),
            ),
        )
    return result


def _complete_object_graph(
    *,
    primary: Sequence[tuple[str, int | None, str, str, object]],
    dependencies: Sequence[object],
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    objects: dict[str, object] = {}
    for value in (*tuple(row[4] for row in primary), *tuple(dependencies)):
        if not is_dataclass(value) or not hasattr(value, "validate"):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "replay dependency is not a validated dataclass authority"
            )
        cast(Any, value).validate()
        receipt = _receipt(value)
        previous = objects.setdefault(receipt, value)
        if type(previous) is not type(value) or _encode_value(
            previous
        ) != _encode_value(value):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "replay dependency receipt resolves to different objects"
            )
    primary_receipts = {_receipt(row[4]) for row in primary}
    supplied_dependency_receipts = {_receipt(value) for value in dependencies}
    if primary_receipts & supplied_dependency_receipts:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "replay dependency duplicates a primary runtime-source object"
        )
    decisions: dict[str, MassiveAdaptiveDecisionRootV1] = {}
    for value in objects.values():
        if not isinstance(value, MassiveAdaptiveDecisionRootV1):
            continue
        previous = decisions.setdefault(value.decision_session_date, value)
        if previous.semantic_receipt_sha256 != value.semantic_receipt_sha256:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source decision date resolves to different roots"
            )
    dependency_edges: dict[str, tuple[str, ...]] = {}
    queue = list(primary_receipts)
    visited: set[str] = set()
    required_auxiliary: set[str] = set()
    while queue:
        receipt = queue.pop()
        if receipt in visited:
            continue
        visited.add(receipt)
        value = objects[receipt]
        expected = _expected_dependencies(
            value=value,
            objects_by_receipt=objects,
            primary_decisions_by_date=decisions,
        )
        dependency_edges[receipt] = tuple(sorted(expected))
        for dependency_receipt, expected_type in expected.items():
            dependency = objects.get(dependency_receipt)
            if dependency is None or type(dependency) is not expected_type:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "runtime-source replay dependency is absent or has the wrong type"
                )
            if dependency_receipt not in primary_receipts:
                required_auxiliary.add(dependency_receipt)
            queue.append(dependency_receipt)
    if supplied_dependency_receipts != required_auxiliary:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source replay dependency inventory is incomplete or contains extras"
        )
    return objects, dependency_edges


def _row(
    *,
    role: str,
    fold_index: int | None,
    logical_key: str,
    role_authority_receipt_sha256: str,
    value: object,
    dependency_receipts: tuple[str, ...],
    snapshot_file_sha256: str,
) -> MassiveAdaptiveRLReplayDependencyV1:
    receipt = _receipt(value)
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_V1_SCHEMA,
        "role": role,
        "fold_index": fold_index,
        "logical_key": logical_key,
        "role_authority_receipt_sha256": role_authority_receipt_sha256,
        "relative_path": _snapshot_relative_path(receipt),
        "file_sha256": snapshot_file_sha256,
        "semantic_receipt_sha256": receipt,
        "domain_type_name": _type_name(value),
        "domain_schema": _domain_schema(role=role, value=value),
        "domain_specification_sha256": _domain_specification(value),
        "implementation_source_sha256": _implementation_source_sha256(type(value)),
        "parser_id": _CODEC_ID,
        "dependency_receipts": dependency_receipts,
    }
    result = MassiveAdaptiveRLReplayDependencyV1(
        **body,  # type: ignore[arg-type]
        receipt_sha256=semantic_sha256(body),
    )
    result.validate()
    return result


def materialize_massive_adaptive_rl_replay_dependency_index_v1(
    *,
    source_root: str | Path,
    manifest: MassiveAdaptiveRLExperimentManifestV3,
    runtime_source_graph_authority: MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1,
    replay_dependencies: Sequence[object],
) -> MassiveAdaptiveRLReplayDependencyIndexV1:
    """Publish a dependency-closed index from one witnessed runtime graph."""

    manifest.validate()
    runtime_source_graph_authority.validate()
    runtime_receipt = runtime_source_graph_authority.runtime_authority_receipt_sha256
    if (
        runtime_receipt is None
        or not runtime_source_graph_authority.source_data_qualified
        or runtime_source_graph_authority.experiment_id != manifest.experiment_id
        or runtime_source_graph_authority.manifest_v3_receipt_sha256
        != manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source reconstruction requires the witnessed graph"
        )
    primary = _primary_bindings(runtime_source_graph_authority)
    objects, dependency_edges = _complete_object_graph(
        primary=primary,
        dependencies=tuple(replay_dependencies),
    )
    root = Path(source_root).resolve()
    snapshot_hashes: dict[str, str] = {}
    for receipt, value in objects.items():
        payload = _snapshot_payload(value)
        relative = _snapshot_relative_path(receipt)
        output = _create_only_output_path(root=root, relative_path=relative)
        _atomic_install(output=output, payload=payload, allow_exact_existing=True)
        snapshot_hashes[receipt] = file_sha256(output)
    rows = [
        _row(
            role=role,
            fold_index=fold_index,
            logical_key=logical_key,
            role_authority_receipt_sha256=role_receipt,
            value=value,
            dependency_receipts=dependency_edges[_receipt(value)],
            snapshot_file_sha256=snapshot_hashes[_receipt(value)],
        )
        for role, fold_index, logical_key, role_receipt, value in primary
    ]
    primary_receipts = {_receipt(row[4]) for row in primary}
    for receipt in sorted(set(objects) - primary_receipts):
        value = objects[receipt]
        rows.append(
            _row(
                role=_DEPENDENCY_ROLE,
                fold_index=None,
                logical_key=f"{_type_name(value)}:{receipt}",
                role_authority_receipt_sha256=receipt,
                value=value,
                dependency_receipts=dependency_edges.get(receipt, ()),
                snapshot_file_sha256=snapshot_hashes[receipt],
            )
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.role == _DEPENDENCY_ROLE,
                row.fold_index if row.fold_index is not None else -1,
                row.role,
                row.logical_key,
            ),
        )
    )
    object_rows = tuple(
        sorted(
            {
                (
                    row.semantic_receipt_sha256,
                    row.relative_path,
                    row.file_sha256,
                    row.domain_type_name,
                )
                for row in ordered
            }
        )
    )
    edges = tuple(
        sorted(
            (row.semantic_receipt_sha256, dependency)
            for row in ordered
            for dependency in row.dependency_receipts
        )
    )
    body = {
        "schema": MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V1_SCHEMA,
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "base_manifest_receipt_sha256": manifest.base_manifest.semantic_receipt_sha256,
        "source_bundle_receipt_sha256": (
            runtime_source_graph_authority.source_bundle_receipt_sha256
        ),
        "persisted_runtime_source_graph_receipt_sha256": (
            runtime_source_graph_authority.semantic_receipt_sha256
        ),
        "runtime_source_graph_witness_receipt_sha256": runtime_receipt,
        "rows": ordered,
        "row_inventory_sha256": semantic_sha256(
            tuple(row.receipt_sha256 for row in ordered)
        ),
        "object_inventory_sha256": semantic_sha256(object_rows),
        "dependency_edge_inventory_sha256": semantic_sha256(edges),
        "committed_source_data_qualified": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
        ),
        "implementation_source_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SOURCE_SHA256
        ),
    }
    provisional = MassiveAdaptiveRLReplayDependencyIndexV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLReplayDependencyIndexV1(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    output = replay_dependency_index_path_v1(
        source_root=root,
        experiment_id=manifest.experiment_id,
    )
    output = _create_only_output_path(
        root=root,
        relative_path=output.relative_to(root).as_posix(),
    )
    _atomic_install(
        output=output,
        payload=asdict(result),
        allow_exact_existing=False,
    )
    return result


def _parse_dependency_row(value: object) -> MassiveAdaptiveRLReplayDependencyV1:
    if not isinstance(value, Mapping):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "adaptive RL replay-dependency row is malformed"
        )
    payload = dict(value)
    dependencies = payload.get("dependency_receipts")
    if not isinstance(dependencies, list):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "adaptive RL replay-dependency edge inventory is malformed"
        )
    payload["dependency_receipts"] = tuple(dependencies)
    result = MassiveAdaptiveRLReplayDependencyV1(**payload)
    result.validate()
    return result


def load_massive_adaptive_rl_replay_dependency_index_v1(
    *, source_root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> MassiveAdaptiveRLReplayDependencyIndexV1:
    """Rehash the package-owned dependency index without reconstructing objects."""

    manifest.validate()
    root = Path(source_root).resolve()
    path = replay_dependency_index_path_v1(
        source_root=root,
        experiment_id=manifest.experiment_id,
    )
    relative = path.relative_to(root).as_posix()
    resolved = _resolve_regular_file(root=root, relative_path=relative)
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except OSError as error:
        raise MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
            "adaptive RL replay-dependency index is temporarily unavailable"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MassiveAdaptiveRLRuntimeSourceIntegrityError(
            "adaptive RL replay-dependency index cannot be decoded"
        ) from error
    if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "adaptive RL replay-dependency index is not canonical JSON"
        )
    payload = dict(value)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "adaptive RL replay-dependency rows are malformed"
        )
    payload["rows"] = tuple(_parse_dependency_row(row) for row in raw_rows)
    result = MassiveAdaptiveRLReplayDependencyIndexV1(**payload)
    result.validate()
    if (
        result.experiment_id != manifest.experiment_id
        or result.manifest_v3_receipt_sha256 != manifest.semantic_receipt_sha256
        or result.base_manifest_receipt_sha256
        != manifest.base_manifest.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "adaptive RL replay-dependency index belongs to another manifest"
        )
    for row in result.rows:
        snapshot = _resolve_regular_file(
            root=root,
            relative_path=row.relative_path,
        )
        if file_sha256(snapshot) != row.file_sha256:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "adaptive RL replay-dependency snapshot hash differs"
            )
    return result


def _load_snapshot_objects(
    *, root: Path, index: MassiveAdaptiveRLReplayDependencyIndexV1
) -> dict[str, object]:
    result: dict[str, object] = {}
    for row in index.rows:
        if row.semantic_receipt_sha256 in result:
            continue
        path = _resolve_regular_file(root=root, relative_path=row.relative_path)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except OSError as error:
            raise MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
                "runtime-source object snapshot is temporarily unavailable"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MassiveAdaptiveRLRuntimeSourceIntegrityError(
                "runtime-source object snapshot cannot be decoded"
            ) from error
        if not isinstance(value, Mapping) or raw != canonical_json_file_bytes(value):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source object snapshot is not canonical JSON"
            )
        parsed = _parse_snapshot(value)
        if (
            _receipt(parsed) != row.semantic_receipt_sha256
            or _type_name(parsed) != row.domain_type_name
            or _domain_schema(role=row.role, value=parsed) != row.domain_schema
            or _domain_specification(parsed) != row.domain_specification_sha256
        ):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source snapshot differs from its dependency row"
            )
        result[row.semantic_receipt_sha256] = parsed
    return result


def _verify_reconstructed_dependency_closure(
    *,
    index: MassiveAdaptiveRLReplayDependencyIndexV1,
    objects: Mapping[str, object],
) -> None:
    primary_rows = tuple(row for row in index.rows if row.role != _DEPENDENCY_ROLE)
    primary = tuple(
        (
            row.role,
            row.fold_index,
            row.logical_key,
            row.role_authority_receipt_sha256,
            objects[row.semantic_receipt_sha256],
        )
        for row in primary_rows
    )
    dependency_receipts = tuple(
        sorted(
            {
                row.semantic_receipt_sha256
                for row in index.rows
                if row.role == _DEPENDENCY_ROLE
            }
        )
    )
    reconstructed, dependency_edges = _complete_object_graph(
        primary=primary,
        dependencies=tuple(objects[receipt] for receipt in dependency_receipts),
    )
    if set(reconstructed) != set(objects) or any(
        row.dependency_receipts != dependency_edges.get(row.semantic_receipt_sha256, ())
        for row in index.rows
    ):
        raise MassiveAdaptiveRLRuntimeSourceDependencyMismatch(
            "runtime-source dependency closure differs after reconstruction"
        )


def _decision_root_views(
    *,
    objects: Mapping[str, object],
    primary_rows: Sequence[MassiveAdaptiveRLReplayDependencyV1],
) -> tuple[
    dict[int, tuple[MassiveAdaptiveDecisionRootV1, ...]],
    dict[str, MassiveAdaptiveDecisionRootV1],
]:
    all_decisions: dict[str, MassiveAdaptiveDecisionRootV1] = {}
    for value in objects.values():
        if type(value) is not MassiveAdaptiveDecisionRootV1:
            continue
        decision = cast(MassiveAdaptiveDecisionRootV1, value)
        previous = all_decisions.setdefault(decision.decision_session_date, decision)
        if previous.semantic_receipt_sha256 != decision.semantic_receipt_sha256:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "runtime-source decision date resolves to different roots"
            )

    decisions_by_fold: dict[int, tuple[MassiveAdaptiveDecisionRootV1, ...]] = {}
    for fold_index in range(4):
        current = tuple(
            cast(MassiveAdaptiveDecisionRootV1, objects[row.semantic_receipt_sha256])
            for row in primary_rows
            if row.role == "decision-root-inventory" and row.fold_index == fold_index
        )
        decisions_by_fold[fold_index] = tuple(
            sorted(current, key=lambda row: row.decision_session_date)
        )
    return decisions_by_fold, all_decisions


def _objects_of_type(
    objects: Mapping[str, object], expected: type[object]
) -> tuple[object, ...]:
    return tuple(value for value in objects.values() if type(value) is expected)


def _reauthorize_dependencies(
    *,
    root: Path,
    objects: dict[str, object],
    primary_decisions_by_date: Mapping[str, MassiveAdaptiveDecisionRootV1],
) -> None:
    for value in _objects_of_type(objects, MassiveAdaptiveDecisionTensorV1):
        tensor = cast(MassiveAdaptiveDecisionTensorV1, value)
        features = tuple(
            cast(MassiveProfitabilityOriginFeaturesV3, objects[receipt])
            for receipt in tensor.feature_semantic_receipts
        )
        origins = tuple(
            cast(MassiveAdaptiveOriginAuthorityV1, objects[receipt])
            for receipt in tensor.action_origin_receipts
        )
        objects[_receipt(tensor)] = authorize_massive_adaptive_decision_tensor_v1(
            root=root,
            tensor=tensor,
            features=features,
            action_origins=origins,
        )
    for value in _objects_of_type(objects, MassiveAdaptiveTrainingAuthorityV1):
        training = cast(MassiveAdaptiveTrainingAuthorityV1, value)
        tensor = cast(
            MassiveAdaptiveDecisionTensorV1,
            objects[training.decision_tensor_receipt_sha256],
        )
        roots = tuple(
            primary_decisions_by_date[date] for date in tensor.decision_session_dates
        )
        objects[_receipt(training)] = build_massive_adaptive_training_authority_v1(
            decision_tensor=tensor,
            decision_roots=roots,
            target_archive=cast(
                MassiveAdaptiveTargetArchiveV1,
                objects[training.target_archive_receipt_sha256],
            ),
            split_plan=cast(
                MassiveAdaptiveSplitPlanV1,
                objects[training.split_plan_receipt_sha256],
            ),
            window_plan=cast(
                MassiveAdaptiveWindowPlanV1,
                objects[training.window_plan_receipt_sha256],
            ),
        )
    for value in _objects_of_type(objects, MassiveAdaptiveCheckpointV1):
        checkpoint = cast(MassiveAdaptiveCheckpointV1, value)
        config = cast(
            MassiveAdaptiveSupervisedTrainingConfigV1,
            objects[checkpoint.training_config_receipt_sha256],
        )
        objects[_receipt(checkpoint)] = authorize_massive_adaptive_checkpoint_v1(
            root=root,
            checkpoint=checkpoint,
            training_authority=cast(
                MassiveAdaptiveTrainingAuthorityV1,
                objects[checkpoint.training_authority_receipt_sha256],
            ),
            decision_tensor_receipt_sha256=checkpoint.decision_tensor_receipt_sha256,
            split_plan_receipt_sha256=checkpoint.split_plan_receipt_sha256,
            window_plan_receipt_sha256=checkpoint.window_plan_receipt_sha256,
            model_spec_receipt_sha256=checkpoint.model_spec_receipt_sha256,
            training_config_receipt_sha256=config.receipt_sha256,
        )
    for value in _objects_of_type(objects, MassiveAdaptiveCausalCheckpointChoiceV1):
        choice = cast(MassiveAdaptiveCausalCheckpointChoiceV1, value)
        rebuilt = build_massive_adaptive_causal_checkpoint_choice_v1(
            checkpoints=tuple(
                cast(MassiveAdaptiveCheckpointV1, objects[receipt])
                for receipt in choice.candidate_checkpoint_receipts
            ),
            training_window_plan=cast(
                MassiveAdaptiveWindowPlanV1,
                objects[choice.training_window_plan_receipt_sha256],
            ),
        )
        if rebuilt.semantic_receipt_sha256 != choice.semantic_receipt_sha256:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "reconstructed causal checkpoint choice differs"
            )
        objects[_receipt(choice)] = rebuilt
    for value in _objects_of_type(objects, MassiveAdaptiveForecastArchiveV1):
        archive = cast(MassiveAdaptiveForecastArchiveV1, value)
        tensor = cast(
            MassiveAdaptiveDecisionTensorV1,
            objects[archive.decision_tensor_receipt_sha256],
        )
        decisions = tuple(
            primary_decisions_by_date[date] for date in tensor.decision_session_dates
        )
        objects[_receipt(archive)] = authorize_massive_adaptive_forecast_archive_v1(
            root=root,
            archive=archive,
            checkpoint=cast(
                MassiveAdaptiveCheckpointV1,
                objects[archive.checkpoint_receipt_sha256],
            ),
            decision_tensor=tensor,
            decision_roots=decisions,
            window_plan=cast(
                MassiveAdaptiveWindowPlanV1,
                objects[archive.window_plan_receipt_sha256],
            ),
            model_spec=cast(
                MassiveAdaptiveAlphaModelSpecV1,
                objects[archive.model_spec_receipt_sha256],
            ),
        )
    for value in _objects_of_type(objects, MassiveAdaptiveForecastCalibrationV2):
        calibration = cast(MassiveAdaptiveForecastCalibrationV2, value)
        objects[_receipt(calibration)] = (
            authorize_massive_adaptive_forecast_calibration_v2(
                root=root,
                calibration=calibration,
                checkpoint=cast(
                    MassiveAdaptiveCheckpointV1,
                    objects[calibration.checkpoint_receipt_sha256],
                ),
                training_forecasts=cast(
                    MassiveAdaptiveForecastArchiveV1,
                    objects[calibration.training_forecast_archive_receipt_sha256],
                ),
                training_targets=cast(
                    MassiveAdaptiveTargetArchiveV1,
                    objects[calibration.training_target_archive_receipt_sha256],
                ),
                training_window_plan=cast(
                    MassiveAdaptiveWindowPlanV1,
                    objects[calibration.training_window_plan_receipt_sha256],
                ),
            )
        )
    for value in _objects_of_type(objects, MassiveAdaptiveRLFitForecastArchiveV1):
        fit_archive = cast(MassiveAdaptiveRLFitForecastArchiveV1, value)
        inference_tensor = cast(
            MassiveAdaptiveDecisionTensorV1,
            objects[fit_archive.inference_tensor_receipt_sha256],
        )
        objects[_receipt(fit_archive)] = (
            authorize_massive_adaptive_rl_fit_forecast_archive_v1(
                root=root,
                archive=fit_archive,
                checkpoint=cast(
                    MassiveAdaptiveCheckpointV1,
                    objects[fit_archive.checkpoint_receipt_sha256],
                ),
                training_window_plan=cast(
                    MassiveAdaptiveWindowPlanV1,
                    objects[fit_archive.training_window_plan_receipt_sha256],
                ),
                inference_tensor=inference_tensor,
                inference_decision_roots=tuple(
                    primary_decisions_by_date[date]
                    for date in inference_tensor.decision_session_dates
                ),
                inference_plan=cast(
                    MassiveAdaptiveRLFitInferencePlanV1,
                    objects[fit_archive.inference_plan_receipt_sha256],
                ),
                split_plan=cast(
                    MassiveAdaptiveSplitPlanV1,
                    objects[fit_archive.split_plan_receipt_sha256],
                ),
                model_spec=cast(
                    MassiveAdaptiveAlphaModelSpecV1,
                    objects[fit_archive.model_spec_receipt_sha256],
                ),
            )
        )


def _rebuild_role_map(
    *,
    index: MassiveAdaptiveRLReplayDependencyIndexV1,
    objects: Mapping[str, object],
) -> dict[tuple[str, int | None], MassiveAdaptiveRLRoleBoundSourceAuthorityV1]:
    grouped: dict[
        tuple[str, int | None], list[MassiveAdaptiveRLReplayDependencyV1]
    ] = {}
    for row in index.rows:
        if row.role == _DEPENDENCY_ROLE:
            continue
        grouped.setdefault((row.role, row.fold_index), []).append(row)
    expected_keys = {
        (role, fold_index)
        for role, spec in MASSIVE_ADAPTIVE_RL_SOURCE_ROLE_REGISTRY_V1.items()
        for fold_index in (range(4) if spec.fold_scoped else (None,))
    }
    if set(grouped) != expected_keys:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "reconstructed runtime-source role inventory is incomplete"
        )
    result: dict[
        tuple[str, int | None], MassiveAdaptiveRLRoleBoundSourceAuthorityV1
    ] = {}
    for key in sorted(
        grouped, key=lambda row: (row[1] is not None, row[1] or -1, row[0])
    ):
        role, fold_index = key
        rows = tuple(sorted(grouped[key], key=lambda row: row.logical_key))
        role_receipts = {row.role_authority_receipt_sha256 for row in rows}
        if len(role_receipts) != 1:
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "reconstructed runtime-source role receipt differs"
            )
        if role in _DOMAIN_INVENTORY_ITEM_TYPES:
            authority: MassiveAdaptiveRLSourceAuthorityProtocol = (
                build_massive_adaptive_rl_typed_authority_inventory_v1(
                    role=role,
                    fold_index=fold_index,
                    items=tuple(
                        cast(
                            MassiveAdaptiveRLSourceAuthorityProtocol,
                            objects[row.semantic_receipt_sha256],
                        )
                        for row in rows
                    ),
                )
            )
        else:
            if len(rows) != 1 or rows[0].logical_key != _ROOT_LOGICAL_KEY:
                raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                    "reconstructed direct runtime-source role differs"
                )
            authority = cast(
                MassiveAdaptiveRLSourceAuthorityProtocol,
                objects[rows[0].semantic_receipt_sha256],
            )
        if _receipt(authority) != next(iter(role_receipts)):
            raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
                "reconstructed runtime-source role authority receipt differs"
            )
        result[key] = bind_massive_adaptive_rl_source_authority_v1(
            role=role,
            fold_index=fold_index,
            authority=authority,
            source_data_qualified=True,
            runtime_source_replayed=True,
        )
    return result


def _inventory_items_from_map(
    runtime_sources: Mapping[
        tuple[str, int | None], MassiveAdaptiveRLRoleBoundSourceAuthorityV1
    ],
    *,
    role: str,
    fold_index: int | None,
) -> tuple[object, ...]:
    authority = runtime_sources[(role, fold_index)].authority
    if (
        not isinstance(authority, MassiveAdaptiveRLTypedAuthorityInventoryV1)
        or authority.runtime_items is None
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "reconstructed runtime-source typed inventory is absent"
        )
    return tuple(authority.runtime_items)


def _reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
    *, source_root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> MassiveAdaptiveRLRuntimeSourcesV1:
    manifest.validate()
    root = Path(source_root).resolve()
    index = load_massive_adaptive_rl_replay_dependency_index_v1(
        source_root=root,
        manifest=manifest,
    )
    source_bundle = load_massive_adaptive_rl_source_bundle_v1(
        source_root=root,
        manifest=manifest.base_manifest,
    )
    generic_graph = load_massive_adaptive_rl_runtime_source_graph_authority_v1(
        source_root=root,
        manifest=manifest,
        source_bundle=source_bundle,
    )
    if (
        index.source_bundle_receipt_sha256 != source_bundle.semantic_receipt_sha256
        or index.persisted_runtime_source_graph_receipt_sha256
        != generic_graph.semantic_receipt_sha256
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "runtime-source reconstruction index lineage differs"
        )
    objects = _load_snapshot_objects(root=root, index=index)
    primary_rows = tuple(row for row in index.rows if row.role != _DEPENDENCY_ROLE)
    _verify_reconstructed_dependency_closure(index=index, objects=objects)
    decisions_by_fold, all_decisions = _decision_root_views(
        objects=objects,
        primary_rows=primary_rows,
    )
    _reauthorize_dependencies(
        root=root,
        objects=objects,
        primary_decisions_by_date=all_decisions,
    )
    runtime_sources = _rebuild_role_map(index=index, objects=objects)
    try:
        authorized_graph = (
            authorize_massive_adaptive_rl_runtime_source_graph_authority_v1(
                authority=generic_graph,
                source_bundle=source_bundle,
                runtime_sources=runtime_sources,
            )
        )
    except MassiveAdaptiveRLRuntimeSourceGraphAuthorityV1Error as error:
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "reconstructed runtime-source graph does not authorize"
        ) from error
    if (
        authorized_graph.runtime_authority_receipt_sha256
        != index.runtime_source_graph_witness_receipt_sha256
    ):
        raise MassiveAdaptiveRLRuntimeSourceReconstructionV1Error(
            "reconstructed runtime-source witness receipt differs"
        )
    folds: list[MassiveAdaptiveRLFoldRuntimeSourcesV1] = []
    for fold_index in range(4):
        fold = MassiveAdaptiveRLFoldRuntimeSourcesV1(
            outer_fold_index=fold_index,
            training_windows=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveWindowPlanV1, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="training-window-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.fold_index,
                )
            ),
            checkpoint_choices=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveCausalCheckpointChoiceV1, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="supervised-checkpoint-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.fold_index,
                )
            ),
            calibrations=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveForecastCalibrationV2, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="calibration-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.fold_index,
                )
            ),
            fit_forecast_archives=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveRLFitForecastArchiveV1, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="fit-forecast-archive-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.block_index,
                )
            ),
            decision_roots=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveDecisionRootV1, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="decision-root-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.decision_session_date,
                )
            ),
            context_origins=tuple(
                sorted(
                    cast(
                        tuple[MassiveAdaptiveContextOriginAuthorityV1, ...],
                        _inventory_items_from_map(
                            runtime_sources,
                            role="context-origin-inventory",
                            fold_index=fold_index,
                        ),
                    ),
                    key=lambda row: row.decision_session_date,
                )
            ),
        )
        fold.validate()
        folds.append(fold)
    partition_items = cast(
        tuple[MassivePersistedPartitionManifestV1, ...],
        _inventory_items_from_map(
            runtime_sources,
            role="persisted-partition-inventory",
            fold_index=None,
        ),
    )
    replay_receipts = tuple(
        sorted(
            row.semantic_receipt_sha256
            for row in index.rows
            if row.role == _DEPENDENCY_ROLE
        )
    )
    body = {
        "experiment_id": manifest.experiment_id,
        "manifest_v3_receipt_sha256": manifest.semantic_receipt_sha256,
        "source_bundle_receipt_sha256": source_bundle.semantic_receipt_sha256,
        "replay_dependency_index_receipt_sha256": index.semantic_receipt_sha256,
        "runtime_source_graph_authority": authorized_graph,
        "session_authority": cast(
            MassiveSessionAuthority,
            runtime_sources[("session-authority", None)].authority,
        ),
        "condition_authority": cast(
            MassiveConditionAuthority,
            runtime_sources[("condition-authority", None)].authority,
        ),
        "persisted_partition_manifests": partition_items,
        "identity_authority": cast(
            PITSecurityUniverseAuthority,
            runtime_sources[("identity-authority", None)].authority,
        ),
        "economic_event_archive": cast(
            MassiveProviderEconomicArchiveAuthorityV6,
            runtime_sources[("economic-event-archive", None)].authority,
        ),
        "daily_input_authority": cast(
            MassiveProfitabilityDailyInputAuthorityV1,
            runtime_sources[("daily-input-authority", None)].authority,
        ),
        "fill_source": cast(
            MassiveAdaptiveFillSourceV1,
            runtime_sources[("fill-source-authority", None)].authority,
        ),
        "split_plan": cast(
            MassiveAdaptiveSplitPlanV1,
            runtime_sources[("split-plan", None)].authority,
        ),
        "folds": tuple(folds),
        "replay_dependency_receipts": replay_receipts,
        "source_data_qualified": True,
        "profitability_reporting_authorized": False,
        "lockbox_access_authorized": False,
        "protocol_receipt_sha256": MASSIVE_ADAPTIVE_ALPHA_V1_RECEIPT_SHA256,
        "specification_sha256": (
            MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256
        ),
        "schema": MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V1_SCHEMA,
    }
    provisional = MassiveAdaptiveRLRuntimeSourcesV1(
        **body,  # type: ignore[arg-type]
        semantic_receipt_sha256="0" * 64,
    )
    result = MassiveAdaptiveRLRuntimeSourcesV1(
        **{
            **body,
            "semantic_receipt_sha256": semantic_sha256(provisional.semantic_unsigned()),
        }  # type: ignore[arg-type]
    )
    result.validate()
    return result


def reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
    *, source_root: str | Path, manifest: MassiveAdaptiveRLExperimentManifestV3
) -> MassiveAdaptiveRLRuntimeSourcesV1:
    """Reconstruct and authorize the complete runtime graph from ``source_root``."""

    try:
        return _reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1(
            source_root=source_root,
            manifest=manifest,
        )
    except MassiveAdaptiveRLRuntimeSourceReconstructionV1Error:
        raise
    except OSError as error:
        raise MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable(
            "persisted runtime-source dependencies are temporarily unavailable"
        ) from error
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MassiveAdaptiveRLRuntimeSourceIntegrityError(
            "persisted runtime-source dependencies do not reconstruct"
        ) from error


__all__ = [
    "MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_INDEX_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_REPLAY_DEPENDENCY_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_OBJECT_SNAPSHOT_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCES_V1_SCHEMA",
    "MASSIVE_ADAPTIVE_RL_RUNTIME_SOURCE_RECONSTRUCTION_V1_SPEC_SHA256",
    "MassiveAdaptiveRLFoldRuntimeSourcesV1",
    "MassiveAdaptiveRLReplayDependencyIndexV1",
    "MassiveAdaptiveRLReplayDependencyV1",
    "MassiveAdaptiveRLRuntimeSourceDependencyMismatch",
    "MassiveAdaptiveRLRuntimeSourceImplementationMismatch",
    "MassiveAdaptiveRLRuntimeSourceIntegrityError",
    "MassiveAdaptiveRLRuntimeSourceReconstructionV1Error",
    "MassiveAdaptiveRLRuntimeSourceTemporarilyUnavailable",
    "MassiveAdaptiveRLRuntimeSourcesV1",
    "load_massive_adaptive_rl_replay_dependency_index_v1",
    "materialize_massive_adaptive_rl_replay_dependency_index_v1",
    "reconstruct_and_authorize_massive_adaptive_rl_runtime_sources_v1",
    "replay_dependency_index_path_v1",
]
