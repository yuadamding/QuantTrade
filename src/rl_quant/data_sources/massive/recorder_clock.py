"""Clock-error authorities for delayed Massive WebSocket recorders."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rl_quant.protocol.canonical_artifact import semantic_sha256


MASSIVE_RECORDER_CLOCK_AUTHORITY_SCHEMA = (
    "rl-quant.massive-recorder-clock-authority-v1"
)


class MassiveRecorderClockError(ValueError):
    """Recorder wall-clock evidence is absent, inconsistent, or too weak."""


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MassiveRecorderClockError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class MassiveRecorderClockAuthority:
    host_id: str
    clock_source: str
    synchronization_protocol: str
    measured_before_capture_offset_ns: int
    measured_after_capture_offset_ns: int
    maximum_absolute_offset_ns: int
    maximum_drift_ns: int
    measurement_source_receipts: tuple[str, ...]
    receipt_sha256: str
    schema: str = MASSIVE_RECORDER_CLOCK_AUTHORITY_SCHEMA

    def unsigned(self) -> dict[str, object]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "receipt_sha256"
        }

    @property
    def maximum_positive_clock_error_ns(self) -> int:
        """Conservative amount added to a local receive timestamp."""

        self.validate()
        return self.maximum_absolute_offset_ns + self.maximum_drift_ns

    def conservative_upper_timestamp_ns(self, local_timestamp_ns: int) -> int:
        if (
            isinstance(local_timestamp_ns, bool)
            or not isinstance(local_timestamp_ns, int)
            or local_timestamp_ns < 0
        ):
            raise MassiveRecorderClockError(
                "local receive timestamp must be nonnegative"
            )
        return local_timestamp_ns + self.maximum_positive_clock_error_ns

    def validate(self) -> None:
        if self.schema != MASSIVE_RECORDER_CLOCK_AUTHORITY_SCHEMA:
            raise MassiveRecorderClockError("recorder clock schema drifted")
        for name in ("host_id", "clock_source", "synchronization_protocol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise MassiveRecorderClockError(f"{name} must be canonical text")
        for name in (
            "measured_before_capture_offset_ns",
            "measured_after_capture_offset_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise MassiveRecorderClockError(f"{name} must be an integer")
        for name in ("maximum_absolute_offset_ns", "maximum_drift_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MassiveRecorderClockError(f"{name} must be nonnegative")
        observed_absolute = max(
            abs(self.measured_before_capture_offset_ns),
            abs(self.measured_after_capture_offset_ns),
        )
        if self.maximum_absolute_offset_ns < observed_absolute:
            raise MassiveRecorderClockError(
                "maximum clock offset understates a measurement"
            )
        observed_drift = abs(
            self.measured_after_capture_offset_ns
            - self.measured_before_capture_offset_ns
        )
        if self.maximum_drift_ns < observed_drift:
            raise MassiveRecorderClockError("maximum clock drift is understated")
        if (
            not self.measurement_source_receipts
            or self.measurement_source_receipts
            != tuple(sorted(set(self.measurement_source_receipts)))
        ):
            raise MassiveRecorderClockError(
                "clock measurement receipts are not canonical"
            )
        for receipt in self.measurement_source_receipts:
            _digest("clock measurement receipt", receipt)
        _digest("recorder clock receipt", self.receipt_sha256)
        if self.receipt_sha256 != semantic_sha256(self.unsigned()):
            raise MassiveRecorderClockError("recorder clock receipt differs")

    @classmethod
    def build(
        cls,
        *,
        host_id: str,
        clock_source: str,
        synchronization_protocol: str,
        measured_before_capture_offset_ns: int,
        measured_after_capture_offset_ns: int,
        maximum_absolute_offset_ns: int,
        maximum_drift_ns: int,
        measurement_source_receipts: tuple[str, ...],
    ) -> MassiveRecorderClockAuthority:
        receipts = tuple(sorted(set(measurement_source_receipts)))
        body = {
            "schema": MASSIVE_RECORDER_CLOCK_AUTHORITY_SCHEMA,
            "host_id": host_id,
            "clock_source": clock_source,
            "synchronization_protocol": synchronization_protocol,
            "measured_before_capture_offset_ns": measured_before_capture_offset_ns,
            "measured_after_capture_offset_ns": measured_after_capture_offset_ns,
            "maximum_absolute_offset_ns": maximum_absolute_offset_ns,
            "maximum_drift_ns": maximum_drift_ns,
            "measurement_source_receipts": receipts,
        }
        value = cls(
            host_id=host_id,
            clock_source=clock_source,
            synchronization_protocol=synchronization_protocol,
            measured_before_capture_offset_ns=measured_before_capture_offset_ns,
            measured_after_capture_offset_ns=measured_after_capture_offset_ns,
            maximum_absolute_offset_ns=maximum_absolute_offset_ns,
            maximum_drift_ns=maximum_drift_ns,
            measurement_source_receipts=receipts,
            receipt_sha256=semantic_sha256(body),
        )
        value.validate()
        return value


__all__ = [
    "MASSIVE_RECORDER_CLOCK_AUTHORITY_SCHEMA",
    "MassiveRecorderClockAuthority",
    "MassiveRecorderClockError",
]
