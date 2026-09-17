"""
Exceptions — Custom exception hierarchy for the Digital Twin system.

All exceptions inherit from PistonEngineError so that callers can
catch the base class when they don't care about the specific type.
"""


class PistonEngineError(Exception):
    """Base exception for all Digital Twin errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ResourceNotFoundException(PistonEngineError):
    """Requested resource was not found."""

    def __init__(self, message: str = "Resource not found", resource_id: str | None = None) -> None:
        super().__init__(message, details={"resource_id": resource_id} if resource_id else {})


class ValidationException(PistonEngineError):
    """Input payload validation error."""


class DigitalTwinError(PistonEngineError):
    """Internal digital twin processing error."""


# ---------------------------------------------------------------------------
# L1 — Data Ingestion
# ---------------------------------------------------------------------------


class InvalidChannelError(PistonEngineError):
    """A telemetry channel value failed validation.

    The system must NOT silently replace the value. Instead it must flag the
    channel as invalid and continue computing unaffected parameters.
    """

    def __init__(self, channel: str, value: float, reason: str) -> None:
        super().__init__(
            f"Channel '{channel}' invalid: value={value}, reason={reason}",
            details={"channel": channel, "value": value, "reason": reason},
        )
        self.channel = channel
        self.value = value
        self.reason = reason


class PacketIntegrityError(PistonEngineError):
    """The incoming data packet failed integrity checks."""


class AdapterConnectionError(PistonEngineError):
    """Failed to connect to a data source adapter."""


# ---------------------------------------------------------------------------
# Architecture Boundary
# ---------------------------------------------------------------------------


class BoundaryViolationError(PistonEngineError):
    """An architectural boundary was violated.

    For example, L2 attempting to access simulator internals from L1.
    This should never occur at runtime — it's a developer error caught by tests.
    """


class RoleViolationError(BoundaryViolationError):
    """An operation attempted to run under an incompatible deployment role.

    For example, attempting to initialize the full ML supervision stack or
    Digital Twin physics on an Edge onboard node.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(PistonEngineError):
    """Invalid or missing configuration."""


# ---------------------------------------------------------------------------
# ML / Models
# ---------------------------------------------------------------------------


class ModelLoadError(PistonEngineError):
    """Failed to load a serialized ML model."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            details={"model_name": model_name, "reason": reason},
        )


class ModelInferenceError(PistonEngineError):
    """ML model inference failed."""


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class ActuationAttemptError(PistonEngineError):
    """CRITICAL: An attempt was made to send an engine control command.

    This system is ADVISORY ONLY. Any code path that would produce an
    actuation command is a severe bug. This exception must never be caught
    and silenced.
    """

    def __init__(self, attempted_action: str) -> None:
        super().__init__(
            f"SAFETY VIOLATION: Attempted actuation command: '{attempted_action}'. "
            "This system is ADVISORY ONLY and must NEVER send engine control commands.",
            details={"attempted_action": attempted_action},
        )
