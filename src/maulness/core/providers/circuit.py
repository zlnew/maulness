import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("maulness.providers.circuit")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Healthy: handles requests normally
    OPEN = "OPEN"            # Tripped: requests fast-bypass to fallback
    HALF_OPEN = "HALF_OPEN"  # Probation: testing single probe request after cooldown


class ErrorClassification(str, Enum):
    RATE_LIMIT = "rate_limit"        # 429, RESOURCE_EXHAUSTED
    TIMEOUT = "timeout"              # TimeoutError, stream idle
    SERVER_ERROR = "server_error"    # 500, 502, 503, UNAVAILABLE
    AUTH_ERROR = "auth_error"        # 401, 403, invalid key
    PAYLOAD_ERROR = "payload_error"  # 400, context length, bad prompt (DO NOT TRIP)
    UNKNOWN = "unknown"


def classify_provider_error(error: Exception) -> tuple[ErrorClassification, str]:
    """Classify an exception and extract human-friendly error string."""
    err_str = str(error)

    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
        return ErrorClassification.RATE_LIMIT, "Rate limit / quota exceeded (429)"

    if isinstance(error, TimeoutError) or "timeout" in err_str.lower():
        return ErrorClassification.TIMEOUT, "Request timed out waiting for model stream"

    if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
        return ErrorClassification.SERVER_ERROR, "Service temporarily unavailable (503)"

    if "502" in err_str or "504" in err_str or "BAD_GATEWAY" in err_str:
        return ErrorClassification.SERVER_ERROR, "Gateway / upstream model error (502/504)"

    if "500" in err_str or "INTERNAL" in err_str:
        return ErrorClassification.SERVER_ERROR, "Internal server error (500)"

    if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
        return ErrorClassification.AUTH_ERROR, "Invalid or unauthorized API key"

    if "401" in err_str or "CreditsError" in err_str:
        return ErrorClassification.AUTH_ERROR, "Authentication failure or insufficient credits (401)"

    if "403" in err_str or "PERMISSION_DENIED" in err_str:
        return ErrorClassification.AUTH_ERROR, "Permission denied / access restricted (403)"

    if "400" in err_str or "INVALID_ARGUMENT" in err_str:
        return ErrorClassification.PAYLOAD_ERROR, "Bad request / payload argument error (400)"

    # Clean generic string
    first_line = err_str.strip().split("\n")[0]
    return ErrorClassification.UNKNOWN, first_line[:100] if len(first_line) > 100 else (first_line or "Unknown error")


@dataclass
class ProviderHealth:
    provider_key: str        # e.g. "gemini:gemini-3.8-flash"
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error_type: Optional[ErrorClassification] = None
    last_error_msg: Optional[str] = None
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    def is_available(self, now: Optional[float] = None) -> bool:
        """Return True if this provider is eligible to receive a request."""
        t = now if now is not None else time.time()
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if t >= self.cooldown_until:
                # Cooldown expired: transition to probation probe
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    "[circuit] Cooldown expired for %s. Transitioning to HALF_OPEN probation.",
                    self.provider_key,
                )
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self, now: Optional[float] = None) -> None:
        """Mark provider healthy and reset circuit breaker to CLOSED."""
        t = now if now is not None else time.time()
        was_degraded = self.state != CircuitState.CLOSED
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.cooldown_until = 0.0
        self.last_success_time = t
        if was_degraded:
            logger.info(
                "[circuit] Provider %s successfully recovered! Circuit reset to CLOSED.",
                self.provider_key,
            )

    def record_failure(self, error: Exception, now: Optional[float] = None) -> tuple[bool, str, int]:
        """Record a failure, update circuit breaker, and return (tripped, friendly_msg, cooldown_seconds)."""
        t = now if now is not None else time.time()
        err_type, friendly_msg = classify_provider_error(error)
        self.last_error_type = err_type
        self.last_error_msg = friendly_msg
        self.last_failure_time = t

        # Do NOT trip circuit on request-specific payload errors (e.g. 400 Bad Request)
        if err_type == ErrorClassification.PAYLOAD_ERROR:
            logger.debug(
                "[circuit] Not tripping circuit for %s on payload error: %s",
                self.provider_key,
                friendly_msg,
            )
            return False, friendly_msg, 0

        self.consecutive_failures += 1
        fails = self.consecutive_failures

        # Determine cooldown with exponential backoff
        if err_type == ErrorClassification.RATE_LIMIT:
            # 60s base, exponential backoff up to 8 min
            base = 60.0
            max_cd = 480.0
            cooldown = min(base * (2 ** (fails - 1)), max_cd)
        elif err_type == ErrorClassification.TIMEOUT:
            # 45s base, exponential backoff up to 4 min
            base = 45.0
            max_cd = 240.0
            cooldown = min(base * (2 ** (fails - 1)), max_cd)
        elif err_type == ErrorClassification.SERVER_ERROR:
            # 30s base, exponential backoff up to 3 min
            base = 30.0
            max_cd = 180.0
            cooldown = min(base * (2 ** (fails - 1)), max_cd)
        elif err_type == ErrorClassification.AUTH_ERROR:
            # 1 hour
            cooldown = 3600.0
        else:
            base = 30.0
            max_cd = 120.0
            cooldown = min(base * (2 ** (fails - 1)), max_cd)

        self.state = CircuitState.OPEN
        self.cooldown_until = t + cooldown

        logger.warning(
            "[circuit] Tripped circuit to OPEN for %s: %s (consecutive failures: %d, cooldown: %ds)",
            self.provider_key,
            friendly_msg,
            fails,
            int(cooldown),
        )
        return True, friendly_msg, int(cooldown)


class ProviderHealthRegistry:
    """In-memory registry tracking circuit breaker health across all providers."""

    _instance: Optional["ProviderHealthRegistry"] = None

    def __init__(self):
        self._registry: dict[str, ProviderHealth] = {}

    @classmethod
    def get_instance(cls) -> "ProviderHealthRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for unit tests)."""
        cls._instance = None

    def get_or_create(self, provider_key: str) -> ProviderHealth:
        if provider_key not in self._registry:
            self._registry[provider_key] = ProviderHealth(provider_key=provider_key)
        return self._registry[provider_key]

    def is_available(self, provider_key: str, now: Optional[float] = None) -> bool:
        return self.get_or_create(provider_key).is_available(now)

    def record_success(self, provider_key: str, now: Optional[float] = None) -> None:
        self.get_or_create(provider_key).record_success(now)

    def record_failure(self, provider_key: str, error: Exception, now: Optional[float] = None) -> tuple[bool, str, int]:
        return self.get_or_create(provider_key).record_failure(error, now)

    def get_status_summary(self) -> list[dict[str, Any]]:
        now = time.time()
        summary = []
        for key, health in sorted(self._registry.items()):
            remaining = max(0, int(health.cooldown_until - now)) if health.state == CircuitState.OPEN else 0
            summary.append({
                "provider_key": key,
                "state": health.state.value,
                "consecutive_failures": health.consecutive_failures,
                "cooldown_remaining_seconds": remaining,
                "last_error": health.last_error_msg,
                "last_success_time": health.last_success_time,
            })
        return summary

    def format_status_text(self) -> str:
        summary = self.get_status_summary()
        if not summary:
            return "No provider health data recorded yet (all healthy)."

        lines = ["**Provider Circuit Health:**"]
        for s in summary:
            st = s["state"]
            if st == CircuitState.CLOSED.value:
                lines.append(f"• `{s['provider_key']}`: **HEALTHY** (CLOSED)")
            elif st == CircuitState.HALF_OPEN.value:
                lines.append(f"• `{s['provider_key']}`: **PROBATION** (HALF_OPEN - next request probes recovery)")
            else:
                lines.append(
                    f"• `{s['provider_key']}`: **COOLDOWN** (OPEN - {s['cooldown_remaining_seconds']}s remaining) — `{s['last_error']}`"
                )
        return "\n".join(lines)

    def reset_all(self) -> None:
        for health in self._registry.values():
            health.record_success()
        logger.info("[circuit] All provider circuits manually reset to CLOSED.")
