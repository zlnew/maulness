import logging
import time
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger("maulness.kernel.budgets")


class BudgetExceededError(Exception):
    """Raised when an operational ceiling (steps, duration, cost) is breached."""
    def __init__(self, reason: str, metric: str, current: float, limit: float):
        super().__init__(f"Budget exceeded: {reason} (Current {metric}: {current}, Limit: {limit})")
        self.reason = reason
        self.metric = metric
        self.current = current
        self.limit = limit


class BudgetLimits(BaseModel):
    """Configurable ceilings for autonomous task execution."""
    max_turns: int = 30
    max_duration_seconds: float = 1500.0  # 25 minutes
    max_cost_usd: float = 2.00            # $2.00 USD


class BudgetGuard:
    """Monitors and enforces operational budgets for autonomous/AFK task execution."""

    def __init__(self, limits: Optional[BudgetLimits] = None):
        self.limits = limits or BudgetLimits()
        self.start_time: float = time.monotonic()
        self.total_turns: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.accumulated_cost_usd: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def record_turn(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record an executed turn and accumulate metrics."""
        self.total_turns += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.accumulated_cost_usd += cost_usd

        self.check_limits()

    def check_limits(self) -> None:
        """Verify that current consumption does not exceed configured limits."""
        # 1. Turn ceiling
        if self.limits.max_turns > 0 and self.total_turns >= self.limits.max_turns:
            msg = f"Turn limit reached ({self.total_turns}/{self.limits.max_turns} turns)"
            logger.warning("[budget] %s", msg)
            raise BudgetExceededError(
                reason=msg,
                metric="turns",
                current=float(self.total_turns),
                limit=float(self.limits.max_turns),
            )

        # 2. Wall-clock duration ceiling
        elapsed = self.elapsed_seconds
        if self.limits.max_duration_seconds > 0 and elapsed >= self.limits.max_duration_seconds:
            msg = f"Wall-clock timeout reached ({elapsed:.1f}s / {self.limits.max_duration_seconds:.1f}s)"
            logger.warning("[budget] %s", msg)
            raise BudgetExceededError(
                reason=msg,
                metric="seconds",
                current=elapsed,
                limit=self.limits.max_duration_seconds,
            )

        # 3. Cost ceiling
        if self.limits.max_cost_usd > 0 and self.accumulated_cost_usd >= self.limits.max_cost_usd:
            msg = f"Cost budget reached (${self.accumulated_cost_usd:.4f} / ${self.limits.max_cost_usd:.2f})"
            logger.warning("[budget] %s", msg)
            raise BudgetExceededError(
                reason=msg,
                metric="cost_usd",
                current=self.accumulated_cost_usd,
                limit=self.limits.max_cost_usd,
            )

    def summary(self) -> dict[str, float | int]:
        return {
            "turns": self.total_turns,
            "max_turns": self.limits.max_turns,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "max_duration_seconds": self.limits.max_duration_seconds,
            "cost_usd": round(self.accumulated_cost_usd, 4),
            "max_cost_usd": self.limits.max_cost_usd,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
