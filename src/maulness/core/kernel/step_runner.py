import hashlib
import inspect
import json
import logging
from typing import Any, Callable, Optional, Union

from maulness.core.kernel.models import AgentEventRecord, KernelEventType, StepResult
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.kernel.step_runner")


def canonical_json(data: Any) -> str:
    """Produce deterministic, whitespace-normalized JSON string for hashing."""
    try:
        if isinstance(data, (dict, list)):
            return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return str(data).strip()
    except Exception:
        return str(data).strip()


class DurableStepRunner:
    """Durable execution engine ensuring step-level memoization and crash resumption.

    Guarantees that steps executed under (task_id, stage, step_index, action_type)
    are recorded as immutable events in SQLite and replayed instantly on crash recovery.
    """

    def __init__(self, storage: Optional[StorageManager] = None):
        self.storage = storage or StorageManager()

    def compute_idempotency_key(
        self,
        task_id: str,
        stage: str,
        step_index: int,
        action_type: Union[KernelEventType, str],
        action_payload: Any,
    ) -> str:
        """Generate a deterministic SHA-256 idempotency key for a step action."""
        type_str = (
            action_type.value
            if isinstance(action_type, KernelEventType)
            else str(action_type)
        )
        payload_repr = canonical_json(action_payload)
        raw_key = f"{task_id}:{stage}:{step_index}:{type_str}:{payload_repr}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def execute_step(
        self,
        task_id: str,
        stage: str,
        step_index: int,
        action_type: KernelEventType,
        action_fn: Callable[..., Any],
        *args: Any,
        action_meta: Optional[Any] = None,
        **kwargs: Any,
    ) -> StepResult:
        """Execute a step with memoization.

        If a step with identical idempotency key already completed in the event store,
        returns the memoized result immediately without re-invoking action_fn.
        Otherwise, executes action_fn, records the event immutably, and returns the result.
        """
        meta_to_hash = action_meta if action_meta is not None else (args, kwargs)
        idempotency_key = self.compute_idempotency_key(
            task_id=task_id,
            stage=stage,
            step_index=step_index,
            action_type=action_type,
            action_payload=meta_to_hash,
        )

        # 1. Check for existing memoized event
        existing = await self.storage.get_agent_event_by_idempotency_key(
            idempotency_key
        )
        if existing:
            logger.info(
                "[kernel] Memoized step hit: task=%s stage=%s step=%s type=%s (id=%s)",
                task_id,
                stage,
                step_index,
                action_type.value,
                existing["id"],
            )
            return StepResult(
                value=existing["payload"],
                from_cache=True,
                event_id=existing["id"],
                idempotency_key=idempotency_key,
                step_index=step_index,
                event_type=action_type.value,
            )

        # 2. Execute the forward action
        logger.debug(
            "[kernel] Executing forward step: task=%s stage=%s step=%s type=%s",
            task_id,
            stage,
            step_index,
            action_type.value,
        )
        if inspect.iscoroutinefunction(action_fn):
            result = await action_fn(*args, **kwargs)
        else:
            res_or_coro = action_fn(*args, **kwargs)
            if inspect.isawaitable(res_or_coro):
                result = await res_or_coro
            else:
                result = res_or_coro

        # 3. Commit immutable event to journal
        event_id = await self.storage.record_agent_event(
            task_id=task_id,
            stage=stage,
            step_index=step_index,
            event_type=action_type.value,
            payload=result,
            idempotency_key=idempotency_key,
        )

        return StepResult(
            value=result,
            from_cache=False,
            event_id=event_id,
            idempotency_key=idempotency_key,
            step_index=step_index,
            event_type=action_type.value,
        )

    async def get_resumption_step_index(self, task_id: str, stage: str) -> int:
        """Get the next step index to execute upon crash resumption."""
        latest = await self.storage.get_latest_agent_step_index(
            task_id=task_id, stage=stage
        )
        return latest + 1 if latest > 0 else 0

    async def get_stage_events(
        self, task_id: str, stage: Optional[str] = None
    ) -> list[AgentEventRecord]:
        """Retrieve chronological event records for a task and stage."""
        raw_events = await self.storage.get_agent_events(task_id=task_id, stage=stage)
        return [
            AgentEventRecord(
                id=e["id"],
                task_id=e["task_id"],
                stage=e["stage"],
                step_index=e["step_index"],
                event_type=e["event_type"],
                payload=e["payload"],
                idempotency_key=e["idempotency_key"],
                created_at=e["created_at"],
            )
            for e in raw_events
        ]
