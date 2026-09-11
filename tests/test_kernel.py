import asyncio
import pytest
from pathlib import Path

from maulness.core.kernel.models import KernelEventType, StepResult
from maulness.core.kernel.step_runner import DurableStepRunner, canonical_json
from maulness.storage.db import StorageManager


@pytest.fixture
async def temp_storage(tmp_path: Path):
    db_path = tmp_path / "test_kernel.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()
    return storage


async def test_canonical_json_determinism():
    d1 = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": True}}
    d2 = {"nested": {"x": True, "y": [1, 2]}, "a": 1, "b": 2}
    assert canonical_json(d1) == canonical_json(d2)


async def test_record_and_query_agent_events(temp_storage: StorageManager):
    event_id = await temp_storage.record_agent_event(
        task_id="task_101",
        stage="planning",
        step_index=0,
        event_type="prompt",
        payload={"query": "Design authentication"},
        idempotency_key="key_001",
    )
    assert event_id > 0

    # Retrieve by idempotency key
    event = await temp_storage.get_agent_event_by_idempotency_key("key_001")
    assert event is not None
    assert event["task_id"] == "task_101"
    assert event["stage"] == "planning"
    assert event["step_index"] == 0
    assert event["payload"] == {"query": "Design authentication"}

    # Non-existent key
    assert await temp_storage.get_agent_event_by_idempotency_key("missing") is None

    # Retrieve by task and stage
    events = await temp_storage.get_agent_events(task_id="task_101", stage="planning")
    assert len(events) == 1
    assert events[0]["id"] == event_id


async def test_step_memoization_skips_execution(temp_storage: StorageManager):
    runner = DurableStepRunner(storage=temp_storage)
    call_count = 0

    def expensive_side_effect(arg_val: str) -> dict:
        nonlocal call_count
        call_count += 1
        return {"status": "computed", "arg": arg_val, "call_count": call_count}

    # Step 0 - First invocation (must execute)
    res1 = await runner.execute_step(
        task_id="task_memo",
        stage="building",
        step_index=0,
        action_type=KernelEventType.TOOL_CALL,
        action_fn=expensive_side_effect,
        arg_val="test_payload",
    )
    assert call_count == 1
    assert res1.from_cache is False
    assert res1.value["call_count"] == 1

    # Step 0 - Second invocation with identical parameters (must be memoized)
    res2 = await runner.execute_step(
        task_id="task_memo",
        stage="building",
        step_index=0,
        action_type=KernelEventType.TOOL_CALL,
        action_fn=expensive_side_effect,
        arg_val="test_payload",
    )
    assert call_count == 1  # side effect was NOT re-invoked
    assert res2.from_cache is True
    assert res2.value["call_count"] == 1
    assert res2.idempotency_key == res1.idempotency_key


async def test_async_step_execution(temp_storage: StorageManager):
    runner = DurableStepRunner(storage=temp_storage)
    executed = False

    async def async_action(x: int, y: int) -> int:
        nonlocal executed
        executed = True
        await asyncio.sleep(0.01)
        return x + y

    res = await runner.execute_step(
        task_id="task_async",
        stage="review",
        step_index=1,
        action_type=KernelEventType.GATE_EVAL,
        action_fn=async_action,
        x=10,
        y=20,
    )
    assert executed is True
    assert res.value == 30
    assert res.from_cache is False

    # Cached recall
    res_cached = await runner.execute_step(
        task_id="task_async",
        stage="review",
        step_index=1,
        action_type=KernelEventType.GATE_EVAL,
        action_fn=async_action,
        x=10,
        y=20,
    )
    assert res_cached.from_cache is True
    assert res_cached.value == 30


async def test_crash_recovery_resumption_flow(temp_storage: StorageManager):
    runner = DurableStepRunner(storage=temp_storage)
    executed_steps = []

    def run_worker_step(step_idx: int) -> str:
        executed_steps.append(step_idx)
        return f"result_of_step_{step_idx}"

    # Phase 1: Simulate steps 0, 1, 2 completing before a crash
    for idx in range(3):
        res = await runner.execute_step(
            task_id="task_recovery",
            stage="building",
            step_index=idx,
            action_type=KernelEventType.TOOL_CALL,
            action_fn=run_worker_step,
            step_idx=idx,
        )
        assert res.from_cache is False

    assert executed_steps == [0, 1, 2]

    # Verify latest step index recorded
    latest = await temp_storage.get_latest_agent_step_index(task_id="task_recovery", stage="building")
    assert latest == 2
    resumption_idx = await runner.get_resumption_step_index(task_id="task_recovery", stage="building")
    assert resumption_idx == 3

    # Phase 2: Simulate process crash and restart.
    # Worker runs the 5-step loop (steps 0..4).
    # Steps 0, 1, 2 should return from cache without executing worker.
    # Steps 3, 4 should execute forward.
    executed_steps.clear()

    results = []
    for idx in range(5):
        res = await runner.execute_step(
            task_id="task_recovery",
            stage="building",
            step_index=idx,
            action_type=KernelEventType.TOOL_CALL,
            action_fn=run_worker_step,
            step_idx=idx,
        )
        results.append(res)

    # Only steps 3 and 4 were physically executed on resumption
    assert executed_steps == [3, 4]
    # Steps 0..2 were loaded from cache
    assert [r.from_cache for r in results] == [True, True, True, False, False]
    # All 5 steps have valid results
    assert [r.value for r in results] == [f"result_of_step_{i}" for i in range(5)]

    # Phase 3: Inspect event timeline
    events = await runner.get_stage_events(task_id="task_recovery", stage="building")
    assert len(events) == 5
    assert [e.step_index for e in events] == [0, 1, 2, 3, 4]
