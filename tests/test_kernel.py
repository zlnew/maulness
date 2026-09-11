import asyncio
import pytest
from pathlib import Path

from maulness.core.kernel.loop import DurableAgentKernel
from maulness.core.kernel.models import KernelEventType, ModelTurnOutput, StepResult
from maulness.core.kernel.step_runner import DurableStepRunner, canonical_json
from maulness.core.profiles import Profile
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


async def test_durable_agent_kernel_turn_memoization(temp_storage: StorageManager, tmp_path: Path):
    profile = Profile(
        identity={"name": "kernel_memo_tester"},
        agent={"provider": "ollama", "model": "llama3"},
        execution={"yolo": True},
    )
    kernel = DurableAgentKernel(profile, storage=temp_storage)

    generator_call_count = 0

    async def mock_turn_generator(messages, tools, session_id, **kwargs) -> ModelTurnOutput:
        nonlocal generator_call_count
        generator_call_count += 1
        if generator_call_count == 1:
            return ModelTurnOutput(
                tool_calls=[{
                    "id": "call_echo_1",
                    "name": "run_command",
                    "arguments": {"command": "echo memo_test_output"},
                }]
            )
        return ModelTurnOutput(content="Task completed successfully.")

    # First run: invokes generator twice (turn 1 tool call, turn 2 synthesis)
    res1 = await kernel.run(
        turn_generator_fn=mock_turn_generator,
        session_id="task_k1_building",
        prompt="Execute echo",
        workspace_path=tmp_path,
    )
    assert generator_call_count == 2
    assert "echo memo_test_output" in res1
    assert "Task completed successfully." in res1

    # Second run with same session_id and prompt: all steps memoized from SQLite
    res2 = await kernel.run(
        turn_generator_fn=mock_turn_generator,
        session_id="task_k1_building",
        prompt="Execute echo",
        workspace_path=tmp_path,
    )
    # Generator was NOT called again
    assert generator_call_count == 2
    assert res2 == res1

    # Check recorded events
    events = await temp_storage.get_agent_events(task_id="task_k1", stage="building")
    assert len(events) >= 4
    event_types = [e["event_type"] for e in events]
    assert "prompt" in event_types
    assert "model_turn" in event_types
    assert "tool_result" in event_types
    assert "final_response" in event_types


async def test_durable_agent_kernel_crash_resumption_mid_session(temp_storage: StorageManager, tmp_path: Path):
    profile = Profile(
        identity={"name": "crash_tester"},
        agent={"provider": "ollama", "model": "llama3"},
        execution={"yolo": True},
    )
    kernel = DurableAgentKernel(profile, storage=temp_storage)

    generator_invocations = []

    async def crashing_turn_generator(messages, tools, session_id, **kwargs) -> ModelTurnOutput:
        generator_invocations.append(len(generator_invocations) + 1)
        if len(generator_invocations) == 1:
            return ModelTurnOutput(
                tool_calls=[{
                    "id": "call_crash_1",
                    "name": "run_command",
                    "arguments": {"command": "echo survived_crash"},
                }]
            )
        elif len(generator_invocations) == 2:
            raise RuntimeError("Simulated process crash mid-session")
        return ModelTurnOutput(content="Resumed after crash and finished.")

    # Phase 1: Run fails on turn 2
    with pytest.raises(RuntimeError, match="Simulated process crash mid-session"):
        await kernel.run(
            turn_generator_fn=crashing_turn_generator,
            session_id="task_crash_building",
            prompt="Run with crash",
            workspace_path=tmp_path,
        )

    assert generator_invocations == [1, 2]

    # Verify that Turn 1 and its tool result were committed
    events = await temp_storage.get_agent_events(task_id="task_crash", stage="building")
    recorded_types = [e["event_type"] for e in events]
    assert "prompt" in recorded_types
    assert "model_turn" in recorded_types
    assert "tool_result" in recorded_types

    # Phase 2: Restart agent on same session with recovering generator
    generator_invocations.clear()

    async def recovered_turn_generator(messages, tools, session_id, **kwargs) -> ModelTurnOutput:
        generator_invocations.append("called")
        return ModelTurnOutput(content="Resumed after crash and finished.")

    final_result = await kernel.run(
        turn_generator_fn=recovered_turn_generator,
        session_id="task_crash_building",
        prompt="Run with crash",
        workspace_path=tmp_path,
    )

    # Only 1 generator call occurred (turn 2 synthesis), turn 1 was replayed from SQLite cache
    assert len(generator_invocations) == 1
    assert "echo survived_crash" in final_result
    assert "Resumed after crash and finished." in final_result

