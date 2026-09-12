import asyncio
import pytest
from pathlib import Path

from maulness.core.kernel.loop import DurableAgentKernel
from maulness.core.kernel.models import KernelEventType, ModelTurnOutput
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


def test_parse_session_scope_variants():
    from maulness.core.kernel.loop import parse_session_scope

    assert parse_session_scope("single") == ("single", "default")
    assert parse_session_scope("custom_session_id") == ("custom", "session_id")
    assert parse_session_scope("task_123_planning") == ("task_123", "planning")


def test_compact_in_flight_tool_messages_invalid_args():
    from maulness.core.kernel.loop import compact_in_flight_tool_messages

    msgs = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "run_command", "arguments": "invalid json"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 250},
        {"role": "tool", "tool_call_id": "c2", "content": "y" * 250},
        {"role": "tool", "tool_call_id": "c3", "content": "z" * 250},
        {"role": "tool", "tool_call_id": "c4", "content": "w" * 250},
    ]

    compact_in_flight_tool_messages(msgs, keep_recent=2)
    # First 2 tool messages should be compacted
    assert "[Tool result for " in msgs[1]["content"]


@pytest.mark.asyncio
async def test_kernel_callbacks_and_tool_events(temp_storage: StorageManager, tmp_path: Path):
    from maulness.core.kernel.loop import DurableAgentKernel

    profile = Profile(identity={"name": "test_agent"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    conv_id = "conv_test_callbacks"
    # Pre-populate conversation history
    await temp_storage.add_conversation_message(conv_id, "user", "Prior question")
    await temp_storage.add_conversation_message(conv_id, "assistant", "Prior answer")

    inits = []
    tool_calls = []

    async def on_init(cid):
        inits.append(cid)

    async def on_tool_call(ev):
        tool_calls.append(ev)

    async def turn_generator(messages, tools, session_id, **kwargs) -> ModelTurnOutput:
        if len(tool_calls) == 0:
            return ModelTurnOutput(
                tool_calls=[{
                    "id": "c1",
                    "name": "run_command",
                    "arguments": "bad json string",
                }]
            )
        return ModelTurnOutput(content="")  # empty content to trigger fallback note

    result = await kernel.run(
        turn_generator_fn=turn_generator,
        session_id="task_cb_test",
        prompt="Test callbacks",
        workspace_path=tmp_path,
        conversation_id=conv_id,
        on_init=on_init,
        on_tool_call=on_tool_call,
    )

    assert len(inits) == 1
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "run_command"
    assert tool_calls[0].args == {"raw": "bad json string"}
    assert "Agent completed tool executions but did not produce a final textual summary" in result


@pytest.mark.asyncio
async def test_kernel_empty_response_raises(temp_storage: StorageManager, tmp_path: Path):
    profile = Profile(identity={"name": "empty_agent"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    async def empty_generator(messages, tools, session_id, **kwargs) -> ModelTurnOutput:
        return ModelTurnOutput(content="[STATUS: COMPLETE]")

    with pytest.raises(RuntimeError, match="returned an empty response"):
        await kernel.run(
            turn_generator_fn=empty_generator,
            session_id="task_empty_test",
            prompt="Nothing",
            workspace_path=tmp_path,
        )



def test_canonical_json_circular_fallback():
    d = {}
    d["self"] = d
    res = canonical_json(d)
    assert "self" in res


@pytest.mark.asyncio
async def test_step_runner_synchronous_returning_awaitable(temp_storage):
    runner = DurableStepRunner(storage=temp_storage)

    def sync_returning_coro():
        async def _inner():
            return "sync_coro_val"
        return _inner()

    res = await runner.execute_step(
        task_id="t_sync",
        stage="s",
        step_index=1,
        action_type=KernelEventType.MODEL_TURN,
        action_fn=sync_returning_coro,
    )
    assert res.value == "sync_coro_val"


@pytest.mark.asyncio
async def test_kernel_replay_from_cache_callbacks(temp_storage, tmp_path):
    profile = Profile(identity={"name": "builder"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    async def gen(messages, tools, session_id, **kwargs):
        return ModelTurnOutput(content="Cached turn content", thought="Cached thought")

    # 1. Run forward execution
    await kernel.run(
        turn_generator_fn=gen,
        session_id="t_cache_cb",
        prompt="hello",
        workspace_path=tmp_path,
    )

    # 2. Run again with on_thought and on_message to exercise cache replay (lines 206, 208)
    replayed_msgs = []
    replayed_thoughts = []
    async def on_msg(ev):
        replayed_msgs.append(ev.delta)
    async def on_th(ev):
        replayed_thoughts.append(ev.delta)

    await kernel.run(
        turn_generator_fn=gen,
        session_id="t_cache_cb",
        prompt="hello",
        workspace_path=tmp_path,
        on_thought=on_th,
        on_message=on_msg,
    )
    assert "Cached turn content" in replayed_msgs
    assert "Cached thought" in replayed_thoughts


@pytest.mark.asyncio
async def test_kernel_loop_intervention_and_breadcrumb_cb(temp_storage, tmp_path):
    profile = Profile(identity={"name": "builder"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    step = 0
    async def loop_gen(messages, tools, session_id, **kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return ModelTurnOutput(
                content="Running command",
                tool_calls=[{"id": "c1", "name": "run_command", "arguments": {"command": "pwd"}}],
            )
        return ModelTurnOutput(content="Loop finished summary")

    msg_deltas = []
    async def on_msg(ev):
        msg_deltas.append(ev.delta)

    # Mock tool execution to return a loop intervention string
    from unittest.mock import patch
    with patch("maulness.core.kernel.loop.execute_tool_call", return_value="[LOOP INTERVENTION: Repetitive tool call detected]"):
        res = await kernel.run(
            turn_generator_fn=loop_gen,
            session_id="t_loop_int",
            prompt="test loop",
            workspace_path=tmp_path,
            on_message=on_msg,
        )
        assert "Loop finished summary" in res
        assert any("LOOP INTERVENTION" in m for m in msg_deltas)


@pytest.mark.asyncio
async def test_kernel_relay_checkpoint_callbacks_and_pause(temp_storage, tmp_path):
    profile = Profile(
        identity={"name": "builder"},
        agent={"provider": "gemini"},
        execution={"max_relays": 2, "max_tool_turns": 1},
    )
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    turn = 0
    async def relay_gen(messages, tools, session_id, **kwargs):
        nonlocal turn
        turn += 1
        if turn in (1, 3):
            return ModelTurnOutput(
                content="Executing tool",
                tool_calls=[{"id": f"c{turn}", "name": "run_command", "arguments": {"command": "pwd"}}],
            )
        return ModelTurnOutput(content="[STATUS: IN_PROGRESS]\nNext: finalize")

    th_deltas = []
    msg_deltas = []
    async def on_th(ev):
        th_deltas.append(ev.delta)
    async def on_msg(ev):
        msg_deltas.append(ev.delta)

    res = await kernel.run(
        turn_generator_fn=relay_gen,
        session_id="t_relay_cb",
        prompt="test relay",
        workspace_path=tmp_path,
        on_thought=on_th,
        on_message=on_msg,
    )
    assert any("Relay Checkpoint" in m for m in msg_deltas)
    assert any("Maximum relay budget" in m for m in msg_deltas)


@pytest.mark.asyncio
async def test_kernel_persist_conversation_failure_handled(temp_storage, tmp_path):
    profile = Profile(identity={"name": "builder"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    async def gen(messages, tools, session_id, **kwargs):
        return ModelTurnOutput(content="Persist test summary")

    from unittest.mock import patch
    with patch.object(temp_storage, "add_conversation_message", side_effect=RuntimeError("disk full")):
        res = await kernel.run(
            turn_generator_fn=gen,
            session_id="t_persist_fail",
            prompt="persist",
            workspace_path=tmp_path,
        )
        assert res == "Persist test summary"

@pytest.mark.asyncio
async def test_kernel_simulated_tool_call(temp_storage, tmp_path):
    profile = Profile(identity={"name": "builder"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    turn = 0
    async def sim_gen(messages, tools, session_id, **kwargs):
        nonlocal turn
        turn += 1
        if turn == 1:
            return ModelTurnOutput(content="> **run_command: pwd**")
        return ModelTurnOutput(content="Finished!")

    res = await kernel.run(
        turn_generator_fn=sim_gen,
        session_id="t_sim_tool",
        prompt="test sim",
        workspace_path=tmp_path,
    )
    assert "Finished!" in res


@pytest.mark.asyncio
async def test_kernel_fallback_note_on_message(temp_storage, tmp_path):
    profile = Profile(identity={"name": "builder"}, agent={"provider": "gemini"})
    kernel = DurableAgentKernel(profile=profile, storage=temp_storage)

    turn = 0
    async def no_text_gen(messages, tools, session_id, **kwargs):
        nonlocal turn
        turn += 1
        if turn == 1:
            return ModelTurnOutput(
                content="",
                tool_calls=[{"id": "c1", "name": "run_command", "arguments": {"command": "pwd"}}],
            )
        return ModelTurnOutput(content="")

    msg_deltas = []
    async def on_msg(ev):
        msg_deltas.append(ev.delta)

    res = await kernel.run(
        turn_generator_fn=no_text_gen,
        session_id="t_no_text",
        prompt="test no text",
        workspace_path=tmp_path,
        on_message=on_msg,
    )
    assert "Agent completed tool executions but did not produce a final textual summary" in res
    assert any("Agent completed tool executions" in m for m in msg_deltas)
