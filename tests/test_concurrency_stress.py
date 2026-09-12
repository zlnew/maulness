import asyncio
import random
import time
from pathlib import Path
import pytest

from maulness.core.debouncer import MessageStreamDebouncer
from maulness.core.models import TaskMode, TaskStatus
from maulness.core.providers.circuit import (
    CircuitState,
    ProviderHealthRegistry,
)
from maulness.core.tools import ActionLoopDetector
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_sqlite_wal_high_concurrency_stress(tmp_path: Path):
    """Stress test SQLite WAL mode with 25 simultaneous concurrent workers.
    Each worker creates a task, updates status, saves sessions/messages, and records events.
    Verifies that WAL + busy_timeout handles concurrent writes without locking errors.
    """
    db_file = tmp_path / "stress.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    worker_count = 25
    events_per_worker = 5
    messages_per_worker = 4

    async def worker_routine(worker_id: int):
        # 1. Create a task
        task_id = f"task_stress_{worker_id:03d}"
        repo_name = f"repo_{worker_id % 5}"
        await storage.create_task(
            task_id=task_id,
            title=f"Concurrent Task {worker_id}",
            repo_name=repo_name,
            workspace_path=str(tmp_path / f"ws_{worker_id}"),
            mode=TaskMode.DIRECT,
        )

        # 2. Transition through statuses with minor jitter
        for next_status in [TaskStatus.BUILDING, TaskStatus.REVIEW, TaskStatus.DONE]:
            await asyncio.sleep(random.uniform(0.001, 0.01))
            await storage.update_task_status(task_id, next_status)

        # 3. Create an agent session
        session_id = f"session_stress_{worker_id:03d}"
        await storage.create_session(
            session_id=session_id,
            task_id=task_id,
            profile="builder",
            engine="direct",
        )

        # 4. Save multiple messages concurrently
        for msg_idx in range(messages_per_worker):
            role = "user" if msg_idx % 2 == 0 else "assistant"
            content = f"Worker {worker_id} message {msg_idx}"
            await storage.add_conversation_message(session_id, role, content)

        # 5. Record agent events with idempotency keys
        for ev_idx in range(events_per_worker):
            idem_key = f"idem_{worker_id}_{ev_idx}"
            await storage.record_agent_event(
                task_id=task_id,
                stage="building",
                step_index=ev_idx,
                event_type="tool_execution",
                payload={"worker": worker_id, "step": ev_idx, "ts": time.time()},
                idempotency_key=idem_key,
            )

        # 6. Read back events and verify
        events = await storage.get_agent_events(task_id)
        assert len(events) == events_per_worker
        return task_id

    # Launch all 25 workers concurrently
    tasks = [worker_routine(i) for i in range(worker_count)]
    completed_ids = await asyncio.gather(*tasks)

    assert len(completed_ids) == worker_count

    # Verify overall DB state integrity
    all_tasks = await storage.list_tasks(limit=100)
    assert len(all_tasks) == worker_count
    for t in all_tasks:
        assert t.status == TaskStatus.DONE

    # Verify all messages exist
    total_messages = 0
    for w_id in range(worker_count):
        session_id = f"session_stress_{w_id:03d}"
        history = await storage.get_conversation_messages(session_id)
        assert len(history) == messages_per_worker
        total_messages += len(history)

    assert total_messages == worker_count * messages_per_worker


@pytest.mark.asyncio
async def test_sqlite_concurrent_idempotency_race(tmp_path: Path):
    """Test race conditions when multiple coroutines try to record the exact same idempotency key."""
    db_file = tmp_path / "idem_race.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    # Pre-create parent task
    await storage.create_task(
        task_id="task_idem_race",
        title="Idempotency Race",
        repo_name="race_repo",
        workspace_path=str(tmp_path),
        mode=TaskMode.DIRECT,
    )

    same_key = "idempotent_step_42"
    competitors = 20

    async def competitor(idx: int):
        await asyncio.sleep(random.uniform(0.001, 0.005))
        return await storage.record_agent_event(
            task_id="task_idem_race",
            stage="building",
            step_index=42,
            event_type="test_idem",
            payload={"competitor": idx},
            idempotency_key=same_key,
        )

    # Run competitors concurrently
    results = await asyncio.gather(*[competitor(i) for i in range(competitors)])
    assert len(results) == competitors

    # Verify that only 1 event exists with this idempotency key
    ev = await storage.get_agent_event_by_idempotency_key(same_key)
    assert ev is not None
    assert ev["idempotency_key"] == same_key

    events = await storage.get_agent_events("task_idem_race")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_circuit_breaker_high_concurrency_storm():
    """Test circuit breaker under high concurrency: 50 concurrent requests recording failures."""
    registry = ProviderHealthRegistry()
    registry.reset_instance()

    provider_key = "stress:gemini-pro"
    error = Exception("429 Resource Exhausted")
    now = 1000.0

    async def fail_task():
        await asyncio.sleep(random.uniform(0.0001, 0.002))
        return registry.record_failure(provider_key, error, now=now)

    # 50 concurrent failures
    results = await asyncio.gather(*[fail_task() for _ in range(50)])

    assert len(results) == 50
    # Circuit should be tripped to OPEN
    health = registry.get_or_create(provider_key)
    assert health.state == CircuitState.OPEN
    assert health.consecutive_failures == 50
    assert registry.is_available(provider_key, now=now + 10.0) is False

    # After cooldown expires, concurrent checks probe to HALF_OPEN
    future_time = health.cooldown_until + 1.0
    assert registry.is_available(provider_key, now=future_time) is True
    assert health.state == CircuitState.HALF_OPEN

    # Concurrent successful requests reset circuit to CLOSED
    async def success_task():
        await asyncio.sleep(random.uniform(0.0001, 0.002))
        registry.record_success(provider_key, now=future_time + 5.0)

    await asyncio.gather(*[success_task() for _ in range(20)])
    assert health.state == CircuitState.CLOSED
    assert health.consecutive_failures == 0
    assert registry.is_available(provider_key) is True


@pytest.mark.asyncio
async def test_debouncer_concurrent_burst_storm():
    """Stress test MessageStreamDebouncer with 25 concurrent writers sending burst deltas."""
    flushed = []

    async def on_flush(text: str, is_final: bool = False, is_overflow: bool = False):
        flushed.append((text, is_final, is_overflow))

    debouncer = MessageStreamDebouncer(flush_callback=on_flush, interval_seconds=0.05)

    writers = 25
    chunks_per_writer = 6

    async def writer(w_id: int):
        for c_idx in range(chunks_per_writer):
            await asyncio.sleep(random.uniform(0.001, 0.005))
            await debouncer.write(f"[W{w_id}C{c_idx}]")

    # Run all 25 writers simultaneously
    await asyncio.gather(*[writer(i) for i in range(writers)])

    # Close debouncer to guarantee final flush
    await debouncer.close()

    # Verify that all 150 chunks are contained in full_text
    full_text = debouncer.full_text
    for w in range(writers):
        for c in range(chunks_per_writer):
            assert f"[W{w}C{c}]" in full_text

    assert len(flushed) >= 1
    assert flushed[-1][1] is True  # final flush was True


@pytest.mark.asyncio
async def test_action_loop_detector_concurrent_sessions():
    """Verify ActionLoopDetector thread/task safety across 30 concurrent agent sessions."""
    detector = ActionLoopDetector(window_size=6, repetition_threshold=3)
    sessions_count = 30

    async def session_loop(s_id: int):
        session_key = f"sess_{s_id}"
        # 1. Non-repeating calls
        for i in range(4):
            loop, _ = detector.record_and_check(session_key, "read_file", {"path": f"file_{i}.txt"})
            assert not loop

        # 2. Repeated calls triggering intervention
        for _ in range(2):
            loop, _ = detector.record_and_check(session_key, "run_command", {"command": "npm test"})

        loop, msg = detector.record_and_check(session_key, "run_command", {"command": "npm test"})
        assert loop
        assert "[LOOP INTERVENTION]" in msg
        detector.clear(session_key)

    await asyncio.gather(*[session_loop(i) for i in range(sessions_count)])
    assert len(detector._history) == 0


@pytest.mark.asyncio
async def test_worktree_concurrent_isolated_environments(tmp_path: Path):
    """Verify that multiple concurrent operations can create isolated git worktrees concurrently without collision."""
    import subprocess
    from maulness.core.worktree import WorktreeManager

    repo_dir = tmp_path / "concurrent_base_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    (repo_dir / "init.txt").write_text("initial commit")
    subprocess.run(["git", "add", "init.txt"], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    wm = WorktreeManager()
    workers = 6

    def run_worker(w_id: int):
        with wm.isolated_worktree(repo_dir, branch_prefix=f"worker-{w_id}") as wt_path:
            assert wt_path.exists()
            assert (wt_path / "init.txt").exists()
            # Write isolated work
            (wt_path / f"worker_{w_id}.txt").write_text(f"data {w_id}")
            time.sleep(0.05)
        # Ensure cleaned up
        assert not wt_path.exists()

    loop = asyncio.get_running_loop()
    await asyncio.gather(*[loop.run_in_executor(None, run_worker, i) for i in range(workers)])


@pytest.mark.asyncio
async def test_pipeline_concurrent_runs(tmp_path: Path):
    """Stress test PipelineOrchestrator running 5 pipelines simultaneously across distinct tasks."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from maulness.core.pipeline import PipelineOrchestrator
    from maulness.core.pipelines import PipelineDefinition, PipelineStage

    storage = StorageManager(db_path=tmp_path / "pipe_stress.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="stress_pipe",
        stages=[
            PipelineStage(name="plan", profile="planner", prompt="Plan {title}"),
            PipelineStage(name="build", profile="builder", prompt="Build {title}"),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Stage done")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)

        async def run_single_pipeline(idx: int):
            ws = tmp_path / f"pipe_ws_{idx}"
            ws.mkdir()
            task = await orchestrator.run_pipeline(
                repo_name=f"repo_{idx}",
                title=f"Concurrent Feature {idx}",
                prompt="Implement feature",
                workspace_path=ws,
                pipeline_def=definition,
                auto_proceed=True,
            )
            assert task.status == TaskStatus.DONE
            return task.id

        results = await asyncio.gather(*[run_single_pipeline(i) for i in range(5)])
        assert len(results) == 5

        tasks = await storage.list_tasks(limit=10)
        assert len(tasks) == 5
        for t in tasks:
            assert t.status == TaskStatus.DONE

