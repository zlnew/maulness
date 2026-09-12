import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
    TaskMode,
    TaskRecord,
    TaskStatus,
)
from maulness.core.runner import TaskRunner
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_runner_direct_execution_success(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "runner_test.db")
    await storage.initialize()

    runner = TaskRunner(storage=storage)

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock()

    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path
        mock_cfg.resolve_repo_path.return_value = tmp_path

        init_called = False

        async def fake_on_init(conv_id: str):
            nonlocal init_called
            init_called = True

        # Simulate provider invoking callbacks
        async def fake_provider_run(**kwargs):
            if "on_init" in kwargs and kwargs["on_init"]:
                await kwargs["on_init"]("test_conv_123")
            if "on_thought" in kwargs and kwargs["on_thought"]:
                await kwargs["on_thought"](AgentThoughtEvent(delta="Thinking...", session_id="s1"))
            if "on_message" in kwargs and kwargs["on_message"]:
                await kwargs["on_message"](AgentMessageEvent(delta="Hello world", session_id="s1"))
            if "on_approval" in kwargs and kwargs["on_approval"]:
                # Test auto-approved action
                ev1 = ApprovalRequestEvent(
                    request_id=1,
                    call_id="call_1",
                    tool_name="read_file",
                    args={"path": "test.txt"},
                    session_id="s1",
                )
                assert await kwargs["on_approval"](ev1) is True

        mock_provider.run.side_effect = fake_provider_run

        task = await runner.run_direct(
            prompt="Do task",
            repo_name="my_repo",
            workspace_path=tmp_path,
            on_init=fake_on_init,
            verbose=True,
        )

        assert task.status == TaskStatus.DONE
        assert init_called is True
        mock_provider.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_direct_with_worktree_and_yolo(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "runner_test.db")
    await storage.initialize()

    mock_wt = MagicMock()
    mock_wt.isolated_worktree.return_value.__enter__.return_value = tmp_path / "wt"

    runner = TaskRunner(storage=storage, worktree_manager=mock_wt)

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock()

    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path

        task = await runner.run_direct(
            prompt="Refactor engine",
            workspace_path=tmp_path,
            use_worktree=True,
            yolo=True,
            verbose=True,
        )

        assert task.status == TaskStatus.DONE
        mock_wt.isolated_worktree.assert_called_once()
        mock_provider.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_terminal_approval_handlers(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "runner_test.db")
    await storage.initialize()
    runner = TaskRunner(storage=storage)

    mock_provider = MagicMock()

    # We will test terminal interactive prompts for:
    # 1. run_command with cwd (answered Yes)
    # 2. write_file with bytes (answered No)
    # 3. generic tool args
    # 4. custom on_approval callback

    approval_events = [
        ApprovalRequestEvent(
            request_id=1,
            call_id="c1",
            tool_name="run_command",
            args={"command": "rm -rf /", "cwd": "/tmp"},
            session_id="s1",
        ),
        ApprovalRequestEvent(
            request_id=2,
            call_id="c2",
            tool_name="write_file",
            args={"path": "/etc/passwd", "bytes": 50},
            session_id="s1",
        ),
        ApprovalRequestEvent(
            request_id=3,
            call_id="c3",
            tool_name="custom_mutator",
            args={"action": "drop_all"},
            session_id="s1",
        ),
    ]

    captured_decisions = []

    async def fake_provider_run(**kwargs):
        on_approval = kwargs["on_approval"]
        # Trigger default terminal approval
        for ev in approval_events:
            dec = await on_approval(ev)
            captured_decisions.append(dec)

    mock_provider.run = AsyncMock(side_effect=fake_provider_run)

    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("builtins.input", side_effect=["y", "n", "yes"]),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path
        task = await runner.run_direct(prompt="Run hazardous commands", workspace_path=tmp_path)
        assert task.status == TaskStatus.DONE
        assert captured_decisions == [True, False, True]


@pytest.mark.asyncio
async def test_runner_custom_callbacks_and_existing_session(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "runner_test.db")
    await storage.initialize()

    # Pre-seed task and session
    await storage.create_task(
        task_id="t_old",
        title="Old Task",
        repo_name="horizonx",
        workspace_path=str(tmp_path),
        mode=TaskMode.DIRECT,
    )
    await storage.create_session(
        session_id="existing_sess_1",
        task_id="t_old",
        profile="builder",
        engine="gemini",
        acp_session_id="acp_prev",
    )

    runner = TaskRunner(storage=storage)
    mock_provider = MagicMock()

    thought_received = []
    message_received = []
    approval_received = []

    async def custom_thought(ev):
        thought_received.append(ev.delta)

    async def custom_message(ev):
        message_received.append(ev.delta)

    async def custom_approval(ev):
        approval_received.append(ev.tool_name)
        return True

    async def fake_provider_run(**kwargs):
        await kwargs["on_thought"](AgentThoughtEvent(delta="Thinking custom", session_id="s1"))
        await kwargs["on_message"](AgentMessageEvent(delta="Message custom", session_id="s1"))
        ev = ApprovalRequestEvent(
            request_id=1,
            call_id="c1",
            tool_name="run_command",
            args={"command": "dangerous_cmd"},
            session_id="s1",
        )
        await kwargs["on_approval"](ev)

    mock_provider.run = AsyncMock(side_effect=fake_provider_run)

    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path
        task = await runner.run_direct(
            prompt="Run custom callback task",
            workspace_path=tmp_path,
            session_id="existing_sess_1",
            on_thought=custom_thought,
            on_message=custom_message,
            on_approval=custom_approval,
        )

        assert task.status == TaskStatus.DONE
        assert thought_received == ["Thinking custom"]
        assert message_received == ["Message custom"]
        assert approval_received == ["run_command"]


@pytest.mark.asyncio
async def test_runner_error_handling(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "runner_test.db")
    await storage.initialize()
    runner = TaskRunner(storage=storage)

    mock_provider = MagicMock()

    # 1. FileNotFoundError
    mock_provider.run = AsyncMock(side_effect=FileNotFoundError("Target file missing"))
    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path
        with pytest.raises(FileNotFoundError):
            await runner.run_direct(prompt="Fail with file not found", workspace_path=tmp_path)

    # 2. Generic Exception
    mock_provider.run = AsyncMock(side_effect=RuntimeError("Provider exploded"))
    with (
        patch("maulness.core.runner.get_provider_for_profile", return_value=mock_provider),
        patch("maulness.core.runner.config") as mock_cfg,
    ):
        mock_cfg.workspace_root = tmp_path
        with pytest.raises(RuntimeError):
            await runner.run_direct(prompt="Fail with runtime error", workspace_path=tmp_path)
