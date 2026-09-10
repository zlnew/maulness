import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from maulness.discord.bot import MaulnessBot
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_bot_slash_commands_registered(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    registered_cmds = {cmd.name for cmd in bot.tree.get_commands()}
    assert "status" in registered_cmds
    assert "profiles" in registered_cmds
    assert "ask" in registered_cmds
    assert "run" in registered_cmds
    assert "task" in registered_cmds
    assert "abort" in registered_cmds


@pytest.mark.asyncio
async def test_bot_abort_handler(tmp_path: Path):
    from maulness.core.models import TaskMode, TaskStatus

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    # Seed an active task
    await storage.create_task(
        task_id="task_to_abort",
        title="Test Abort",
        repo_name="horizonx",
        workspace_path="/tmp",
        mode=TaskMode.DIRECT,
    )

    # Mock an asyncio task in active_tasks
    mock_async_task = MagicMock()
    mock_async_task.done.return_value = False
    bot.active_tasks["task_to_abort"] = mock_async_task

    # Call abort logic through command callback
    abort_cmd = next(c for c in bot.tree.get_commands() if c.name == "abort")

    mock_interaction = AsyncMock()
    mock_interaction.user.id = 12345
    mock_interaction.channel_id = 999
    mock_interaction.channel = MagicMock()
    bot.owner_id = 12345

    await abort_cmd.callback(mock_interaction, task_id="task_to_abort")

    mock_async_task.cancel.assert_called_once()
    updated_task = await storage.get_task("task_to_abort")
    assert updated_task.status == TaskStatus.FAILED
