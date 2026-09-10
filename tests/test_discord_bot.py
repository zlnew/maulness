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
    assert "thread" in registered_cmds


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


@pytest.mark.asyncio
async def test_bot_should_handle_message_routing(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.home_channel_id = 1111
    bot.forum_channel_id = 2222
    mock_user = MagicMock()
    mock_user.id = 9999
    bot._connection.user = mock_user

    # 1. Message in home channel
    msg1 = MagicMock()
    msg1.channel.id = 1111
    msg1.channel.parent_id = None
    msg1.mentions = []
    msg1.content = "hello bot"
    assert bot._should_handle_message(msg1) is True

    # 2. Message in thread of forum channel
    msg2 = MagicMock()
    msg2.channel.id = 3333
    msg2.channel.parent_id = 2222
    msg2.mentions = []
    msg2.content = "working on task"
    assert bot._should_handle_message(msg2) is True

    # 3. Message mentioning bot in unrelated channel
    msg3 = MagicMock()
    msg3.channel.id = 8888
    msg3.channel.parent_id = None
    msg3.mentions = [bot.user]
    msg3.content = "<@9999> check this"
    assert bot._should_handle_message(msg3) is True

    # 4. Message in unrelated channel with no mention
    msg4 = MagicMock()
    msg4.channel.id = 8888
    msg4.channel.parent_id = None
    msg4.mentions = []
    msg4.content = "random chat"
    assert bot._should_handle_message(msg4) is False

