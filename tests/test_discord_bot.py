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
    expected_cmds = {
        "new", "stop", "interrupt", "queue", "context", "compact",
        "pipeline", "profile", "yolo", "worktree", "usage", "help", "thread", "providers"
    }
    assert registered_cmds == expected_cmds, f"Slash command mismatch: {registered_cmds ^ expected_cmds}"

    removed_cmds = {
        "status", "profiles", "ask", "run", "task", "abort",
        "reset", "clear", "model", "reasoning", "memory"
    }
    for cmd in removed_cmds:
        assert cmd not in registered_cmds, f"Removed command /{cmd} should not be registered"


@pytest.mark.asyncio
async def test_bot_stop_handler(tmp_path: Path):
    from maulness.core.models import TaskMode, TaskStatus

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    # Seed an active task
    await storage.create_task(
        task_id="task_to_stop",
        title="Test Stop",
        repo_name="horizonx",
        workspace_path="/tmp",
        mode=TaskMode.DIRECT,
    )

    # Mock an asyncio task in active_tasks
    mock_async_task = MagicMock()
    mock_async_task.done.return_value = False
    bot.active_tasks["task_to_stop"] = mock_async_task

    # Call stop logic through command callback
    stop_cmd = next(c for c in bot.tree.get_commands() if c.name == "stop")

    mock_interaction = AsyncMock()
    mock_interaction.user.id = 12345
    mock_interaction.channel_id = 999
    mock_interaction.channel = MagicMock()
    bot.owner_id = 12345

    await stop_cmd.callback(mock_interaction, task_id="task_to_stop")

    mock_async_task.cancel.assert_called_once()
    updated_task = await storage.get_task("task_to_stop")
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

    # 5. Satellite profile (e.g. life) should ignore mentions outside its home/forum channels
    bot_life = MaulnessBot(storage=storage, profile_name="life")
    bot_life.home_channel_id = 5555
    bot_life._connection.user = mock_user
    assert bot_life._should_handle_message(msg3) is False


@pytest.mark.asyncio
async def test_bot_new_and_reset_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    # Seed channel conversation in memory and DB
    bot.channel_conversations[1234] = "conv-uuid-1234"
    await storage.set_channel_conversation(1234, "conv-uuid-1234", "default")
    assert await storage.get_channel_conversation(1234) == "conv-uuid-1234"

    # Call /new command callback
    new_cmd = next(c for c in bot.tree.get_commands() if c.name == "new")
    mock_interaction = AsyncMock()
    mock_interaction.user.id = 12345
    mock_interaction.channel_id = 1234
    bot.owner_id = 12345

    await new_cmd.callback(mock_interaction)

    assert 1234 not in bot.channel_conversations
    assert await storage.get_channel_conversation(1234) is None
    mock_interaction.response.send_message.assert_called_once()
    embed = mock_interaction.response.send_message.call_args[1]["embed"]
    assert "Session Reset" in embed.title


@pytest.mark.asyncio
async def test_bot_context_and_profile_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    bot.channel_conversations[555] = "active-uuid-999"

    context_cmd = next(c for c in bot.tree.get_commands() if c.name == "context")
    mock_interaction = AsyncMock()
    mock_interaction.user.id = 12345
    mock_interaction.channel_id = 555
    bot.owner_id = 12345

    await context_cmd.callback(mock_interaction)
    mock_interaction.response.send_message.assert_called_once()
    embed = mock_interaction.response.send_message.call_args[1]["embed"]
    assert "Active Session Context" in embed.title

    # Test /profile command
    profile_cmd = next(c for c in bot.tree.get_commands() if c.name == "profile")
    mock_interaction_prof = AsyncMock()
    mock_interaction_prof.user.id = 12345
    mock_interaction_prof.channel_id = 555

    # Switch profile for channel
    await profile_cmd.callback(mock_interaction_prof, name="builder")
    assert bot.channel_profile_overrides[555].name == "builder"
    assert 555 not in bot.channel_conversations


@pytest.mark.asyncio
async def test_bot_yolo_worktree_usage_compact_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    # 1. Test /yolo toggle
    yolo_cmd = next(c for c in bot.tree.get_commands() if c.name == "yolo")
    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 100
    await yolo_cmd.callback(mock_int, enabled=None)
    assert bot.channel_yolo[100] is True
    await yolo_cmd.callback(mock_int, enabled=False)
    assert bot.channel_yolo[100] is False

    # 2. Test /worktree toggle
    worktree_cmd = next(c for c in bot.tree.get_commands() if c.name == "worktree")
    await worktree_cmd.callback(mock_int, enabled=True)
    assert bot.channel_worktree[100] is True

    # 3. Test /usage
    usage_cmd = next(c for c in bot.tree.get_commands() if c.name == "usage")
    bot.total_prompts = 5
    bot.total_chars_out = 4000
    await usage_cmd.callback(mock_int)
    mock_int.response.send_message.assert_called()
    embed = mock_int.response.send_message.call_args[1]["embed"]
    assert "Session Metrics & Usage" in embed.title

    # 4. Test /compact
    compact_cmd = next(c for c in bot.tree.get_commands() if c.name == "compact")
    bot.channel_conversations[100] = "uuid-to-compact"
    await compact_cmd.callback(mock_int)
    assert 100 not in bot.channel_conversations
    latest_mem = await storage.get_latest_session_memory(f"discord_100_{int(bot.session_start_time)}")
    # Memory persisted


@pytest.mark.asyncio
async def test_bot_on_message_text_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345
    bot.channel_conversations[777] = "conv-777"
    await storage.set_channel_conversation(777, "conv-777", "default")

    # Mock message for /new typed in text
    msg = AsyncMock()
    msg.author.bot = False
    msg.author.id = 12345
    msg.channel.id = 777
    msg.content = "/new"

    await bot.on_message(msg)

    assert 777 not in bot.channel_conversations
    assert await storage.get_channel_conversation(777) is None
    msg.channel.send.assert_called_once()


def test_format_discord_markdown():
    from maulness.discord.bot import format_discord_markdown

    sample = "Discord $\\rightarrow$ maulness $\\rightarrow$ Antigravity $\\rightarrow$ Filesystem"
    assert format_discord_markdown(sample) == "Discord -> maulness -> Antigravity -> Filesystem"

    backticked = "`pipeline: git $\\rightarrow$ build`"
    assert format_discord_markdown(backticked) == "`pipeline: git -> build`"

    comparisons = "x $\\le$ 10 and y $\\ge$ 20, z $\\approx$ 5"
    assert format_discord_markdown(comparisons) == "x <= 10 and y >= 20, z ~ 5"

    math_symbols = "3 $\\times$ 4 $\\pm$ 0.1"
    assert format_discord_markdown(math_symbols) == "3 * 4 +/- 0.1"



