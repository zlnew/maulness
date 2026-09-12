import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
        "new",
        "stop",
        "interrupt",
        "queue",
        "context",
        "compact",
        "pipeline",
        "profile",
        "yolo",
        "worktree",
        "usage",
        "help",
        "thread",
        "providers",
    }
    assert registered_cmds == expected_cmds, (
        f"Slash command mismatch: {registered_cmds ^ expected_cmds}"
    )

    removed_cmds = {
        "status",
        "profiles",
        "ask",
        "run",
        "task",
        "abort",
        "reset",
        "clear",
        "model",
        "reasoning",
        "memory",
    }
    for cmd in removed_cmds:
        assert cmd not in registered_cmds, (
            f"Removed command /{cmd} should not be registered"
        )


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
    latest_mem = await storage.get_latest_session_memory(
        f"discord_100_{int(bot.session_start_time)}"
    )
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
    assert (
        format_discord_markdown(sample)
        == "Discord -> maulness -> Antigravity -> Filesystem"
    )

    backticked = "`pipeline: git $\\rightarrow$ build`"
    assert format_discord_markdown(backticked) == "`pipeline: git -> build`"

    comparisons = "x $\\le$ 10 and y $\\ge$ 20, z $\\approx$ 5"
    assert format_discord_markdown(comparisons) == "x <= 10 and y >= 20, z ~ 5"

    math_symbols = "3 $\\times$ 4 $\\pm$ 0.1"
    assert format_discord_markdown(math_symbols) == "3 * 4 +/- 0.1"

    # Header softening
    headers = "# Main Header\nText\n## Sub Header\nMore text\n### Already Small"
    expected_headers = (
        "### Main Header\nText\n### Sub Header\nMore text\n### Already Small"
    )
    assert format_discord_markdown(headers) == expected_headers

    # Code block comments are preserved (not treated as headers)
    code_block = "```python\n# This is a comment\nx = 1\n```"
    assert format_discord_markdown(code_block) == code_block

    # Markdown table converted to monospace block
    table_sample = (
        "Here is a table:\n| Name | Role |\n| --- | --- |\n| Maul | Lead |\nDone."
    )
    formatted_table = format_discord_markdown(table_sample)
    assert (
        "```text\n| Name | Role |\n| --- | --- |\n| Maul | Lead |\n```"
        in formatted_table
    )
    assert "Here is a table:" in formatted_table
    assert "Done." in formatted_table


@pytest.mark.asyncio
async def test_bot_interrupt_command(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    # Mock active task
    mock_active = MagicMock()
    mock_active.done.return_value = False
    bot.active_tasks["task_999"] = mock_active
    bot.channel_tasks[555] = "task_999"

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 555
    mock_int.channel = AsyncMock()

    interrupt_cmd = next(c for c in bot.tree.get_commands() if c.name == "interrupt")
    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await interrupt_cmd.callback(mock_int, prompt="New direction")
        mock_active.cancel.assert_called_once()
        mock_int.response.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_bot_queue_command(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    queue_cmd = next(c for c in bot.tree.get_commands() if c.name == "queue")

    # 1. Busy channel
    mock_active = MagicMock()
    mock_active.done.return_value = False
    bot.active_tasks["task_busy"] = mock_active
    bot.channel_tasks[444] = "task_busy"

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 444
    mock_int.channel = AsyncMock()

    await queue_cmd.callback(mock_int, prompt="Queued work")
    assert "Queued work" in bot.channel_queues[444]
    assert "Prompt Queued" in mock_int.response.send_message.call_args[0][0]

    # 2. Idle channel
    mock_int_idle = AsyncMock()
    mock_int_idle.user.id = 12345
    mock_int_idle.channel_id = 333
    mock_int_idle.channel = AsyncMock()

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await queue_cmd.callback(mock_int_idle, prompt="Immediate work")
        assert (
            "Dispatching prompt directly"
            in mock_int_idle.response.send_message.call_args[0][0]
        )


@pytest.mark.asyncio
async def test_bot_help_and_providers_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    mock_int = AsyncMock()
    mock_int.user.id = 12345

    # 1. Help
    help_cmd = next(c for c in bot.tree.get_commands() if c.name == "help")
    await help_cmd.callback(mock_int)
    embed = mock_int.response.send_message.call_args[1]["embed"]
    assert "Help & Commands" in embed.title

    # 2. Providers status
    prov_cmd = next(c for c in bot.tree.get_commands() if c.name == "providers")
    await prov_cmd.callback(mock_int, action="status")
    embed2 = mock_int.response.send_message.call_args[1]["embed"]
    assert "Provider Circuit Breaker" in embed2.title

    # 3. Providers reset
    await prov_cmd.callback(mock_int, action="reset")
    assert "reset to CLOSED" in mock_int.response.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_bot_thread_command(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    thread_cmd = next(c for c in bot.tree.get_commands() if c.name == "thread")

    mock_text_channel = MagicMock(spec=discord.TextChannel)
    mock_created_thread = AsyncMock(spec=discord.Thread)
    mock_created_thread.id = 8888
    mock_created_thread.mention = "<#8888>"
    mock_text_channel.create_thread = AsyncMock(return_value=mock_created_thread)

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel = mock_text_channel

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await thread_cmd.callback(
            mock_int, name="Feature Thread", message="Initial spec"
        )
        mock_text_channel.create_thread.assert_awaited_once()
        mock_created_thread.send.assert_awaited_once()
        mock_int.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_bot_pipeline_command_in_thread(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    pipe_cmd = next(c for c in bot.tree.get_commands() if c.name == "pipeline")

    mock_thread = AsyncMock(spec=discord.Thread)
    mock_thread.id = 9999
    mock_thread.parent = None
    mock_thread.send = AsyncMock()

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel = mock_thread
    mock_int.channel_id = 9999

    bot.pipeline.run_pipeline = AsyncMock(return_value=MagicMock(id="pipe_task_1"))
    bot._update_forum_tags = AsyncMock()

    await pipe_cmd.callback(
        mock_int,
        prompt="Build feature in repo",
        repo="horizonx",
        title="Feature Pipeline",
        pipeline_name="standard",
        worktree=True,
        yolo=True,
    )
    mock_int.response.defer.assert_awaited_once()
    mock_int.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_bot_yolo_and_worktree_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 777

    yolo_cmd = next(c for c in bot.tree.get_commands() if c.name == "yolo")
    # Toggle on
    await yolo_cmd.callback(mock_int)
    assert bot.channel_yolo[777] is True
    # Toggle off
    await yolo_cmd.callback(mock_int)
    assert bot.channel_yolo[777] is False
    # Explicit True
    await yolo_cmd.callback(mock_int, enabled=True)
    assert bot.channel_yolo[777] is True

    worktree_cmd = next(c for c in bot.tree.get_commands() if c.name == "worktree")
    # Toggle on
    await worktree_cmd.callback(mock_int)
    assert bot.channel_worktree[777] is True
    # Explicit False
    await worktree_cmd.callback(mock_int, enabled=False)
    assert bot.channel_worktree[777] is False


@pytest.mark.asyncio
async def test_bot_new_and_context_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 888

    # Set up active conversation and task
    bot.channel_conversations[888] = "old-conv-uuid"
    mock_task = MagicMock()
    mock_task.done.return_value = False
    bot.active_tasks["task-888"] = mock_task
    bot.channel_tasks[888] = "task-888"

    # 1. /new
    new_cmd = next(c for c in bot.tree.get_commands() if c.name == "new")
    await new_cmd.callback(mock_int)
    assert 888 not in bot.channel_conversations
    assert mock_task.cancel.called

    # 2. /context
    context_cmd = next(c for c in bot.tree.get_commands() if c.name == "context")
    await context_cmd.callback(mock_int)
    embed = mock_int.response.send_message.call_args[1]["embed"]
    assert "Active Session Context" in embed.title


@pytest.mark.asyncio
async def test_bot_compact_and_usage_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 999

    # 1. /compact
    compact_cmd = next(c for c in bot.tree.get_commands() if c.name == "compact")
    await compact_cmd.callback(mock_int)
    embed = mock_int.response.send_message.call_args[1]["embed"]
    assert "Context Compacted" in embed.title

    # 2. /usage
    usage_cmd = next(c for c in bot.tree.get_commands() if c.name == "usage")
    await usage_cmd.callback(mock_int)
    embed2 = mock_int.response.send_message.call_args[1]["embed"]
    assert "Session Metrics & Usage" in embed2.title


@pytest.mark.asyncio
async def test_bot_interrupt_and_profile_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 555
    mock_int.channel = AsyncMock()

    # 1. /interrupt
    mock_task = MagicMock()
    mock_task.done.return_value = False
    bot.active_tasks["chat_555"] = mock_task
    bot.channel_tasks[555] = "chat_555"

    interrupt_cmd = next(c for c in bot.tree.get_commands() if c.name == "interrupt")
    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock):
        await interrupt_cmd.callback(mock_int, prompt="New direction")
        assert mock_task.cancel.called

    # 2. /profile list
    profile_cmd = next(c for c in bot.tree.get_commands() if c.name == "profile")
    await profile_cmd.callback(mock_int, name=None)
    embed = mock_int.response.send_message.call_args[1]["embed"]
    assert "Available Agent Profiles" in embed.title

    # 3. /profile switch
    await profile_cmd.callback(mock_int, name="builder")
    assert bot.channel_profile_overrides[555].name == "builder"

    # 4. Autocomplete
    import discord

    mock_int.response.is_done = lambda: False
    ns = discord.app_commands.Namespace(
        mock_int, {}, [{"name": "name", "value": "build", "type": 3}]
    )
    await profile_cmd._invoke_autocomplete(mock_int, "name", ns)
    mock_int.response.autocomplete.assert_awaited_once()
    choices = mock_int.response.autocomplete.call_args[0][0]
    assert any("builder" in c.value for c in choices)


def _make_mock_channel(cid: int = 1010):
    ch = AsyncMock()
    ch.id = cid
    typing_cm = MagicMock()
    typing_cm.__aenter__ = AsyncMock()
    typing_cm.__aexit__ = AsyncMock(return_value=None)
    ch.typing = MagicMock(return_value=typing_cm)
    return ch


@pytest.mark.asyncio
async def test_bot_execute_chat_prompt_lifecycle(tmp_path: Path):
    from maulness.core.models import (
        AgentThoughtEvent,
        AgentMessageEvent,
        AgentToolCallEvent,
    )

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(1010)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # Mock provider
    class DummyProvider:
        def __init__(self):
            self.last_conversation_id = "conv-xyz-789"
            self.primary = "primary_engine"
            self.last_used_provider = MagicMock()
            self.last_used_provider.profile = MagicMock(
                provider="ollama", model="qwen2.5-coder:7b", command=None
            )

        async def run(self, **kwargs):
            # 1. Trigger on_init
            if kwargs.get("on_init"):
                await kwargs["on_init"]("conv-xyz-789")

            # 2. Trigger on_thought with fallback alert
            if kwargs.get("on_thought"):
                await kwargs["on_thought"](
                    AgentThoughtEvent(
                        delta="Switching to fallback provider ollama", session_id="test"
                    )
                )

            # 3. Trigger on_tool_call
            if kwargs.get("on_tool_call"):
                await kwargs["on_tool_call"](
                    AgentToolCallEvent(
                        call_id="c1",
                        tool_name="run_command",
                        args={"command": "git status"},
                        session_id="test",
                    )
                )
                await kwargs["on_tool_call"](
                    AgentToolCallEvent(
                        call_id="c2",
                        tool_name="write_file",
                        args={"path": "main.py"},
                        session_id="test",
                    )
                )
                await kwargs["on_tool_call"](
                    AgentToolCallEvent(
                        call_id="c3",
                        tool_name="custom_tool",
                        args={"foo": "bar"},
                        session_id="test",
                    )
                )

            # 4. Trigger on_message chunk
            if kwargs.get("on_message"):
                await kwargs["on_message"](
                    AgentMessageEvent(delta="Hello from agent", session_id="test")
                )

            return "Hello from agent"

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=DummyProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Initial test prompt",
            author_mention="<@12345>",
            session_key="chat_1010",
        )

    assert bot.channel_conversations[1010] == "conv-xyz-789"
    assert await storage.get_channel_conversation(1010) == "conv-xyz-789"
    mock_msg.edit.assert_called()


@pytest.mark.asyncio
async def test_bot_execute_chat_prompt_approvals(tmp_path: Path):
    from maulness.core.models import ApprovalRequestEvent

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(2020)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # 1. Channel YOLO auto-approve
    bot.channel_yolo[2020] = True
    approval_result_yolo = None

    class YoloProvider:
        async def run(self, **kwargs):
            nonlocal approval_result_yolo
            on_app = kwargs.get("on_approval")
            ev = ApprovalRequestEvent(
                request_id=1,
                call_id="c1",
                tool_name="run_command",
                args={"command": "rm -rf"},
                session_id="s1",
            )
            approval_result_yolo = await on_app(ev)
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=YoloProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Run command",
            author_mention="<@12345>",
            session_key="chat_2020",
        )
    assert approval_result_yolo is True

    # 2. Interactive approval granted
    bot.channel_yolo[2020] = False
    approval_result_interactive = None

    class InteractiveProvider:
        async def run(self, **kwargs):
            nonlocal approval_result_interactive
            on_app = kwargs.get("on_approval")
            ev = ApprovalRequestEvent(
                request_id=2,
                call_id="c2",
                tool_name="run_command",
                args={"command": "pytest", "cwd": "/tmp"},
                session_id="s2",
            )
            approval_result_interactive = await on_app(ev)
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=InteractiveProvider(),
    ):
        with patch("asyncio.wait_for", AsyncMock(return_value=True)):
            await bot._execute_chat_prompt(
                channel=mock_channel,
                prompt="Run command",
                author_mention="<@12345>",
                session_key="chat_2020",
            )
    assert approval_result_interactive is True

    # 3. Interactive approval denied / timeout
    class DeniedProvider:
        async def run(self, **kwargs):
            on_app = kwargs.get("on_approval")
            ev_write = ApprovalRequestEvent(
                request_id=3,
                call_id="c3",
                tool_name="write_file",
                args={"path": "foo.txt", "bytes": 10},
                session_id="s3",
            )
            res_denied = await on_app(ev_write)
            assert res_denied is False

            ev_custom = ApprovalRequestEvent(
                request_id=4,
                call_id="c4",
                tool_name="custom",
                args={"arg": 1},
                session_id="s4",
            )
            res_custom = await on_app(ev_custom)
            assert res_custom is False
            return "Denied"

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=DeniedProvider()
    ):
        with patch(
            "asyncio.wait_for", AsyncMock(side_effect=[False, asyncio.TimeoutError()])
        ):
            await bot._execute_chat_prompt(
                channel=mock_channel,
                prompt="Run command",
                author_mention="<@12345>",
                session_key="chat_2020",
            )


@pytest.mark.asyncio
async def test_bot_execute_chat_prompt_compacted_memory_and_errors(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(3030)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # Seed compacted session memory
    await storage.save_session_memory(
        session_id="discord_3030_12345",
        summary="User prefers strict type hints and zero emojis.",
        token_count=100,
    )

    received_prompt = ""

    class CheckPromptProvider:
        async def run(self, **kwargs):
            nonlocal received_prompt
            received_prompt = kwargs.get("prompt", "")
            return "Acknowledge"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=CheckPromptProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Hello world",
            author_mention="<@12345>",
            session_key="chat_3030",
        )

    assert "[PREVIOUS COMPACTED SESSION CONTEXT]" in received_prompt
    assert "User prefers strict type hints" in received_prompt

    # Test error handling during execution
    class ErrorProvider:
        async def run(self, **kwargs):
            raise RuntimeError("Provider connection failed completely")

    mock_channel_err = _make_mock_channel(4040)
    mock_msg_err = AsyncMock()
    mock_channel_err.send.return_value = mock_msg_err

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=ErrorProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_err,
            prompt="Failing turn",
            author_mention="<@12345>",
            session_key="chat_4040",
        )

    # Error message rendered
    mock_msg_err.edit.assert_called()
    assert "Provider connection failed completely" in str(mock_msg_err.edit.call_args)


@pytest.mark.asyncio
async def test_bot_prefix_commands(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345
    bot._connection.user = MagicMock(id=9999)

    def create_msg(content: str, channel_id: int = 5050):
        msg = AsyncMock()
        msg.author.bot = False
        msg.author.id = 12345
        msg.author.mention = "<@12345>"
        msg.channel = _make_mock_channel(channel_id)
        msg.channel.parent_id = None
        msg.mentions = []
        msg.content = content
        return msg

    # 1. !stop on idle channel
    msg = create_msg("!stop")
    await bot.on_message(msg)
    assert "No active task" in msg.channel.send.call_args[0][0]

    # 2. !stop on active task
    mock_task = MagicMock()
    mock_task.done.return_value = False
    bot.active_tasks["chat_5050"] = mock_task
    bot.channel_tasks[5050] = "chat_5050"
    msg = create_msg("!stop")
    await bot.on_message(msg)
    assert mock_task.cancel.called
    assert "stopped immediately" in msg.channel.send.call_args[0][0]

    # 3. !interrupt without prompt
    msg = create_msg("!interrupt")
    await bot.on_message(msg)
    assert "Usage: `!interrupt <prompt>`" in msg.channel.send.call_args[0][0]

    # 4. !interrupt with prompt
    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock):
        msg = create_msg("!interrupt Steer now")
        await bot.on_message(msg)
        assert "Interrupted previous task" in msg.channel.send.call_args[0][0]

    # 5. !queue without prompt
    msg = create_msg("!queue")
    await bot.on_message(msg)
    assert "Usage: `!queue <prompt>`" in msg.channel.send.call_args[0][0]

    # 6. !queue with busy channel
    bot.active_tasks["chat_5050"] = mock_task
    bot.channel_tasks[5050] = "chat_5050"
    msg = create_msg("!queue Follow-up task")
    await bot.on_message(msg)
    assert "Prompt Queued" in msg.channel.send.call_args[0][0]
    assert "Follow-up task" in bot.channel_queues[5050]

    # 7. !context
    msg = create_msg("!context")
    await bot.on_message(msg)
    assert "Active Session Context" in msg.channel.send.call_args[1]["embed"].title

    # 8. !compact
    msg = create_msg("!compact")
    await bot.on_message(msg)
    assert "Context Compacted" in msg.channel.send.call_args[1]["embed"].title

    # 9. !profile
    msg = create_msg("!profile")
    await bot.on_message(msg)
    assert "Available Agent Profiles" in msg.channel.send.call_args[1]["embed"].title

    msg_switch = create_msg("!profile builder")
    await bot.on_message(msg_switch)
    assert bot.channel_profile_overrides[5050].name == "builder"

    # 10. !yolo & !worktree
    msg_yolo = create_msg("!yolo")
    await bot.on_message(msg_yolo)
    assert bot.channel_yolo[5050] is True

    msg_wt = create_msg("!worktree")
    await bot.on_message(msg_wt)
    assert bot.channel_worktree[5050] is True

    # 11. !usage & !help
    msg_u = create_msg("!usage")
    await bot.on_message(msg_u)
    assert "Session Metrics & Usage" in msg_u.channel.send.call_args[1]["embed"].title

    msg_h = create_msg("!help")
    await bot.on_message(msg_h)
    assert "Help & Commands" in msg_h.channel.send.call_args[1]["embed"].title

    # 12. Standard prompt execution
    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        msg_norm = create_msg("Can you refactor this module?")
        await bot.on_message(msg_norm)
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_bot_pipeline_full_lifecycle(tmp_path: Path):
    import discord
    from maulness.core.models import TaskMode, TaskStatus
    from maulness.core.pipelines import PipelineStage

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    pipe_cmd = next(c for c in bot.tree.get_commands() if c.name == "pipeline")

    # Mock TextChannel creating a Thread
    mock_channel = AsyncMock(spec=discord.TextChannel)
    mock_channel.id = 6060
    mock_thread = AsyncMock(spec=discord.Thread)
    mock_thread.id = 7070
    mock_thread.parent = None
    mock_channel.create_thread = AsyncMock(return_value=mock_thread)

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel = mock_channel
    mock_int.channel_id = 6060

    # Seed task record and events
    task_rec = await storage.create_task(
        task_id="pipe_lifecycle_1",
        title="Implement feature",
        repo_name="horizonx",
        workspace_path=str(tmp_path),
        mode=TaskMode.MULTI,
    )
    await storage.update_task_status("pipe_lifecycle_1", TaskStatus.DONE)
    await storage.record_agent_event(
        task_id="pipe_lifecycle_1",
        stage="init",
        step_index=0,
        event_type="worktree_branch",
        payload={"branch": "feat/isolated-worktree"},
    )
    await storage.record_agent_event(
        task_id="pipe_lifecycle_1",
        stage="review",
        step_index=1,
        event_type="final_response",
        payload={"content": "All tests pass. Score: 10/10 PASS"},
    )

    async def mock_run_pipe(*args, **kwargs):
        # Call stage start & finish
        stage = PipelineStage(name="planning", profile="planner", prompt="Plan feature")
        if kwargs.get("on_stage_start"):
            await kwargs["on_stage_start"](stage, 1, 3)
        if kwargs.get("on_stage_finish"):
            await kwargs["on_stage_finish"](stage, "Stage output markdown " * 50)
        if kwargs.get("on_approval"):
            app_res = await kwargs["on_approval"](
                MagicMock(tool_name="git_commit", args={"msg": "feat"})
            )
            assert app_res is True
        return task_rec

    bot.pipeline.run_pipeline = AsyncMock(side_effect=mock_run_pipe)

    await pipe_cmd.callback(
        mock_int,
        prompt="Implement robust feature",
        repo="horizonx",
        title="Feature Implementation",
        pipeline_name="standard",
        worktree=True,
        yolo=True,
    )

    mock_channel.create_thread.assert_awaited_once()
    mock_thread.send.assert_called()


@pytest.mark.asyncio
async def test_bot_update_forum_tags(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")

    # Non-forum parent -> early return
    thread = AsyncMock(spec=discord.Thread)
    thread.parent = MagicMock(spec=discord.TextChannel)
    await bot._update_forum_tags(thread, "Done")
    thread.edit.assert_not_called()

    # ForumChannel parent
    forum = MagicMock(spec=discord.ForumChannel)
    tag_planning = MagicMock(name="Planning")
    tag_planning.name = "Planning"
    tag_done = MagicMock(name="Done")
    tag_done.name = "Done"
    forum.available_tags = [tag_planning, tag_done]

    thread.parent = forum
    thread.applied_tags = [tag_planning]

    await bot._update_forum_tags(thread, "Done")
    thread.edit.assert_awaited_once_with(applied_tags=[tag_done])


@pytest.mark.asyncio
async def test_bot_app_command_error_handler(tmp_path: Path):
    from discord.app_commands import AppCommandError

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    interaction = AsyncMock()
    interaction.channel_id = 999
    interaction.response.is_done = lambda: False

    # 1. Response not done yet
    await bot.tree.on_error(interaction, AppCommandError("Test command failure"))
    interaction.response.send_message.assert_awaited_once()

    # 2. Response already done (followup)
    interaction_done = AsyncMock()
    interaction_done.channel_id = 999
    interaction_done.response.is_done = lambda: True
    await bot.tree.on_error(interaction_done, AppCommandError("Test followup error"))
    interaction_done.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_git_branch_exceptions_and_empty(tmp_path: Path):
    from maulness.discord.bot import get_git_branch, format_discord_markdown

    # Non-existent path causes Exception -> returns "none"
    assert get_git_branch(tmp_path / "nonexistent_dir_123") == "none"

    # Empty string or None in format_discord_markdown
    assert format_discord_markdown("") == ""
    assert format_discord_markdown(None) is None


@pytest.mark.asyncio
async def test_bot_init_variants_and_resolve_channel_ids(tmp_path: Path):
    from maulness.config import config

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()

    # Default init without profile_name or profile
    bot = MaulnessBot(storage=storage)
    assert bot.profile_name == "default"
    assert len(bot.profiles) == 1

    # Single profile object passed
    prof_custom = bot.profile_manager.get_profile("builder")
    bot_custom = MaulnessBot(storage=storage, profile=prof_custom)
    assert bot_custom.profile_name == "builder"

    # Test int parsing exception in _resolve_* methods
    bot.bound_profile.env_vars["DISCORD_HOME_CHANNEL"] = "not_an_int"
    bot.bound_profile.env_vars["DISCORD_FORUM_CHANNEL_ID"] = "not_an_int"
    bot.bound_profile.env_vars["OWNER_DISCORD_ID"] = "not_an_int"
    bot.bound_profile.env_vars["DISCORD_GUILD_ID"] = "not_an_int"

    # When invalid int string, falls back to config or None
    assert bot._resolve_home_channel_id() == config.discord_forum_channel_id
    assert bot._resolve_forum_channel_id() == config.discord_forum_channel_id
    assert bot._resolve_owner_id() == config.owner_discord_id
    assert bot._resolve_guild_id() == config.discord_guild_id

    # When valid int string
    bot.bound_profile.env_vars["DISCORD_HOME_CHANNEL"] = "111"
    bot.bound_profile.env_vars["DISCORD_FORUM_CHANNEL_ID"] = "222"
    bot.bound_profile.env_vars["OWNER_DISCORD_ID"] = "333"
    bot.bound_profile.env_vars["DISCORD_GUILD_ID"] = "444"

    assert bot._resolve_home_channel_id() == 111
    assert bot._resolve_forum_channel_id() == 222
    assert bot._resolve_owner_id() == 333
    assert bot._resolve_guild_id() == 444


@pytest.mark.asyncio
async def test_bot_setup_hook_and_lifecycle(tmp_path: Path):
    import discord
    from maulness.discord import bot as bot_module

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")

    # 1. setup_hook when storage.list_channel_conversations raises
    bot.storage.list_channel_conversations = AsyncMock(
        side_effect=RuntimeError("Storage offline")
    )
    bot.tree.sync = AsyncMock()
    bot.tree.copy_global_to = MagicMock()
    bot.guild_id = None
    bot.bound_profile.env_vars["DISCORD_BOT_TOKEN"] = "mock_token_12345678901234567890"

    bot.tree.clear_commands(guild=None)
    await bot.setup_hook()
    bot.tree.sync.assert_awaited_once()

    # 2. setup_hook when token already in _synced_tokens
    token_key = "mock_token_12345678901234567890"[:24]
    assert token_key in bot_module._synced_tokens
    bot.tree.sync.reset_mock()
    bot.tree.clear_commands(guild=None)
    await bot.setup_hook()
    bot.tree.sync.assert_not_called()

    # 3. setup_hook with guild_id set
    bot_module._synced_tokens.remove(token_key)
    bot.guild_id = 98765
    bot.storage.list_channel_conversations = AsyncMock(return_value={10: "uuid-10"})
    bot.tree.clear_commands(guild=None)
    await bot.setup_hook()
    bot.tree.copy_global_to.assert_called_once()
    assert bot.channel_conversations == {10: "uuid-10"}

    # 4. on_ready
    bot._connection.user = MagicMock(id=12345, name="MaulnessBot")
    await bot.on_ready()

    # 5. on_interaction
    int_app = MagicMock(spec=discord.Interaction)
    int_app.type = discord.InteractionType.application_command
    int_app.command.name = "status"
    int_app.channel_id = 123
    int_app.user.id = 456
    await bot.on_interaction(int_app)

    int_other = MagicMock(spec=discord.Interaction)
    int_other.type = discord.InteractionType.component
    await bot.on_interaction(int_other)


@pytest.mark.asyncio
async def test_bot_resolve_profile_advanced(tmp_path: Path):
    from maulness.config import config

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")

    # Gateway routes test
    config.gateway_profile_routes = [
        {"chat_id": "7001", "profile": "default"},
        {"chat_id": "not_an_int", "profile": "default"},
    ]
    matched = bot.resolve_profile_for_channel(channel_id=7001)
    assert matched is not None
    assert matched.name == "default"

    # Profile DISCORD_HOME_CHANNEL with invalid format
    prof = bot.profile_manager.get_profile("builder")
    prof.env_vars["DISCORD_HOME_CHANNEL"] = "bad_int"
    bot.profiles.append(prof)
    assert bot.resolve_profile_for_channel(channel_id=999999) is None

    # Fallback to bound profile if home or forum channel matches
    bot.home_channel_id = 8888
    assert bot.resolve_profile_for_channel(channel_id=8888) == bot.bound_profile

    # Message routing: message with only bot mention (empty prompt)
    bot._connection.user = MagicMock(id=9999)
    empty_mention_msg = AsyncMock()
    empty_mention_msg.author.bot = False
    empty_mention_msg.author.id = bot.owner_id or 12345
    empty_mention_msg.channel.id = 8888
    empty_mention_msg.channel.parent_id = None
    empty_mention_msg.mentions = [bot.user]
    empty_mention_msg.content = "<@9999>"

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await bot.on_message(empty_mention_msg)
        mock_exec.assert_not_called()

    # Message from bot itself
    bot_msg = AsyncMock()
    bot_msg.author.bot = True
    await bot.on_message(bot_msg)

    bot_self_msg = AsyncMock()
    bot_self_msg.author.bot = False
    bot_self_msg.author.id = 9999
    await bot.on_message(bot_self_msg)

    # Message from non-owner
    bot.owner_id = 12345
    unauth_msg = AsyncMock()
    unauth_msg.author.bot = False
    unauth_msg.author.id = 67890
    await bot.on_message(unauth_msg)


@pytest.mark.asyncio
async def test_slash_commands_unauthorized(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    commands_to_test = [
        ("new", {}),
        ("stop", {"task_id": "test"}),
        ("interrupt", {"prompt": "test"}),
        ("queue", {"prompt": "test"}),
        ("context", {}),
        ("compact", {}),
        ("help", {}),
        ("thread", {"name": "test"}),
        ("profile", {"name": "default"}),
        ("pipeline", {"prompt": "test"}),
        ("yolo", {"enabled": True}),
        ("worktree", {"enabled": True}),
        ("usage", {}),
        ("providers", {"action": "status"}),
    ]

    for cmd_name, kwargs in commands_to_test:
        cmd = next(c for c in bot.tree.get_commands() if c.name == cmd_name)
        interaction = AsyncMock()
        interaction.user.id = 99999  # unauthorized!
        interaction.channel_id = 1010
        await cmd.callback(interaction, **kwargs)
        interaction.response.send_message.assert_awaited_with(
            "Unauthorized", ephemeral=True
        )


@pytest.mark.asyncio
async def test_bot_execute_chat_prompt_advanced_branches(tmp_path: Path):
    from maulness.core.models import (
        AgentThoughtEvent,
        AgentMessageEvent,
        AgentToolCallEvent,
    )

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(8080)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # 1. Overflow chunk splitting in flush_chunk + edit failures
    class OverflowProvider:
        async def run(self, **kwargs):
            # Fallback thought
            on_th = kwargs.get("on_thought")
            await on_th(
                AgentThoughtEvent(
                    delta="Switching to fallback provider gemini", session_id="s"
                )
            )
            # Tool call
            on_tc = kwargs.get("on_tool_call")
            await on_tc(
                AgentToolCallEvent(
                    call_id="c1",
                    tool_name="run_command",
                    args={"command": "echo 1"},
                    session_id="s",
                )
            )
            # Large message chunk (> 2500 chars)
            on_msg = kwargs.get("on_message")
            await on_msg(AgentMessageEvent(delta="Z" * 3000, session_id="s"))
            return "Done"

    mock_msg.edit.side_effect = [RuntimeError("Edit failed"), None, None]
    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=OverflowProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Test overflow",
            author_mention="<@12345>",
            session_key="chat_8080",
        )

    # 2. CancelledError during run
    class CancelledProvider:
        async def run(self, **kwargs):
            raise asyncio.CancelledError()

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=CancelledProvider(),
    ):
        with pytest.raises(asyncio.CancelledError):
            await bot._execute_chat_prompt(
                channel=mock_channel,
                prompt="Cancelled test",
                author_mention="<@12345>",
                session_key="chat_8080",
            )

    # 3. Empty output generated -> "(No visible output was generated by the model)"
    class EmptyProvider:
        async def run(self, **kwargs):
            return ""

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=EmptyProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Empty test",
            author_mention="<@12345>",
            session_key="chat_8080",
        )

    # 4. Storage errors during on_init and storage persist
    class StorageErrorProvider:
        async def run(self, **kwargs):
            on_init = kwargs.get("on_init")
            await on_init("conv-err-uuid")
            return "Done"

    bot.storage.set_channel_conversation = AsyncMock(
        side_effect=RuntimeError("Storage disk error")
    )
    bot.storage.get_latest_compacted_memory = AsyncMock(
        side_effect=RuntimeError("Storage read error")
    )

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=StorageErrorProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Storage error test",
            author_mention="<@12345>",
            session_key="chat_8080",
        )

    # 5. Queued prompts execution loop
    bot.channel_queues[8080] = ["Next queued prompt"]
    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=EmptyProvider()
    ):
        with patch.object(
            bot, "_execute_chat_prompt", new_callable=AsyncMock
        ) as mock_exec:
            # Run a prompt that will trigger the queued prompt
            await bot._execute_chat_prompt(
                channel=mock_channel,
                prompt="First prompt",
                author_mention="<@12345>",
                session_key="chat_8080",
            )
            # Give event loop a cycle to schedule the background task
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_bot_stop_handler_extra_branches(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    stop_cmd = next(c for c in bot.tree.get_commands() if c.name == "stop")

    # 1. No active task
    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 1010
    await stop_cmd.callback(mock_int)
    mock_int.response.send_message.assert_awaited_with(
        "[Info] No active task or agent execution running in this channel. (Use `/new` to reset context)",
        ephemeral=True,
    )

    # 2. Task already done
    mock_task_done = MagicMock()
    mock_task_done.done.return_value = True
    bot.active_tasks["done_task"] = mock_task_done
    bot.channel_tasks[1010] = "done_task"

    await stop_cmd.callback(mock_int)
    mock_int.response.send_message.assert_awaited_with(
        "[Info] Execution `done_task` is already finished.", ephemeral=True
    )

    # 3. Active non-chat task in a Thread
    mock_task_running = MagicMock()
    mock_task_running.done.return_value = False
    bot.active_tasks["pipeline_task_99"] = mock_task_running
    bot.channel_tasks[2020] = "pipeline_task_99"

    mock_thread_int = AsyncMock()
    mock_thread_int.user.id = 12345
    mock_thread_int.channel_id = 2020
    mock_thread_int.channel = MagicMock(spec=discord.Thread)
    bot._update_forum_tags = AsyncMock()

    await stop_cmd.callback(mock_thread_int, task_id="pipeline_task_99")
    mock_task_running.cancel.assert_called_once()
    bot._update_forum_tags.assert_awaited_once_with(mock_thread_int.channel, "Failed")


@pytest.mark.asyncio
async def test_bot_thread_command_variants(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    thread_cmd = next(c for c in bot.tree.get_commands() if c.name == "thread")

    # 1. In ForumChannel
    mock_forum = AsyncMock(spec=discord.ForumChannel)
    mock_thread = AsyncMock(spec=discord.Thread)
    mock_thread.id = 3030
    mock_thread.mention = "<#3030>"
    mock_thread_with_msg = MagicMock(thread=mock_thread)
    mock_forum.create_thread = AsyncMock(return_value=mock_thread_with_msg)

    mock_int_forum = AsyncMock()
    mock_int_forum.user.id = 12345
    mock_int_forum.channel = mock_forum

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await thread_cmd.callback(
            mock_int_forum, name="Forum Topic", message="First query"
        )
        mock_forum.create_thread.assert_awaited_once()
        mock_exec.assert_awaited_once()

    # 2. In Thread already
    mock_int_thread = AsyncMock()
    mock_int_thread.user.id = 12345
    mock_thread_existing = AsyncMock(spec=discord.Thread)
    mock_thread_existing.id = 4040
    mock_thread_existing.mention = "<#4040>"
    mock_int_thread.channel = mock_thread_existing

    await thread_cmd.callback(mock_int_thread, name="Sub Thread", message=None)
    mock_int_thread.followup.send.assert_awaited_once()

    # 3. In unsupported channel (e.g. DMChannel)
    mock_int_dm = AsyncMock()
    mock_int_dm.user.id = 12345
    mock_int_dm.channel = MagicMock(spec=discord.DMChannel)

    await thread_cmd.callback(mock_int_dm, name="DM Thread")
    assert "Cannot create a thread" in mock_int_dm.followup.send.call_args[0][0]


@pytest.mark.asyncio
async def test_bot_pipeline_command_advanced_branches(tmp_path: Path):
    import discord
    from maulness.core.models import TaskMode, TaskStatus
    from maulness.core.pipelines import PipelineStage, VerificationGate

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    pipe_cmd = next(c for c in bot.tree.get_commands() if c.name == "pipeline")

    # In a Forum Thread: infer repo from applied_tags
    mock_parent_forum = MagicMock(spec=discord.ForumChannel)
    mock_thread = AsyncMock(spec=discord.Thread)
    mock_thread.id = 5050
    mock_thread.parent = mock_parent_forum
    tag_repo = MagicMock()
    tag_repo.name = "horizonx"
    mock_thread.applied_tags = [tag_repo]
    mock_thread.send = AsyncMock()

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel = mock_thread
    mock_int.channel_id = 5050

    task_rec = await storage.create_task(
        task_id="pipe_infer_1",
        title="Inferred Repo",
        repo_name="horizonx",
        workspace_path=str(tmp_path),
        mode=TaskMode.MULTI,
    )
    await storage.update_task_status("pipe_infer_1", TaskStatus.FAILED)

    async def mock_run_pipe(*args, **kwargs):
        gate = VerificationGate(command="pytest")
        stage = PipelineStage(
            name="review",
            profile="reviewer",
            prompt="Audit code",
            verification_gate=gate,
        )
        # on_stage_start with verification gate
        await kwargs["on_stage_start"](stage, 1, 1)
        # on_stage_finish with short snippet (< 900 chars)
        await kwargs["on_stage_finish"](stage, "Short review snippet")
        # on_thought and on_message callbacks
        await kwargs["on_thought"](MagicMock())
        await kwargs["on_message"](MagicMock())
        # on_approval with YOLO False
        with patch("asyncio.wait_for", AsyncMock(return_value=True)):
            app_ok = await kwargs["on_approval"](MagicMock(tool_name="git", args={}))
            assert app_ok is True
        return task_rec

    bot.pipeline.run_pipeline = AsyncMock(side_effect=mock_run_pipe)
    bot._update_forum_tags = AsyncMock()

    await pipe_cmd.callback(
        mock_int,
        prompt="Check inferred repo pipeline",
        repo=None,  # triggers inference
        yolo=False,
    )

    bg_task = bot.active_tasks.get("pipeline_5050")
    if bg_task:
        await bg_task

    mock_thread.send.assert_called()


@pytest.mark.asyncio
async def test_bot_update_forum_tags_exception(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")

    forum = MagicMock(spec=discord.ForumChannel)
    forum.available_tags = []
    thread = AsyncMock(spec=discord.Thread)
    thread.parent = forum
    thread.applied_tags = []
    thread.edit.side_effect = RuntimeError("Discord API error")

    # Does not raise, catches exception and logs debug
    await bot._update_forum_tags(thread, "Done")


@pytest.mark.asyncio
async def test_bot_init_profiles_list(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage)
    p1 = bot.profile_manager.get_profile("builder")
    p2 = bot.profile_manager.get_profile("default")

    b = MaulnessBot(storage=storage, profiles=[p1, p2])
    assert b.profile_name == "default"
    assert b.bound_profile == p2


@pytest.mark.asyncio
async def test_bot_resolve_profile_for_channel_matching_home_forum(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    p = bot.profile_manager.get_profile("builder")
    p.env_vars["DISCORD_HOME_CHANNEL"] = "9999"
    bot.profiles.append(p)
    assert bot.resolve_profile_for_channel(9999) == p


@pytest.mark.asyncio
async def test_bot_flush_chunk_edge_cases(tmp_path: Path):
    from maulness.core.models import AgentMessageEvent

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(7171)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # 1. current_msg.edit fails, channel.send fails (lines 339-340)
    class EditAndSendFailProvider:
        async def run(self, **kwargs):
            on_msg = kwargs.get("on_message")
            await on_msg(AgentMessageEvent(delta="short delta", session_id="s"))
            return "Done"

    mock_msg.edit.side_effect = RuntimeError("Edit failed")
    mock_channel.send.side_effect = [mock_msg, RuntimeError("Send failed")]

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=EditAndSendFailProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Prompt",
            author_mention="<@12345>",
            session_key="chat_7171",
        )

    # 2. Long text (>1950 chars) where edit fails, send succeeds on first_part (line 350) and fails on overflow (lines 359-361)
    mock_channel_long = _make_mock_channel(7272)
    mock_msg_long = AsyncMock()
    mock_msg_long.edit.side_effect = RuntimeError("Edit first part failed")
    mock_channel_long.send.side_effect = [
        mock_msg_long,  # initial placeholder
        mock_msg_long,  # first part send succeeds -> line 350 active_msgs.append
        RuntimeError("Send overflow failed"),  # overflow chunk fails -> line 359-361
    ]

    class LongChunkProvider:
        async def run(self, **kwargs):
            on_msg = kwargs.get("on_message")
            await on_msg(AgentMessageEvent(delta="Y" * 4000, session_id="s"))
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=LongChunkProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_long,
            prompt="Prompt long",
            author_mention="<@12345>",
            session_key="chat_7272",
        )
    # 2b. Long text where both edit and send fail on first_part (lines 351-352)
    mock_channel_fail_both = _make_mock_channel(7274)
    mock_msg_fail = AsyncMock()
    mock_msg_fail.edit.side_effect = RuntimeError("Edit first part fail")
    mock_channel_fail_both.send.side_effect = [
        mock_msg_fail,
        RuntimeError("Send first part also fail"),
    ]
    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=LongChunkProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_fail_both,
            prompt="Prompt fail both",
            author_mention="<@12345>",
            session_key="chat_7274",
        )

    # 3. Stream overflow chunk with <= 1950 chars (lines 332-333)
    mock_channel_overflow = _make_mock_channel(7273)
    mock_msg_overflow = AsyncMock()
    mock_channel_overflow.send.return_value = mock_msg_overflow

    class MidOverflowProvider:
        async def run(self, **kwargs):
            on_msg = kwargs.get("on_message")
            # Write 1850 chars to trigger debouncer max_chunk_size overflow flush
            await on_msg(AgentMessageEvent(delta="M" * 1850, session_id="s"))
            await asyncio.sleep(0.01)
            return "Final bit"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=MidOverflowProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_overflow,
            prompt="Mid overflow",
            author_mention="<@12345>",
            session_key="chat_7273",
        )


@pytest.mark.asyncio
async def test_bot_on_approval_cleanup_and_errors(tmp_path: Path):
    from maulness.core.models import (
        AgentToolCallEvent,
        ApprovalRequestEvent,
        AgentMessageEvent,
    )
    from maulness.core.debouncer import MessageStreamDebouncer

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    # 1. Tool call edit failure (lines 390-391)
    mock_channel_tool = _make_mock_channel(7372)
    mock_msg_tool = AsyncMock()
    mock_msg_tool.edit.side_effect = RuntimeError("Tool edit failed")
    mock_channel_tool.send.return_value = mock_msg_tool

    class ToolEditFailProvider:
        async def run(self, **kwargs):
            on_tc = kwargs.get("on_tool_call")
            await on_tc(
                AgentToolCallEvent(
                    call_id="c1",
                    tool_name="run_command",
                    args={"command": "ls"},
                    session_id="s",
                )
            )
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=ToolEditFailProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_tool,
            prompt="Tool prompt",
            author_mention="<@12345>",
            session_key="chat_7372",
        )

    # 2. Debouncer has text, close fails, channel.send status_text fails (lines 442-446, 459-460)
    mock_channel = _make_mock_channel(7373)
    mock_msg = AsyncMock()
    mock_channel.send.side_effect = [
        mock_msg,  # placeholder
        mock_msg,  # approval card
        RuntimeError("Status text send failed"),  # lines 459-460
    ]

    class ApprovalWithTextProvider:
        async def run(self, **kwargs):
            on_msg = kwargs.get("on_message")
            await on_msg(AgentMessageEvent(delta="Prior output text", session_id="s"))

            on_app = kwargs.get("on_approval")
            with patch.object(
                MessageStreamDebouncer,
                "close",
                AsyncMock(side_effect=RuntimeError("Close fail")),
            ):
                with patch("asyncio.wait_for", AsyncMock(return_value=True)):
                    await on_app(
                        ApprovalRequestEvent(
                            request_id=1,
                            call_id="c1",
                            tool_name="run_command",
                            args={},
                            session_id="s",
                        )
                    )
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=ApprovalWithTextProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Approval prompt",
            author_mention="<@12345>",
            session_key="chat_7373",
        )

    # 3. Debouncer empty, delete fails (lines 438-440)
    mock_channel_empty = _make_mock_channel(7374)
    mock_msg_empty = AsyncMock()
    mock_msg_empty.delete.side_effect = RuntimeError("Delete fail")
    mock_channel_empty.send.side_effect = [
        mock_msg_empty,
        mock_msg_empty,
        mock_msg_empty,
    ]

    class ApprovalEmptyProvider:
        async def run(self, **kwargs):
            on_app = kwargs.get("on_approval")
            with patch("asyncio.wait_for", AsyncMock(return_value=False)):
                await on_app(
                    ApprovalRequestEvent(
                        request_id=2,
                        call_id="c2",
                        tool_name="run_command",
                        args={},
                        session_id="s",
                    )
                )
            return "Done"

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=ApprovalEmptyProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_empty,
            prompt="Empty debouncer tool",
            author_mention="<@12345>",
            session_key="chat_7374",
        )


@pytest.mark.asyncio
async def test_bot_storage_save_errors_and_cancellation(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    mock_channel = _make_mock_channel(7474)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    # 1. storage.set_channel_conversation raises after run (lines 528-529)
    class PostRunProvider:
        def __init__(self):
            self.last_conversation_id = "conv-persisted"

        async def run(self, **kwargs):
            return "All done"

    bot.storage.set_channel_conversation = AsyncMock(
        side_effect=RuntimeError("Storage disk full")
    )
    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=PostRunProvider()
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel,
            prompt="Persist prompt",
            author_mention="<@12345>",
            session_key="chat_7474",
        )

    # 2. CancelledError with channel.send failing (lines 534-535)
    class CancelFailProvider:
        async def run(self, **kwargs):
            raise asyncio.CancelledError()

    mock_channel_cancel = _make_mock_channel(7575)
    mock_channel_cancel.send.side_effect = [
        mock_msg,
        RuntimeError("Cancel send failed"),
    ]
    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=CancelFailProvider(),
    ):
        with pytest.raises(asyncio.CancelledError):
            await bot._execute_chat_prompt(
                channel=mock_channel_cancel,
                prompt="Cancel fail",
                author_mention="<@12345>",
                session_key="chat_7575",
            )

    # 3. Generic Exception with edit and send both failing (lines 544-547)
    class GenericErrProvider:
        async def run(self, **kwargs):
            raise RuntimeError("Fatal generic crash")

    mock_channel_err = _make_mock_channel(7676)
    mock_msg_err = AsyncMock()
    mock_msg_err.edit.side_effect = RuntimeError("Edit error msg failed")
    mock_channel_err.send.side_effect = [
        mock_msg_err,
        RuntimeError("Send error msg failed"),
    ]

    with patch(
        "maulness.discord.bot.get_provider_for_profile",
        return_value=GenericErrProvider(),
    ):
        await bot._execute_chat_prompt(
            channel=mock_channel_err,
            prompt="Generic err",
            author_mention="<@12345>",
            session_key="chat_7676",
        )


@pytest.mark.asyncio
async def test_bot_autonomous_memory_extraction_and_queues(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    # Seed 5 messages in storage for conv_id using add_conversation_message
    bot.channel_conversations[7777] = "conv-extract-1"
    for i in range(5):
        await storage.add_conversation_message(
            conversation_id="conv-extract-1",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Conversation turn {i}",
        )

    mock_channel = _make_mock_channel(7777)
    mock_msg = AsyncMock()
    mock_channel.send.return_value = mock_msg

    class DummyProvider:
        async def run(self, **kwargs):
            return "Response"

    # Queue next prompt so lines 580-581 run
    bot.channel_queues[7777] = ["Queued auto-prompt"]

    with patch(
        "maulness.discord.bot.get_provider_for_profile", return_value=DummyProvider()
    ):
        with patch(
            "maulness.core.memory.AutonomousMemoryExtractor.extract_and_update",
            AsyncMock(side_effect=RuntimeError("Extractor test error")),
        ):
            await bot._execute_chat_prompt(
                channel=mock_channel,
                prompt="Prompt that triggers fact extraction",
                author_mention="<@12345>",
                session_key="chat_7777",
            )
            # Allow background tasks to run
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_bot_on_message_prefix_remaining(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    bot.owner_id = 12345

    # 1. !new when task is running (lines 659-661)
    mock_task = MagicMock()
    mock_task.done.return_value = False
    bot.active_tasks["task_running"] = mock_task
    bot.channel_tasks[8888] = "task_running"

    msg_new = AsyncMock()
    msg_new.author.bot = False
    msg_new.author.id = 12345
    msg_new.channel.id = 8888
    msg_new.content = "!new"

    await bot.on_message(msg_new)
    mock_task.cancel.assert_called_once()

    # 2. !stop when task is already done (line 688)
    mock_task_done = MagicMock()
    mock_task_done.done.return_value = True
    bot.active_tasks["task_done"] = mock_task_done
    bot.channel_tasks[8888] = "task_done"

    msg_stop = AsyncMock()
    msg_stop.author.bot = False
    msg_stop.author.id = 12345
    msg_stop.channel.id = 8888
    msg_stop.content = "!stop"

    await bot.on_message(msg_stop)
    assert "already finished" in msg_stop.channel.send.call_args[0][0]

    # 3. !queue when channel is idle (lines 726-728)
    msg_q = AsyncMock()
    msg_q.author.bot = False
    msg_q.author.id = 12345
    msg_q.channel.id = 8888
    msg_q.content = "!queue Direct execution prompt"

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await bot.on_message(msg_q)
        assert "Dispatching prompt directly" in msg_q.channel.send.call_args[0][0]

    # 4. !compact when old_id was set (line 813)
    bot.channel_conversations[8888] = "old-conv-compact"
    msg_compact = AsyncMock()
    msg_compact.author.bot = False
    msg_compact.author.id = 12345
    msg_compact.channel.id = 8888
    msg_compact.content = "!compact"

    await bot.on_message(msg_compact)
    embed = msg_compact.channel.send.call_args[1]["embed"]
    assert embed.footer.text == "Compacted session: old-conv-compact"

    # 5. Normal message without mention in unmapped channel -> returns None (line 896)
    msg_unmapped = AsyncMock()
    msg_unmapped.author.bot = False
    msg_unmapped.author.id = 12345
    msg_unmapped.channel.id = 99999
    msg_unmapped.channel.parent_id = None
    msg_unmapped.mentions = []
    msg_unmapped.content = "Just random chit-chat"

    with patch.object(bot, "_execute_chat_prompt", new_callable=AsyncMock) as mock_exec:
        await bot.on_message(msg_unmapped)
        mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_bot_update_forum_tags_preserves_non_status(tmp_path: Path):
    import discord

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")

    forum = MagicMock(spec=discord.ForumChannel)
    tag_repo = MagicMock(name="repo")
    tag_repo.name = "horizonx"
    tag_done = MagicMock(name="done")
    tag_done.name = "Done"
    forum.available_tags = [tag_repo, tag_done]

    thread = AsyncMock(spec=discord.Thread)
    thread.parent = forum
    thread.applied_tags = [tag_repo]

    await bot._update_forum_tags(thread, "Done")
    thread.edit.assert_awaited_once_with(applied_tags=[tag_repo, tag_done])


@pytest.mark.asyncio
async def test_on_app_command_error_exception(tmp_path: Path):
    from discord.app_commands import AppCommandError

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()

    interaction = AsyncMock()
    interaction.response.is_done = lambda: False
    interaction.response.send_message.side_effect = RuntimeError("Failed to send error")

    # Should not raise exception
    await bot.tree.on_error(interaction, AppCommandError("Command crashed"))


@pytest.mark.asyncio
async def test_bot_stop_handler_storage_update_error(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345

    stop_cmd = next(c for c in bot.tree.get_commands() if c.name == "stop")
    mock_task = MagicMock()
    mock_task.done.return_value = False
    bot.active_tasks["pipeline_task_err"] = mock_task
    bot.channel_tasks[9191] = "pipeline_task_err"

    bot.storage.update_task_status = AsyncMock(
        side_effect=RuntimeError("DB update failed")
    )
    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel_id = 9191

    await stop_cmd.callback(mock_int, task_id="pipeline_task_err")
    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_bot_pipeline_forum_target_parent_and_errors(tmp_path: Path):
    import discord
    from maulness.core.models import TaskMode, TaskStatus

    storage = StorageManager(db_path=tmp_path / "bot_test.db")
    await storage.initialize()
    bot = MaulnessBot(storage=storage, profile_name="default")
    await bot._register_slash_commands()
    bot.owner_id = 12345
    bot.forum_channel_id = 6666

    pipe_cmd = next(c for c in bot.tree.get_commands() if c.name == "pipeline")

    mock_forum = AsyncMock(spec=discord.ForumChannel)
    tag_pipe = MagicMock()
    tag_pipe.name = "pipeline"
    tag_plan = MagicMock()
    tag_plan.name = "planning"
    tag_repo = MagicMock()
    tag_repo.name = "horizonx"
    mock_forum.available_tags = [tag_pipe, tag_plan, tag_repo]

    mock_created_thread = AsyncMock(spec=discord.Thread)
    mock_created_thread.id = 7777
    mock_created_thread.mention = "<#7777>"
    mock_forum.create_thread.return_value = MagicMock(thread=mock_created_thread)

    bot.get_channel = MagicMock(return_value=None)
    bot.fetch_channel = AsyncMock(return_value=mock_forum)

    mock_channel_text = AsyncMock(spec=discord.TextChannel)
    mock_channel_text.id = 8888

    mock_int = AsyncMock()
    mock_int.user.id = 12345
    mock_int.channel = mock_channel_text
    mock_int.channel_id = 8888

    # 1. Pipeline execution with branch event as str and review event as str
    task_rec = await storage.create_task(
        task_id="pipe_complete_2",
        title="Complete 2",
        repo_name="horizonx",
        workspace_path=str(tmp_path),
        mode=TaskMode.MULTI,
    )
    await storage.update_task_status("pipe_complete_2", TaskStatus.DONE)
    await storage.record_agent_event(
        task_id="pipe_complete_2",
        stage="init",
        step_index=0,
        event_type="worktree_branch",
        payload="feat/str-branch",
    )
    await storage.record_agent_event(
        task_id="pipe_complete_2",
        stage="review",
        step_index=1,
        event_type="final_response",
        payload="Simple string review scorecard",
    )

    async def mock_run_pipe(*args, **kwargs):
        # Trigger on_approval timeout
        if kwargs.get("on_approval"):
            with patch(
                "asyncio.wait_for", AsyncMock(side_effect=asyncio.TimeoutError())
            ):
                res = await kwargs["on_approval"](MagicMock(tool_name="tool", args={}))
                assert res is False
        return task_rec

    bot.pipeline.run_pipeline = AsyncMock(side_effect=mock_run_pipe)
    bot._update_forum_tags = AsyncMock()
    # Pre-seed active_tasks to hit line 1543
    bot.active_tasks["pipe_complete_2"] = MagicMock()

    await pipe_cmd.callback(
        mock_int,
        prompt="Execute forum parent pipeline",
        repo="horizonx",
        yolo=False,
    )

    bg_task = bot.active_tasks.get("pipeline_7777")
    if bg_task:
        await bg_task

    mock_created_thread.send.assert_called()
    assert "pipe_complete_2" not in bot.active_tasks  # line 1543 hit

    # 2. Pipeline with storage.get_agent_events error (line 1530)
    bot.pipeline.run_pipeline = AsyncMock(return_value=task_rec)
    with patch.object(
        bot.storage,
        "get_agent_events",
        AsyncMock(side_effect=RuntimeError("Events DB fail")),
    ):
        await pipe_cmd.callback(
            mock_int,
            prompt="Events fail test",
            repo="horizonx",
            yolo=True,
        )
        bg_task_ev = bot.active_tasks.get("pipeline_7777")
        if bg_task_ev:
            await bg_task_ev

    # 3. Pipeline CancelledError
    bot.pipeline.run_pipeline = AsyncMock(side_effect=asyncio.CancelledError())
    await pipe_cmd.callback(
        mock_int,
        prompt="Cancel pipeline test",
        repo="horizonx",
    )
    bg_task_c = bot.active_tasks.get("pipeline_7777")
    if bg_task_c:
        await bg_task_c

    # 4. Pipeline Generic Exception
    bot.pipeline.run_pipeline = AsyncMock(side_effect=RuntimeError("Pipeline boom"))
    await pipe_cmd.callback(
        mock_int,
        prompt="Error pipeline test",
        repo="horizonx",
    )
    bg_task_e = bot.active_tasks.get("pipeline_7777")
    if bg_task_e:
        await bg_task_e

    # 5. Pipeline fetch_channel exception (lines 1352-1353)
    bot.fetch_channel = AsyncMock(side_effect=RuntimeError("Fetch discord fail"))
    mock_channel_direct = AsyncMock(spec=discord.TextChannel)
    mock_channel_direct.id = 9999
    mock_channel_direct.create_thread = AsyncMock(return_value=mock_created_thread)
    mock_int_direct = AsyncMock()
    mock_int_direct.user.id = 12345
    mock_int_direct.channel = mock_channel_direct
    mock_int_direct.channel_id = 9999

    bot.pipeline.run_pipeline = AsyncMock(return_value=task_rec)
    await pipe_cmd.callback(
        mock_int_direct,
        prompt="Direct text channel pipeline test",
        repo="horizonx",
    )


@pytest.mark.asyncio
async def test_bot_handle_afk_suspension_actions(tmp_path: Path):
    from maulness.core.models import TaskMode, TaskStatus

    storage = StorageManager(db_path=tmp_path / "bot_afk.db")
    await storage.initialize()

    bot = MaulnessBot(storage=storage, profile_name="default")

    task = await storage.create_task(
        task_id="task_afk_1",
        title="AFK Task",
        repo_name="horizonx",
        workspace_path=str(tmp_path),
        mode=TaskMode.MULTI,
    )
    await storage.update_task_status(task.id, TaskStatus.SUSPENDED_AFK)

    mock_channel = AsyncMock()
    mock_resume = AsyncMock()

    # 1. Action: resume
    fut_resume = asyncio.Future()
    fut_resume.set_result("resume")
    await bot._handle_afk_suspension(
        task_record=task,
        future=fut_resume,
        channel=mock_channel,
        resume_coro_fn=mock_resume,
    )
    mock_channel.send.assert_awaited()
    assert "Resuming" in mock_channel.send.call_args[0][0]

    # 2. Action: merge
    fut_merge = asyncio.Future()
    fut_merge.set_result("merge")
    await bot._handle_afk_suspension(
        task_record=task,
        future=fut_merge,
        channel=mock_channel,
    )
    updated_task = await storage.get_task(task.id)
    assert updated_task.status == TaskStatus.DONE
    assert "signed off and marked DONE" in mock_channel.send.call_args[0][0]

    # 3. Action: abort
    fut_abort = asyncio.Future()
    fut_abort.set_result("abort")
    await bot._handle_afk_suspension(
        task_record=task,
        future=fut_abort,
        channel=mock_channel,
    )
    aborted_task = await storage.get_task(task.id)
    assert aborted_task.status == TaskStatus.FAILED
    assert "aborted and marked FAILED" in mock_channel.send.call_args[0][0]
