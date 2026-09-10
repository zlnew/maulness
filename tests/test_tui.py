import asyncio
import pytest
from pathlib import Path

from maulness.cli.tui import (
    CommandPaletteModal,
    DiffModal,
    HelpModal,
    MaulnessTUIApp,
)
from maulness.config import config


@pytest.mark.asyncio
async def test_tui_neovim_modes_and_scrolling():
    app = MaulnessTUIApp(initial_profile="builder")
    async with app.run_test() as pilot:
        # Initial mode is insert
        assert app.mode == "insert"
        statusline = app.query_one("#vim-statusline")
        assert "INSERT" in str(statusline.render())

        # Press Escape to enter NORMAL mode
        await pilot.press("escape")
        await pilot.pause()
        assert app.mode == "normal"
        assert "NORMAL" in str(statusline.render())

        # Test Vim scrolling keys in normal mode
        await pilot.press("j", "k", "G", "g", "g")
        await pilot.pause()
        assert app.mode == "normal"

        # Press 'i' to re-enter INSERT mode
        await pilot.press("i")
        await pilot.pause()
        assert app.mode == "insert"
        assert "INSERT" in str(statusline.render())


@pytest.mark.asyncio
async def test_tui_lsp_autocomplete():
    from textual.widgets import OptionList

    app = MaulnessTUIApp(initial_profile="default")
    async with app.run_test() as pilot:
        assert app.mode == "insert"
        popup = app.query_one("#autocomplete-popup", OptionList)
        assert not popup.has_class("visible")

        # Type '/' in insert mode to trigger LSP autocomplete popup
        await pilot.press("slash")
        await pilot.pause()
        assert popup.has_class("visible")
        assert popup.option_count > 0

        # Type 'c' -> input is now '/c', popup filters
        await pilot.press("c")
        await pilot.pause()
        assert popup.has_class("visible")

        # Navigate down and complete with tab
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        # Input should have completed a /c command and popup hidden
        inp = app.query_one("#chat-input")
        assert inp.value.startswith("/c")
        assert not popup.has_class("visible")


@pytest.mark.asyncio
async def test_tui_shell_escape_key():
    app = MaulnessTUIApp(initial_profile="builder")
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert app.mode == "normal"

        # Press '!' to enter insert mode with '! ' prepended
        await pilot.press("exclamation_mark")
        await pilot.pause()
        assert app.mode == "insert"
        inp = app.query_one("#chat-input")
        assert inp.value == "! "


@pytest.mark.asyncio
async def test_workspace_resolution_arbitrary_dir():
    # Test from home directory (~ or /home/zlnew)
    home_dir = Path("/home/zlnew")
    repo_name, ws_path = config.get_current_workspace(home_dir)
    assert repo_name == "zlnew"
    assert ws_path == home_dir

    # Resolving repo path should NEVER raise FileNotFoundError
    resolved = config.resolve_repo_path(repo_name)
    assert resolved.exists()
    assert resolved.is_dir()


@pytest.mark.asyncio
async def test_tui_chat_execution_from_home(tmp_path: Path, monkeypatch):
    class MockProvider:
        async def run(self, prompt, **kwargs):
            on_msg = kwargs.get("on_message")
            if on_msg:
                from maulness.core.models import AgentMessageEvent
                await on_msg(AgentMessageEvent(delta="Mock response content", session_id="test"))
            return "Mock response content"

    from maulness.core import runner as runner_module
    monkeypatch.setattr(runner_module, "get_provider_for_profile", lambda p: MockProvider())

    app = MaulnessTUIApp(initial_profile="default")
    # Simulate being in /home/zlnew
    app.repo_name = "zlnew"
    app.workspace_path = Path("/home/zlnew")

    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "Hello assistant"
        await pilot.press("enter")
        await pilot.pause(0.5)

        # Confirm agent card exists and completed without errors
        from maulness.cli.tui import AgentCard
        agent_card = app.query_one(AgentCard)
        assert "Mock response content" in "".join(agent_card.message_text)
        assert "Completed" in str(agent_card.status_label.render())


@pytest.mark.asyncio
async def test_tui_dynamic_commands_discovery(tmp_path: Path):
    from maulness.core.pipelines import PipelineManager
    from maulness.core.profiles import ProfileManager

    # Create a custom YAML pipeline
    custom_pipe_yaml = """
name: my_custom_review
description: "My custom multi-stage review pipeline"
stages:
  - name: stage1
    profile: planner
    prompt: "Plan {title}"
"""
    (tmp_path / "my_custom_review.yaml").write_text(custom_pipe_yaml)

    app = MaulnessTUIApp()
    app.pipeline_manager = PipelineManager(pipelines_dir=tmp_path)
    commands = app.get_dynamic_commands()
    command_keys = [c[0] for c in commands]

    # Verify custom pipeline appears automatically in commands
    assert "/pipeline my_custom_review " in command_keys
    # Verify standard pipelines appear
    assert "/pipeline standard " in command_keys
    # Verify profiles appear dynamically
    assert "/profile builder" in command_keys
    assert "/profile planner" in command_keys


@pytest.mark.asyncio
async def test_tui_normal_mode_slash_triggers_autocomplete():
    from textual.widgets import OptionList

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert app.mode == "normal"

        # Press '/' in normal mode
        await pilot.press("slash")
        await pilot.pause()

        # Should enter insert mode and open floating LSP autocomplete popup
        assert app.mode == "insert"
        popup = app.query_one("#autocomplete-popup", OptionList)
        assert popup.has_class("visible")
        assert popup.option_count > 0

        # Press escape to dismiss popup
        await pilot.press("escape")
        await pilot.pause()
        assert not popup.has_class("visible")


@pytest.mark.asyncio
async def test_tui_new_slash_commands():
    from maulness.cli.tui import SystemCard

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input")

        # 1. Test /usage command
        inp.value = "/usage"
        await pilot.press("enter")
        await pilot.pause()
        cards = list(app.query(SystemCard))
        assert any("Session Metrics & Usage" in str(c.card_title) for c in cards)

        # 2. Test /context command
        inp.value = "/context"
        await pilot.press("enter")
        await pilot.pause()
        cards = list(app.query(SystemCard))
        assert any("Runtime & Workspace Context" in str(c.card_title) for c in cards)

        # 3. Test /compact command
        inp.value = "/compact"
        await pilot.press("enter")
        await pilot.pause()
        cards = list(app.query(SystemCard))
        assert any("Context Compacted" in str(c.card_title) for c in cards)

        # 4. Test /queue command
        app.is_busy = True
        inp.value = "/queue test queued prompt"
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.prompt_queue) == 1
        assert app.prompt_queue[0] == "test queued prompt"

        # 5. Test /new command resets session
        app.active_acp_session_id = "test_acp_session"
        inp.value = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.active_acp_session_id is None
        cards = list(app.query(SystemCard))
        assert any("New Session" in str(c.card_title) for c in cards)

        # 6. Verify unified command taxonomy
        cmd_keys = [c[0] for c in app.get_dynamic_commands()]
        assert "/new" in cmd_keys
        assert "/stop" in cmd_keys
        assert "/diff" not in cmd_keys
        assert "/cancel" not in cmd_keys
        assert "/clear" not in cmd_keys


@pytest.mark.asyncio
async def test_profile_workspace_configuration():
    from maulness.core.profiles import Profile, ProfileManager

    pm = ProfileManager()
    # Profile with inherit workspace
    p_inherit = Profile(identity={"name": "p_inherit"}, agent={"command": "agy"}, execution={"workspace": "inherit"})
    resolved_inherit = pm.resolve_workspace_for_profile(p_inherit, Path("/home/zlnew/www/personal"))
    assert resolved_inherit == Path("/home/zlnew/www/personal")

    # Profile with specific workspace repo
    p_custom = Profile(identity={"name": "p_custom"}, agent={"command": "agy"}, execution={"workspace": "repo/expense-tracker"})
    resolved_custom = pm.resolve_workspace_for_profile(p_custom, Path("/home/zlnew/www/personal"))
    assert resolved_custom.name == "expense-tracker"


@pytest.mark.asyncio
async def test_tui_task_cancellation():
    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        app.is_busy = True
        app._update_statusline()
        statusline = app.query_one("#vim-statusline")
        assert "BUSY" in str(statusline.render())
        assert "STOP" in str(statusline.render())

        # Press Ctrl+C while busy to cancel
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_busy is False
        assert "NORMAL" in str(statusline.render()) or "INSERT" in str(statusline.render())


@pytest.mark.asyncio
async def test_tui_prompt_queue_and_interrupt(monkeypatch):
    executed_prompts = []

    class MockProvider:
        async def run(self, prompt, **kwargs):
            executed_prompts.append(prompt)
            on_msg = kwargs.get("on_message")
            if on_msg:
                from maulness.core.models import AgentMessageEvent
                await on_msg(AgentMessageEvent(delta=f"Result for: {prompt}", session_id="test"))
            return f"Result for: {prompt}"

    from maulness.core import runner as runner_module
    monkeypatch.setattr(runner_module, "get_provider_for_profile", lambda p: MockProvider())

    app = MaulnessTUIApp(initial_profile="default")
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input")

        # 1. Test queue auto-dispatch
        app.prompt_queue.append("queued follow-up")
        inp.value = "first prompt"
        await pilot.press("enter")
        # Wait for first prompt to run and auto-drain queued prompt
        await pilot.pause(0.5)

        assert "first prompt" in executed_prompts
        assert "queued follow-up" in executed_prompts
        assert len(app.prompt_queue) == 0

        # 2. Test /interrupt command
        inp.value = "/interrupt steering prompt"
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert "steering prompt" in executed_prompts


@pytest.mark.asyncio
async def test_compact_sqlite_persistence(tmp_path: Path):
    from maulness.storage.db import StorageManager

    db_path = tmp_path / "test_maulness.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    # Verify saving and retrieving session memory
    row_id = await storage.save_session_memory(
        session_id="test_session_123",
        summary="Key decisions: Built TUI autocomplete, resolved per-profile workspace.",
        token_count=150,
    )
    assert row_id is not None

    latest = await storage.get_latest_session_memory("test_session_123")
    assert latest is not None
    assert "Built TUI autocomplete" in latest
