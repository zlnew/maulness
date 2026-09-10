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
async def test_tui_command_palette():
    app = MaulnessTUIApp(initial_profile="builder")
    async with app.run_test() as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert app.mode == "normal"

        # Press '/' to open command palette
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteModal)

        # Filter for diff
        await pilot.press("d", "i", "f", "f")
        await pilot.pause()
        # Hit Enter to select /diff
        await pilot.press("enter")
        await pilot.pause()

        # Should have opened DiffModal
        assert isinstance(app.screen, DiffModal)

        # Close diff modal with Esc
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, DiffModal)


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
async def test_tui_typing_slash_opens_palette():
    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        # In insert mode, typing '/' triggers the command palette
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteModal)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, CommandPaletteModal)


@pytest.mark.asyncio
async def test_tui_task_cancellation():
    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        app.is_busy = True
        app._update_statusline()
        statusline = app.query_one("#vim-statusline")
        assert "BUSY" in str(statusline.render())
        assert "CANCEL" in str(statusline.render())

        # Press Ctrl+C while busy to cancel
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_busy is False
        assert "NORMAL" in str(statusline.render()) or "INSERT" in str(statusline.render())
