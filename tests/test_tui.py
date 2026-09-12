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
    app.profile_manager = ProfileManager(profiles_dir=tmp_path / "profiles_empty")
    commands = app.get_dynamic_commands()
    command_keys = [c[0] for c in commands]

    # Verify custom pipeline appears automatically in commands
    assert "/pipeline my_custom_review " in command_keys
    # Verify standard pipelines appear
    assert "/pipeline standard " in command_keys
    # Verify profiles appear dynamically (fallback to templates when dir is empty)
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


@pytest.mark.asyncio
async def test_tui_modals_direct_interaction(tmp_path: Path):
    from unittest.mock import MagicMock
    from maulness.core.models import ApprovalRequestEvent
    from maulness.cli.tui import (
        ApprovalModal,
        GateModal,
        HelpModal,
        ProfileModal,
        CommandPaletteModal,
        DiffModal,
    )
    from maulness.core.profiles import Profile

    # 1. ApprovalModal
    ev_cmd = ApprovalRequestEvent(request_id=1, call_id="c1", tool_name="run_command", args={"command": "git status", "cwd": "/tmp"}, session_id="s1")
    modal_cmd = ApprovalModal(ev_cmd)
    modal_cmd.dismiss = MagicMock()
    modal_cmd.action_allow()
    modal_cmd.dismiss.assert_called_with(True)
    modal_cmd.action_deny()
    modal_cmd.dismiss.assert_called_with(False)

    ev_file = ApprovalRequestEvent(request_id=2, call_id="c2", tool_name="write_file", args={"path": "foo.py", "bytes": 128}, session_id="s2")
    modal_file = ApprovalModal(ev_file)
    btn_ev_allow = MagicMock(button=MagicMock(id="btn-allow"))
    modal_file.dismiss = MagicMock()
    modal_file.on_button_pressed(btn_ev_allow)
    modal_file.dismiss.assert_called_with(True)

    ev_custom = ApprovalRequestEvent(request_id=3, call_id="c3", tool_name="custom", args={"flag": True}, session_id="s3")
    modal_custom = ApprovalModal(ev_custom)
    btn_ev_deny = MagicMock(button=MagicMock(id="btn-deny"))
    modal_custom.dismiss = MagicMock()
    modal_custom.on_button_pressed(btn_ev_deny)
    modal_custom.dismiss.assert_called_with(False)

    # 2. GateModal
    gate_modal = GateModal("Proceed to building stage?")
    gate_modal.dismiss = MagicMock()
    gate_modal.action_proceed()
    gate_modal.dismiss.assert_called_with(True)
    gate_modal.action_stop()
    gate_modal.dismiss.assert_called_with(False)
    btn_gate = MagicMock(button=MagicMock(id="btn-proceed"))
    gate_modal.on_button_pressed(btn_gate)
    gate_modal.dismiss.assert_called_with(True)

    # 3. HelpModal
    help_modal = HelpModal()
    help_modal.dismiss = MagicMock()
    help_modal.action_close()
    help_modal.dismiss.assert_called_with(None)
    help_modal.on_button_pressed(MagicMock())
    help_modal.dismiss.assert_called_with(None)

    # 4. ProfileModal
    p1 = Profile(identity={"name": "builder", "description": "Senior builder"}, agent={"command": "agy"})
    p2 = Profile(identity={"name": "reviewer", "description": "Code auditor"}, agent={"command": "agy"})
    prof_modal = ProfileModal(profiles=[p1, p2], current_profile="builder")
    prof_modal.dismiss = MagicMock()
    prof_modal.action_cancel()
    prof_modal.dismiss.assert_called_with(None)
    prof_modal.on_option_list_option_selected(MagicMock(option=MagicMock(id="reviewer")))
    prof_modal.dismiss.assert_called_with("reviewer")

    # 5. CommandPaletteModal
    cmds = [
        ("/new", "Start fresh session"),
        ("/pipeline standard", "Run standard pipeline"),
        ("/profile builder", "Switch to builder"),
    ]
    palette = CommandPaletteModal(commands=cmds, initial_query="/new")
    palette.dismiss = MagicMock()
    palette.action_cancel()
    palette.dismiss.assert_called_with(None)
    palette.on_option_list_option_selected(MagicMock(option=MagicMock(id="/new")))
    palette.dismiss.assert_called_with("/new")

    # 6. DiffModal
    diff_modal = DiffModal(workspace_path=tmp_path)
    diff_modal.dismiss = MagicMock()
    diff_modal.action_close()
    diff_modal.dismiss.assert_called_with(None)
    diff_modal.on_button_pressed(MagicMock())
    diff_modal.dismiss.assert_called_with(None)


@pytest.mark.asyncio
async def test_tui_cards_state_and_rendering():
    from maulness.cli.tui import AgentCard, PipelineCard

    # 1. AgentCard
    agent_card = AgentCard(profile_name="builder", model_info="ollama:qwen")
    agent_card.append_thought("Analyzing requirements...")
    assert "Analyzing requirements..." in agent_card.thought_text[0]

    # Fallback delta in thought
    agent_card.append_thought("Switching to fallback provider ollama")
    assert any("fallback" in t for t in agent_card.thought_text)

    # Tool call recording across different argument shapes
    agent_card.record_tool_call("run_command", {"command": "npm test"})
    agent_card.record_tool_call("run_command", {"CommandLine": "cargo test"})
    agent_card.record_tool_call("read_file", {"path": "src/lib.rs"})
    agent_card.record_tool_call("search", {"repo_path": "/home/user/repo"})
    agent_card.record_tool_call("edit", {"TargetFile": "/tmp/a.py"})
    agent_card.record_tool_call("view", {"AbsolutePath": "/var/log/syslog"})

    # Streaming message and final flush
    agent_card.append_message("Here is the solution.", force_render=True)
    agent_card.flush_final()
    agent_card.set_status("[green]Done[/green]")

    # 2. PipelineCard
    pipe_card = PipelineCard(pipeline_name="standard", goal="Implement feature")
    pipe_card.update_stage("planning", "planner", 1, 3)
    pipe_card.append_output("Created architectural plan.")
    pipe_card.finish("[green]Finished[/green]")


@pytest.mark.asyncio
async def test_tui_additional_commands_and_shell_execution(tmp_path: Path):
    from maulness.cli.tui import MaulnessTUIApp, SystemCard, UserCard
    from maulness.core.providers.circuit import ProviderHealthRegistry
    from textual.widgets import Input

    app = MaulnessTUIApp(initial_profile="default")
    app.workspace_path = tmp_path

    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input")

        # 1. /providers status and reset
        inp.value = "/providers"
        await pilot.press("enter")
        await pilot.pause()

        inp.value = "/providers reset"
        await pilot.press("enter")
        await pilot.pause()

        # 2. /yolo toggle
        inp.value = "/yolo"
        await pilot.press("enter")
        await pilot.pause()
        assert app.yolo_mode is True
        inp.value = "/yolo"
        await pilot.press("enter")
        await pilot.pause()
        assert app.yolo_mode is False

        # 3. /worktree toggle
        inp.value = "/worktree"
        await pilot.press("enter")
        await pilot.pause()
        assert app.use_worktree is True
        inp.value = "/worktree"
        await pilot.press("enter")
        await pilot.pause()
        assert app.use_worktree is False

        # 4. /profile builder switch by argument
        inp.value = "/profile builder"
        await pilot.press("enter")
        await pilot.pause()
        assert app.current_profile == "builder"

        # 5. Shell command escape: !echo "tui shell test"
        inp.value = '!echo "tui shell test"'
        await pilot.press("enter")
        await pilot.pause()
        sys_cards = list(app.query(SystemCard))
        assert any("tui shell test" in str(c.card_content) for c in sys_cards)

        # 6. /interrupt with missing prompt
        await app.on_input_submitted(Input.Submitted(inp, "/interrupt"))
        await pilot.pause()
        sys_cards = list(app.query(SystemCard))
        assert any("Usage: /interrupt <prompt>" in str(c.card_content) for c in sys_cards)

        # 7. /queue with missing prompt
        await app.on_input_submitted(Input.Submitted(inp, "/queue"))
        await pilot.pause()
        sys_cards = list(app.query(SystemCard))
        assert any("Usage: /queue <prompt>" in str(c.card_content) for c in sys_cards)

        # 8. /stop on idle
        inp.value = "/stop"
        await pilot.press("enter")
        await pilot.pause()
        sys_cards = list(app.query(SystemCard))
        assert any("No task is currently running" in str(c.card_content) for c in sys_cards)


def test_tui_helpers():
    from maulness.cli.tui import get_git_branch, get_daemon_status
    from unittest.mock import patch

    assert get_git_branch(Path("/nonexistent_path_404")) == "detached"
    with patch("subprocess.run", side_effect=Exception("Systemd fail")):
        assert get_daemon_status() == "[dim]standby[/dim]"


def test_agent_card_flush_thought_only():
    from maulness.cli.tui import AgentCard

    card = AgentCard(profile_name="default")
    card.append_thought("Deep thinking...")
    card.flush_final()
    assert "Thought for" in str(card.thought_static.render())


@pytest.mark.asyncio
async def test_tui_modals_mounted_lifecycle(tmp_path: Path):
    from unittest.mock import MagicMock
    from maulness.cli.tui import (
        MaulnessTUIApp,
        HelpModal,
        DiffModal,
        ProfileModal,
        CommandPaletteModal,
        ApprovalModal,
        GateModal,
    )
    from maulness.core.models import ApprovalRequestEvent

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        # 1. HelpModal
        app.action_show_help()
        await pilot.pause()
        assert isinstance(app.screen, HelpModal)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpModal)

        # 2. DiffModal
        app.action_view_diff()
        await pilot.pause()
        assert isinstance(app.screen, DiffModal)
        diff_modal = app.screen
        diff_modal.action_scroll_down()
        diff_modal.action_scroll_up()
        diff_modal.action_page_down()
        diff_modal.action_page_up()
        diff_modal.action_scroll_end()
        diff_modal.action_scroll_home()
        await pilot.press("q")
        await pilot.pause()
        assert not isinstance(app.screen, DiffModal)

        # 3. ProfileModal
        app.action_select_profile()
        await pilot.pause()
        assert isinstance(app.screen, ProfileModal)
        # Select profile option to cover lines 1171-1175 and dismiss
        app.screen.dismiss("builder")
        await pilot.pause()
        assert not isinstance(app.screen, ProfileModal)
        assert app.current_profile == "builder"

        # 4. CommandPaletteModal
        app.push_screen(
            CommandPaletteModal(commands=app.get_dynamic_commands(), initial_query="new")
        )
        await pilot.pause()
        assert isinstance(app.screen, CommandPaletteModal)
        pal = app.screen
        pal.on_input_changed(MagicMock(value="pipeline"))
        await pilot.press("down")
        await pilot.press("up")
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, CommandPaletteModal)

        # 5. ApprovalModal
        ev = ApprovalRequestEvent(
            request_id=1,
            call_id="c1",
            tool_name="write_file",
            args={"path": "a.txt", "bytes": 10},
            session_id="s",
        )
        app.push_screen(ApprovalModal(ev))
        await pilot.pause()
        assert isinstance(app.screen, ApprovalModal)
        await pilot.press("y")
        await pilot.pause()
        assert not isinstance(app.screen, ApprovalModal)

        # 6. GateModal
        app.push_screen(GateModal("Confirm proceed?"))
        await pilot.pause()
        assert isinstance(app.screen, GateModal)
        await pilot.press("y")
        await pilot.pause()
        assert not isinstance(app.screen, GateModal)


@pytest.mark.asyncio
async def test_tui_autocomplete_advanced_branches():
    from unittest.mock import MagicMock
    from textual.widgets import OptionList, Input
    from textual.widgets.option_list import Option

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        popup = app.query_one("#autocomplete-popup", OptionList)
        inp = app.query_one("#chat-input", Input)

        # 1. /pipeline sub-command completion (lines 944-955)
        inp.value = "/pipeline "
        await pilot.pause()
        assert popup.has_class("visible")

        # 2. /profile sub-command completion (lines 958-970)
        inp.value = "/profile "
        await pilot.pause()
        assert popup.has_class("visible")

        # Non-matching profile arg -> hides popup (lines 969-970)
        inp.value = "/profile nonexistent_profile_xyz"
        await pilot.pause()
        assert not popup.has_class("visible")

        # 3. Non-matching unknown slash command (lines 987-988)
        inp.value = "/unknown_nonexistent_command_123"
        await pilot.pause()
        assert not popup.has_class("visible")

        # 4. Key navigation inside autocomplete: ctrl+n and ctrl+p (lines 1062, 1067-1072)
        inp.value = "/"
        await pilot.pause()
        assert popup.has_class("visible")
        await pilot.press("ctrl+n")
        await pilot.press("ctrl+p")
        await pilot.press("escape")
        await pilot.pause()
        assert not popup.has_class("visible")

        # 5. Tab / ctrl+k in insert mode updates autocomplete (lines 1146-1148)
        inp.value = "/con"
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert popup.has_class("visible")

        # 6. Option selected event directly (lines 1049-1050)
        opt = Option("label", id="/context")
        app.on_option_list_option_selected(MagicMock(option_list=popup, option=opt))


@pytest.mark.asyncio
async def test_tui_normal_mode_and_commands_advanced():
    from textual.widgets import Input
    from maulness.cli.tui import SystemCard

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input", Input)

        # 1. Normal mode keys: d, u, p, ?, q (lines 1106, 1108, 1130-1135)
        await pilot.press("escape")
        await pilot.pause()
        assert app.mode == "normal"

        await pilot.press("d", "u")  # page down / page up
        await pilot.pause()

        # Press 'p' to open ProfileModal, then escape
        await pilot.press("p")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # Press '?' to open HelpModal, then escape
        await pilot.press("question_mark")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # Return to insert
        await pilot.press("i")
        await pilot.pause()

        # 2. Empty input submitted (line 1186)
        await app.on_input_submitted(Input.Submitted(inp, "   "))

        # 3. Busy state prompts (lines 1212-1213, 1231-1233, 1374-1386)
        app.is_busy = True
        inp.value = "/stop"
        await pilot.press("enter")
        await pilot.pause()

        app.is_busy = True
        inp.value = "/interrupt new direction"
        await pilot.press("enter")
        await pilot.pause()

        app.is_busy = True
        inp.value = "some regular prompt while busy"
        await pilot.press("enter")
        await pilot.pause()
        cards = list(app.query(SystemCard))
        assert any("Busy" in str(c.card_title) for c in cards)
        app.is_busy = False

        # 4. /queue when not busy (lines 1260-1261)
        inp.value = "/queue prompt when idle"
        await pilot.press("enter")
        await pilot.pause()

        # 5. /help, /profile without args, exit (lines 1414-1425, 1390-1391)
        inp.value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        inp.value = "/profile"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # 6. Shell execution error (lines 1444-1445)
        inp.value = "!exit 42"
        await pilot.press("enter")
        await pilot.pause()

        # 7. action_quit_app (line 1155)
        app.action_quit_app()


@pytest.mark.asyncio
async def test_tui_execute_direct_and_pipeline_full_lifecycle(tmp_path: Path):
    from unittest.mock import AsyncMock, MagicMock, patch
    from maulness.cli.tui import MaulnessTUIApp, PipelineCard
    from maulness.core.models import (
        AgentThoughtEvent,
        AgentMessageEvent,
        AgentToolCallEvent,
        ApprovalRequestEvent,
        TaskStatus,
    )
    from maulness.core.pipelines import PipelineStage

    app = MaulnessTUIApp(initial_profile="default")
    app.workspace_path = tmp_path

    # 1. _execute_direct callbacks
    async def mock_run_direct(*args, **kwargs):
        on_init = kwargs.get("on_init")
        if on_init:
            await on_init("conv_tui_123")
        on_thought = kwargs.get("on_thought")
        if on_thought:
            await on_thought(AgentThoughtEvent(delta="Thinking...", session_id="s"))
        on_tool_call = kwargs.get("on_tool_call")
        if on_tool_call:
            await on_tool_call(
                AgentToolCallEvent(call_id="c1", tool_name="run_command", args={"command": "ls"}, session_id="s")
            )
        on_message = kwargs.get("on_message")
        if on_message:
            await on_message(AgentMessageEvent(delta="Direct result", session_id="s"))
        on_approval = kwargs.get("on_approval")
        if on_approval:
            with patch.object(app, "push_screen_wait", AsyncMock(return_value=True)):
                await on_approval(
                    ApprovalRequestEvent(request_id=1, call_id="c1", tool_name="run_command", args={}, session_id="s")
                )
        return MagicMock(status=TaskStatus.DONE)

    app.runner.run_direct = AsyncMock(side_effect=mock_run_direct)

    # Pre-populate prompt_queue so line 1532-1534 runs
    app.prompt_queue = ["Queued direct task"]

    async with app.run_test() as pilot:
        inp = app.query_one("#chat-input")
        inp.value = "Direct task trigger"
        await pilot.press("enter")
        await pilot.pause(0.5)

    # 2. _execute_direct CancelledError and Exception
    app_cancel = MaulnessTUIApp(initial_profile="default")
    app_cancel.workspace_path = tmp_path
    app_cancel.runner.run_direct = AsyncMock(side_effect=asyncio.CancelledError())
    async with app_cancel.run_test() as pilot:
        inp = app_cancel.query_one("#chat-input")
        inp.value = "Cancel test"
        await pilot.press("enter")
        await pilot.pause(0.2)

    app_err = MaulnessTUIApp(initial_profile="default")
    app_err.workspace_path = tmp_path
    app_err.runner.run_direct = AsyncMock(side_effect=RuntimeError("Direct run failure"))
    async with app_err.run_test() as pilot:
        inp = app_err.query_one("#chat-input")
        inp.value = "Err test"
        await pilot.press("enter")
        await pilot.pause(0.2)

    # 3. _execute_pipeline lifecycle (lines 1451-1460, 1537-1600)
    app_pipe = MaulnessTUIApp(initial_profile="default")
    app_pipe.workspace_path = tmp_path

    async def mock_run_pipeline(*args, **kwargs):
        on_stage_start = kwargs.get("on_stage_start")
        if on_stage_start:
            await on_stage_start(PipelineStage(name="planning", profile="planner", prompt="Plan"), 1, 3)
        on_thought = kwargs.get("on_thought")
        if on_thought:
            await on_thought(AgentThoughtEvent(delta="Stage plan thinking", session_id="s"))
        on_message = kwargs.get("on_message")
        if on_message:
            await on_message(AgentMessageEvent(delta="Stage plan message", session_id="s"))
        on_gate = kwargs.get("on_gate")
        if on_gate:
            with patch.object(app_pipe, "push_screen_wait", AsyncMock(return_value=True)):
                await on_gate("Proceed?")
        on_approval = kwargs.get("on_approval")
        if on_approval:
            with patch.object(app_pipe, "push_screen_wait", AsyncMock(return_value=True)):
                await on_approval(
                    ApprovalRequestEvent(request_id=1, call_id="c1", tool_name="tool", args={}, session_id="s")
                )
        return MagicMock(status=TaskStatus.DONE)

    app_pipe.orchestrator.run_pipeline = AsyncMock(side_effect=mock_run_pipeline)
    app_pipe.runner.run_direct = AsyncMock(return_value=MagicMock(status=TaskStatus.DONE))
    app_pipe.prompt_queue = ["Queued after pipeline"]

    async with app_pipe.run_test() as pilot:
        inp = app_pipe.query_one("#chat-input")
        inp.value = "/pipeline standard Implement login feature"
        await pilot.press("enter")
        await pilot.pause(0.5)

        p_card = app_pipe.query_one(PipelineCard)
        assert "Finished" in str(p_card.status_label.render())

    # 4. _execute_pipeline CancelledError and Exception
    app_pipe_cancel = MaulnessTUIApp(initial_profile="default")
    app_pipe_cancel.workspace_path = tmp_path
    app_pipe_cancel.orchestrator.run_pipeline = AsyncMock(side_effect=asyncio.CancelledError())
    async with app_pipe_cancel.run_test() as pilot:
        inp = app_pipe_cancel.query_one("#chat-input")
        inp.value = "/pipeline standard Cancel pipe"
        await pilot.press("enter")
        await pilot.pause(0.2)

    app_pipe_err = MaulnessTUIApp(initial_profile="default")
    app_pipe_err.workspace_path = tmp_path
    app_pipe_err.orchestrator.run_pipeline = AsyncMock(side_effect=RuntimeError("Pipeline crash"))
    async with app_pipe_err.run_test() as pilot:
        inp = app_pipe_err.query_one("#chat-input")
        inp.value = "/pipeline standard Error pipe"
        await pilot.press("enter")
        await pilot.pause(0.2)


@pytest.mark.asyncio
async def test_tui_complete_coverage_edges(tmp_path):
    from unittest.mock import MagicMock, patch, AsyncMock
    from textual.widgets import OptionList, Input
    from textual.widgets.option_list import Option
    from textual import events
    from maulness.cli.tui import (
        MaulnessTUIApp,
        CommandPaletteModal,
        ApprovalModal,
        SystemCard,
    )
    from maulness.core.models import ApprovalRequestEvent, TaskStatus

    # 1. CommandPaletteModal edge branches (lines 103, 109, 131, 141-147)
    commands = [("/pipeline", "Run pipeline"), ("/profile", "Switch profile"), ("/new", "New session")]
    pal = CommandPaletteModal(commands=commands, initial_query="")

    app = MaulnessTUIApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(pal)
        await pilot.pause()

        pal._populate_options("")
        pal._populate_options("eline")
        ol = pal.query_one("#palette-options", OptionList)
        ol.highlighted = None
        ev_down = events.Key(key="down", character=None)
        pal.on_key(ev_down)
        assert ol.highlighted == 0

        ev_enter = events.Key(key="enter", character=None)
        pal.on_key(ev_enter)
        await pilot.pause()

        pal2 = CommandPaletteModal(commands=commands, initial_query="")
        app.push_screen(pal2)
        await pilot.pause()
        ol2 = pal2.query_one("#palette-options", OptionList)
        ol2.highlighted = None
        ev_tab = events.Key(key="tab", character=None)
        pal2.on_key(ev_tab)
        await pilot.pause()

        # 2. ApprovalModal edge branches (lines 179-183, 190-194)
        ev_cmd = ApprovalRequestEvent(
            request_id=1, call_id="c1", tool_name="run_command", args={"command": "ls -la", "cwd": "/tmp"}, session_id="s"
        )
        modal_cmd = ApprovalModal(ev_cmd)
        app.push_screen(modal_cmd)
        await pilot.pause()
        modal_cmd.dismiss(True)
        await pilot.pause()

        class Unserializable:
            pass

        ev_custom = ApprovalRequestEvent(
            request_id=2, call_id="c2", tool_name="custom_tool", args={"key": "val", "bad": Unserializable()}, session_id="s"
        )
        modal_custom = ApprovalModal(ev_custom)
        app.push_screen(modal_custom)
        await pilot.pause()
        modal_custom.dismiss(False)
        await pilot.pause()

    # 3. Exception pass blocks in app methods (lines 904-905, 912-913, 936-937, 1082-1083, 1207-1208)
    app2 = MaulnessTUIApp()
    async with app2.run_test() as pilot:
        await pilot.pause()
        with patch.object(app2, "query_one", side_effect=RuntimeError("DOM query fail")):
            app2.set_mode("normal")
            app2._update_statusline()
            app2._update_top_bar()
            app2.on_key(events.Key(key="a", character="a"))

            def mock_push_screen(screen, callback=None):
                if callback:
                    callback("planner")

            with patch.object(app2, "push_screen", side_effect=mock_push_screen):
                app2.action_select_profile()

    # 4. _apply_autocomplete when popup.highlighted is None (line 1032)
    app3 = MaulnessTUIApp()
    async with app3.run_test() as pilot:
        await pilot.pause()
        popup = app3.query_one("#autocomplete-popup", OptionList)
        popup.add_class("visible")
        popup.add_option(Option("test", id="test"))
        popup.highlighted = None
        assert app3._apply_autocomplete() is False

        ev_down = events.Key(key="down", character=None)
        app3.on_key(ev_down)
        assert popup.highlighted == 0

        app3.set_mode("normal")
        ev_q = events.Key(key="q", character="q")
        with patch.object(app3, "exit") as mock_exit:
            app3.on_key(ev_q)
            mock_exit.assert_called_once()

        import time
        app3._last_g_time = time.time()
        ev_g = events.Key(key="g", character="g")
        app3.on_key(ev_g)
        assert app3._last_g_time == 0.0

        app3.set_mode("insert")
        popup.add_class("visible")
        ev_esc = events.Key(key="escape", character=None)
        app3.on_key(ev_esc)
        assert not popup.has_class("visible")

    # 5. Unmounted app on_key exception (lines 1081-1082)
    app_unmounted = MaulnessTUIApp()
    app_unmounted.on_key(events.Key(key="a", character="a"))

    # 6. Submitted slash commands: /exit, /help, /profile, /pipeline fallback, shell error
    app4 = MaulnessTUIApp()
    app4.workspace_path = tmp_path
    app4.orchestrator.run_pipeline = AsyncMock(return_value=MagicMock(status=TaskStatus.DONE))
    async with app4.run_test() as pilot:
        await pilot.pause()
        inp = app4.query_one("#chat-input", Input)

        with patch.object(app4, "exit") as mock_exit:
            await app4.on_input_submitted(Input.Submitted(inp, "/exit"))
            mock_exit.assert_called_once()

        with patch.object(app4, "action_show_help") as mock_help:
            await app4.on_input_submitted(Input.Submitted(inp, "/help"))
            mock_help.assert_called_once()

        with patch.object(app4, "action_select_profile") as mock_sel_prof:
            await app4.on_input_submitted(Input.Submitted(inp, "/profile"))
            mock_sel_prof.assert_called_once()

        await app4.on_input_submitted(Input.Submitted(inp, "/profile planner"))
        assert app4.current_profile == "planner"

        with patch("subprocess.run", side_effect=OSError("Command spawn failed")):
            await app4.on_input_submitted(Input.Submitted(inp, "!failing_binary"))

        await app4.on_input_submitted(Input.Submitted(inp, "/pipeline build a whole website"))
        await pilot.pause(0.2)


