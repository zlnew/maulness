import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.pipeline import PipelineOrchestrator
from maulness.core.pipelines import PipelineManager, PipelineStage
from maulness.core.profiles import ProfileManager
from maulness.core.runner import TaskRunner

logger = logging.getLogger("maulness.cli.tui")


def get_git_branch(workspace_path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "detached"


def get_daemon_status() -> str:
    try:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", "maulness.service"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.stdout.strip() == "active":
            return "[#a6e3a1]● active[/#a6e3a1]"
    except Exception:
        pass
    return "[#6c7086]○ standby[/#6c7086]"


# ==============================================================================
# Modal Dialogs (All centered)
# ==============================================================================
class CommandPaletteModal(ModalScreen[Optional[str]]):
    """Centered Neovim Telescope-style Command Palette for slash commands & actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, commands: list[tuple[str, str]], initial_query: str = ""):
        super().__init__()
        self.commands = commands
        self.initial_query = initial_query.lstrip("/")

    def compose(self) -> ComposeResult:
        with Container(classes="palette-dialog"):
            yield Label("[bold #bb9af7]󰍉 Command Palette[/bold #bb9af7] [dim #737aa2](Type to filter, ↑/↓ to navigate, Enter/Tab to select, Esc to close)[/dim #737aa2]", classes="palette-title")
            yield Input(value=self.initial_query, placeholder="Filter commands (/pipeline, /profile, /diff, !<cmd>)...", id="palette-filter")
            ol = OptionList(id="palette-options")
            yield ol

    def on_mount(self) -> None:
        self._populate_options(self.initial_query)
        self.query_one("#palette-filter", Input).focus()

    def _populate_options(self, query: str) -> None:
        ol = self.query_one("#palette-options", OptionList)
        ol.clear_options()
        q = query.lower().strip()

        scored = []
        for cmd, desc in self.commands:
            cmd_clean = cmd.strip()
            if not q:
                scored.append((0, cmd, desc))
            elif cmd_clean.lower() == f"/{q}" or cmd_clean.lower() == q:
                scored.append((100, cmd, desc))
            elif cmd_clean.lower().startswith(f"/{q}") or cmd_clean.lower().startswith(q):
                scored.append((80, cmd, desc))
            elif q in cmd.lower():
                scored.append((50, cmd, desc))
            elif q in desc.lower():
                scored.append((10, cmd, desc))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, cmd, desc in scored:
            prompt_markup = f"[bold #7aa2f7]{cmd}[/bold #7aa2f7] [dim #737aa2]— {desc}[/dim #737aa2]"
            ol.add_option(Option(prompt_markup, id=cmd))

        if ol.option_count > 0:
            ol.highlighted = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate_options(event.value.strip())

    def on_key(self, event: events.Key) -> None:
        ol = self.query_one("#palette-options", OptionList)
        if event.key in ("down", "ctrl+n"):
            event.prevent_default()
            event.stop()
            if ol.option_count > 0:
                if ol.highlighted is None:
                    ol.highlighted = 0
                elif ol.highlighted < ol.option_count - 1:
                    ol.highlighted += 1
        elif event.key in ("up", "ctrl+p"):
            event.prevent_default()
            event.stop()
            if ol.option_count > 0:
                if ol.highlighted is not None and ol.highlighted > 0:
                    ol.highlighted -= 1
        elif event.key in ("enter", "tab"):
            event.prevent_default()
            event.stop()
            if ol.highlighted is not None and ol.option_count > 0:
                opt = ol.get_option_at_index(ol.highlighted)
                self.dismiss(str(opt.id))
            else:
                self.dismiss(None)
        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalModal(ModalScreen[bool]):
    """Centered HITL Tool Approval Modal Dialog."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("enter", "allow", "Allow"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, event: ApprovalRequestEvent):
        super().__init__()
        self.event = event

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("[bold #e0af68]⚠️  HITL Tool Approval Required[/bold #e0af68]", classes="modal-title")
            yield Label(f"[bold #7aa2f7]Tool:[/bold #7aa2f7] {self.event.tool_name}", classes="modal-row")
            try:
                formatted_args = json.dumps(self.event.args, indent=2)
            except Exception:
                formatted_args = str(self.event.args)
            yield Static(Syntax(formatted_args, "json", theme="monokai"), classes="args-box")
            with Horizontal(classes="modal-buttons"):
                yield Button("Allow Once (y)", variant="success", id="btn-allow")
                yield Button("Deny (n)", variant="error", id="btn-deny")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-allow")


class GateModal(ModalScreen[bool]):
    """Centered Pipeline Stage Confirmation Gate Modal."""

    BINDINGS = [
        Binding("y", "proceed", "Proceed"),
        Binding("n", "stop", "Stop"),
        Binding("enter", "proceed", "Proceed"),
        Binding("escape", "stop", "Stop"),
    ]

    def __init__(self, prompt: str):
        super().__init__()
        self.gate_prompt = prompt

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("[bold #bb9af7]🚦 Pipeline Confirmation Gate[/bold #bb9af7]", classes="modal-title")
            yield Label(self.gate_prompt, classes="modal-row")
            with Horizontal(classes="modal-buttons"):
                yield Button("Proceed (y)", variant="primary", id="btn-proceed")
                yield Button("Stop (n)", variant="default", id="btn-stop")

    def action_proceed(self) -> None:
        self.dismiss(True)

    def action_stop(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-proceed")


class DiffModal(ModalScreen[None]):
    """Centered Neovim-styled scrollable git diff inspector."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
        Binding("j", "scroll_down", "Scroll Down"),
        Binding("k", "scroll_up", "Scroll Up"),
        Binding("d", "page_down", "Half Page Down"),
        Binding("u", "page_up", "Half Page Up"),
        Binding("G", "scroll_end", "Scroll End"),
        Binding("g", "scroll_home", "Scroll Home"),
    ]

    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path

    def compose(self) -> ComposeResult:
        with Container(classes="diff-dialog"):
            yield Label("[bold #7aa2f7]🔍 Git Diff HEAD[/bold #7aa2f7] [dim #737aa2](j/k to scroll, q or Esc to close)[/dim #737aa2]")
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
            )
            diff_text = res.stdout.strip() or "(No uncommitted diffs detected in current workspace)"
            with VerticalScroll(id="diff-scroll-view"):
                yield Static(Syntax(diff_text, "diff", theme="monokai"))
            with Horizontal(classes="modal-buttons"):
                yield Button("Close (q / Esc)", variant="default", id="btn-close")

    def action_close(self) -> None:
        self.dismiss(None)

    def action_scroll_down(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_up()

    def action_page_down(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_page_down()

    def action_page_up(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_page_up()

    def action_scroll_end(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_end()

    def action_scroll_home(self) -> None:
        self.query_one("#diff-scroll-view", VerticalScroll).scroll_home()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class ProfileModal(ModalScreen[Optional[str]]):
    """Centered Profile selector modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, profiles: list, current_profile: str):
        super().__init__()
        self.profiles = profiles
        self.current_profile = current_profile

    def compose(self) -> ComposeResult:
        with Container(classes="palette-dialog"):
            yield Label("[bold #bb9af7]Select Active Profile[/bold #bb9af7] [dim #737aa2](Enter to switch, Esc to cancel)[/dim #737aa2]", classes="palette-title")
            ol = OptionList(id="profile-options")
            yield ol

    def on_mount(self) -> None:
        ol = self.query_one("#profile-options", OptionList)
        for idx, p in enumerate(self.profiles):
            is_active = " [bold #a6e3a1]● ACTIVE[/bold #a6e3a1]" if p.name == self.current_profile else ""
            label = f"[bold #7aa2f7]{p.name}[/bold #7aa2f7] ({p.provider}) — [dim #737aa2]{p.description[:50]}[/dim #737aa2]{is_active}"
            ol.add_option(Option(label, id=p.name))
            if p.name == self.current_profile:
                ol.highlighted = idx
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpModal(ModalScreen[None]):
    """Centered Help overlay modal."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        help_markdown = """
# Maulness Terminal TUI — Neovim Keybindings & Commands

### Neovim Modal Navigation
- **NORMAL Mode** (Green status badge):
  - `i` or `a` : Enter **INSERT** mode (focus input bar)
  - `/` : Open **Command Palette** (centered Telescope-style slash picker)
  - `j` / `k` : Scroll chat view down / up
  - `d` / `u` : Half-page scroll down / up (`Ctrl+D` / `Ctrl+U`)
  - `G` : Scroll to bottom of chat
  - `gg` : Scroll to top of chat
  - `: ` or `!` : Enter INSERT mode pre-filled with `! ` (shell command)
  - `p` : Open Profile Switcher modal
  - `?` : Open this Help reference
  - `q` : Quit Maulness

- **INSERT Mode** (Blue status badge):
  - Type prompt naturally (executed by active profile)
  - `/` : Opens centered Autocomplete Command Palette immediately
  - `Escape` : Exit to **NORMAL** mode (or stop/cancel active task)
  - `Tab` or `Ctrl+K` : Open Command Palette
  - `Enter` : Submit prompt / execute command
  - `Ctrl+C` : Immediately cancel and stop running task

### Dynamic Slash Commands (Loaded from YAML)
- `/pipeline <name> <goal>` : Execute declarative pipeline (`standard`, `quick`, `plan_only`, `audit`, or custom)
- `/profile <name>` : Switch active profile (`builder`, `planner`, `reviewer`, `default`, or custom)
- `/diff` : Centered full-screen scrollable git diff inspector
- `!<command>` : Execute local workspace shell command (e.g. `!git status`, `!pytest`)
- `/clear` : Clear chat transcript
- `/exit` : Quit
"""
        with Container(classes="diff-dialog"):
            with VerticalScroll():
                yield Static(RichMarkdown(help_markdown))
            with Horizontal(classes="modal-buttons"):
                yield Button("Close (q / Esc)", variant="default", id="btn-close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


# ==============================================================================
# Message Cards
# ==============================================================================
class UserCard(Static):
    """Card displaying user prompt."""

    def __init__(self, prompt: str, repo: str, branch: str):
        super().__init__(classes="user-card")
        self.prompt = prompt
        self.repo = repo
        self.branch = branch

    def compose(self) -> ComposeResult:
        yield Label(f"[bold #7aa2f7]Maul[/bold #7aa2f7] [dim #565f89]({self.repo}:{self.branch})[/dim #565f89]", classes="card-header")
        yield Static(self.prompt, classes="card-body")


class SystemCard(Static):
    """Card displaying system messages or shell command execution."""

    def __init__(self, title: str, content: str, is_error: bool = False):
        super().__init__(classes="system-card")
        self.card_title = title
        self.card_content = content
        self.is_error = is_error

    def compose(self) -> ComposeResult:
        color = "#f7768e" if self.is_error else "#e0af68"
        yield Label(f"[bold {color}]{self.card_title}[/bold {color}]", classes="card-header")
        yield Static(self.card_content, classes="card-body")


class AgentCard(Static):
    """Card displaying streaming thought, tool calls, and assistant response."""

    def __init__(self, profile_name: str):
        super().__init__(classes="agent-card")
        self.profile_name = profile_name
        self.thought_text: list[str] = []
        self.message_text: list[str] = []
        self.status_label = Label("[dim #737aa2]󰑮 Thinking...[/dim #737aa2]")
        self.thought_static = Static("", classes="thought-box")
        self.message_static = Static("", classes="card-body")

    def compose(self) -> ComposeResult:
        yield Label(f"[bold #bb9af7]Maulness[/bold #bb9af7] [dim #565f89]({self.profile_name})[/dim #565f89]", classes="card-header")
        yield self.thought_static
        yield self.message_static
        yield self.status_label

    def append_thought(self, delta: str) -> None:
        self.thought_text.append(delta)
        text = "".join(self.thought_text)
        self.thought_static.update(f"[dim italic #737aa2]💭 {text[-200:].strip()}[/dim italic #737aa2]")

    def append_message(self, delta: str) -> None:
        self.message_text.append(delta)
        text = "".join(self.message_text)
        self.message_static.update(RichMarkdown(text))

    def set_status(self, status: str) -> None:
        self.status_label.update(status)


class PipelineCard(Static):
    """Card displaying declarative pipeline multi-stage progress."""

    def __init__(self, pipeline_name: str, goal: str):
        super().__init__(classes="agent-card")
        self.pipeline_name = pipeline_name
        self.goal = goal
        self.stages_static = Static("")
        self.status_label = Label("[dim #737aa2]󰑮 Initializing pipeline...[/dim #737aa2]")
        self.output_static = Static("", classes="card-body")

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold #bb9af7]Pipeline: {self.pipeline_name}[/bold #bb9af7] — [dim #7aa2f7]{self.goal[:60]}[/dim #7aa2f7]",
            classes="card-header",
        )
        yield self.stages_static
        yield self.output_static
        yield self.status_label

    def update_stage(self, stage_name: str, profile: str, current: int, total: int) -> None:
        self.stages_static.update(
            f"[bold #e0af68]▶ Stage {current}/{total}:[/bold #e0af68] [bold #c0caf5]{stage_name}[/bold #c0caf5] ([#bb9af7]{profile}[/#bb9af7])"
        )
        self.status_label.update(f"[dim #737aa2]Running stage {current} of {total}...[/dim #737aa2]")

    def append_output(self, delta: str) -> None:
        self.output_static.update(RichMarkdown(delta))

    def finish(self, status: str) -> None:
        self.status_label.update(status)


# ==============================================================================
# Main Textual App
# ==============================================================================
class MaulnessTUIApp(App):
    """Full-screen Neovim-styled terminal harness for Maulness."""

    TITLE = "Maulness ⚡"
    CSS = """
    Screen {
        background: #1a1b26;
        color: #c0caf5;
    }

    ModalScreen {
        align: center middle;
    }

    #top-bar {
        dock: top;
        height: 3;
        background: #16161e;
        color: #a9b1d6;
        padding: 0 2;
        border-bottom: solid #24283b;
    }

    #chat-view {
        height: 1fr;
        padding: 1 2;
        background: #1a1b26;
    }

    .user-card {
        background: #24283b;
        border: round #7aa2f7;
        padding: 1 2;
        margin: 1 0;
    }

    .agent-card {
        background: #1f2335;
        border: round #bb9af7;
        padding: 1 2;
        margin: 1 0;
    }

    .system-card {
        background: #1f2335;
        border: round #414868;
        padding: 1 2;
        margin: 1 0;
    }

    .card-header {
        margin-bottom: 1;
    }

    .card-body {
        margin-top: 0;
    }

    .thought-box {
        color: #737aa2;
        background: #16161e;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick #414868;
    }

    #bottom-container {
        dock: bottom;
        height: 6;
        background: #16161e;
        border-top: solid #24283b;
        padding: 0 1;
    }

    #vim-statusline {
        height: 1;
        background: #16161e;
        color: #a9b1d6;
        padding: 0 1;
    }

    #input-row {
        height: 4;
        background: #1f2335;
        border: tall #414868;
        padding: 0 1;
    }

    #input-row.focused-insert {
        border: tall #7aa2f7;
    }

    #chat-input {
        width: 1fr;
        background: transparent;
        border: none;
        color: #c0caf5;
    }

    #chat-input:focus {
        border: none;
    }

    /* Modals (All Centered) */
    .palette-dialog {
        width: 75%;
        max-height: 22;
        background: #1f2335;
        border: double #bb9af7;
        padding: 1 2;
    }

    .palette-title {
        margin-bottom: 1;
    }

    #palette-filter {
        background: #16161e;
        border: tall #414868;
        color: #c0caf5;
        margin-bottom: 1;
    }

    #palette-options {
        max-height: 12;
        background: #16161e;
        border: tall #24283b;
    }

    .modal-dialog {
        width: 70%;
        max-height: 80%;
        background: #1f2335;
        border: thick #bb9af7;
        padding: 1 2;
    }

    .diff-dialog {
        width: 90%;
        height: 90%;
        background: #16161e;
        border: thick #7aa2f7;
        padding: 1 2;
    }

    .modal-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .modal-row {
        margin-bottom: 1;
    }

    .args-box {
        max-height: 12;
        overflow-y: scroll;
        margin-bottom: 1;
    }

    #diff-scroll-view {
        height: 1fr;
        margin: 1 0;
    }

    .modal-buttons {
        height: 3;
        align: right middle;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "select_profile", "Profile"),
        Binding("ctrl+d", "view_diff", "Diff"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("f1", "show_help", "Help"),
    ]

    def __init__(self, initial_profile: str = "builder"):
        super().__init__()
        self.repo_name, self.workspace_path = config.get_current_workspace()
        self.current_profile = initial_profile
        self.profile_manager = ProfileManager()
        self.pipeline_manager = PipelineManager()
        self.runner = TaskRunner(profile_manager=self.profile_manager)
        self.orchestrator = PipelineOrchestrator(
            profile_manager=self.profile_manager,
            pipeline_manager=self.pipeline_manager,
        )
        self.is_busy = False
        self.mode = "insert"  # "normal" or "insert"
        self._last_g_time = 0.0
        self.active_worker = None

    def compose(self) -> ComposeResult:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()

        yield Static(
            f"[bold #f7768e]⚡ MAULNESS[/bold #f7768e] │ [bold #a6e3a1]repo:[/bold #a6e3a1] {self.repo_name} │ "
            f"[bold #89dceb]branch:[/bold #89dceb] {branch} │ [bold #cba6f7]profile:[/bold #cba6f7] {self.current_profile} │ "
            f"{daemon_str}",
            id="top-bar",
        )

        with VerticalScroll(id="chat-view"):
            yield Static(
                f"[dim #737aa2]Welcome back, Maul. Scoped to [bold #a6e3a1]{self.workspace_path}[/bold #a6e3a1].\n"
                f"• Natural language executes with [bold #cba6f7]{self.current_profile}[/bold #cba6f7].\n"
                f"• Press [bold #7aa2f7]/[/bold #7aa2f7] or [bold #7aa2f7]Tab[/bold #7aa2f7] for centered Command Palette.\n"
                f"• Press [bold #7aa2f7]Esc[/bold #7aa2f7] for NORMAL mode ([#a6e3a1]j/k[/#a6e3a1] scroll, [#a6e3a1]i[/#a6e3a1] insert, [#a6e3a1]?[/#a6e3a1] help).\n"
                f"• Press [bold #f7768e]Ctrl+C[/bold #f7768e] or [bold #f7768e]Esc[/bold #f7768e] to CANCEL running tasks at any time.[/dim #737aa2]",
                classes="system-card",
            )

        with Container(id="bottom-container"):
            yield Static("", id="vim-statusline")
            with Horizontal(id="input-row", classes="focused-insert"):
                yield Input(
                    placeholder=f"Ask {self.current_profile} or type / for commands, !<cmd>...",
                    id="chat-input",
                )

    def on_mount(self) -> None:
        self.set_mode("insert")
        self._update_statusline()
        self._update_top_bar()

    def get_dynamic_commands(self) -> list[tuple[str, str]]:
        """Dynamically build list of available slash commands from discovered YAML pipelines & profiles."""
        commands: list[tuple[str, str]] = []

        # 1. Pipelines from PipelineManager (custom user YAMLs + templates)
        for pipe in self.pipeline_manager.list_pipelines():
            stages_summary = " ➔ ".join(s.name for s in pipe.stages)
            desc = pipe.description or f"Pipeline {pipe.name}"
            if stages_summary:
                desc = f"{desc} ({stages_summary})"
            commands.append((f"/pipeline {pipe.name} ", desc))

        # 2. Profiles from ProfileManager (custom user YAMLs + templates)
        for prof in self.profile_manager.list_profiles():
            desc = prof.description or f"Profile {prof.name}"
            commands.append((f"/profile {prof.name}", f"Switch active profile to {prof.name} ({prof.provider}) — {desc[:45]}"))

        # 3. Global actions and shell helpers
        commands.extend([
            ("/diff", "Inspect uncommitted git changes in current workspace"),
            ("!git status", "Shell: Check working tree and git status"),
            ("!pytest", "Shell: Run pytest test suite"),
            ("/clear", "Clear chat history and transcript view"),
            ("/help", "Show keyboard shortcuts and command reference"),
            ("/exit", "Exit Maulness interactive session"),
        ])
        return commands

    def set_mode(self, new_mode: str) -> None:
        self.mode = new_mode
        chat_input = self.query_one("#chat-input", Input)
        input_row = self.query_one("#input-row", Horizontal)

        if self.mode == "insert":
            input_row.add_class("focused-insert")
            chat_input.focus()
        else:
            input_row.remove_class("focused-insert")
            self.set_focus(None)

        self._update_statusline()

    def _update_statusline(self) -> None:
        statusline = self.query_one("#vim-statusline", Static)
        if self.is_busy:
            mode_badge = "[bold #16161e on #f7768e] BUSY [/bold #16161e on #f7768e]"
            hints = "[bold #f7768e]Ctrl+C / Esc: STOP / CANCEL TASK[/bold #f7768e]"
        elif self.mode == "insert":
            mode_badge = "[bold #16161e on #7aa2f7] INSERT [/bold #16161e on #7aa2f7]"
            hints = "[dim #737aa2]Esc: normal mode │ /: command palette │ Enter: send[/dim #737aa2]"
        else:
            mode_badge = "[bold #16161e on #a6e3a1] NORMAL [/bold #16161e on #a6e3a1]"
            hints = "[dim #737aa2]i: insert │ /: commands │ j/k: scroll │ d/u: page │ gg/G: top/bottom │ ?: help │ q: quit[/dim #737aa2]"

        statusline.update(f"{mode_badge}  {hints}")

    def _update_top_bar(self) -> None:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()
        status_text = "[#e0af68]busy[/#e0af68]" if self.is_busy else "[#a6e3a1]idle[/#a6e3a1]"
        top_bar = self.query_one("#top-bar", Static)
        top_bar.update(
            f"[bold #f7768e]⚡ MAULNESS[/bold #f7768e] │ [bold #a6e3a1]repo:[/bold #a6e3a1] {self.repo_name} │ "
            f"[bold #89dceb]branch:[/bold #89dceb] {branch} │ [bold #cba6f7]profile:[/bold #cba6f7] {self.current_profile} │ "
            f"{daemon_str} │ {status_text}"
        )

    def cancel_active_task(self) -> None:
        """Cancel and stop the currently running direct or pipeline task."""
        if self.active_worker and not self.active_worker.is_finished:
            self.active_worker.cancel()
            logger.info("Cancelled active worker task upon user request.")
        self.is_busy = False
        self._update_top_bar()
        self._update_statusline()

    def on_key(self, event: events.Key) -> None:
        # Stop / Cancel Running Task when busy
        if self.is_busy:
            if event.key in ("ctrl+c", "escape"):
                event.prevent_default()
                event.stop()
                self.cancel_active_task()
                return

        chat_view = self.query_one("#chat-view", VerticalScroll)

        # NORMAL MODE KEY HANDLING
        if self.mode == "normal":
            if event.key in ("j", "down"):
                chat_view.scroll_down()
            elif event.key in ("k", "up"):
                chat_view.scroll_up()
            elif event.key in ("d", "ctrl+d"):
                chat_view.scroll_page_down()
            elif event.key in ("u", "ctrl+u"):
                chat_view.scroll_page_up()
            elif event.key == "G":
                chat_view.scroll_end()
            elif event.key == "g":
                now = time.time()
                if now - self._last_g_time < 1.5:
                    chat_view.scroll_home()
                    self._last_g_time = 0.0
                else:
                    self._last_g_time = now
            elif event.key in ("i", "a", "enter"):
                self.set_mode("insert")
            elif event.key == "slash":
                self.open_command_palette()
            elif event.key in ("colon", "exclamation_mark"):
                chat_input = self.query_one("#chat-input", Input)
                chat_input.value = "! "
                self.set_mode("insert")
            elif event.key == "p":
                self.action_select_profile()
            elif event.key in ("question_mark", "f1"):
                self.action_show_help()
            elif event.key == "q":
                self.exit()
            return

        # INSERT MODE KEY HANDLING
        if self.mode == "insert":
            if event.key == "escape":
                self.set_mode("normal")
            elif event.key in ("tab", "ctrl+k"):
                chat_input = self.query_one("#chat-input", Input)
                val = chat_input.value
                self.open_command_palette(val if val.startswith("/") else "")

    def on_input_changed(self, event: Input.Changed) -> None:
        # If user typed leading slash as the very first character, pop open command palette
        if event.value == "/":
            self.open_command_palette(initial_query="")

    def open_command_palette(self, initial_query: str = "") -> None:
        commands = self.get_dynamic_commands()

        def _on_command_selected(selected: Optional[str]) -> None:
            if not selected:
                self.set_mode("insert")
                return

            # Direct action commands
            if selected == "/diff":
                self.action_view_diff()
                self.set_mode("normal")
                return
            if selected == "/clear":
                self.action_clear_chat()
                self.set_mode("normal")
                return
            if selected == "/help":
                self.action_show_help()
                self.set_mode("normal")
                return
            if selected == "/exit":
                self.exit()
                return

            # Profile switch from command
            if selected.startswith("/profile "):
                p_name = selected[len("/profile ") :].strip()
                self.current_profile = p_name
                self._update_top_bar()
                chat_input = self.query_one("#chat-input", Input)
                chat_input.value = ""
                self.set_mode("insert")
                return

            # Fill input and switch to insert mode
            chat_input = self.query_one("#chat-input", Input)
            chat_input.value = selected
            self.set_mode("insert")
            chat_input.cursor_position = len(selected)

        self.push_screen(CommandPaletteModal(commands, initial_query), callback=_on_command_selected)

    def action_quit_app(self) -> None:
        self.exit()

    def action_clear_chat(self) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()

    def action_view_diff(self) -> None:
        self.push_screen(DiffModal(self.workspace_path))

    def action_select_profile(self) -> None:
        profiles = self.profile_manager.list_profiles()

        def _on_profile_selected(selected: Optional[str]):
            if selected:
                self.current_profile = selected
                self._update_top_bar()
                chat_input = self.query_one("#chat-input", Input)
                chat_input.placeholder = f"Ask {self.current_profile} or type / for commands, !<cmd>..."
            self.set_mode("normal")

        self.push_screen(ProfileModal(profiles, self.current_profile), callback=_on_profile_selected)

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_text = event.value.strip()
        if not raw_text:
            return

        chat_input = self.query_one("#chat-input", Input)
        chat_input.value = ""

        # Cancellation commands
        if raw_text in ("/stop", "/cancel", "stop", "cancel") and self.is_busy:
            self.cancel_active_task()
            return

        if self.is_busy:
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(SystemCard("Busy", "Agent is currently processing a task. Press Ctrl+C or Esc to cancel.", is_error=True))
            chat_view.scroll_end(animate=False)
            return

        # Direct Slash Commands
        if raw_text in ("/exit", "/quit", "exit", "quit"):
            self.exit()
            return

        if raw_text == "/clear":
            self.action_clear_chat()
            return

        if raw_text == "/diff":
            self.action_view_diff()
            return

        if raw_text in ("/help", "help"):
            self.action_show_help()
            return

        if raw_text.startswith("/profile"):
            parts = raw_text.split(" ", 1)
            if len(parts) > 1 and parts[1].strip():
                self.current_profile = parts[1].strip()
                self._update_top_bar()
                chat_input.placeholder = f"Ask {self.current_profile} or type / for commands, !<cmd>..."
            else:
                self.action_select_profile()
            return

        # Local Shell Command Escape: !<cmd>
        if raw_text.startswith("!"):
            cmd = raw_text[1:].strip()
            branch = get_git_branch(self.workspace_path)
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(UserCard(raw_text, self.repo_name, branch))
            try:
                res = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                )
                output = res.stdout or res.stderr or "(Command executed with no output)"
                await chat_view.mount(SystemCard(f"$ {cmd} (exit {res.returncode})", output))
            except Exception as e:
                await chat_view.mount(SystemCard(f"$ {cmd} (failed)", str(e), is_error=True))
            chat_view.scroll_end(animate=False)
            return

        # Pipeline Command: /pipeline [name] <goal>
        if raw_text.startswith("/pipeline "):
            pipeline_body = raw_text[len("/pipeline ") :].strip()
            parts = pipeline_body.split(" ", 1)
            known_pipelines = {p.name for p in self.pipeline_manager.list_pipelines()}
            if len(parts) == 2 and parts[0] in known_pipelines:
                p_name, goal = parts[0], parts[1].strip()
            else:
                p_name, goal = "standard", pipeline_body

            self.active_worker = self.run_worker(self._execute_pipeline(p_name, goal), exclusive=True)
            return

        # Default Natural Language Prompt -> Direct Execution with Active Profile
        self.active_worker = self.run_worker(self._execute_direct(raw_text), exclusive=True)

    async def _execute_direct(self, prompt: str) -> None:
        self.is_busy = True
        self._update_top_bar()
        self._update_statusline()
        chat_view = self.query_one("#chat-view", VerticalScroll)

        branch = get_git_branch(self.workspace_path)
        user_card = UserCard(prompt, self.repo_name, branch)
        agent_card = AgentCard(self.current_profile)

        await chat_view.mount(user_card)
        await chat_view.mount(agent_card)
        chat_view.scroll_end(animate=False)

        async def on_thought(event: AgentThoughtEvent):
            agent_card.append_thought(event.delta)
            chat_view.scroll_end(animate=False)

        async def on_message(event: AgentMessageEvent):
            agent_card.append_message(event.delta)
            chat_view.scroll_end(animate=False)

        async def on_approval(event: ApprovalRequestEvent) -> bool:
            return await self.push_screen_wait(ApprovalModal(event))

        try:
            task = await self.runner.run_direct(
                repo_name=self.repo_name,
                prompt=prompt,
                workspace_path=self.workspace_path,
                profile_name=self.current_profile,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
                verbose=False,
            )
            agent_card.set_status(f"[#a6e3a1]✓ Completed ({task.status.value})[/#a6e3a1]")
        except asyncio.CancelledError:
            agent_card.set_status("[bold #f7768e]⏹ Stopped by user[/bold #f7768e]")
        except Exception as e:
            agent_card.set_status(f"[#f7768e]✗ Failed: {e}[/#f7768e]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            self._update_statusline()
            chat_view.scroll_end(animate=False)

    async def _execute_pipeline(self, pipeline_name: str, goal: str) -> None:
        self.is_busy = True
        self._update_top_bar()
        self._update_statusline()
        chat_view = self.query_one("#chat-view", VerticalScroll)

        branch = get_git_branch(self.workspace_path)
        await chat_view.mount(UserCard(f"/pipeline {pipeline_name} {goal}", self.repo_name, branch))
        p_card = PipelineCard(pipeline_name, goal)
        await chat_view.mount(p_card)
        chat_view.scroll_end(animate=False)

        async def on_gate(prompt: str) -> bool:
            return await self.push_screen_wait(GateModal(prompt))

        async def on_stage_start(stage: PipelineStage, current: int, total: int):
            p_card.update_stage(stage.name, stage.profile, current, total)
            chat_view.scroll_end(animate=False)

        async def on_thought(event: AgentThoughtEvent):
            pass

        async def on_message(event: AgentMessageEvent):
            p_card.append_output(event.delta)
            chat_view.scroll_end(animate=False)

        async def on_approval(event: ApprovalRequestEvent) -> bool:
            return await self.push_screen_wait(ApprovalModal(event))

        try:
            task = await self.orchestrator.run_pipeline(
                repo_name=self.repo_name,
                title=goal[:80],
                prompt=goal,
                workspace_path=self.workspace_path,
                pipeline_name=pipeline_name,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
                on_gate=on_gate,
                on_stage_start=on_stage_start,
                verbose=False,
            )
            p_card.finish(f"[#a6e3a1]✓ Pipeline Finished ({task.status.value})[/#a6e3a1]")
        except asyncio.CancelledError:
            p_card.finish("[bold #f7768e]⏹ Pipeline stopped by user[/bold #f7768e]")
        except Exception as e:
            p_card.finish(f"[#f7768e]✗ Pipeline Failed: {e}[/#f7768e]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            self._update_statusline()
            chat_view.scroll_end(animate=False)
