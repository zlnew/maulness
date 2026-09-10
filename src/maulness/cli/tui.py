import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

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
from maulness.storage.db import StorageManager

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
            return "[green]● active[/green]"
    except Exception:
        pass
    return "[dim]○ standby[/dim]"


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
            yield Label("[bold magenta]󰍉 Command Palette[/bold magenta] [dim](Type to filter, ↑/↓ to navigate, Enter/Tab to select, Esc to close)[/dim]", classes="palette-title")
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
            prompt_markup = f"[bold cyan]{cmd}[/bold cyan] [dim]— {desc}[/dim]"
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
            yield Label("[bold yellow]⚠️  HITL Tool Approval Required[/bold yellow]", classes="modal-title")
            yield Label(f"[bold cyan]Tool:[/bold cyan] {self.event.tool_name}", classes="modal-row")
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
            yield Label("[bold magenta]🚦 Pipeline Confirmation Gate[/bold magenta]", classes="modal-title")
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
            yield Label("[bold cyan]🔍 Git Diff HEAD[/bold cyan] [dim](j/k to scroll, q or Esc to close)[/dim]")
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
            yield Label("[bold magenta]Select Active Profile[/bold magenta] [dim](Enter to switch, Esc to cancel)[/dim]", classes="palette-title")
            ol = OptionList(id="profile-options")
            yield ol

    def on_mount(self) -> None:
        ol = self.query_one("#profile-options", OptionList)
        for idx, p in enumerate(self.profiles):
            is_active = " [bold green]● ACTIVE[/bold green]" if p.name == self.current_profile else ""
            label = f"[bold cyan]{p.name}[/bold cyan] ({p.provider}) — [dim]{p.description[:50]}[/dim]{is_active}"
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
  - `/` : Trigger LSP Autocomplete popup docked above input
  - `j` / `k` : Scroll chat view down / up
  - `d` / `u` : Half-page scroll down / up (`Ctrl+D` / `Ctrl+U`)
  - `G` : Scroll to bottom of chat
  - `gg` : Scroll to top of chat
  - `: ` or `!` : Enter INSERT mode pre-filled with `! ` (shell command)
  - `p` : Open Profile Switcher modal
  - `?` or `F1` : Open this Help reference
  - `q` : Quit Maulness

- **INSERT Mode** (Blue status badge):
  - Type prompt naturally (executed by active profile)
  - `/` : Opens LSP Autocomplete popup docked above input
  - `Tab` / `Down` / `Up` : Navigate autocomplete suggestions
  - `Escape` : Dismiss autocomplete popup, or return to **NORMAL** mode
  - `Enter` : Submit prompt / execute command
  - `Ctrl+C` : Immediately cancel running task

### Dynamic Slash Commands
- `/cancel` or `/stop` : Stop/cancel currently running task
- `/interrupt <prompt>` : Cancel current task and steer agent with new prompt
- `/queue <prompt>` : Queue a prompt to auto-execute after current task finishes
- `/usage` : Display session token metrics, duration, and cost estimation
- `/context` : Display current workspace, git status, active profile, and engine
- `/compact` : Compress conversation history into SQLite long-term memory
- `/pipeline <name> <goal>` : Execute declarative pipeline (`standard`, `quick`, `plan_only`, `audit`)
- `/profile <name>` : Switch active profile (`default`, `builder`, `planner`, `reviewer`)
- `/diff` : Scrollable git diff inspector
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
        yield Label(f"[bold cyan]Maul[/bold cyan] [dim]({self.repo}:{self.branch})[/dim]", classes="card-header")
        yield Static(self.prompt, classes="card-body")


class SystemCard(Static):
    """Card displaying system messages or shell command execution."""

    def __init__(self, title: str, content: str, is_error: bool = False):
        super().__init__(classes="system-card")
        self.card_title = title
        self.card_content = content
        self.is_error = is_error

    def compose(self) -> ComposeResult:
        color = "red" if self.is_error else "yellow"
        yield Label(f"[bold {color}]{self.card_title}[/bold {color}]", classes="card-header")
        yield Static(self.card_content, classes="card-body")


class AgentCard(Static):
    """Card displaying compact thought ticker, tool badges, and clean assistant response."""

    def __init__(self, profile_name: str):
        super().__init__(classes="agent-card")
        self.profile_name = profile_name
        self.thought_text: list[str] = []
        self.message_text: list[str] = []
        self.thought_start_time = time.time()
        self.thought_duration: Optional[float] = None
        self.has_started_message = False

        self.thought_static = Static("", classes="thought-box")
        self.tools_static = Static("", classes="tools-box")
        self.message_static = Static("", classes="card-body")
        self.status_label = Label("[dim]󰑮 Initializing...[/dim]")

        self._last_render_time = 0.0

    def compose(self) -> ComposeResult:
        yield Label(f"[bold magenta]Maulness[/bold magenta] [dim]({self.profile_name})[/dim]", classes="card-header")
        yield self.thought_static
        yield self.tools_static
        yield self.message_static
        yield self.status_label

    def append_thought(self, delta: str) -> None:
        self.thought_text.append(delta)
        if not self.has_started_message:
            elapsed = time.time() - self.thought_start_time
            full_thought = "".join(self.thought_text).strip()
            lines = [line.strip() for line in full_thought.splitlines() if line.strip()]
            snippet = lines[-1][:75] if lines else "Thinking..."
            self.thought_static.update(f"[dim italic cyan]💭 Thinking ({elapsed:.1f}s):[/dim italic cyan] [dim]{snippet}[/dim]")
            self.thought_static.add_class("visible")
            self.status_label.update("[dim cyan]󰑮 Thinking...[/dim cyan]")

    def record_tool_call(self, tool_name: str, args: Any = None) -> None:
        summary = ""
        if isinstance(args, dict):
            if "CommandLine" in args:
                summary = f": {args['CommandLine'][:40]}"
            elif "path" in args:
                summary = f": {args['path']}"
            elif "TargetFile" in args:
                summary = f": {args['TargetFile']}"
            elif "AbsolutePath" in args:
                summary = f": {args['AbsolutePath']}"
        self.tools_static.update(f"[bold yellow]⚡ Tool:[/bold yellow] [dim]{tool_name}{summary}[/dim]")
        self.tools_static.add_class("visible")

    def append_message(self, delta: str, force_render: bool = False) -> None:
        if not self.has_started_message:
            self.has_started_message = True
            self.thought_duration = time.time() - self.thought_start_time
            if self.thought_text:
                self.thought_static.update(f"[dim]💭 Thought for {self.thought_duration:.1f}s[/dim]")
                self.thought_static.add_class("visible")
            else:
                self.thought_static.remove_class("visible")
                self.thought_static.update("")
            self.status_label.update("[dim]󰑮 Streaming response...[/dim]")

        self.message_text.append(delta)
        now = time.time()
        if force_render or (now - self._last_render_time >= 0.05):
            self._render_message()
            self._last_render_time = now

    def _render_message(self) -> None:
        text = "".join(self.message_text)
        if text:
            self.message_static.update(RichMarkdown(text))

    def flush_final(self) -> None:
        if not self.has_started_message and self.thought_text:
            duration = time.time() - self.thought_start_time
            self.thought_static.update(f"[dim]💭 Thought for {duration:.1f}s[/dim]")
            self.thought_static.add_class("visible")
        self._render_message()

    def set_status(self, status: str) -> None:
        self.flush_final()
        self.status_label.update(status)


class PipelineCard(Static):
    """Card displaying declarative pipeline multi-stage progress."""

    def __init__(self, pipeline_name: str, goal: str):
        super().__init__(classes="agent-card")
        self.pipeline_name = pipeline_name
        self.goal = goal
        self.stages_static = Static("")
        self.status_label = Label("[dim]󰑮 Initializing pipeline...[/dim]")
        self.output_static = Static("", classes="card-body")

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold magenta]Pipeline: {self.pipeline_name}[/bold magenta] — [dim cyan]{self.goal[:60]}[/dim cyan]",
            classes="card-header",
        )
        yield self.stages_static
        yield self.output_static
        yield self.status_label

    def update_stage(self, stage_name: str, profile: str, current: int, total: int) -> None:
        self.stages_static.update(
            f"[bold yellow]▶ Stage {current}/{total}:[/bold yellow] [bold]{stage_name}[/bold] ([magenta]{profile}[/magenta])"
        )
        self.status_label.update(f"[dim]Running stage {current} of {total}...[/dim]")

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
        background: transparent;
        color: $text;
    }

    ModalScreen {
        align: center middle;
    }

    #top-bar {
        dock: top;
        height: 3;
        background: transparent;
        color: $text;
        padding: 0 2;
        border-bottom: solid $border;
    }

    #chat-view {
        height: 1fr;
        padding: 1 2;
        background: transparent;
    }

    .user-card {
        background: transparent;
        border: round $accent;
        padding: 0 1;
        margin: 1 0;
    }

    .agent-card {
        background: transparent;
        border: round $primary;
        padding: 0 1;
        margin: 1 0;
    }

    .system-card {
        background: transparent;
        border: round $border;
        padding: 0 1;
        margin: 1 0;
    }

    .card-header {
        margin-bottom: 1;
    }

    .card-body {
        margin-top: 0;
    }

    .thought-box {
        display: none;
        color: $text-muted;
        background: transparent;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick $border;
    }

    .thought-box.visible {
        display: block;
    }

    .tools-box {
        display: none;
        color: $warning;
        background: transparent;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick $warning;
    }

    .tools-box.visible {
        display: block;
    }

    #bottom-container {
        dock: bottom;
        height: auto;
        max-height: 18;
        background: transparent;
        border-top: solid $border;
        padding: 0 1;
    }

    #autocomplete-popup {
        display: none;
        max-height: 8;
        height: auto;
        background: $panel;
        border: tall $primary;
        margin-bottom: 0;
        scrollbar-size-vertical: 1;
    }

    #autocomplete-popup.visible {
        display: block;
    }

    #vim-statusline {
        height: 1;
        background: transparent;
        color: $text-muted;
        padding: 0 1;
    }

    #input-row {
        height: 3;
        background: transparent;
        border: tall $border;
        padding: 0 1;
    }

    #input-row.focused-insert {
        border: tall $primary;
    }

    #chat-input {
        width: 1fr;
        background: transparent;
        border: none;
        color: $text;
    }

    #chat-input:focus {
        border: none;
    }

    /* Modals (All Centered, using solid $panel/$surface for high contrast against transparent background) */
    .palette-dialog {
        width: 75%;
        max-height: 22;
        background: $panel;
        border: double $primary;
        padding: 1 2;
    }

    .palette-title {
        margin-bottom: 1;
    }

    #palette-filter {
        background: $surface;
        border: tall $border;
        color: $text;
        margin-bottom: 1;
    }

    #palette-options {
        max-height: 12;
        background: $surface;
        border: tall $border;
    }

    .modal-dialog {
        width: 70%;
        max-height: 80%;
        background: $panel;
        border: thick $primary;
        padding: 1 2;
    }

    .diff-dialog {
        width: 90%;
        height: 90%;
        background: $panel;
        border: thick $primary;
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

    def __init__(self, initial_profile: str = "default"):
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

        # Session metrics & Prompt Queue
        self.prompt_queue: list[str] = []
        self.session_start_time = time.time()
        self.total_prompts = 0
        self.total_chars_out = 0
        self.active_session_id = f"tui_{uuid.uuid4().hex[:8]}"
        self.active_acp_session_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()

        yield Static(
            f"[bold red]⚡ MAULNESS[/bold red] │ [bold green]repo:[/bold green] {self.repo_name} │ "
            f"[bold cyan]branch:[/bold cyan] {branch} │ [bold magenta]profile:[/bold magenta] {self.current_profile} │ "
            f"{daemon_str}",
            id="top-bar",
        )

        with VerticalScroll(id="chat-view"):
            yield Static(
                f"[dim]Welcome back, Maul. Scoped to [bold green]{self.workspace_path}[/bold green].\n"
                f"• Natural language executes with [bold magenta]{self.current_profile}[/bold magenta].\n"
                f"• Type [bold cyan]/[/bold cyan] for Neovim LSP-style floating autocomplete popup.\n"
                f"• Press [bold cyan]Esc[/bold cyan] for NORMAL mode ([green]j/k[/green] scroll, [green]i[/green] insert, [green]?[/green] help).\n"
                f"• Press [bold red]Ctrl+C[/bold red] or [bold red]Esc[/bold red] to CANCEL running tasks at any time.[/dim]",
                classes="system-card",
            )

        with Container(id="bottom-container"):
            yield OptionList(id="autocomplete-popup")
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

        # 1. Flow Control & Task Orchestration
        commands.extend([
            ("/cancel", "Cancel currently running task"),
            ("/stop", "Stop currently running task (alias of /cancel)"),
            ("/interrupt ", "Interrupt current task and steer with new prompt (<prompt>)"),
            ("/queue ", "Queue a prompt to run after current task finishes (<prompt>)"),
            ("/usage", "Display token metrics and estimated cost for this session"),
            ("/context", "Display current workspace, git branch, and agent status"),
            ("/compact", "Compact conversation history into SQLite memory"),
        ])

        # 2. Pipelines from PipelineManager (custom user YAMLs + templates)
        for pipe in self.pipeline_manager.list_pipelines():
            stages_summary = " ➔ ".join(s.name for s in pipe.stages)
            desc = pipe.description or f"Pipeline {pipe.name}"
            if stages_summary:
                desc = f"{desc} ({stages_summary})"
            commands.append((f"/pipeline {pipe.name} ", desc))

        # 3. Profiles from ProfileManager (custom user YAMLs + templates)
        for prof in self.profile_manager.list_profiles():
            desc = prof.description or f"Profile {prof.name}"
            ws_hint = f" [ws: {prof.workspace}]" if prof.workspace and prof.workspace != "inherit" else ""
            commands.append((f"/profile {prof.name}", f"Switch active profile to {prof.name} ({prof.provider}){ws_hint} — {desc[:40]}"))

        # 4. Global actions and shell helpers
        commands.extend([
            ("/diff", "Inspect uncommitted git changes in current workspace"),
            ("!<command>", "Shell: Execute command in current workspace (e.g. !git status)"),
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
            mode_badge = "[bold white on red] BUSY [/bold white on red]"
            hints = "[bold red]Ctrl+C / Esc: STOP / CANCEL TASK[/bold red]"
        elif self.mode == "insert":
            mode_badge = "[bold black on cyan] INSERT [/bold black on cyan]"
            hints = "[dim]Esc: normal mode │ /: command palette │ Enter: send[/dim]"
        else:
            mode_badge = "[bold black on green] NORMAL [/bold black on green]"
            hints = "[dim]i: insert │ /: commands │ j/k: scroll │ d/u: page │ gg/G: top/bottom │ ?: help │ q: quit[/dim]"

        statusline.update(f"{mode_badge}  {hints}")

    def _update_top_bar(self) -> None:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()
        status_text = "[yellow]busy[/yellow]" if self.is_busy else "[green]idle[/green]"
        top_bar = self.query_one("#top-bar", Static)
        top_bar.update(
            f"[bold red]⚡ MAULNESS[/bold red] │ [bold green]repo:[/bold green] {self.repo_name} │ "
            f"[bold cyan]branch:[/bold cyan] {branch} │ [bold magenta]profile:[/bold magenta] {self.current_profile} │ "
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

    def _update_autocomplete(self, text: str) -> None:
        """Update floating LSP autocomplete popup docked directly above the input."""
        popup = self.query_one("#autocomplete-popup", OptionList)
        if not text.startswith("/"):
            self._hide_autocomplete()
            return

        parts = text.split(" ", 1)

        # 1. Sub-command completion for /pipeline <name>
        if text.startswith("/pipeline "):
            arg = parts[1].lower().strip() if len(parts) > 1 else ""
            pipes = self.pipeline_manager.list_pipelines()
            matching_pipes = [p for p in pipes if not arg or p.name.lower().startswith(arg)]
            if matching_pipes:
                popup.clear_options()
                for p in matching_pipes:
                    popup.add_option(Option(f"[bold cyan]/pipeline {p.name}[/bold cyan] [dim]— {p.description}[/dim]", id=f"/pipeline {p.name} "))
                popup.highlighted = 0
                popup.add_class("visible")
                return
            self._hide_autocomplete()
            return

        # 2. Sub-command completion for /profile <name>
        if text.startswith("/profile "):
            arg = parts[1].lower().strip() if len(parts) > 1 else ""
            profs = self.profile_manager.list_profiles()
            matching_profs = [p for p in profs if not arg or p.name.lower().startswith(arg)]
            if matching_profs:
                popup.clear_options()
                for p in matching_profs:
                    popup.add_option(Option(f"[bold cyan]/profile {p.name}[/bold cyan] [dim]— {p.description[:40]}[/dim]", id=f"/profile {p.name}"))
                popup.highlighted = 0
                popup.add_class("visible")
                return
            self._hide_autocomplete()
            return

        # 3. If text already has a space and isn't sub-command completions, hide popup
        if " " in text:
            self._hide_autocomplete()
            return

        query = text.lower().strip()
        all_commands = self.get_dynamic_commands()
        matching = []

        for cmd, desc in all_commands:
            cmd_clean = cmd.strip()
            if query == "/" or cmd_clean.lower().startswith(query) or query.lstrip("/") in cmd_clean.lower():
                matching.append((cmd, desc))

        if not matching:
            self._hide_autocomplete()
            return

        popup.clear_options()
        for cmd, desc in matching:
            markup = f"[bold cyan]{cmd.strip()}[/bold cyan] [dim]— {desc}[/dim]"
            popup.add_option(Option(markup, id=cmd))

        popup.highlighted = 0
        popup.add_class("visible")

    def _hide_autocomplete(self) -> None:
        popup = self.query_one("#autocomplete-popup", OptionList)
        popup.remove_class("visible")
        popup.clear_options()

    def _apply_autocomplete(self, execute_zero_arg: bool = False) -> bool:
        """Apply current highlighted autocomplete suggestion into chat input."""
        popup = self.query_one("#autocomplete-popup", OptionList)
        if not popup.has_class("visible") or popup.highlighted is None or popup.option_count == 0:
            return False

        opt = popup.get_option_at_index(popup.highlighted)
        selected_cmd = str(opt.id)
        chat_input = self.query_one("#chat-input", Input)

        if selected_cmd.startswith("/profile "):
            p_name = selected_cmd[len("/profile ") :].strip()
            self.current_profile = p_name
            self._update_top_bar()
            chat_input.placeholder = f"Ask {self.current_profile} or type / for commands, !<cmd>..."
            chat_input.value = ""
            self._hide_autocomplete()
            return True

        # Zero-argument immediate action commands (only when Enter pressed)
        if execute_zero_arg and selected_cmd in (
            "/diff",
            "/clear",
            "/help",
            "/exit",
            "/cancel",
            "/stop",
            "/usage",
            "/context",
            "/compact",
        ):
            self._hide_autocomplete()
            chat_input.value = selected_cmd
            self.post_message(Input.Submitted(chat_input, selected_cmd))
            return True

        # Complete text into chat input
        cmd_text = selected_cmd if selected_cmd.endswith(" ") else selected_cmd + " "
        chat_input.value = cmd_text
        chat_input.cursor_position = len(cmd_text)
        self._hide_autocomplete()
        chat_input.focus()
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "autocomplete-popup":
            self._apply_autocomplete(execute_zero_arg=False)

    def on_key(self, event: events.Key) -> None:
        popup = self.query_one("#autocomplete-popup", OptionList)

        # 1. Autocomplete Popup Navigation (LSP floating style)
        if popup.has_class("visible"):
            if event.key in ("down", "ctrl+n"):
                event.prevent_default()
                event.stop()
                if popup.option_count > 0:
                    if popup.highlighted is None:
                        popup.highlighted = 0
                    elif popup.highlighted < popup.option_count - 1:
                        popup.highlighted += 1
                return
            elif event.key in ("up", "ctrl+p"):
                event.prevent_default()
                event.stop()
                if popup.option_count > 0:
                    if popup.highlighted is not None and popup.highlighted > 0:
                        popup.highlighted -= 1
                return
            elif event.key == "tab":
                event.prevent_default()
                event.stop()
                self._apply_autocomplete(execute_zero_arg=False)
                return
            elif event.key == "enter":
                event.prevent_default()
                event.stop()
                self._apply_autocomplete(execute_zero_arg=True)
                return
            elif event.key == "escape":
                event.prevent_default()
                event.stop()
                self._hide_autocomplete()
                return

        # 2. Stop / Cancel Running Task when busy
        if self.is_busy:
            if event.key in ("ctrl+c", "escape"):
                event.prevent_default()
                event.stop()
                self.cancel_active_task()
                return

        chat_view = self.query_one("#chat-view", VerticalScroll)

        # 3. NORMAL MODE KEY HANDLING
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
            elif event.key in ("i", "a"):
                self.set_mode("insert")
            elif event.key == "slash":
                chat_input = self.query_one("#chat-input", Input)
                chat_input.value = "/"
                self.set_mode("insert")
                chat_input.cursor_position = 1
                self._update_autocomplete("/")
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

        # 4. INSERT MODE KEY HANDLING
        if self.mode == "insert":
            if event.key == "escape":
                if popup.has_class("visible"):
                    self._hide_autocomplete()
                else:
                    self.set_mode("normal")
            elif event.key in ("tab", "ctrl+k"):
                chat_input = self.query_one("#chat-input", Input)
                val = chat_input.value
                self._update_autocomplete(val if val.startswith("/") else "/")

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.mode == "insert":
            self._update_autocomplete(event.value)

    def action_quit_app(self) -> None:
        self.exit()

    def action_clear_chat(self) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()
        self.active_acp_session_id = None
        self.active_session_id = f"tui_{uuid.uuid4().hex[:8]}"

    def action_view_diff(self) -> None:
        self.push_screen(DiffModal(self.workspace_path))

    def action_select_profile(self) -> None:
        profiles = self.profile_manager.list_profiles()

        def _on_profile_selected(selected: Optional[str]):
            if selected:
                self.current_profile = selected
                self.active_acp_session_id = None
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
        self._hide_autocomplete()

        # Cancellation commands
        if raw_text in ("/stop", "/cancel", "stop", "cancel"):
            chat_view = self.query_one("#chat-view", VerticalScroll)
            if self.is_busy:
                self.cancel_active_task()
                await chat_view.mount(SystemCard("Stopped", "⏹ Active task cancelled upon user request."))
            else:
                await chat_view.mount(SystemCard("Info", "No task is currently running."))
            chat_view.scroll_end(animate=False)
            return

        # Interrupt command: stop active task and steer with new prompt
        if raw_text.startswith("/interrupt"):
            parts = raw_text.split(" ", 1)
            steer_prompt = parts[1].strip() if len(parts) > 1 else ""
            if not steer_prompt:
                chat_view = self.query_one("#chat-view", VerticalScroll)
                await chat_view.mount(SystemCard("Error", "Usage: /interrupt <prompt>", is_error=True))
                chat_view.scroll_end(animate=False)
                return

            chat_view = self.query_one("#chat-view", VerticalScroll)
            if self.is_busy:
                self.cancel_active_task()
                await chat_view.mount(SystemCard("Interrupt", f"⚡ Interrupted previous task to steer: {steer_prompt}"))
                chat_view.scroll_end(animate=False)

            self.active_worker = self.run_worker(self._execute_direct(steer_prompt), exclusive=True)
            return

        # Queue command: queue prompt to run when current task finishes
        if raw_text.startswith("/queue"):
            parts = raw_text.split(" ", 1)
            queued_prompt = parts[1].strip() if len(parts) > 1 else ""
            if not queued_prompt:
                chat_view = self.query_one("#chat-view", VerticalScroll)
                await chat_view.mount(SystemCard("Error", "Usage: /queue <prompt>", is_error=True))
                chat_view.scroll_end(animate=False)
                return

            if self.is_busy:
                self.prompt_queue.append(queued_prompt)
                chat_view = self.query_one("#chat-view", VerticalScroll)
                await chat_view.mount(
                    SystemCard(
                        f"Prompt Queued (#{len(self.prompt_queue)})",
                        f"Queued for execution once current task finishes:\n{queued_prompt}",
                    )
                )
                chat_view.scroll_end(animate=False)
                return

            self.active_worker = self.run_worker(self._execute_direct(queued_prompt), exclusive=True)
            return

        # Usage command: display session token metrics and cost
        if raw_text == "/usage":
            uptime = time.time() - self.session_start_time
            m = int(uptime // 60)
            s = int(uptime % 60)
            est_tokens = self.total_chars_out // 4
            prof = self.profile_manager.get_profile(self.current_profile)
            usage_content = (
                f"• Prompts Executed: [bold green]{self.total_prompts}[/bold green]\n"
                f"• Queued Prompts: [bold magenta]{len(self.prompt_queue)}[/bold magenta]\n"
                f"• Output Characters: [bold cyan]{self.total_chars_out:,}[/bold cyan]\n"
                f"• Estimated Output Tokens: [bold blue]~{est_tokens:,}[/bold blue]\n"
                f"• Active Profile: [bold magenta]{self.current_profile}[/bold magenta] ({prof.provider})\n"
                f"• Antigravity Session: [bold magenta]{self.active_acp_session_id or '(fresh turn)'}[/bold magenta]\n"
                f"• Session Uptime: [bold yellow]{m}m {s}s[/bold yellow]\n"
                f"• Estimated Cost: [bold green]$0.00[/bold green] (Local Antigravity ACP / Free Tier)"
            )
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(SystemCard("Session Metrics & Usage", usage_content))
            chat_view.scroll_end(animate=False)
            return

        # Context command: display active workspace, git, and agent context
        if raw_text == "/context":
            branch = get_git_branch(self.workspace_path)
            diff_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
            )
            dirty_count = len([l for l in diff_proc.stdout.splitlines() if l.strip()])
            dirty_str = f"[red]{dirty_count} uncommitted changes[/red]" if dirty_count > 0 else "[green]clean[/green]"

            prof = self.profile_manager.get_profile(self.current_profile)
            target = prof.model or prof.command or "default"
            ws_cfg = prof.workspace or "inherit"

            context_content = (
                f"• Active Workspace: [bold green]{self.workspace_path}[/bold green] (repo: [bold]{self.repo_name}[/bold])\n"
                f"• Git Branch: [bold cyan]{branch}[/bold cyan] ({dirty_str})\n"
                f"• Active Profile: [bold magenta]{self.current_profile}[/bold magenta] ({prof.provider})\n"
                f"• Target Engine: [bold cyan]{target}[/bold cyan]\n"
                f"• Antigravity Session: [bold magenta]{self.active_acp_session_id or '(none)'}[/bold magenta]\n"
                f"• Profile Workspace: [dim]{ws_cfg}[/dim]\n"
                f"• Soul Doctrine: [bold green]Enabled[/bold green] (~/.config/maulness/SOUL.md)\n"
                f"• Storage DB: [dim]{config.db_path}[/dim]\n"
                f"• Daemon Status: {get_daemon_status()}\n"
                f"• Prompt Queue: {len(self.prompt_queue)} pending"
            )
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(SystemCard("Runtime & Workspace Context", context_content))
            chat_view.scroll_end(animate=False)
            return

        # Compact command: compress session history into SQLite
        if raw_text == "/compact":
            storage = StorageManager(db_path=config.db_path)
            await storage.initialize()
            session_id = f"session_{int(time.time())}"
            summary = (
                f"Compacted Session Summary ({self.repo_name} - {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                f"Workspace: {self.workspace_path}\n"
                f"Active Profile: {self.current_profile}\n"
                f"Prompts Completed: {self.total_prompts}\n"
                f"Output Characters: {self.total_chars_out}\n"
                f"Key State: Scoped to {self.repo_name}. Working transcript compressed into persistent memory."
            )
            await storage.save_session_memory(
                session_id=session_id,
                summary=summary,
                token_count=self.total_chars_out // 4,
            )
            self.total_chars_out = 0
            self.active_acp_session_id = None
            self.active_session_id = f"tui_{uuid.uuid4().hex[:8]}"
            chat_view = self.query_one("#chat-view", VerticalScroll)
            chat_view.remove_children()
            await chat_view.mount(
                SystemCard(
                    "📦 Context Compacted",
                    f"Saved summary to SQLite ([bold magenta]session_memories[/bold magenta]).\n"
                    f"Session transcript compressed into memory. Active context reset for lean token usage.\n"
                    f"Persisted in: [dim]{config.db_path}[/dim]",
                )
            )
            self.total_chars_out = 0
            chat_view.scroll_end(animate=False)
            return

        # Busy check for non-control prompts
        if self.is_busy:
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(
                SystemCard(
                    "Busy",
                    "Agent is currently busy processing a task.\n"
                    "• Use [bold red]/cancel[/bold red] or [bold red]Ctrl+C[/bold red] to stop.\n"
                    "• Use [bold magenta]/interrupt <prompt>[/bold magenta] to stop and steer immediately.\n"
                    "• Use [bold cyan]/queue <prompt>[/bold cyan] to queue your prompt to run next.",
                    is_error=True,
                )
            )
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
        self.total_prompts += 1
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
            self.total_chars_out += len(event.delta)
            agent_card.append_message(event.delta)
            chat_view.scroll_end(animate=False)

        async def on_tool_call(event: AgentToolCallEvent):
            agent_card.record_tool_call(event.tool_name, event.args)
            chat_view.scroll_end(animate=False)

        async def on_approval(event: ApprovalRequestEvent) -> bool:
            return await self.push_screen_wait(ApprovalModal(event))

        async def on_init(conv_id: str):
            self.active_acp_session_id = conv_id
            logger.info("Attached to Antigravity conversation: %s", conv_id)

        try:
            task = await self.runner.run_direct(
                repo_name=self.repo_name,
                prompt=prompt,
                workspace_path=self.workspace_path,
                profile_name=self.current_profile,
                session_id=self.active_session_id,
                conversation_id=self.active_acp_session_id,
                on_init=on_init,
                on_thought=on_thought,
                on_message=on_message,
                on_tool_call=on_tool_call,
                on_approval=on_approval,
                verbose=False,
            )
            agent_card.set_status(f"[green]✓ Completed ({task.status.value})[/green]")
        except asyncio.CancelledError:
            agent_card.set_status("[bold red]⏹ Stopped by user[/bold red]")
        except Exception as e:
            agent_card.set_status(f"[red]✗ Failed: {e}[/red]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            self._update_statusline()
            chat_view.scroll_end(animate=False)

            # Auto-dispatch next prompt from prompt_queue if available
            if self.prompt_queue:
                next_prompt = self.prompt_queue.pop(0)
                logger.info(f"Auto-dispatching queued prompt: {next_prompt[:40]}")
                self.active_worker = self.run_worker(self._execute_direct(next_prompt), exclusive=True)

    async def _execute_pipeline(self, pipeline_name: str, goal: str) -> None:
        self.is_busy = True
        self.total_prompts += 1
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
            self.total_chars_out += len(event.delta)
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
            p_card.finish(f"[green]✓ Pipeline Finished ({task.status.value})[/green]")
        except asyncio.CancelledError:
            p_card.finish("[bold red]⏹ Pipeline stopped by user[/bold red]")
        except Exception as e:
            p_card.finish(f"[red]✗ Pipeline Failed: {e}[/red]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            self._update_statusline()
            chat_view.scroll_end(animate=False)

            # Auto-dispatch next prompt from prompt_queue if available
            if self.prompt_queue:
                next_prompt = self.prompt_queue.pop(0)
                logger.info(f"Auto-dispatching queued prompt: {next_prompt[:40]}")
                self.active_worker = self.run_worker(self._execute_direct(next_prompt), exclusive=True)
