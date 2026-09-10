import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

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
            return "[green]daemon: active[/green]"
    except Exception:
        pass
    return "[dim]daemon: off[/dim]"


# ==============================================================================
# Modal Dialogs
# ==============================================================================
class ApprovalModal(ModalScreen[bool]):
    """HITL Tool Approval Modal Dialog."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, event: ApprovalRequestEvent):
        super().__init__()
        self.event = event

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("[bold yellow]⚠️  HITL Approval Required[/bold yellow]", classes="modal-title")
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
        if event.button.id == "btn-allow":
            self.dismiss(True)
        else:
            self.dismiss(False)


class GateModal(ModalScreen[bool]):
    """Pipeline Stage Confirmation Gate Modal."""

    BINDINGS = [
        Binding("y", "proceed", "Proceed"),
        Binding("n", "stop", "Stop"),
        Binding("escape", "stop", "Stop"),
    ]

    def __init__(self, prompt: str):
        super().__init__()
        self.gate_prompt = prompt

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("[bold magenta]🚦 Pipeline Stage Gate[/bold magenta]", classes="modal-title")
            yield Label(self.gate_prompt, classes="modal-row")
            with Horizontal(classes="modal-buttons"):
                yield Button("Proceed (y)", variant="primary", id="btn-proceed")
                yield Button("Stop (n)", variant="default", id="btn-stop")

    def action_proceed(self) -> None:
        self.dismiss(True)

    def action_stop(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-proceed":
            self.dismiss(True)
        else:
            self.dismiss(False)


class DiffModal(ModalScreen[None]):
    """Full-screen / large scrollable git diff inspector."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, workspace_path: Path):
        super().__init__()
        self.workspace_path = workspace_path

    def compose(self) -> ComposeResult:
        with Container(classes="diff-dialog"):
            yield Label("[bold cyan]🔍 Workspace Uncommitted Changes (git diff HEAD)[/bold cyan]")
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
            )
            diff_text = res.stdout.strip() or "(No uncommitted diffs detected in working directory)"
            with VerticalScroll(classes="diff-scroll"):
                yield Static(Syntax(diff_text, "diff", theme="monokai"))
            with Horizontal(classes="modal-buttons"):
                yield Button("Close (Esc)", variant="primary", id="btn-close")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class ProfileModal(ModalScreen[Optional[str]]):
    """Profile selector modal."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, profiles: list, current_profile: str):
        super().__init__()
        self.profiles = profiles
        self.current_profile = current_profile

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("[bold magenta]Select Active Agent Profile[/bold magenta]", classes="modal-title")
            with VerticalScroll():
                for p in self.profiles:
                    is_active = " [green]●[/green]" if p.name == self.current_profile else ""
                    yield Button(
                        f"{p.name} ({p.provider}) — {p.description[:40]}{is_active}",
                        id=f"prof_{p.name}",
                        classes="profile-button",
                    )
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", variant="default", id="btn-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("prof_"):
            selected = event.button.id[len("prof_") :]
            self.dismiss(selected)
        else:
            self.dismiss(None)


class HelpModal(ModalScreen[None]):
    """Help overlay modal."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        help_markdown = """
# Maulness TUI Shortcuts & Commands

### Keyboard Shortcuts
- **Ctrl + P**: Switch Active Profile
- **Ctrl + D**: View Git Diff Modal
- **Ctrl + L**: Clear Chat Transcript
- **Ctrl + Q**: Quit Application
- **F1**: Show This Help

### Slash Commands
- `/pipeline [name] <goal>` : Execute declarative pipeline (`standard`, `quick`, `plan_only`, `audit`)
- `/profile <name>` : Switch profile on the fly (`builder`, `planner`, `reviewer`, `default`)
- `/diff` : Inspect uncommitted git diff
- `/clear` : Clear chat window
- `/exit` or `/quit` : Exit Maulness
- `!<command>` : Execute local workspace shell command (e.g. `!git status`, `!pytest`)

### Natural Language
Any plain text typed without a `/` or `!` is executed immediately by the **Active Profile** (default: `builder` via Antigravity ACP).
"""
        with Container(classes="diff-dialog"):
            with VerticalScroll():
                yield Static(RichMarkdown(help_markdown))
            with Horizontal(classes="modal-buttons"):
                yield Button("Close (Esc)", variant="primary", id="btn-close")

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
        yield Label(f"[bold blue]Maul[/bold blue] ([cyan]{self.repo}:{self.branch}[/cyan])", classes="user-header")
        yield Static(self.prompt)


class SystemCard(Static):
    """Card displaying system notifications or shell command results."""

    def __init__(self, title: str, content: str, is_error: bool = False):
        super().__init__(classes="system-card")
        self.card_title = title
        self.card_content = content
        self.is_error = is_error

    def compose(self) -> ComposeResult:
        color = "red" if self.is_error else "yellow"
        yield Label(f"[bold {color}]{self.card_title}[/bold {color}]")
        yield Static(self.card_content)


class AgentCard(Static):
    """Card displaying streaming thought, tool calls, and assistant response."""

    def __init__(self, profile_name: str):
        super().__init__(classes="agent-card")
        self.profile_name = profile_name
        self.thought_text: list[str] = []
        self.message_text: list[str] = []
        self.status_label = Label("[dim]Thinking...[/dim]")
        self.thought_static = Static("", classes="thought-box")
        self.message_static = Static("")

    def compose(self) -> ComposeResult:
        yield Label(f"[bold magenta]Maulness[/bold magenta] ([magenta]{self.profile_name}[/magenta])", classes="agent-header")
        yield self.thought_static
        yield self.message_static
        yield self.status_label

    def append_thought(self, delta: str) -> None:
        self.thought_text.append(delta)
        text = "".join(self.thought_text)
        self.thought_static.update(f"[dim italic grey70]💭 {text[-200:].strip()}[/dim italic grey70]")

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
        self.status_label = Label("[dim]Initializing pipeline...[/dim]")
        self.output_static = Static("")

    def compose(self) -> ComposeResult:
        yield Label(
            f"[bold magenta]Pipeline: {self.pipeline_name}[/bold magenta] — [cyan]{self.goal[:60]}[/cyan]",
            classes="agent-header",
        )
        yield self.stages_static
        yield self.output_static
        yield self.status_label

    def update_stage(self, stage_name: str, profile: str, current: int, total: int) -> None:
        self.stages_static.update(
            f"[bold yellow]▶ Stage {current}/{total}:[/bold yellow] [bold white]{stage_name}[/bold white] ([magenta]{profile}[/magenta])"
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
    """Full-screen interactive Textual terminal harness for Maulness."""

    TITLE = "Maulness ⚡"
    CSS = """
    Screen {
        background: #0f172a;
        color: #f1f5f9;
    }

    #top-bar {
        dock: top;
        height: 3;
        background: #1e293b;
        color: #e2e8f0;
        padding: 0 2;
        border-bottom: solid #334155;
    }

    #chat-view {
        height: 1fr;
        padding: 1 2;
    }

    .user-card {
        background: #1e293b;
        border: round #3b82f6;
        padding: 1 2;
        margin: 1 0;
    }

    .user-header {
        color: #60a5fa;
        text-style: bold;
        margin-bottom: 0;
    }

    .agent-card {
        background: #1e1b4b;
        border: round #8b5cf6;
        padding: 1 2;
        margin: 1 0;
    }

    .agent-header {
        color: #c084fc;
        text-style: bold;
        margin-bottom: 1;
    }

    .thought-box {
        color: #94a3b8;
        background: #090d16;
        padding: 0 1;
        margin-bottom: 1;
        border-left: thick #64748b;
    }

    .system-card {
        background: #1c1917;
        border: round #78716c;
        padding: 1 2;
        margin: 1 0;
    }

    #bottom-bar {
        dock: bottom;
        height: 3;
        background: #1e293b;
        padding: 0 1;
        border-top: solid #334155;
    }

    #chat-input {
        width: 1fr;
        background: #0f172a;
        border: none;
        color: #f8fafc;
    }

    #chat-input:focus {
        border: none;
    }

    .modal-dialog {
        width: 70%;
        max-height: 80%;
        background: #1e293b;
        border: thick #8b5cf6;
        padding: 1 2;
    }

    .diff-dialog {
        width: 85%;
        height: 85%;
        background: #0f172a;
        border: thick #3b82f6;
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

    .diff-scroll {
        height: 1fr;
        margin: 1 0;
    }

    .modal-buttons {
        height: 3;
        align: right middle;
    }

    .profile-button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "select_profile", "Switch Profile"),
        Binding("ctrl+d", "view_diff", "View Diff"),
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

    def compose(self) -> ComposeResult:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()

        yield Static(
            f"[bold red]⚡ MAULNESS[/bold red] │ [bold green]repo:[/bold green] {self.repo_name} │ "
            f"[bold cyan]branch:[/bold cyan] {branch} │ [bold magenta]profile:[/bold magenta] {self.current_profile} │ "
            f"{daemon_str} │ [dim]F1: Help, Ctrl+P: Profile, Ctrl+D: Diff, Ctrl+Q: Quit[/dim]",
            id="top-bar",
        )

        with VerticalScroll(id="chat-view"):
            yield Static(
                f"[dim]Welcome back, Maul. Harness active in [green]{self.workspace_path}[/green].\n"
                f"Plain text runs with [magenta]{self.current_profile}[/magenta]. "
                f"Use [cyan]/pipeline <goal>[/cyan] for multi-stage pipelines or [cyan]!<cmd>[/cyan] for shell commands.[/dim]",
                classes="system-card",
            )

        with Horizontal(id="bottom-bar"):
            yield Input(
                placeholder=f"Ask {self.current_profile} or /pipeline, /profile, /diff, !<cmd>...",
                id="chat-input",
            )

    def _update_top_bar(self) -> None:
        branch = get_git_branch(self.workspace_path)
        daemon_str = get_daemon_status()
        status_text = "[yellow]busy[/yellow]" if self.is_busy else "[green]idle[/green]"
        top_bar = self.query_one("#top-bar", Static)
        top_bar.update(
            f"[bold red]⚡ MAULNESS[/bold red] │ [bold green]repo:[/bold green] {self.repo_name} │ "
            f"[bold cyan]branch:[/bold cyan] {branch} │ [bold magenta]profile:[/bold magenta] {self.current_profile} │ "
            f"{daemon_str} │ [dim]{status_text}[/dim]"
        )

    def action_quit_app(self) -> None:
        self.exit()

    def action_clear_chat(self) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()

    def action_view_diff(self) -> None:
        self.push_screen(DiffModal(self.workspace_path))

    async def action_select_profile(self) -> None:
        profiles = self.profile_manager.list_profiles()
        selected = await self.push_screen_wait(ProfileModal(profiles, self.current_profile))
        if selected:
            self.current_profile = selected
            self._update_top_bar()
            chat_input = self.query_one("#chat-input", Input)
            chat_input.placeholder = f"Ask {self.current_profile} or /pipeline, /profile, /diff, !<cmd>..."

    def action_show_help(self) -> None:
        self.push_screen(HelpModal())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        raw_text = event.value.strip()
        if not raw_text:
            return

        chat_input = self.query_one("#chat-input", Input)
        chat_input.value = ""

        if self.is_busy:
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(SystemCard("Busy", "Agent is currently processing a task. Please wait.", is_error=True))
            chat_view.scroll_end(animate=False)
            return

        # Handle Slash Commands & Shell Escapes
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
                chat_input.placeholder = f"Ask {self.current_profile} or /pipeline, /profile, /diff, !<cmd>..."
            else:
                await self.action_select_profile()
            return

        # Local Shell Command Escape: !<cmd>
        if raw_text.startswith("!"):
            cmd = raw_text[1:].strip()
            chat_view = self.query_one("#chat-view", VerticalScroll)
            await chat_view.mount(UserCard(raw_text, self.repo_name, get_git_branch(self.workspace_path)))
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
            # Check if first part matches known pipeline name
            known_pipelines = {p.name for p in self.pipeline_manager.list_pipelines()}
            if len(parts) == 2 and parts[0] in known_pipelines:
                p_name, goal = parts[0], parts[1].strip()
            else:
                p_name, goal = "standard", pipeline_body

            self.run_worker(self._execute_pipeline(p_name, goal), exclusive=True)
            return

        # Default Natural Language Prompt -> Direct Execution with Active Profile
        self.run_worker(self._execute_direct(raw_text), exclusive=True)

    async def _execute_direct(self, prompt: str) -> None:
        self.is_busy = True
        self._update_top_bar()
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
                profile_name=self.current_profile,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
                verbose=False,
            )
            agent_card.set_status(f"[green]✓ Completed ({task.status.value})[/green]")
        except Exception as e:
            agent_card.set_status(f"[red]✗ Failed: {e}[/red]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            chat_view.scroll_end(animate=False)

    async def _execute_pipeline(self, pipeline_name: str, goal: str) -> None:
        self.is_busy = True
        self._update_top_bar()
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
                pipeline_name=pipeline_name,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
                on_gate=on_gate,
                on_stage_start=on_stage_start,
                verbose=False,
            )
            p_card.finish(f"[green]✓ Pipeline Finished ({task.status.value})[/green]")
        except Exception as e:
            p_card.finish(f"[red]✗ Pipeline Failed: {e}[/red]")
        finally:
            self.is_busy = False
            self._update_top_bar()
            chat_view.scroll_end(animate=False)
