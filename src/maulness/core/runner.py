import asyncio
import uuid
from pathlib import Path
from typing import Callable, Coroutine, Optional

from rich.console import Console
from rich.panel import Panel

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
    TaskMode,
    TaskRecord,
    TaskStatus,
)
from maulness.core.approvals import ApprovalClassifier
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.worktree import WorktreeManager
from maulness.storage.db import StorageManager

console = Console()


class TaskRunner:
    """Orchestrates task execution over Profiles, Multi-Providers, and SQLite storage."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        profile_manager: Optional[ProfileManager] = None,
        worktree_manager: Optional[WorktreeManager] = None,
    ):
        self.storage = storage or StorageManager(db_path=config.db_path)
        self.profile_manager = profile_manager or ProfileManager()
        self.worktree_manager = worktree_manager or WorktreeManager()

    async def run_direct(
        self,
        prompt: str,
        repo_name: Optional[str] = None,
        workspace_path: Optional[Path] = None,
        profile_name: str = "builder",
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        origin_platform: str = "cli",
        origin_channel_id: Optional[str | int] = None,
        origin_thread_id: Optional[str | int] = None,
        use_worktree: bool = False,
        yolo: bool = False,
        on_init: Optional[Callable[[str], Coroutine]] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine]] = None,
        verbose: bool = True,
    ) -> TaskRecord:
        """Execute a task in Direct Mode with the selected profile and conversation persistence."""
        await self.storage.initialize()

        profile = self.profile_manager.get_profile(profile_name)
        if repo_name:
            fallback_workspace = Path(workspace_path).resolve() if workspace_path else config.resolve_repo_path(repo_name)
        else:
            fallback_workspace = Path(workspace_path).resolve() if workspace_path else config.workspace_root
        target_workspace = self.profile_manager.resolve_workspace_for_profile(profile, fallback_workspace)
        effective_repo = repo_name or (target_workspace.name if target_workspace != config.workspace_root else None)
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        # Record task in database
        task = await self.storage.create_task(
            task_id=task_id,
            title=prompt[:80],
            repo_name=effective_repo,
            workspace_path=str(target_workspace),
            mode=TaskMode.DIRECT,
            origin_platform=origin_platform,
            origin_channel_id=origin_channel_id,
            origin_thread_id=origin_thread_id,
        )

        session_rec_id = session_id or f"session_{uuid.uuid4().hex[:10]}"
        session = await self.storage.get_session(session_rec_id)
        if not session:
            session = await self.storage.create_session(
                session_id=session_rec_id,
                task_id=task_id,
                profile=profile.name,
                engine=profile.provider,
                acp_session_id=conversation_id,
            )

        effective_conv_id = conversation_id or (session.acp_session_id if session else None)

        async def internal_init_handler(conv_id: str):
            self.last_conversation_id = conv_id
            if session_rec_id:
                await self.storage.update_session_acp_id(session_rec_id, conv_id)
            if on_init:
                await on_init(conv_id)

        if verbose:
            yolo_badge = " [bold red][YOLO][/bold red]" if yolo else ""
            console.print(
                f"[bold cyan][*] Task {task_id} initialized for [green]{repo_name}[/green] "
                f"using profile [magenta]{profile.name}[/magenta] ([dim]{profile.provider}[/dim]){yolo_badge}[/bold cyan]"
            )
            console.print(f"[dim]Workspace: {target_workspace}[/dim]\n")

        # Set task to BUILDING
        await self.storage.update_task_status(task_id, TaskStatus.BUILDING)

        classifier = ApprovalClassifier(yolo_mode=yolo, rule_engine=profile.get_rule_engine())

        # Default terminal approval handler if none provided
        async def terminal_approval_handler(event: ApprovalRequestEvent) -> bool:
            if classifier.should_auto_approve(event.tool_name, event.args):
                if verbose:
                    console.print(
                        f"[dim cyan][Auto-Approved{' / YOLO' if yolo else ''}] {event.tool_name}[/dim cyan]"
                    )
                return True

            if on_approval:
                return await on_approval(event)

            details = f"[bold yellow]Tool:[/bold yellow] {event.tool_name}\n"
            if event.tool_name == "run_command":
                details += f"[bold green]Command:[/bold green] {event.args.get('command')}\n"
                if "cwd" in event.args:
                    details += f"[dim]Directory: {event.args.get('cwd')}[/dim]"
            elif event.tool_name in ("write_file", "read_file"):
                details += f"[bold magenta]Path:[/bold magenta] {event.args.get('path')}\n"
                if "bytes" in event.args:
                    details += f"[dim]Size: {event.args.get('bytes')} bytes[/dim]"
            else:
                details += f"[bold yellow]Arguments:[/bold yellow] {event.args}"

            console.print(
                Panel(
                    details.strip(),
                    title="Approval Required (HITL)",
                    border_style="yellow",
                )
            )
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, input, "Approve execution? [y/N]: ")
            approved = answer.strip().lower() in ("y", "yes")
            if approved:
                console.print("[green]Approved[/green]\n")
            else:
                console.print("[red]Denied[/red]\n")
            return approved

        async def default_thought_handler(event: AgentThoughtEvent):
            if on_thought:
                await on_thought(event)
            else:
                console.print(f"[dim italic grey70][thought] {event.delta}[/dim italic grey70]")

        async def default_message_handler(event: AgentMessageEvent):
            if on_message:
                await on_message(event)
            else:
                console.print(event.delta, end="")

        try:
            provider = get_provider_for_profile(profile)

            async def _execute_on(ws: Path):
                await provider.run(
                    session_id=task_id,
                    prompt=prompt,
                    workspace_path=ws,
                    conversation_id=effective_conv_id,
                    on_init=internal_init_handler,
                    on_thought=default_thought_handler,
                    on_message=default_message_handler,
                    on_tool_call=on_tool_call,
                    on_approval=terminal_approval_handler,
                )

            if use_worktree:
                with self.worktree_manager.isolated_worktree(
                    target_workspace, branch_prefix=f"task-{task_id[:6]}"
                ) as wt_path:
                    if verbose:
                        console.print(f"[dim]Isolated Git Worktree: {wt_path}[/dim]")
                    await _execute_on(wt_path)
            else:
                await _execute_on(target_workspace)

            await self.storage.update_task_status(task_id, TaskStatus.DONE)
            if verbose:
                console.print(f"\n[bold green]Task {task_id} completed successfully.[/bold green]")

        except FileNotFoundError as e:
            if verbose:
                console.print(f"[bold red]Execution error: {e}[/bold red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            raise
        except Exception as e:
            if verbose:
                console.print(f"\n[bold red]Task failed: {e}[/bold red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            raise

        fetched = await self.storage.get_task(task_id)
        return fetched or task
