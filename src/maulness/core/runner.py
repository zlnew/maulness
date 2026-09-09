import asyncio
import uuid
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
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.storage.db import StorageManager

console = Console()


class TaskRunner:
    """Orchestrates task execution over Profiles, Multi-Providers, and SQLite storage."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        profile_manager: Optional[ProfileManager] = None,
    ):
        self.storage = storage or StorageManager(db_path=config.db_path)
        self.profile_manager = profile_manager or ProfileManager()

    async def run_direct(
        self,
        repo_name: str,
        prompt: str,
        profile_name: str = "builder",
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine]] = None,
    ) -> TaskRecord:
        """Execute a task in Direct Mode with the selected profile."""
        await self.storage.initialize()

        workspace_path = config.resolve_repo_path(repo_name)
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        profile = self.profile_manager.get_profile(profile_name)

        # Record task in database
        task = await self.storage.create_task(
            task_id=task_id,
            title=prompt[:80],
            repo_name=repo_name,
            workspace_path=str(workspace_path),
            mode=TaskMode.DIRECT,
        )

        console.print(
            f"[bold cyan][*] Task {task_id} initialized for [green]{repo_name}[/green] "
            f"using profile [magenta]{profile.name}[/magenta] ([dim]{profile.provider}[/dim])[/bold cyan]"
        )
        console.print(f"[dim]Workspace: {workspace_path}[/dim]\n")

        # Set task to BUILDING
        await self.storage.update_task_status(task_id, TaskStatus.BUILDING)

        # Default terminal approval handler if none provided
        async def terminal_approval_handler(event: ApprovalRequestEvent) -> bool:
            if on_approval:
                return await on_approval(event)

            console.print(
                Panel(
                    f"[bold yellow]Tool:[/bold yellow] {event.tool_name}\n"
                    f"[bold yellow]Arguments:[/bold yellow] {event.args}",
                    title="⚠️  Approval Required (HITL)",
                    border_style="yellow",
                )
            )
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, input, "Approve execution? [y/N]: ")
            approved = answer.strip().lower() in ("y", "yes")
            if approved:
                console.print("[green]✓ Approved[/green]\n")
            else:
                console.print("[red]✗ Denied[/red]\n")
            return approved

        async def default_thought_handler(event: AgentThoughtEvent):
            if on_thought:
                await on_thought(event)
            else:
                console.print(f"[dim italic grey70]💭 {event.delta}[/dim italic grey70]")

        async def default_message_handler(event: AgentMessageEvent):
            if on_message:
                await on_message(event)
            else:
                console.print(event.delta, end="", flush=True)

        try:
            provider = get_provider_for_profile(profile)
            await provider.run(
                session_id=task_id,
                prompt=prompt,
                workspace_path=workspace_path,
                on_thought=default_thought_handler,
                on_message=default_message_handler,
                on_tool_call=on_tool_call,
                on_approval=terminal_approval_handler,
            )

            await self.storage.update_task_status(task_id, TaskStatus.DONE)
            console.print(f"\n[bold green][+] Task {task_id} completed successfully.[/bold green]")

        except FileNotFoundError as e:
            console.print(f"[bold red][x] Execution error: {e}[/bold red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
        except Exception as e:
            console.print(f"\n[bold red][x] Task failed: {e}[/bold red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)

        fetched = await self.storage.get_task(task_id)
        return fetched or task
