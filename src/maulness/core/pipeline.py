import asyncio
import subprocess
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


class PipelineOrchestrator:
    """Coordinates Multi-Route Kanban lifecycle: Planner ➔ Builder ➔ Reviewer."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        profile_manager: Optional[ProfileManager] = None,
    ):
        self.storage = storage or StorageManager(db_path=config.db_path)
        self.profile_manager = profile_manager or ProfileManager()

    async def run_pipeline(
        self,
        repo_name: str,
        title: str,
        prompt: str,
        discord_thread_id: Optional[int] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine]] = None,
        auto_proceed: bool = False,
    ) -> TaskRecord:
        """Execute the full 3-stage assembly line: Planner ➔ Builder ➔ Reviewer."""
        await self.storage.initialize()
        workspace_path = config.resolve_repo_path(repo_name)
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        # 1. Initialize Task
        task = await self.storage.create_task(
            task_id=task_id,
            title=title,
            repo_name=repo_name,
            workspace_path=str(workspace_path),
            mode=TaskMode.MULTI,
            discord_thread_id=discord_thread_id,
        )

        console.print(f"[bold magenta][*] Starting Pipeline Task {task_id} for [green]{repo_name}[/green][/bold magenta]")
        console.print(f"[dim]Goal: {title}[/dim]\n")

        # ==========================================
        # STAGE 1: PLANNING
        # ==========================================
        console.print("[bold blue]── Stage 1: Planning (Shipwright Architect) ──[/bold blue]")
        await self.storage.update_task_status(task_id, TaskStatus.PLANNING)

        planner_profile = self.profile_manager.get_profile("planner")
        planner_provider = get_provider_for_profile(planner_profile)

        planning_prompt = (
            f"You are drafting an implementation plan for repository '{repo_name}'.\n"
            f"Goal: {title}\n\n"
            f"Details & Requirements:\n{prompt}\n\n"
            f"Please output a structured markdown plan covering proposed changes, "
            f"files to modify/create, risk assessment, and verification steps."
        )

        plan_output = await planner_provider.run(
            session_id=f"{task_id}_plan",
            prompt=planning_prompt,
            workspace_path=workspace_path,
            on_thought=on_thought,
            on_message=on_message,
        )

        console.print(Panel(plan_output, title="📋 Draft Implementation Plan", border_style="blue"))

        # Plan Approval Gate
        if not auto_proceed:
            loop = asyncio.get_running_loop()
            answer = await loop.run_in_executor(None, input, "\nProceed to Building stage with Antigravity ACP? [Y/n]: ")
            if answer.strip().lower() in ("n", "no"):
                console.print("[yellow][!] Pipeline paused by user at Planning stage.[/yellow]")
                return task

        # ==========================================
        # STAGE 2: BUILDING (Antigravity ACP)
        # ==========================================
        console.print("\n[bold yellow]── Stage 2: Building (Antigravity Senior Deckhand) ──[/bold yellow]")
        await self.storage.update_task_status(task_id, TaskStatus.BUILDING)

        builder_profile = self.profile_manager.get_profile("builder")
        builder_provider = get_provider_for_profile(builder_profile)

        build_prompt = (
            f"Goal: {title}\n\n"
            f"Approved Plan:\n{plan_output}\n\n"
            f"Execute the changes, inspect diffs, and run test suites to verify."
        )

        try:
            await builder_provider.run(
                session_id=f"{task_id}_build",
                prompt=build_prompt,
                workspace_path=workspace_path,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
            )
            console.print("[green]✓ Building stage complete.[/green]")
        except Exception as e:
            console.print(f"[red][x] Building stage failed: {e}[/red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            return task

        # ==========================================
        # STAGE 3: REVIEW (Independent Diff Audit)
        # ==========================================
        console.print("\n[bold cyan]── Stage 3: Review (Independent Diff Auditor) ──[/bold cyan]")
        await self.storage.update_task_status(task_id, TaskStatus.REVIEW)

        # Grab git diff
        diff_res = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
        )
        git_diff = diff_res.stdout or "(No uncommitted diffs detected)"

        reviewer_profile = self.profile_manager.get_profile("reviewer")
        reviewer_provider = get_provider_for_profile(reviewer_profile)

        review_prompt = (
            f"Review the following changes made for task '{title}':\n\n"
            f"Git Diff:\n```diff\n{git_diff[:4000]}\n```\n\n"
            f"Provide an audit scorecard: PASS or REWORK with reasons."
        )

        review_output = await reviewer_provider.run(
            session_id=f"{task_id}_review",
            prompt=review_prompt,
            workspace_path=workspace_path,
            on_thought=on_thought,
            on_message=on_message,
        )

        console.print(Panel(review_output, title="🔍 Review Audit Scorecard", border_style="cyan"))

        await self.storage.update_task_status(task_id, TaskStatus.DONE)
        console.print(f"\n[bold green][+] Pipeline Task {task_id} successfully completed![/bold green]")

        fetched = await self.storage.get_task(task_id)
        return fetched or task
