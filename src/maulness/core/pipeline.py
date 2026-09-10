import asyncio
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from rich.console import Console
from rich.panel import Panel

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    ApprovalRequestEvent,
    TaskMode,
    TaskRecord,
    TaskStatus,
)
from maulness.core.pipelines import PipelineDefinition, PipelineManager, PipelineStage
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.worktree import WorktreeManager
from maulness.storage.db import StorageManager

console = Console()


class PipelineOrchestrator:
    """Coordinates declarative multi-stage Kanban pipelines driven by YAML definitions."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        profile_manager: Optional[ProfileManager] = None,
        pipeline_manager: Optional[PipelineManager] = None,
        worktree_manager: Optional[WorktreeManager] = None,
    ):
        self.storage = storage or StorageManager(db_path=config.db_path)
        self.profile_manager = profile_manager or ProfileManager()
        self.pipeline_manager = pipeline_manager or PipelineManager()
        self.worktree_manager = worktree_manager or WorktreeManager()

    async def run_pipeline(
        self,
        repo_name: str,
        title: str,
        prompt: str,
        workspace_path: Optional[Path] = None,
        pipeline_name: str = "standard",
        pipeline_def: Optional[PipelineDefinition] = None,
        discord_thread_id: Optional[int] = None,
        use_worktree: bool = False,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine]] = None,
        on_gate: Optional[Callable[[str], Coroutine[Any, Any, bool]]] = None,
        on_stage_start: Optional[Callable[[PipelineStage, int, int], Coroutine]] = None,
        on_stage_finish: Optional[Callable[[PipelineStage, str], Coroutine]] = None,
        auto_proceed: bool = False,
        verbose: bool = True,
    ) -> TaskRecord:
        """Execute a declarative pipeline across defined stages."""
        await self.storage.initialize()
        target_workspace = Path(workspace_path).resolve() if workspace_path else config.resolve_repo_path(repo_name)
        task_id = f"task_{uuid.uuid4().hex[:10]}"

        # Load pipeline definition
        definition = pipeline_def or self.pipeline_manager.get_pipeline(pipeline_name)

        # 1. Initialize Task Record
        task = await self.storage.create_task(
            task_id=task_id,
            title=title,
            repo_name=repo_name,
            workspace_path=str(target_workspace),
            mode=TaskMode.MULTI,
            discord_thread_id=discord_thread_id,
        )

        if verbose:
            console.print(
                f"[bold magenta][*] Starting Pipeline '{definition.name}' (Task {task_id}) for [green]{repo_name}[/green][/bold magenta]"
            )
            console.print(f"[dim]Goal: {title}[/dim]\n")

        async def _execute_stages(effective_workspace: Path) -> TaskRecord:
            # Context store shared and mutated across stages
            context: dict[str, Any] = {
                "repo_name": repo_name,
                "workspace_path": str(effective_workspace),
                "title": title,
                "prompt": prompt,
                "git_diff": "",
                "previous_output": "",
            }

            total_stages = len(definition.stages)

            for idx, stage in enumerate(definition.stages, start=1):
                # Check gate before stage begins if specified
                if stage.gate and not auto_proceed:
                    proceed = True
                    if on_gate:
                        proceed = await on_gate(stage.gate.prompt)
                    else:
                        loop = asyncio.get_running_loop()
                        ans = await loop.run_in_executor(
                            None, input, f"\n{stage.gate.prompt} [Y/n]: "
                        )
                        proceed = ans.strip().lower() not in ("n", "no")

                    if not proceed:
                        if verbose:
                            console.print(
                                f"[yellow][!] Pipeline paused at stage '{stage.name}' by user.[/yellow]"
                            )
                        return task

                # Update DB status
                status_enum = TaskStatus.BUILDING
                try:
                    status_enum = TaskStatus(stage.status)
                except ValueError:
                    pass
                await self.storage.update_task_status(task_id, status_enum)

                if verbose:
                    console.print(
                        f"\n[bold cyan]── Stage {idx}/{total_stages}: {stage.name.title()} ({stage.profile}) ──[/bold cyan]"
                    )

                if on_stage_start:
                    await on_stage_start(stage, idx, total_stages)

                # If stage requires git diff, refresh it now
                if stage.requires_diff:
                    diff_res = subprocess.run(
                        ["git", "diff", "HEAD"],
                        cwd=str(effective_workspace),
                        capture_output=True,
                        text=True,
                    )
                    context["git_diff"] = diff_res.stdout or "(No uncommitted diffs detected)"

                # Render stage prompt
                stage_prompt = self.pipeline_manager.render_stage_prompt(stage, context)

                profile = self.profile_manager.get_profile(stage.profile)
                stage_workspace = self.profile_manager.resolve_workspace_for_profile(
                    profile, effective_workspace
                )
                provider = get_provider_for_profile(profile)

                try:
                    stage_output = await provider.run(
                        session_id=f"{task_id}_{stage.name}",
                        prompt=stage_prompt,
                        workspace_path=stage_workspace,
                        on_thought=on_thought,
                        on_message=on_message,
                        on_approval=on_approval if stage.requires_approval else None,
                    )
                except Exception as e:
                    if verbose:
                        console.print(f"[red][x] Stage '{stage.name}' failed: {e}[/red]")
                    await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                    raise

                # Store output in context
                context["previous_output"] = stage_output
                if stage.output_key:
                    context[stage.output_key] = stage_output

                if on_stage_finish:
                    await on_stage_finish(stage, stage_output)

                if verbose:
                    console.print(f"[green]✓ Stage '{stage.name}' complete.[/green]")

            # All stages finished successfully
            await self.storage.update_task_status(task_id, TaskStatus.DONE)
            if verbose:
                console.print(
                    f"\n[bold green][+] Pipeline '{definition.name}' completed successfully![/bold green]"
                )

            fetched = await self.storage.get_task(task_id)
            return fetched or task

        if use_worktree:
            with self.worktree_manager.isolated_worktree(
                target_workspace, branch_prefix=f"pipe-{task_id[:6]}"
            ) as wt_path:
                if verbose:
                    console.print(f"[dim]Running pipeline in isolated worktree: {wt_path}[/dim]")
                return await _execute_stages(wt_path)
        else:
            return await _execute_stages(target_workspace)
