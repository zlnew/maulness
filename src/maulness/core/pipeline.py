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
from maulness.core.pipelines import (
    PipelineDefinition,
    PipelineManager,
    PipelineStage,
    check_is_fail_verdict,
    check_is_pass_verdict,
    check_is_rework_verdict,
)
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
        yolo: bool = False,
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
        effective_auto_proceed = auto_proceed or yolo
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
                "rework_feedback_block": "",
                "reviewer_feedback": "",
            }

            total_stages = len(definition.stages)
            stage_map = {s.name: i for i, s in enumerate(definition.stages)}
            rework_counts: dict[str, int] = {}
            stage_executions: dict[str, int] = {}

            stage_index = 0
            while stage_index < total_stages:
                stage = definition.stages[stage_index]
                display_idx = stage_index + 1

                # Check gate before stage begins if specified
                if stage.gate and not effective_auto_proceed:
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
                        f"\n[bold cyan]── Stage {display_idx}/{total_stages}: {stage.name.title()} ({stage.profile}) ──[/bold cyan]"
                    )

                if on_stage_start:
                    await on_stage_start(stage, display_idx, total_stages)

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

                exec_count = stage_executions.get(stage.name, 0)
                stage_executions[stage.name] = exec_count + 1
                session_suffix = f"_r{exec_count}" if exec_count > 0 else ""
                session_id = f"{task_id}_{stage.name}{session_suffix}"

                try:
                    stage_output = await provider.run(
                        session_id=session_id,
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
                    console.print(f"[green]Stage '{stage.name}' complete.[/green]")

                # Evaluate transitions
                if stage.transitions:
                    # 1. Rework check
                    if stage.transitions.rework_target and check_is_rework_verdict(stage_output):
                        cur_reworks = rework_counts.get(stage.name, 0)
                        if cur_reworks < stage.transitions.max_reworks:
                            rework_counts[stage.name] = cur_reworks + 1
                            target_name = stage.transitions.rework_target
                            if target_name in stage_map:
                                if verbose:
                                    console.print(
                                        f"[bold yellow][!] Stage '{stage.name}' requested REWORK "
                                        f"(Cycle {rework_counts[stage.name]}/{stage.transitions.max_reworks}). "
                                        f"Looping back to '{target_name}'...[/bold yellow]"
                                    )
                                context["reviewer_feedback"] = stage_output
                                context["rework_feedback_block"] = (
                                    f"\n## Reviewer Feedback (Rework Cycle {rework_counts[stage.name]}/{stage.transitions.max_reworks})\n"
                                    f"{stage_output}\n\n"
                                    f"Address all issues highlighted in the review feedback above.\n"
                                )
                                stage_index = stage_map[target_name]
                                continue
                            else:
                                if verbose:
                                    console.print(
                                        f"[red][x] Rework target '{target_name}' not found in pipeline stages.[/red]"
                                    )
                                await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                                return await self.storage.get_task(task_id) or task
                        else:
                            if verbose:
                                console.print(
                                    f"[bold red][x] Maximum rework attempts reached ({stage.transitions.max_reworks}) "
                                    f"for stage '{stage.name}'. Halting pipeline for user steering.[/bold red]"
                                )
                            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                            return await self.storage.get_task(task_id) or task

                    # 2. Fail check
                    if stage.transitions.fail_target and check_is_fail_verdict(stage_output):
                        target_name = stage.transitions.fail_target
                        if target_name in stage_map:
                            stage_index = stage_map[target_name]
                            continue
                        else:
                            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                            return await self.storage.get_task(task_id) or task

                    # 3. Explicit pass target check
                    if stage.transitions.pass_target:
                        target_name = stage.transitions.pass_target
                        if target_name in stage_map:
                            context["rework_feedback_block"] = ""
                            stage_index = stage_map[target_name]
                            continue

                # Normal sequential progression
                context["rework_feedback_block"] = ""
                stage_index += 1

            # All stages finished successfully
            await self.storage.update_task_status(task_id, TaskStatus.DONE)
            if verbose:
                console.print(
                    f"\n[bold green][+] Pipeline '{definition.name}' completed successfully![/bold green]"
                )

            fetched = await self.storage.get_task(task_id)
            return fetched or task

        effective_use_worktree = use_worktree or any(s.use_worktree for s in definition.stages)
        if effective_use_worktree:
            with self.worktree_manager.isolated_worktree(
                target_workspace, branch_prefix=f"pipe-{task_id[:6]}"
            ) as wt_path:
                if verbose:
                    console.print(f"[dim]Running pipeline in isolated worktree: {wt_path}[/dim]")
                return await _execute_stages(wt_path)
        else:
            return await _execute_stages(target_workspace)
