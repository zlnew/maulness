import asyncio
import json
import logging
import re
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
    ChangeSet,
    Milestone,
    MilestonePlan,
    ReviewVerdict,
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
from maulness.core.kernel.gates import DeterministicGateRunner
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.worktree import WorktreeManager
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.pipeline")
console = Console()


def sync_task_ledger(
    workspace: Path,
    task_id: str,
    title: str,
    definition: PipelineDefinition,
    current_stage_idx: int,
    rework_counts: dict[str, int],
    context: dict[str, Any],
) -> str:
    """Create or update durable .maulness/task.md in target workspace."""
    ledger_dir = workspace / ".maulness"
    ledger_path = ledger_dir / "task.md"
    try:
        ledger_dir.mkdir(parents=True, exist_ok=True)
        total_stages = len(definition.stages)
        current_stage = definition.stages[current_stage_idx] if current_stage_idx < total_stages else None

        stage_lines = []
        for idx, s in enumerate(definition.stages):
            if idx < current_stage_idx:
                prefix = "- [x]"
                state_str = "Passed"
            elif idx == current_stage_idx:
                prefix = "- [/]"
                reworks = rework_counts.get(s.name, 0)
                rework_str = f" (Rework {reworks})" if reworks > 0 else ""
                state_str = f"Active{rework_str}"
            else:
                prefix = "- [ ]"
                state_str = "Pending"
            stage_lines.append(f"{prefix} **{s.name.title()}** (`{s.profile}`) - {state_str}")

        stages_checklist = "\n".join(stage_lines)
        notes = context.get("reviewer_feedback") or context.get("prompt") or "(No additional notes)"

        content = (
            f"# Task Ledger: {title}\n\n"
            f"- **Task ID:** `{task_id}`\n"
            f"- **Pipeline:** `{definition.name}`\n"
            f"- **Active Stage:** `{current_stage.name if current_stage else 'Complete'}`\n\n"
            f"## Pipeline Execution Progress\n"
            f"{stages_checklist}\n\n"
            f"## Active Objective & Feedback\n"
            f"{notes}\n"
        )
        ledger_path.write_text(content, encoding="utf-8")
        return content
    except Exception as e:
        logger.warning("Failed to sync task ledger to '%s': %s", ledger_path, e)
        return ""


def parse_milestone_plan(plan_text: str, task_id: str) -> Optional[MilestonePlan]:
    """Parse structured JSON or markdown milestone list from planner output."""
    if not plan_text:
        return None

    # 1. Try finding fenced or raw JSON MilestonePlan
    json_candidates = []
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", plan_text, re.DOTALL)
    if json_match:
        json_candidates.append(json_match.group(1))

    # Also check if text has { "milestones": ... }
    m_block = re.search(r"(\{\s*\"(?:task_id|milestones)\".*?\})", plan_text, re.DOTALL)
    if m_block:
        json_candidates.append(m_block.group(1))

    for cand in json_candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and "milestones" in data:
                return MilestonePlan.model_validate(data)
        except Exception:
            pass

    # 2. Heuristic extraction from markdown milestone headings (e.g. "### Milestone 1: ...")
    milestones: list[Milestone] = []
    blocks = re.split(r"(?m)^#+\s*(?:Milestone\s*\d+|M\d+)\s*[:\-]\s*", plan_text)
    if len(blocks) > 1:
        headers = re.findall(r"(?m)^#+\s*(?:Milestone\s*\d+|M\d+)\s*[:\-]\s*([^\n]+)", plan_text)
        for idx, (head, body) in enumerate(zip(headers, blocks[1:], strict=False), 1):
            cmd_match = re.search(r"(?:Verification|Test|Command)\s*:\s*`([^`]+)`", body)
            ver_cmd = cmd_match.group(1) if cmd_match else ""
            crit_match = re.search(r"(?:Criteria|Acceptance)\s*:\s*([^\n]+)", body)
            crit = crit_match.group(1).strip() if crit_match else body.strip()[:200]
            milestones.append(
                Milestone(
                    id=f"M{idx}",
                    title=head.strip(),
                    verification_command=ver_cmd,
                    acceptance_criteria=crit,
                )
            )

    if milestones:
        return MilestonePlan(task_id=task_id, summary=plan_text[:300], milestones=milestones)

    return None


class PipelineOrchestrator:
    """Coordinates declarative multi-stage Kanban pipelines driven by YAML definitions."""

    def __init__(
        self,
        storage: Optional[StorageManager] = None,
        profile_manager: Optional[ProfileManager] = None,
        pipeline_manager: Optional[PipelineManager] = None,
        worktree_manager: Optional[WorktreeManager] = None,
        gate_runner: Optional[DeterministicGateRunner] = None,
    ):
        self.storage = storage or StorageManager(db_path=config.db_path)
        self.profile_manager = profile_manager or ProfileManager()
        self.pipeline_manager = pipeline_manager or PipelineManager()
        self.worktree_manager = worktree_manager or WorktreeManager()
        self.gate_runner = gate_runner or DeterministicGateRunner(storage=self.storage)


    async def run_pipeline(
        self,
        title: str,
        prompt: str,
        repo_name: Optional[str] = None,
        workspace_path: Optional[Path] = None,
        pipeline_name: str = "standard",
        pipeline_def: Optional[PipelineDefinition] = None,
        origin_platform: str = "cli",
        origin_channel_id: Optional[str | int] = None,
        origin_thread_id: Optional[str | int] = None,
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
        target_workspace = Path(workspace_path).resolve() if workspace_path else (config.resolve_repo_path(repo_name) if repo_name else config.workspace_root)
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
            origin_platform=origin_platform,
            origin_channel_id=origin_channel_id,
            origin_thread_id=origin_thread_id,
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
                "task_ledger": "",
                "gate_verification": "",
                "gate_failures": "",
                "last_gate_result": {},
            }


            total_stages = len(definition.stages)
            stage_map = {s.name: i for i, s in enumerate(definition.stages)}
            rework_counts: dict[str, int] = {}
            stage_executions: dict[str, int] = {}
            stage_checkpoints: dict[str, str] = {}

            stage_index = 0
            while stage_index < total_stages:
                stage = definition.stages[stage_index]
                display_idx = stage_index + 1

                # Sync durable task ledger in workspace
                context["task_ledger"] = sync_task_ledger(
                    effective_workspace,
                    task_id,
                    title,
                    definition,
                    stage_index,
                    rework_counts,
                    context,
                )

                # Capture pre-stage git checkpoint if explicitly requested or rollback configured
                should_checkpoint = (
                    stage.checkpoint_before_stage
                    or bool(stage.transitions and stage.transitions.rollback_on_rework)
                )
                if should_checkpoint and self.worktree_manager.is_git_repo(effective_workspace):
                    exec_count = stage_executions.get(stage.name, 0)
                    cp_label = f"{task_id}_{stage.name}_r{exec_count}"
                    cp_hash = self.worktree_manager.create_checkpoint(effective_workspace, cp_label)
                    if cp_hash:
                        stage_checkpoints[stage.name] = cp_hash

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

                if stage.is_gate_only:
                    stage_output = ""
                else:
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

                    # Phase 2: Check if stage runs milestone sub-sessions
                    plan_candidate = context.get("plan") or context.get("previous_output") or ""
                    parsed_plan = parse_milestone_plan(plan_candidate, task_id)

                    if (stage.run_milestones or (stage.profile == "builder" and parsed_plan and len(parsed_plan.milestones) > 1)):
                        if verbose:
                            console.print(
                                f"[bold magenta][*] Launching Milestone Sub-Sessions ({len(parsed_plan.milestones)} milestones)[/bold magenta]"
                            )

                        milestone_outputs = []
                        for m_idx, milestone in enumerate(parsed_plan.milestones, 1):
                            if verbose:
                                console.print(
                                    f"\n[cyan]── Milestone {m_idx}/{len(parsed_plan.milestones)}: {milestone.title} ({milestone.id}) ──[/cyan]"
                                )

                            # Context Flushing: Fresh minimal prompt turn carrying only high-level plan, current goal, and git stat
                            diff_stat_res = subprocess.run(
                                ["git", "diff", "--stat", "HEAD"],
                                cwd=str(stage_workspace),
                                capture_output=True,
                                text=True,
                            )
                            git_stat = diff_stat_res.stdout.strip() or "(Clean worktree)"

                            milestone_prompt = (
                                f"# High-Level Task Objective\n{title}\n\n"
                                f"## Active Milestone ({milestone.id}): {milestone.title}\n"
                                f"Acceptance Criteria: {milestone.acceptance_criteria}\n"
                            )
                            if milestone.verification_command:
                                milestone_prompt += f"Verification Command: `{milestone.verification_command}`\n"
                            milestone_prompt += (
                                f"\n## Cumulative Worktree Changes\n{git_stat}\n\n"
                                f"Implement this milestone using your available tools. Focus strictly on {milestone.title}."
                            )

                            m_session_id = f"{session_id}_{milestone.id}"
                            try:
                                m_output = await provider.run(
                                    session_id=m_session_id,
                                    prompt=milestone_prompt,
                                    workspace_path=stage_workspace,
                                    on_thought=on_thought,
                                    on_message=on_message,
                                    on_approval=on_approval if stage.requires_approval else None,
                                    max_tool_turns=10,  # Hard limit per milestone sub-session
                                )
                                milestone_outputs.append(f"### {milestone.id}: {milestone.title}\n{m_output}")

                                # Run milestone verification command if provided
                                if milestone.verification_command and self.worktree_manager.is_git_repo(stage_workspace):
                                    v_res = subprocess.run(
                                        milestone.verification_command,
                                        shell=True,
                                        cwd=str(stage_workspace),
                                        capture_output=True,
                                        text=True,
                                        timeout=60,
                                    )
                                    if v_res.returncode != 0:
                                        if verbose:
                                            console.print(
                                                f"[bold yellow][!] Milestone {milestone.id} verification failed. "
                                                "Attempting git rollback...[/bold yellow]"
                                            )
                                        self.worktree_manager.rollback_to_previous_milestone(stage_workspace)
                                    else:
                                        self.worktree_manager.milestone_checkpoint(stage_workspace, milestone.id, milestone.title)
                                else:
                                    if self.worktree_manager.is_git_repo(stage_workspace):
                                        self.worktree_manager.milestone_checkpoint(stage_workspace, milestone.id, milestone.title)

                            except Exception as m_err:
                                if verbose:
                                    console.print(f"[red][x] Milestone {milestone.id} error: {m_err}[/red]")
                                if self.worktree_manager.is_git_repo(stage_workspace):
                                    self.worktree_manager.rollback_to_previous_milestone(stage_workspace)
                                raise

                        stage_output = "\n\n".join(milestone_outputs)
                    else:
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

                # Execute deterministic verification gate if configured
                if stage.verification_gate:
                    gate_cfg = stage.verification_gate
                    gate_cwd = (
                        Path(gate_cfg.cwd).resolve()
                        if gate_cfg.cwd
                        else effective_workspace
                    )
                    step_idx = await self.storage.get_latest_agent_step_index(task_id, stage.name) + 1
                    gate_res = await self.gate_runner.run_gate(
                        task_id=task_id,
                        stage=stage.name,
                        step_index=step_idx,
                        command=gate_cfg.command,
                        workspace_path=gate_cwd,
                        timeout_seconds=gate_cfg.timeout_seconds,
                        sandbox_mode=gate_cfg.sandbox_mode,
                    )
                    context["last_gate_result"] = gate_res.to_dict()

                    if not gate_res.passed:
                        context["gate_failures"] = gate_res.to_feedback_prompt()
                        if verbose:
                            console.print(
                                f"[bold red][!] Deterministic verification gate failed for stage '{stage.name}': "
                                f"{gate_res.summary}[/bold red]"
                            )

                        if gate_cfg.auto_rework_on_fail:
                            cur_reworks = rework_counts.get(stage.name, 0)
                            max_reworks = stage.transitions.max_reworks if stage.transitions else 2
                            rework_target = (
                                stage.transitions.rework_target
                                if stage.transitions and stage.transitions.rework_target
                                else stage.name
                            )

                            if cur_reworks < max_reworks:
                                rework_counts[stage.name] = cur_reworks + 1
                                if verbose:
                                    console.print(
                                        f"[bold yellow][!] Deterministic verification triggered REWORK "
                                        f"(Cycle {rework_counts[stage.name]}/{max_reworks}). "
                                        f"Looping back to '{rework_target}'...[/bold yellow]"
                                    )

                                if stage.transitions and stage.transitions.rollback_on_rework:
                                    target_cp = stage_checkpoints.get(rework_target)
                                    if target_cp:
                                        rolled_back = self.worktree_manager.rollback_to_checkpoint(
                                            effective_workspace, target_cp
                                        )
                                        if rolled_back and verbose:
                                            console.print(
                                                f"[bold yellow][!] Rolled back working tree to pre-stage checkpoint "
                                                f"{target_cp[:8]} for clean rework restart.[/bold yellow]"
                                            )

                                context["reviewer_feedback"] = gate_res.to_feedback_prompt()
                                context["rework_feedback_block"] = (
                                    f"\n## Deterministic Verification Gate Failure "
                                    f"(Rework Cycle {rework_counts[stage.name]}/{max_reworks})\n"
                                    f"{gate_res.to_feedback_prompt()}\n\n"
                                )

                                if rework_target in stage_map:
                                    stage_index = stage_map[rework_target]
                                    continue
                                else:
                                    if verbose:
                                        console.print(
                                            f"[red][x] Rework target '{rework_target}' not found in pipeline stages.[/red]"
                                        )
                                    await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                                    return await self.storage.get_task(task_id) or task
                            else:
                                if verbose:
                                    console.print(
                                        f"[bold red][x] Maximum rework attempts reached ({max_reworks}) "
                                        f"for deterministic verification on '{stage.name}'. Halting pipeline.[/bold red]"
                                    )
                                await self.storage.update_task_status(task_id, TaskStatus.FAILED)
                                return await self.storage.get_task(task_id) or task
                    else:
                        context["gate_verification"] = (
                            f"Deterministic verification gate passed for `{gate_cfg.command}`: {gate_res.summary}"
                        )
                        context["gate_failures"] = ""
                        if verbose:
                            console.print(
                                f"[bold green][+] Deterministic verification gate passed: {gate_res.summary}[/bold green]"
                            )
                        if stage.is_gate_only:
                            stage_output = gate_res.summary


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
                                # Rollback working tree if configured on stage transitions
                                if stage.transitions.rollback_on_rework:
                                    target_cp = stage_checkpoints.get(target_name)
                                    if target_cp:
                                        rolled_back = self.worktree_manager.rollback_to_checkpoint(
                                            effective_workspace, target_cp
                                        )
                                        if rolled_back and verbose:
                                            console.print(
                                                f"[bold yellow][!] Rolled back working tree to pre-stage checkpoint "
                                                f"{target_cp[:8]} for clean rework restart.[/bold yellow]"
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

            # Final sync for task ledger marking completion
            sync_task_ledger(
                effective_workspace,
                task_id,
                title,
                definition,
                total_stages,
                rework_counts,
                context,
            )

            if verbose:
                console.print(
                    f"\n[bold green][+] Pipeline '{definition.name}' completed successfully![/bold green]"
                )

            fetched = await self.storage.get_task(task_id)
            return fetched or task

        if use_worktree is not None:
            effective_use_worktree = use_worktree
        else:
            effective_use_worktree = any(s.use_worktree for s in definition.stages)

        if effective_use_worktree and self.worktree_manager.is_git_repo(target_workspace):
            with self.worktree_manager.isolated_worktree(
                target_workspace, branch_prefix=f"pipe-{task_id[:6]}"
            ) as wt_path:
                if verbose:
                    console.print(f"[dim]Running pipeline in isolated worktree: {wt_path}[/dim]")

                wt_branch = None
                try:
                    b_res = subprocess.run(
                        ["git", "branch", "--show-current"],
                        cwd=str(wt_path),
                        capture_output=True,
                        text=True,
                    )
                    if b_res.returncode == 0 and b_res.stdout.strip():
                        wt_branch = b_res.stdout.strip()
                except Exception:
                    pass

                result = await _execute_stages(wt_path)

                if wt_branch:
                    try:
                        await self.storage.record_agent_event(
                            task_id=task_id,
                            stage="pipeline",
                            step_index=0,
                            event_type="worktree_branch",
                            payload={"branch": wt_branch},
                        )
                    except Exception as ev_err:
                        logger.debug("Failed to record worktree_branch event: %s", ev_err)

                return result
        else:
            return await _execute_stages(target_workspace)
