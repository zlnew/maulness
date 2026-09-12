import asyncio
import json
import logging
import re
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
    RepoGotcha,
    TaskMode,
    TaskRecord,
    TaskRetrospective,
    TaskStatus,
)
from maulness.core.approvals import ApprovalClassifier
from maulness.core.kernel.budgets import BudgetExceededError, BudgetGuard
from maulness.core.kernel.gates import TestFreezeGate, TestTamperingDetectedError
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.worktree import WorktreeManager
from maulness.storage.db import StorageManager

console = Console()
logger = logging.getLogger("maulness.runner")


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
            fallback_workspace = (
                Path(workspace_path).resolve()
                if workspace_path
                else config.resolve_repo_path(repo_name)
            )
        else:
            fallback_workspace = (
                Path(workspace_path).resolve()
                if workspace_path
                else config.workspace_root
            )
        target_workspace = self.profile_manager.resolve_workspace_for_profile(
            profile, fallback_workspace
        )
        effective_repo = repo_name or (
            target_workspace.name if target_workspace != config.workspace_root else None
        )
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

        effective_conv_id = conversation_id or (
            session.acp_session_id if session else None
        )

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

        classifier = ApprovalClassifier(
            yolo_mode=yolo, rule_engine=profile.get_rule_engine()
        )

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
                details += (
                    f"[bold green]Command:[/bold green] {event.args.get('command')}\n"
                )
                if "cwd" in event.args:
                    details += f"[dim]Directory: {event.args.get('cwd')}[/dim]"
            elif event.tool_name in ("write_file", "read_file"):
                details += (
                    f"[bold magenta]Path:[/bold magenta] {event.args.get('path')}\n"
                )
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
            answer = await loop.run_in_executor(
                None, input, "Approve execution? [y/N]: "
            )
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
                console.print(
                    f"[dim italic grey70][thought] {event.delta}[/dim italic grey70]"
                )

        async def default_message_handler(event: AgentMessageEvent):
            if on_message:
                await on_message(event)
            else:
                console.print(event.delta, end="")

        budget_guard = BudgetGuard()

        # Pre-task Query: Inject active commit-anchored gotchas
        active_gotchas = await self.storage.get_active_gotchas(
            str(target_workspace), limit=3
        )
        effective_prompt = prompt
        if active_gotchas:
            gotcha_lines = ["\n[Known Workspace Gotchas]:"]
            for g in active_gotchas:
                gotcha_lines.append(
                    f"- **{g.component}**: {g.symptom} -> {g.resolution} (commit {g.commit_hash[:7]})"
                )
            effective_prompt = f"{prompt}\n" + "\n".join(gotcha_lines)

        try:
            provider = get_provider_for_profile(profile)

            last_result: str = ""

            async def _execute_on(ws: Path):
                nonlocal last_result
                # Ensure baseline tests are hashed on actual workspace/worktree
                active_freeze = TestFreezeGate(ws)

                async def wrapped_tool_call(event: AgentToolCallEvent):
                    # Check operational budget before tool execution
                    budget_guard.record_turn()
                    if on_tool_call:
                        await on_tool_call(event)

                last_result = await provider.run(
                    session_id=task_id,
                    prompt=effective_prompt,
                    workspace_path=ws,
                    conversation_id=effective_conv_id,
                    on_init=internal_init_handler,
                    on_thought=default_thought_handler,
                    on_message=default_message_handler,
                    on_tool_call=wrapped_tool_call,
                    on_approval=terminal_approval_handler,
                )

                # Final test integrity freeze verification
                active_freeze.verify_no_tampering()

            if use_worktree:
                with self.worktree_manager.isolated_worktree(
                    target_workspace, branch_prefix=f"task-{task_id[:6]}"
                ) as wt_path:
                    if verbose:
                        console.print(f"[dim]Isolated Git Worktree: {wt_path}[/dim]")
                    await _execute_on(wt_path)
            else:
                await _execute_on(target_workspace)

            # Invalidate gotchas if modified files touched components
            current_head_hash = "unknown"
            if self.worktree_manager.is_git_repo(target_workspace):
                import subprocess

                diff_res = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD~1"],
                    cwd=str(target_workspace),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if diff_res.returncode == 0 and diff_res.stdout.strip():
                    mod_files = [
                        f.strip() for f in diff_res.stdout.splitlines() if f.strip()
                    ]
                    await self.storage.invalidate_gotchas_for_files(
                        str(target_workspace), mod_files
                    )

                rev_res = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(target_workspace),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if rev_res.returncode == 0 and rev_res.stdout.strip():
                    current_head_hash = rev_res.stdout.strip()[:10]

            # G1: Post-task LLM gotcha extraction turn
            try:
                extract_prompt = (
                    "Extract 1-2 non-obvious environmental gotchas, repository quirks, or build/test fixes "
                    f"discovered during this task execution:\n\nTask: {prompt}\n\n"
                    f"Result: {last_result[:2000]}\n\n"
                    'Format response strictly as JSON array: [{"component": "...", "symptom": "...", "resolution": "..."}] '
                    "or reply with 'NONE' if no non-obvious environmental or tool quirks were discovered."
                )
                extract_session = f"extract_{task_id}_{uuid.uuid4().hex[:6]}"
                gotcha_text = await provider.run(
                    session_id=extract_session,
                    prompt=extract_prompt,
                    workspace_path=target_workspace,
                    max_tool_turns=1,
                )
                if gotcha_text and "NONE" not in gotcha_text.upper():
                    m = re.search(r"\[.*\]", gotcha_text, flags=re.DOTALL)
                    if m:
                        items = json.loads(m.group(0))
                        if isinstance(items, list):
                            for item in items[:2]:
                                if (
                                    isinstance(item, dict)
                                    and item.get("component")
                                    and item.get("symptom")
                                    and item.get("resolution")
                                ):
                                    g_id = f"gotcha_{uuid.uuid4().hex[:8]}"
                                    gotcha = RepoGotcha(
                                        id=g_id,
                                        repo_path=str(target_workspace),
                                        component=str(item["component"])[:100],
                                        symptom=str(item["symptom"])[:300],
                                        resolution=str(item["resolution"])[:300],
                                        commit_hash=current_head_hash,
                                    )
                                    await self.storage.save_repo_gotcha(gotcha)
                                    logger.info(
                                        "Saved extracted gotcha for %s: %s",
                                        target_workspace,
                                        gotcha.component,
                                    )
            except Exception as extract_err:
                logger.debug(
                    "Gotcha extraction turn skipped or failed: %s", extract_err
                )

            # G12: Record enriched task retrospective
            clean_prompt = prompt.replace("\n", " ").strip()
            retro_summary = (
                f"Task: {clean_prompt[:60]}... | "
                f"Turns: {budget_guard.total_turns} | "
                f"Cost: ${budget_guard.accumulated_cost_usd:.4f} | Status: DONE"
            )
            retro = TaskRetrospective(
                task_id=task_id,
                repo_path=str(target_workspace),
                summary=retro_summary,
                passed=True,
                total_steps=budget_guard.total_turns,
                cost_usd=budget_guard.accumulated_cost_usd,
            )
            await self.storage.save_task_retrospective(retro)

            await self.storage.update_task_status(task_id, TaskStatus.DONE)
            if verbose:
                console.print(
                    f"\n[bold green]Task {task_id} completed successfully.[/bold green]"
                )

        except BudgetExceededError as e:
            if verbose:
                console.print(
                    f"\n[bold yellow]Task suspended (budget exceeded): {e}[/bold yellow]"
                )
            await self.storage.save_task_retrospective(
                TaskRetrospective(
                    task_id=task_id,
                    repo_path=str(target_workspace),
                    summary=f"Suspended: {e}",
                    passed=False,
                    total_steps=budget_guard.total_turns,
                    cost_usd=budget_guard.accumulated_cost_usd,
                )
            )
            await self.storage.update_task_status(task_id, TaskStatus.SUSPENDED_AFK)
            raise
        except TestTamperingDetectedError as e:
            if verbose:
                console.print(
                    f"\n[bold red]Task failed (test tampering): {e}[/bold red]"
                )
            await self.storage.save_task_retrospective(
                TaskRetrospective(
                    task_id=task_id,
                    repo_path=str(target_workspace),
                    summary=f"Failed (tampering): {e}",
                    passed=False,
                    total_steps=budget_guard.total_turns,
                    cost_usd=budget_guard.accumulated_cost_usd,
                )
            )
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            raise
        except FileNotFoundError as e:
            if verbose:
                console.print(f"[bold red]Execution error: {e}[/bold red]")
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            raise
        except Exception as e:
            if verbose:
                console.print(f"\n[bold red]Task failed: {e}[/bold red]")
            await self.storage.save_task_retrospective(
                TaskRetrospective(
                    task_id=task_id,
                    repo_path=str(target_workspace),
                    summary=f"Failed: {e}",
                    passed=False,
                    total_steps=budget_guard.total_turns,
                    cost_usd=budget_guard.accumulated_cost_usd,
                )
            )
            await self.storage.update_task_status(task_id, TaskStatus.FAILED)
            raise

        fetched = await self.storage.get_task(task_id)
        return fetched or task
