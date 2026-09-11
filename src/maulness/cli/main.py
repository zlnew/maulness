import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from maulness.config import config
from maulness.core.pipeline import PipelineOrchestrator
from maulness.core.pipelines import PipelineManager
from maulness.core.profiles import ProfileManager
from maulness.core.runner import TaskRunner
from maulness.storage.db import StorageManager

app = typer.Typer(
    name="maulness",
    help="Maulness: Personal AI Agent Harness (Antigravity ACP + Multi-Provider + Discord)",
    add_completion=False,
)
daemon_app = typer.Typer(help="Manage the background daemon systemd service")
task_app = typer.Typer(help="Inspect tasks and execution history")
session_app = typer.Typer(help="Inspect active and past agent sessions")
profile_app = typer.Typer(help="Inspect agent execution profiles")
discord_app = typer.Typer(help="Manage Discord bot gateways")
soul_app = typer.Typer(help="Inspect and edit SOUL.md personal doctrines")
memory_app = typer.Typer(help="Inspect and edit MEMORY.md workspace knowledge")
user_app = typer.Typer(help="Inspect and edit USER.md working style and preferences")
db_app = typer.Typer(help="Database maintenance and migration commands")

app.add_typer(daemon_app, name="daemon")
app.add_typer(task_app, name="task")
app.add_typer(session_app, name="session")
app.add_typer(profile_app, name="profile")
app.add_typer(discord_app, name="discord")
app.add_typer(soul_app, name="soul")
app.add_typer(memory_app, name="memory")
app.add_typer(user_app, name="user")
app.add_typer(db_app, name="db")

console = Console()


@app.command()
def status():
    """Display the live status of Maulness harness, daemon, and active sessions."""
    table = Table(title="Maulness System Status", border_style="bright_blue")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Details", style="white")

    # Check systemd user service status
    daemon_status_text = "[red]Inactive[/red]"
    try:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", "maulness.service"],
            capture_output=True,
            text=True,
        )
        if res.stdout.strip() == "active":
            daemon_status_text = "[green]Active (Running)[/green]"
    except Exception:
        daemon_status_text = "[yellow]Unknown[/yellow]"

    table.add_row("Daemon Service", daemon_status_text, "systemd user unit maulness.service")

    # Check Discord
    if config.gateway_multiplex_profiles:
        allowlist = config.gateway_multiplex_profile_allowlist
        detail = f"Multiplexing ({', '.join(allowlist) if allowlist else 'all'})"
        table.add_row("Discord Gateway", "[green]Multiplexing[/green]", detail)
    elif config.has_discord:
        table.add_row("Discord Gateway", "[green]Configured[/green]", "Single Default Profile")
    else:
        table.add_row(
            "Discord Gateway",
            "[yellow]Standby[/yellow]",
            "Waiting for credentials in ~/.config/maulness/env or profile .env",
        )

    # Check Antigravity binary
    agy_binary = config.agy_cmd[0]
    agy_installed = (
        subprocess.run(["which", agy_binary], capture_output=True).returncode == 0
    )
    if agy_installed:
        table.add_row("Antigravity ACP", "[green]Ready[/green]", f"Binary at {agy_binary}")
    else:
        table.add_row(
            "Antigravity ACP",
            "[yellow]Pending[/yellow]",
            f"Binary '{agy_binary}' not on PATH",
        )

    # Check Profiles
    pm = ProfileManager()
    profiles = pm.list_profiles()
    table.add_row(
        "Active Profiles",
        f"[green]{len(profiles)} Loaded[/green]",
        ", ".join(p.name for p in profiles),
    )

    # Context Workspace (cwd)
    repo_name, repo_path = config.get_current_workspace()
    table.add_row("Current Workspace", f"[green]{repo_name}[/green]", str(repo_path))

    # Database
    db_exists = config.db_path.exists()
    table.add_row(
        "Database",
        "[green]Ready[/green]" if db_exists else "[yellow]Uninitialized[/yellow]",
        str(config.db_path),
    )

    console.print(table)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Instruction or task to perform"),
    repo: Optional[str] = typer.Option(
        None, "-r", "--repo", help="Target repository (defaults to current directory)"
    ),
    profile: str = typer.Option("builder", "-p", "--profile", help="Execution profile"),
    worktree: bool = typer.Option(
        False, "-w", "--worktree", help="Execute inside an isolated git worktree"
    ),
    yolo: bool = typer.Option(
        False, "-y", "--yolo", help="YOLO mode: bypass human confirmation on mutating tools"
    ),
):
    """Execute a direct task on a repository with the specified profile."""
    runner = TaskRunner()
    current_repo, current_workspace = config.get_current_workspace()
    effective_repo = repo or current_repo
    target_workspace = config.resolve_repo_path(repo) if repo else current_workspace

    asyncio.run(
        runner.run_direct(
            repo_name=effective_repo,
            prompt=prompt,
            workspace_path=target_workspace,
            profile_name=profile,
            use_worktree=worktree,
            yolo=yolo,
        )
    )


def run_plain_chat(initial_profile: str = "default"):
    """Plain REPL interactive fallback without the full-screen TUI."""
    repo_name, workspace_path = config.get_current_workspace()
    current_profile = initial_profile
    yolo_mode = False
    worktree_mode = False
    runner = TaskRunner()
    pipeline_manager = PipelineManager()
    orchestrator = PipelineOrchestrator(pipeline_manager=pipeline_manager)

    console.print(
        Panel(
            f"[bold cyan]Maulness Interactive Workspace Chat (Plain REPL)[/bold cyan]\n"
            f"Repository: [green]{repo_name}[/green] ([dim]{workspace_path}[/dim])\n"
            f"Active Profile: [magenta]{current_profile}[/magenta]\n\n"
            f"[bold yellow]Commands:[/bold yellow]\n"
            f"  [cyan]/pipeline [name] <goal>[/cyan]  Run declarative pipeline (standard, quick, plan_only, audit)\n"
            f"  [cyan]/profile <name>[/cyan]         Switch profile on the fly (builder, planner, default, reviewer)\n"
            f"  [cyan]/yolo[/cyan]                   Toggle YOLO mode (bypass human confirmation)\n"
            f"  [cyan]/worktree[/cyan]               Toggle isolated Git worktree execution\n"
            f"  [cyan]/diff[/cyan]                   Inspect current uncommitted git changes\n"
            f"  [cyan]!<cmd>[/cyan]                  Run workspace shell command (e.g. !git status, !pytest)\n"
            f"  [cyan]/clear[/cyan]                  Clear screen\n"
            f"  [cyan]/exit[/cyan]                   Exit session\n\n"
            f"[dim]Tip: Plain natural language input executes directly with the active profile ({current_profile}).[/dim]",
            border_style="cyan",
        )
    )

    active_conversation_id: Optional[str] = None
    session_id: str = f"cli_{uuid.uuid4().hex[:8]}"

    while True:
        try:
            status_badges = []
            if yolo_mode:
                status_badges.append("[red]YOLO[/red]")
            if worktree_mode:
                status_badges.append("[cyan]WT[/cyan]")
            badge_str = f" [{','.join(status_badges)}]" if status_badges else ""

            user_input = console.input(
                f"\n[bold green]maulness[/bold green] ([cyan]{repo_name}[/cyan]:[magenta]{current_profile}[/magenta]{badge_str}) > "
            )
            raw_prompt = user_input.strip()
            if not raw_prompt:
                continue

            # Standard control commands
            if raw_prompt in ("/exit", "/quit", "exit", "quit"):
                console.print("[yellow]Exiting chat session. Goodbye Maul![/yellow]")
                break

            if raw_prompt == "/clear":
                console.clear()
                active_conversation_id = None
                session_id = f"cli_{uuid.uuid4().hex[:8]}"
                continue

            if raw_prompt == "/yolo":
                yolo_mode = not yolo_mode
                state = "[bold red]ENABLED (Auto-approving mutating tools)[/bold red]" if yolo_mode else "[bold green]DISABLED (HITL confirmations active)[/bold green]"
                console.print(f"[*] YOLO mode: {state}")
                continue

            if raw_prompt == "/worktree":
                worktree_mode = not worktree_mode
                state = "[bold cyan]ENABLED (Isolated git worktree)[/bold cyan]" if worktree_mode else "[dim]DISABLED (Direct repo workspace)[/dim]"
                console.print(f"[*] Worktree isolation: {state}")
                continue

            if raw_prompt == "/diff":
                res = subprocess.run(
                    ["git", "diff", "HEAD"],
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                )
                diff = res.stdout or "(No uncommitted changes)"
                console.print(Syntax(diff, "diff", theme="monokai"))
                continue

            if raw_prompt.startswith("/profile"):
                parts = raw_prompt.split(" ", 1)
                if len(parts) > 1 and parts[1].strip():
                    current_profile = parts[1].strip()
                    active_conversation_id = None
                    console.print(
                        f"[green]Switched active profile to [magenta]{current_profile}[/magenta][/green]"
                    )
                else:
                    pm = ProfileManager()
                    profiles = pm.list_profiles()
                    console.print(
                        f"[magenta]Available profiles:[/magenta] {', '.join(p.name for p in profiles)}"
                    )
                continue

            # Local shell escape: !<cmd>
            if raw_prompt.startswith("!"):
                cmd = raw_prompt[1:].strip()
                res = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                )
                output = res.stdout or res.stderr or "(No output)"
                console.print(Panel(output, title=f"$ {cmd} (exit {res.returncode})", border_style="yellow"))
                continue

            # Declarative pipeline command: /pipeline [name] <goal>
            if raw_prompt.startswith(("/pipeline ", "/task ")):
                pipeline_body = raw_prompt.split(" ", 1)[1].strip()
                parts = pipeline_body.split(" ", 1)
                known_pipelines = {p.name for p in pipeline_manager.list_pipelines()}
                if len(parts) == 2 and parts[0] in known_pipelines:
                    p_name, goal = parts[0], parts[1].strip()
                else:
                    p_name, goal = "standard", pipeline_body

                console.print(f"[bold magenta][*] Launching pipeline '{p_name}': [white]{goal}[/white][/bold magenta]\n")
                asyncio.run(
                    orchestrator.run_pipeline(
                        repo_name=repo_name,
                        title=goal[:80],
                        prompt=goal,
                        workspace_path=workspace_path,
                        pipeline_name=p_name,
                        use_worktree=worktree_mode,
                        yolo=yolo_mode,
                        auto_proceed=yolo_mode,
                    )
                )
                continue

            async def on_init(conv_id: str):
                nonlocal active_conversation_id
                active_conversation_id = conv_id

            # Direct prompt turn with active profile (no /code prefix required)
            asyncio.run(
                runner.run_direct(
                    repo_name=repo_name,
                    prompt=raw_prompt,
                    workspace_path=workspace_path,
                    profile_name=current_profile,
                    session_id=session_id,
                    conversation_id=active_conversation_id,
                    use_worktree=worktree_mode,
                    yolo=yolo_mode,
                    on_init=on_init,
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session terminated.[/yellow]")
            break


@app.command()
def chat(
    profile: str = typer.Option(
        "default",
        "-p",
        "--profile",
        help="Initial profile to chat with (default, builder, planner, reviewer)",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        "--no-tui",
        help="Run in plain text REPL mode instead of the full terminal TUI",
    ),
):
    """Start an interactive workspace chat session scoped to current directory."""
    if not plain and sys.stdin.isatty():
        from maulness.cli.tui import MaulnessTUIApp

        app = MaulnessTUIApp(initial_profile=profile)
        app.run()
    else:
        run_plain_chat(initial_profile=profile)


# ==========================================
# Task Management (Strictly list & show)
# ==========================================
@task_app.command("list")
def task_list(limit: int = typer.Option(20, "-n", "--limit", help="Number of tasks to show")):
    """List recent and active tasks."""
    storage = StorageManager(db_path=config.db_path)

    async def _list():
        await storage.initialize()
        return await storage.list_tasks(limit=limit)

    tasks = asyncio.run(_list())
    if not tasks:
        console.print("[dim]No tasks recorded yet. Run `maulness chat` to start one.[/dim]")
        return

    table = Table(title="Recent & Active Tasks", border_style="cyan")
    table.add_column("Task ID", style="bold")
    table.add_column("Repo", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Title")

    for t in tasks:
        status_color = {
            "done": "green",
            "building": "yellow",
            "planning": "blue",
            "failed": "red",
        }.get(t.status.value, "white")
        table.add_row(
            t.id,
            t.repo_name or "[dim]general[/dim]",
            t.mode.value,
            f"[{status_color}]{t.status.value}[/{status_color}]",
            t.title,
        )

    console.print(table)


@task_app.command("show")
def task_show(task_id: str = typer.Argument(..., help="Task ID to inspect")):
    """Show details of a specific task."""
    storage = StorageManager(db_path=config.db_path)

    async def _get():
        await storage.initialize()
        return await storage.get_task(task_id)

    task = asyncio.run(_get())
    if not task:
        console.print(f"[red][x] Task '{task_id}' not found.[/red]")
        return

    origin_str = f"{task.origin_platform}"
    if task.origin_channel_id:
        origin_str += f" (channel: {task.origin_channel_id}"
        if task.origin_thread_id:
            origin_str += f", thread: {task.origin_thread_id}"
        origin_str += ")"

    console.print(
        Panel(
            f"[bold cyan]ID:[/bold cyan] {task.id}\n"
            f"[bold cyan]Origin:[/bold cyan] {origin_str}\n"
            f"[bold cyan]Repository:[/bold cyan] {task.repo_name or 'None (General)'}\n"
            f"[bold cyan]Workspace:[/bold cyan] {task.workspace_path or 'None'}\n"
            f"[bold cyan]Mode:[/bold cyan] {task.mode.value}\n"
            f"[bold cyan]Status:[/bold cyan] {task.status.value}\n"
            f"[bold cyan]Title:[/bold cyan] {task.title}",
            title=f"Task Details — {task.id}",
            border_style="cyan",
        )
    )


@task_app.command("abort")
def task_abort(task_id: str = typer.Argument(..., help="Task ID to abort")):
    """Immediately aborts an active task and closes associated sessions."""
    from maulness.core.models import TaskStatus

    storage = StorageManager(db_path=config.db_path)

    async def _abort():
        await storage.initialize()
        task = await storage.get_task(task_id)
        if not task:
            console.print(f"[red][x] Task '{task_id}' not found.[/red]")
            return
        await storage.update_task_status(task_id, TaskStatus.FAILED)
        sessions = await storage.list_sessions(limit=50)
        closed_count = 0
        for s in sessions:
            if s.task_id == task_id and s.status == "active":
                await storage.update_session_status(s.id, "closed")
                if s.pid:
                    try:
                        os.kill(s.pid, 9)
                    except ProcessLookupError:
                        pass
                closed_count += 1
        console.print(
            f"[green]Aborted task '{task_id}' (closed {closed_count} sessions).[/green]"
        )

    asyncio.run(_abort())


# ==========================================
# Session Management (list & show)
# ==========================================
@session_app.command("list")
def session_list(limit: int = typer.Option(20, "-n", "--limit", help="Number of sessions to show")):
    """List active and historical agent sessions."""
    storage = StorageManager(db_path=config.db_path)

    async def _list():
        await storage.initialize()
        return await storage.list_sessions(limit=limit)

    sessions = asyncio.run(_list())
    if not sessions:
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    table = Table(title="Agent Sessions", border_style="green")
    table.add_column("Session ID", style="bold")
    table.add_column("Profile", style="magenta")
    table.add_column("Engine", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Task ID", style="dim")

    for s in sessions:
        table.add_row(
            s.id,
            s.profile,
            s.engine,
            s.status,
            s.task_id or "-",
        )

    console.print(table)


@session_app.command("show")
def session_show(session_id: str = typer.Argument(..., help="Session ID to inspect")):
    """Show details of a specific agent session."""
    storage = StorageManager(db_path=config.db_path)

    async def _get():
        await storage.initialize()
        return await storage.get_session(session_id)

    session = asyncio.run(_get())
    if not session:
        console.print(f"[red][x] Session '{session_id}' not found.[/red]")
        return

    console.print(
        Panel(
            f"[bold cyan]ID:[/bold cyan] {session.id}\n"
            f"[bold cyan]Profile:[/bold cyan] {session.profile}\n"
            f"[bold cyan]Engine:[/bold cyan] {session.engine}\n"
            f"[bold cyan]Status:[/bold cyan] {session.status}\n"
            f"[bold cyan]Task ID:[/bold cyan] {session.task_id or 'None'}\n"
            f"[bold cyan]ACP Session ID:[/bold cyan] {session.acp_session_id or 'None'}\n"
            f"[bold cyan]Process PID:[/bold cyan] {session.pid or 'None'}",
            title=f"Session Details — {session.id}",
            border_style="green",
        )
    )


# ==========================================
# Profile Management (list & show)
# ==========================================
@profile_app.command("list")
def profile_list():
    """List all available profiles."""
    pm = ProfileManager()
    profiles = pm.list_profiles()

    table = Table(title="Available Agent Profiles", border_style="magenta")
    table.add_column("Profile", style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Model / Command", style="yellow")
    table.add_column("Description")

    for p in profiles:
        target = p.model or p.command or "-"
        table.add_row(p.name, p.provider, target, p.description)

    console.print(table)


@profile_app.command("show")
def profile_show(name: str = typer.Argument(..., help="Profile name to inspect")):
    """Display specification, doctrine, and skills for a profile."""
    pm = ProfileManager()
    profile = pm.get_profile(name)

    data = profile.model_dump(mode="json", exclude={"env_vars", "soul_content"})
    yaml_str = yaml.dump(data, sort_keys=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"Profile Configuration — {profile.name}", border_style="magenta"))

    if profile.soul_content:
        console.print(Panel(profile.soul_content, title="Role Operating Doctrine (SOUL.md)", border_style="cyan"))

    from maulness.core.skills import SkillManager
    skills = SkillManager().list_skills(profile_skills_dir=profile.skills_dir)
    if skills:
        skill_lines = []
        for sname, s in skills.items():
            is_local = profile.skills_dir and s.path.is_relative_to(profile.skills_dir)
            tag = "[magenta](profile-extending)[/magenta]" if is_local else "[dim](root-default)[/dim]"
            skill_lines.append(f"• [bold #a6e3a1]{sname}[/bold #a6e3a1] {tag}: {s.description}")
        console.print(Panel("\n".join(skill_lines), title=f"Effective Skills ({len(skills)})", border_style="green"))


# ==========================================
# Daemon Management (start, stop, restart, status, logs)
# ==========================================
@daemon_app.command("status")
def daemon_status():
    """Check background daemon systemd status."""
    subprocess.run(["systemctl", "--user", "status", "maulness.service"])


@daemon_app.command("start")
def daemon_start():
    """Start the background daemon service."""
    res = subprocess.run(["systemctl", "--user", "start", "maulness.service"])
    if res.returncode == 0:
        console.print("[green][+] Started maulness.service successfully.[/green]")
    else:
        console.print("[red][x] Failed to start maulness.service.[/red]")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the background daemon service."""
    res = subprocess.run(["systemctl", "--user", "stop", "maulness.service"])
    if res.returncode == 0:
        console.print("[yellow][!] Stopped maulness.service.[/yellow]")
    else:
        console.print("[red][x] Failed to stop maulness.service.[/red]")


@daemon_app.command("restart")
def daemon_restart():
    """Restart the background daemon service."""
    res = subprocess.run(["systemctl", "--user", "restart", "maulness.service"])
    if res.returncode == 0:
        console.print("[green][+] Restarted maulness.service.[/green]")
    else:
        console.print("[red][x] Failed to restart maulness.service.[/red]")


@daemon_app.command("logs")
def daemon_logs(
    follow: bool = typer.Option(True, "-f", "--follow", help="Follow log output in real time"),
    lines: int = typer.Option(50, "-n", "--lines", help="Number of lines to show"),
):
    """View daemon logs via journalctl."""
    cmd = ["journalctl", "--user", "-u", "maulness.service", f"-n{lines}"]
    if follow:
        cmd.append("-f")
    subprocess.run(cmd)


# ==========================================
# Discord Gateway (run foreground)
# ==========================================
@discord_app.command("run")
def discord_run(
    profile: Optional[str] = typer.Option(None, "-p", "--profile", help="Target specific profile"),
):
    """Run Discord gateway adapter in the foreground."""
    from maulness.daemon.service import main as service_main
    asyncio.run(service_main(profile_filter=profile))


# ==========================================
# Soul Doctrine Management (show, edit)
# ==========================================
@soul_app.command("show")
def soul_show(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global doctrine)"
    ),
):
    """Print SOUL.md personal doctrine."""
    pm = ProfileManager()
    if profile:
        prof = pm.get_profile(profile)
        if prof.soul_content:
            console.print(
                Panel(
                    prof.soul_content,
                    title=f"SOUL.md ({prof.name})",
                    border_style="magenta",
                )
            )
        else:
            console.print(f"[dim]No profile-specific SOUL.md found for '{profile}'.[/dim]")
    else:
        root_soul = config.config_dir / "SOUL.md"
        if root_soul.exists():
            console.print(
                Panel(
                    root_soul.read_text(encoding="utf-8"),
                    title="Maulness Global SOUL.md",
                    border_style="cyan",
                )
            )
        else:
            tpl_soul = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "templates"
                / "SOUL.md"
            )
            if tpl_soul.exists():
                console.print(
                    Panel(
                        tpl_soul.read_text(encoding="utf-8"),
                        title="Template SOUL.md",
                        border_style="cyan",
                    )
                )
            else:
                console.print("[dim]No SOUL.md found.[/dim]")


@soul_app.command("edit")
def soul_edit(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global doctrine)"
    ),
):
    """Open SOUL.md in $EDITOR."""
    editor = os.getenv("EDITOR", "nano")
    if profile:
        target = config.config_dir / "profiles" / profile / "SOUL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(f"# {profile.title()} Persona Doctrine\n")
    else:
        target = config.config_dir / "SOUL.md"
        if not target.exists():
            tpl_soul = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "templates"
                / "SOUL.md"
            )
            if tpl_soul.exists():
                target.write_text(tpl_soul.read_text(encoding="utf-8"))
            else:
                target.write_text("# Maul Personal Agent Doctrine\n")

    subprocess.run([editor, str(target)])


# ==========================================
# Workspace Memory Management (show, edit)
# ==========================================
@memory_app.command("show")
def memory_show(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global memory)"
    ),
):
    """Print durable MEMORY.md workspace knowledge."""
    pm = ProfileManager()
    if profile:
        prof = pm.get_profile(profile)
        if prof.memory_content:
            console.print(
                Panel(
                    prof.memory_content,
                    title=f"MEMORY.md ({prof.name})",
                    border_style="cyan",
                )
            )
        else:
            console.print(f"[dim]No profile-specific MEMORY.md found for '{profile}'.[/dim]")
    else:
        root_mem = config.config_dir / "MEMORY.md"
        if root_mem.exists():
            console.print(
                Panel(
                    root_mem.read_text(encoding="utf-8"),
                    title="Maulness Workspace Knowledge (MEMORY.md)",
                    border_style="cyan",
                )
            )
        else:
            console.print("[dim]No MEMORY.md found.[/dim]")


@memory_app.command("edit")
def memory_edit(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global memory)"
    ),
):
    """Open MEMORY.md in $EDITOR."""
    editor = os.getenv("EDITOR", "nano")
    if profile:
        target = config.config_dir / "profiles" / profile / "MEMORY.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(f"# {profile.title()} Workspace Memory\n")
    else:
        target = config.config_dir / "MEMORY.md"
        if not target.exists():
            tpl = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "MEMORY.md"
            if tpl.exists():
                target.write_text(tpl.read_text(encoding="utf-8"))
            else:
                target.write_text("# Workspace Knowledge (MEMORY.md)\n")

    subprocess.run([editor, str(target)])


# ==========================================
# User Profile & Working Style (show, edit)
# ==========================================
@user_app.command("show")
def user_show(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global user profile)"
    ),
):
    """Print user preferences and working style (USER.md)."""
    pm = ProfileManager()
    if profile:
        prof = pm.get_profile(profile)
        if prof.user_content:
            console.print(
                Panel(
                    prof.user_content,
                    title=f"USER.md ({prof.name})",
                    border_style="green",
                )
            )
        else:
            console.print(f"[dim]No profile-specific USER.md found for '{profile}'.[/dim]")
    else:
        root_user = config.config_dir / "USER.md"
        if root_user.exists():
            console.print(
                Panel(
                    root_user.read_text(encoding="utf-8"),
                    title="User Profile & Style (USER.md)",
                    border_style="green",
                )
            )
        else:
            console.print("[dim]No USER.md found.[/dim]")


@user_app.command("edit")
def user_edit(
    profile: Optional[str] = typer.Option(
        None, "-p", "--profile", help="Profile name (defaults to global user profile)"
    ),
):
    """Open USER.md in $EDITOR."""
    editor = os.getenv("EDITOR", "nano")
    if profile:
        target = config.config_dir / "profiles" / profile / "USER.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(f"# {profile.title()} User Profile\n")
    else:
        target = config.config_dir / "USER.md"
        if not target.exists():
            tpl = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "USER.md"
            if tpl.exists():
                target.write_text(tpl.read_text(encoding="utf-8"))
            else:
                target.write_text("# User Profile (USER.md)\n")

    subprocess.run([editor, str(target)])


# ==========================================
# Database Maintenance (migrate, reset)
# ==========================================
@db_app.command("migrate")
def db_migrate():
    """Apply SQLite DDL schemas to ensure database integrity."""
    storage = StorageManager(db_path=config.db_path)
    asyncio.run(storage.initialize())
    console.print(f"[green]Applied database DDL schemas at {config.db_path}[/green]")


@db_app.command("reset")
def db_reset():
    """Reset orphan active sessions and mark pending/in-flight tasks as failed."""
    storage = StorageManager(db_path=config.db_path)

    async def _reset():
        await storage.initialize()
        res = await storage.reset_stale_sessions()
        console.print(
            f"[green]Stale database state reset:[/green] "
            f"closed {res['sessions_closed']} sessions, marked {res['tasks_failed']} tasks failed."
        )

    asyncio.run(_reset())


def main():
    app()


if __name__ == "__main__":
    main()
