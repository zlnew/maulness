import asyncio
import os
import subprocess
import sys
from pathlib import Path
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

app.add_typer(daemon_app, name="daemon")
app.add_typer(task_app, name="task")
app.add_typer(session_app, name="session")
app.add_typer(profile_app, name="profile")

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
    if config.has_discord:
        table.add_row("Discord Gateway", "[green]Configured[/green]", "Token & Owner ID loaded")
    else:
        table.add_row(
            "Discord Gateway",
            "[yellow]Standby[/yellow]",
            "Waiting for credentials in ~/.config/maulness/env",
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


def run_plain_chat(initial_profile: str = "builder"):
    """Plain REPL interactive fallback without the full-screen TUI."""
    repo_name, workspace_path = config.get_current_workspace()
    current_profile = initial_profile
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
            f"  [cyan]/diff[/cyan]                   Inspect current uncommitted git changes\n"
            f"  [cyan]!<cmd>[/cyan]                  Run workspace shell command (e.g. !git status, !pytest)\n"
            f"  [cyan]/clear[/cyan]                  Clear screen\n"
            f"  [cyan]/exit[/cyan]                   Exit session\n\n"
            f"[dim]Tip: Plain natural language input executes directly with the active profile ({current_profile}).[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input(
                f"\n[bold green]maulness[/bold green] ([cyan]{repo_name}[/cyan]:[magenta]{current_profile}[/magenta]) > "
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
                        auto_proceed=False,
                    )
                )
                continue

            # Direct prompt turn with active profile (no /code prefix required)
            asyncio.run(
                runner.run_direct(
                    repo_name=repo_name,
                    prompt=raw_prompt,
                    workspace_path=workspace_path,
                    profile_name=current_profile,
                )
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session terminated.[/yellow]")
            break


@app.command()
def chat(
    profile: str = typer.Option(
        "builder",
        "-p",
        "--profile",
        help="Initial profile to chat with (builder, planner, default, reviewer)",
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
            t.repo_name,
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

    console.print(
        Panel(
            f"[bold cyan]ID:[/bold cyan] {task.id}\n"
            f"[bold cyan]Repository:[/bold cyan] {task.repo_name}\n"
            f"[bold cyan]Workspace:[/bold cyan] {task.workspace_path}\n"
            f"[bold cyan]Mode:[/bold cyan] {task.mode.value}\n"
            f"[bold cyan]Status:[/bold cyan] {task.status.value}\n"
            f"[bold cyan]Discord Thread:[/bold cyan] {task.discord_thread_id or 'None'}\n"
            f"[bold cyan]Title:[/bold cyan] {task.title}",
            title=f"Task Details — {task.id}",
            border_style="cyan",
        )
    )


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
    """Display YAML specification for a profile."""
    pm = ProfileManager()
    profile = pm.get_profile(name)

    data = profile.model_dump()
    yaml_str = yaml.dump(data, sort_keys=False)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"Profile — {profile.name}", border_style="magenta"))


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


def main():
    app()


if __name__ == "__main__":
    main()
