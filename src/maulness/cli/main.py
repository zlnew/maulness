import asyncio
import os
import subprocess
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from maulness.config import config
from maulness.core.profiles import ProfileManager
from maulness.core.runner import TaskRunner
from maulness.core.soul import get_soul_content
from maulness.storage.db import StorageManager

app = typer.Typer(
    name="maulness",
    help="Maulness: Personal AI Agent Harness (Antigravity ACP + Multi-Provider + Discord)",
    add_completion=False,
)
daemon_app = typer.Typer(help="Manage the background daemon systemd service")
task_app = typer.Typer(help="Manage tasks and execution history")
profile_app = typer.Typer(help="Manage agent execution profiles")
soul_app = typer.Typer(help="Manage SOUL.md personal doctrine")

app.add_typer(daemon_app, name="daemon")
app.add_typer(task_app, name="task")
app.add_typer(profile_app, name="profile")
app.add_typer(soul_app, name="soul")

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
    table.add_row("Active Profiles", f"[green]{len(profiles)} Loaded[/green]", ", ".join(p.name for p in profiles))

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
    repo: str = typer.Argument(..., help="Target repository name (e.g. expense-tracker, horizonx)"),
    prompt: str = typer.Argument(..., help="The prompt or instruction to execute"),
    profile: str = typer.Option("builder", "-p", "--profile", help="Profile to execute with (default, planner, builder, reviewer)"),
):
    """Execute a task in Direct Mode using the specified profile."""
    runner = TaskRunner()
    try:
        asyncio.run(runner.run_direct(repo_name=repo, prompt=prompt, profile_name=profile))
    except KeyboardInterrupt:
        console.print("\n[yellow][!] Aborted by user.[/yellow]")


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


@soul_app.command("show")
def soul_show():
    """Display active SOUL.md personal doctrine."""
    soul = get_soul_content()
    console.print(Panel(soul, title="SOUL.md — Personal Operating Doctrine", border_style="green"))


@soul_app.command("edit")
def soul_edit():
    """Open SOUL.md in default terminal editor."""
    soul_file = config.config_dir / "SOUL.md"
    editor = os.getenv("EDITOR", "nano")
    subprocess.run([editor, str(soul_file)])


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


@task_app.command("create")
def task_create(
    repo: str = typer.Argument(..., help="Target repository name"),
    title: str = typer.Argument(..., help="Task title / headline"),
    prompt: str = typer.Argument(..., help="Detailed requirements and constraints"),
    auto_proceed: bool = typer.Option(False, "--yes", "-y", help="Auto-proceed through planning gate"),
):
    """Create and execute a task in Multi-Route Mode (Planner ➔ Builder ➔ Reviewer)."""
    from maulness.core.pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    try:
        asyncio.run(
            orchestrator.run_pipeline(
                repo_name=repo,
                title=title,
                prompt=prompt,
                auto_proceed=auto_proceed,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow][!] Pipeline aborted by user.[/yellow]")


@task_app.command("list")
def task_list(limit: int = typer.Option(20, "-n", "--limit", help="Number of tasks to show")):
    """List recent and active tasks."""
    storage = StorageManager(db_path=config.db_path)

    async def _list():
        await storage.initialize()
        return await storage.list_tasks(limit=limit)

    tasks = asyncio.run(_list())
    if not tasks:
        console.print("[dim]No tasks recorded yet. Run `maulness run <repo> <prompt>` to start one.[/dim]")
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


def main():
    app()


if __name__ == "__main__":
    main()
