import asyncio
import subprocess
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from maulness.config import config
from maulness.core.runner import TaskRunner
from maulness.storage.db import StorageManager

app = typer.Typer(
    name="maulness",
    help="Maulness: Personal AI Agent Harness (Antigravity ACP + Gemini + Discord)",
    add_completion=False,
)
daemon_app = typer.Typer(help="Manage the background daemon systemd service")
task_app = typer.Typer(help="Manage tasks and execution history")

app.add_typer(daemon_app, name="daemon")
app.add_typer(task_app, name="task")

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
):
    """Execute a task in Direct Mode (solo Antigravity ACP loop)."""
    runner = TaskRunner()
    try:
        asyncio.run(runner.run_direct(repo_name=repo, prompt=prompt))
    except KeyboardInterrupt:
        console.print("\n[yellow][!] Aborted by user.[/yellow]")


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
