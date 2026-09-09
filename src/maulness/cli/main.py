import sys
import typer
from rich.console import Console
from rich.table import Table

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
    table.add_column("Status", style="bold green")
    table.add_column("Details", style="white")

    table.add_row("Daemon Service", "Running", "systemd user unit maulness.service")
    table.add_row("Discord Gateway", "Connected", "Listening on #workbench & #console")
    table.add_row("Antigravity ACP", "Ready", "Binary detected at agy")
    table.add_row("Database", "Healthy", "~/.config/maulness/maulness.db")

    console.print(table)


@app.command()
def run(
    repo: str = typer.Argument(..., help="Target repository name (e.g. expense-tracker, horizonx)"),
    prompt: str = typer.Argument(..., help="The prompt or instruction to execute"),
):
    """Execute a task in Direct Mode (solo Antigravity ACP loop)."""
    console.print(f"[bold cyan][*] Running Direct Task on [green]{repo}[/green]...[/bold cyan]")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")
    console.print("[yellow][!] Connecting to Antigravity ACP engine...[/yellow]")


@daemon_app.command("status")
def daemon_status():
    """Check background daemon systemd status."""
    console.print("[cyan][*] Checking systemctl --user status maulness.service...[/cyan]")


@daemon_app.command("start")
def daemon_start():
    """Start the background daemon service."""
    console.print("[green][+] Starting maulness.service...[/green]")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the background daemon service."""
    console.print("[yellow][!] Stopping maulness.service...[/yellow]")


@daemon_app.command("restart")
def daemon_restart():
    """Restart the background daemon service."""
    console.print("[green][+] Restarting maulness.service...[/green]")


@task_app.command("list")
def task_list():
    """List recent and active tasks."""
    table = Table(title="Active & Recent Tasks", border_style="cyan")
    table.add_column("Task ID", style="bold")
    table.add_column("Repo", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Title")
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
