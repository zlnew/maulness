import asyncio
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner

console = Console()


class TerminalLiveRenderer:
    """Renders streaming agent thoughts and markdown message tokens in real time

    with syntax highlighting, animated spinners, and panel formatting.
    """

    def __init__(self, title: str = "Agent Output"):
        self.title = title
        self.thoughts: list[str] = []
        self.tokens: list[str] = []
        self._live: Live = None
        self._is_thinking = True

    def __enter__(self):
        self._live = Live(
            self._render_view(),
            console=console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def on_thought(self, delta: str):
        """Append thought delta and refresh live display."""
        self._is_thinking = True
        self.thoughts.append(delta)
        if self._live:
            self._live.update(self._render_view())

    def on_token(self, delta: str):
        """Append message token and refresh live markdown display."""
        self._is_thinking = False
        self.tokens.append(delta)
        if self._live:
            self._live.update(self._render_view())

    def _render_view(self):
        content = []

        # Render thought section if present
        if self.thoughts:
            thought_text = "".join(self.thoughts)
            if self._is_thinking:
                spinner = Spinner("dots", text=f" [dim italic]Thinking: {thought_text[-120:].strip()}[/dim italic]")
                content.append(spinner)
            else:
                content.append(f"[dim italic grey70][thought] {thought_text[-150:].strip()}[/dim italic grey70]")

        # Render message tokens as Markdown
        msg_text = "".join(self.tokens)
        if msg_text.strip():
            content.append(Markdown(msg_text))
        elif not self.thoughts:
            content.append(Spinner("dots", text=" [dim]Waiting for agent response...[/dim]"))

        return Panel(
            content[0] if len(content) == 1 else content[-1],
            title=self.title,
            border_style="magenta",
        )
