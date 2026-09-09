from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.config import config
from maulness.core.acp import AcpClient
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider


class AcpProvider(BaseProvider):
    """Antigravity ACP stdio subprocess provider."""

    async def run(
        self,
        session_id: str,
        prompt: str,
        workspace_path: Optional[Path] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]] = None,
    ) -> str:
        cmd = self.profile.command.split() if self.profile.command else config.agy_cmd
        cwd = str(workspace_path) if workspace_path else str(config.workspace_root)

        client = await AcpClient.spawn(command=cmd, cwd=cwd)
        client.on_thought = on_thought
        client.on_message = on_message
        client.on_tool_call = on_tool_call
        client.on_approval_request = on_approval

        accumulated_text = []

        async def internal_msg_handler(event: AgentMessageEvent):
            accumulated_text.append(event.delta)
            if on_message:
                await on_message(event)

        client.on_message = internal_msg_handler

        try:
            workspace_uri = Path(cwd).as_uri()
            await client.initialize(client_name="maulness", workspace_uri=workspace_uri)

            # Send prompt
            await client.prompt(session_id=session_id, prompt_text=prompt)
        finally:
            await client.stop()

        return "".join(accumulated_text)
