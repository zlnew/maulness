from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile


class BaseProvider(ABC):
    """Abstract base provider for agent execution engines."""

    def __init__(self, profile: Profile):
        self.profile = profile

    @abstractmethod
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
        """Execute a prompt and stream responses through registered callbacks."""
        pass
