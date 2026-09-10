import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider

logger = logging.getLogger("maulness.providers.fallback")


class FallbackProviderChain(BaseProvider):
    """Executes requests across a primary provider with fallback chain on rate-limits or errors."""

    def __init__(self, primary: BaseProvider, fallbacks: list[BaseProvider]):
        super().__init__(primary.profile)
        self.primary = primary
        self.fallbacks = fallbacks

    async def run(
        self,
        session_id: str,
        prompt: str,
        workspace_path: Optional[Path] = None,
        conversation_id: Optional[str] = None,
        on_init: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]] = None,
    ) -> str:
        chain = [self.primary] + self.fallbacks
        last_error: Optional[Exception] = None

        for idx, provider in enumerate(chain):
            try:
                if idx > 0:
                    logger.warning(
                        "Attempting fallback provider %s (%s) for session %s after failure",
                        provider.profile.name,
                        provider.profile.provider,
                        session_id,
                    )
                return await provider.run(
                    session_id=session_id,
                    prompt=prompt,
                    workspace_path=workspace_path,
                    conversation_id=conversation_id,
                    on_init=on_init,
                    on_thought=on_thought,
                    on_message=on_message,
                    on_tool_call=on_tool_call,
                    on_approval=on_approval,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Provider %s (%s) failed for session %s: %s",
                    provider.profile.name,
                    provider.profile.provider,
                    session_id,
                    e,
                )
                if idx == len(chain) - 1:
                    raise last_error

        if last_error:
            raise last_error
        return ""
