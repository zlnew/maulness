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


def format_user_friendly_error(error: Exception) -> str:
    """Extract a concise, human-readable summary from various provider exceptions."""
    err_str = str(error)
    # Check status codes
    if "500" in err_str:
        return "Internal server error (500) from upstream model provider"
    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        return "Rate limit / quota exceeded (429)"
    if "401" in err_str or "CreditsError" in err_str:
        return "Authentication failure or insufficient credits (401)"
    if "403" in err_str or "PERMISSION_DENIED" in err_str:
        return "Permission denied / model access restricted (403)"
    if "404" in err_str or "NOT_FOUND" in err_str:
        return "Model not found or deprecated (404)"
    if isinstance(error, TimeoutError) or "timeout" in err_str.lower():
        return "Request timed out waiting for model stream"
    if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
        return "Invalid or unauthorized API key"

    # Clean up JSON or multi-line blobs
    first_line = err_str.strip().split("\n")[0]
    if len(first_line) > 120:
        return first_line[:117] + "…"
    return first_line or "Unknown provider error"


class FallbackProviderChain(BaseProvider):
    """Executes requests across a primary provider with fallback chain on rate-limits or errors."""

    def __init__(self, primary: BaseProvider, fallbacks: list[BaseProvider]):
        super().__init__(primary.profile)
        self.primary = primary
        self.fallbacks = fallbacks
        self.last_used_provider: Optional[BaseProvider] = None
        self.chain_errors: list[dict[str, Any]] = []

    @property
    def last_conversation_id(self) -> Optional[str]:
        target = self.last_used_provider or self.primary
        return getattr(target, "last_conversation_id", None)

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
        self.last_used_provider = None
        self.chain_errors = []
        last_error: Optional[Exception] = None

        for idx, provider in enumerate(chain):
            prov_name = provider.profile.name
            prov_type = provider.profile.provider
            prov_model = provider.profile.model or provider.profile.command or "default"
            try:
                if idx > 0:
                    logger.warning(
                        "Attempting fallback provider %s (%s:%s) for session %s after failure",
                        prov_name,
                        prov_type,
                        prov_model,
                        session_id,
                    )
                res = await provider.run(
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
                self.last_used_provider = provider
                return res
            except Exception as e:
                last_error = e
                friendly_msg = format_user_friendly_error(e)
                self.chain_errors.append({
                    "profile": prov_name,
                    "provider": prov_type,
                    "model": prov_model,
                    "error": friendly_msg,
                    "raw_error": str(e),
                })
                logger.warning(
                    "Provider %s (%s:%s) failed for session %s: %s",
                    prov_name,
                    prov_type,
                    prov_model,
                    session_id,
                    friendly_msg,
                )

                # Emit user-facing fallback thought notice if there is another provider in chain
                if idx < len(chain) - 1:
                    next_p = chain[idx + 1]
                    next_tag = f"{next_p.profile.provider}:{next_p.profile.model or next_p.profile.command or 'default'}"
                    notice = (
                        f"⚠️ Provider [{prov_type}:{prov_model}] failed ({friendly_msg}). "
                        f"Switching to fallback [{next_tag}]..."
                    )
                    if on_thought:
                        await on_thought(
                            AgentThoughtEvent(delta=f"{notice}\n", session_id=session_id)
                        )

                if idx == len(chain) - 1:
                    summary = "\n".join(
                        f"• {item['provider']}:{item['model']} ➔ {item['error']}"
                        for item in self.chain_errors
                    )
                    raise RuntimeError(f"All configured providers failed:\n{summary}") from last_error

        if last_error:
            raise last_error
        return ""
