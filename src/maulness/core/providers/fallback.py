import logging
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.circuit import (
    ProviderHealthRegistry,
    classify_provider_error,
)
from maulness.core.tools import clear_turn_tools, get_turn_executed_tools

logger = logging.getLogger("maulness.providers.fallback")


def format_user_friendly_error(error: Exception) -> str:
    """Extract a concise, human-readable summary from various provider exceptions."""
    _, friendly = classify_provider_error(error)
    return friendly


class FallbackProviderChain(BaseProvider):
    """Executes requests across a primary provider with fallback chain on rate-limits or errors."""

    def __init__(
        self,
        primary: BaseProvider,
        fallbacks: list[BaseProvider],
        registry: Optional[ProviderHealthRegistry] = None,
    ):
        super().__init__(primary.profile)
        self.primary = primary
        self.fallbacks = fallbacks
        self.registry = registry or ProviderHealthRegistry.get_instance()
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
        on_thought: Optional[
            Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_message: Optional[
            Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_tool_call: Optional[
            Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_approval: Optional[
            Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]
        ] = None,
    ) -> str:
        registry = self.registry
        full_chain = [self.primary] + self.fallbacks
        self.last_used_provider = None
        self.chain_errors = []
        last_error: Optional[Exception] = None

        def get_prov_key(p: BaseProvider) -> str:
            return f"{p.profile.provider}:{p.profile.model or p.profile.command or 'default'}"

        now = time.time()
        available_chain: list[BaseProvider] = []
        bypassed_info: list[tuple[BaseProvider, str, int]] = []

        # 1. Candidate filtering via circuit breaker state
        for p in full_chain:
            pkey = get_prov_key(p)
            health = registry.get_or_create(pkey)
            if health.is_available(now):
                available_chain.append(p)
            else:
                rem = max(1, int(health.cooldown_until - now))
                bypassed_info.append((p, health.last_error_msg or "Circuit OPEN", rem))
                logger.info(
                    "[router] Fast-bypassing %s (circuit OPEN, %ds cooldown remaining: %s)",
                    pkey,
                    rem,
                    health.last_error_msg,
                )

        # If all configured providers are in cooldown, route to soonest-expiring
        if not available_chain:
            soonest_p = min(
                full_chain,
                key=lambda p: registry.get_or_create(get_prov_key(p)).cooldown_until,
            )
            available_chain = [soonest_p]
            logger.warning(
                "[router] All providers in cooldown! Attempting soonest-expiring provider: %s",
                get_prov_key(soonest_p),
            )

        # Emit user-facing thought notice when degraded providers were fast-bypassed
        if bypassed_info and on_thought:
            first_active = available_chain[0]
            first_tag = get_prov_key(first_active)
            bypassed_notes = [
                f"[{get_prov_key(bp[0])}] ({bp[1]}, {bp[2]}s cooldown)"
                for bp in bypassed_info
            ]
            notice = f"Routing directly to [{first_tag}]. Bypassed: {', '.join(bypassed_notes)}.\n"
            await on_thought(AgentThoughtEvent(delta=notice, session_id=session_id))

        try:
            for idx, provider in enumerate(available_chain):
                prov_name = provider.profile.name
                prov_type = provider.profile.provider
                prov_model = (
                    provider.profile.model or provider.profile.command or "default"
                )
                prov_key = get_prov_key(provider)
                try:
                    effective_prompt = prompt
                    if idx > 0 or bypassed_info:
                        logger.warning(
                            "Attempting fallback provider %s (%s) for session %s after failure/bypass",
                            prov_name,
                            prov_key,
                            session_id,
                        )
                        completed_tools = get_turn_executed_tools(session_id)
                        if completed_tools:
                            tools_summary = "\n".join(
                                f"- Tool `{t['name']}` with arguments `{t['args']}` returned:\n```\n{t['result'][:1500]}\n```"
                                for t in completed_tools
                            )
                            effective_prompt = (
                                f"{prompt}\n\n"
                                f"[SYSTEM NOTE: The following tool(s) were already executed during this request:\n"
                                f"{tools_summary}\n"
                                f"Do NOT re-execute these tools. Formulate your final response directly using the output above.]"
                            )

                    res = await provider.run(
                        session_id=session_id,
                        prompt=effective_prompt,
                        workspace_path=workspace_path,
                        conversation_id=conversation_id,
                        on_init=on_init,
                        on_thought=on_thought,
                        on_message=on_message,
                        on_tool_call=on_tool_call,
                        on_approval=on_approval,
                    )
                    if not res or not res.strip():
                        raise RuntimeError(
                            f"Provider {prov_key} completed but returned an empty response"
                        )

                    # Succeeded: reset circuit breaker
                    registry.record_success(prov_key)
                    self.last_used_provider = provider
                    return res
                except Exception as e:
                    last_error = e
                    tripped, friendly_msg, cd = registry.record_failure(prov_key, e)
                    self.chain_errors.append(
                        {
                            "profile": prov_name,
                            "provider": prov_type,
                            "model": prov_model,
                            "error": friendly_msg,
                            "raw_error": str(e),
                        }
                    )
                    logger.warning(
                        "Provider %s (%s) failed for session %s: %s (cooldown: %ds)",
                        prov_name,
                        prov_key,
                        session_id,
                        friendly_msg,
                        cd,
                    )

                    # Emit user-facing fallback thought notice if there is another provider in available_chain
                    if idx < len(available_chain) - 1:
                        next_p = available_chain[idx + 1]
                        next_tag = get_prov_key(next_p)
                        notice = (
                            f"Provider [{prov_key}] failed ({friendly_msg}). "
                            f"Switching to fallback [{next_tag}]..."
                        )
                        if on_thought:
                            await on_thought(
                                AgentThoughtEvent(
                                    delta=f"{notice}\n", session_id=session_id
                                )
                            )

                    if idx == len(available_chain) - 1:
                        summary = "\n".join(
                            f"- {item['provider']}:{item['model']} -> {item['error']}"
                            for item in self.chain_errors
                        )
                        raise RuntimeError(
                            f"All configured providers failed:\n{summary}"
                        ) from last_error

            if last_error:
                raise last_error
            return ""
        finally:
            clear_turn_tools(session_id)
