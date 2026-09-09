import json
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
import httpx

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider


class UnifiedApiProvider(BaseProvider):
    """Streaming HTTP provider supporting OpenAI, Anthropic, and OpenRouter APIs."""

    BASE_URLS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
    }

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
        api_key = self.profile.get_api_key()
        if not api_key:
            raise ValueError(
                f"Missing API key for profile '{self.profile.name}'. "
                f"Set {self.profile.api_key_env} in ~/.config/maulness/env"
            )

        provider_type = self.profile.provider.lower()
        if provider_type == "anthropic":
            return await self._stream_anthropic(session_id, prompt, api_key, on_message)
        else:
            return await self._stream_openai_compatible(
                provider_type, session_id, prompt, api_key, on_message
            )

    async def _stream_openai_compatible(
        self,
        provider_type: str,
        session_id: str,
        prompt: str,
        api_key: str,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]],
    ) -> str:
        url = self.BASE_URLS.get(provider_type, self.BASE_URLS["openrouter"])
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if provider_type == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/zlnew/maulness"
            headers["X-Title"] = "Maulness"

        payload = {
            "model": self.profile.model or "gpt-4o",
            "messages": [
                {"role": "system", "content": self.profile.effective_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "stream": True,
        }

        accumulated = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        accumulated.append(delta)
                        if on_message:
                            await on_message(
                                AgentMessageEvent(delta=delta, session_id=session_id)
                            )

        return "".join(accumulated)

    async def _stream_anthropic(
        self,
        session_id: str,
        prompt: str,
        api_key: str,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]],
    ) -> str:
        url = self.BASE_URLS["anthropic"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.profile.model or "claude-3-7-sonnet-20250219",
            "system": self.profile.effective_system_prompt(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.profile.max_tokens,
            "temperature": self.profile.temperature,
            "stream": True,
        }

        accumulated = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event_data = json.loads(line[6:])
                    event_type = event_data.get("type")
                    if event_type == "content_block_delta":
                        delta = event_data.get("delta", {}).get("text", "")
                        if delta:
                            accumulated.append(delta)
                            if on_message:
                                await on_message(
                                    AgentMessageEvent(delta=delta, session_id=session_id)
                                )

        return "".join(accumulated)
