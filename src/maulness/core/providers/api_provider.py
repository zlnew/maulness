import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
import httpx

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider


class UnifiedApiProvider(BaseProvider):
    """Streaming HTTP provider supporting OpenAI, Anthropic, OpenRouter, OpenCode, DeepSeek, and Ollama APIs."""

    BASE_URLS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "opencode": "https://api.opencode.ai/v1/chat/completions",
        "opencode_go": "https://api.opencode.ai/v1/chat/completions",
        "opencode_zen": "https://api.opencode.ai/v1/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "ollama": "http://localhost:11434/v1/chat/completions",
    }

    @property
    def last_conversation_id(self) -> Optional[str]:
        return None

    @property
    def base_url(self) -> Optional[str]:
        if self.profile.base_url:
            return self.profile.base_url
        endpoint = self.BASE_URLS.get(self.profile.provider.lower().strip())
        if endpoint:
            for suffix in ("/chat/completions", "/messages"):
                if endpoint.endswith(suffix):
                    return endpoint[:-len(suffix)]
        return endpoint

    @property
    def api_key(self) -> Optional[str]:
        return self.profile.get_api_key()

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
        **kwargs: Any,
    ) -> str:
        api_key = self.profile.get_api_key()
        provider_type = self.profile.provider.lower().strip()

        # Ollama usually does not require an API key
        if not api_key and provider_type != "ollama":
            key_name = self.profile.api_key_env or f"{self.profile.provider.upper()}_API_KEY"
            raise ValueError(
                f"Missing API key for profile '{self.profile.name}'. "
                f"Set {key_name} in ~/.config/maulness/env or profile .env"
            )

        if provider_type == "anthropic":
            return await self._stream_anthropic(session_id, prompt, api_key or "", on_message)
        else:
            return await self._stream_openai_compatible(
                provider_type, session_id, prompt, api_key or "", on_message
            )

    async def _stream_openai_compatible(
        self,
        provider_type: str,
        session_id: str,
        prompt: str,
        api_key: str,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]],
    ) -> str:
        url = self.profile.base_url or self.BASE_URLS.get(provider_type, self.BASE_URLS["openai"])
        if not url.endswith("/chat/completions"):
            url = f"{url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if provider_type == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/zlnew/maulness"
            headers["X-Title"] = "Maulness"

        default_model = "gpt-4o"
        if provider_type.startswith("opencode") or provider_type == "deepseek":
            default_model = "deepseek-ai/deepseek-coder-v3"

        payload = {
            "model": self.profile.model or default_model,
            "messages": [
                {"role": "system", "content": self.profile.effective_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "stream": True,
        }

        idle_timeout = config.stream_idle_timeout_seconds
        accumulated = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                lines_iter = response.aiter_lines().__aiter__()
                while True:
                    try:
                        line = await asyncio.wait_for(lines_iter.__anext__(), timeout=idle_timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Stream idle watchdog triggered: no response received for {int(idle_timeout)}s"
                        )

                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

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
        url = self.profile.base_url or self.BASE_URLS["anthropic"]
        if not url.endswith("/messages"):
            url = f"{url.rstrip('/')}/messages"
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

        idle_timeout = config.stream_idle_timeout_seconds
        accumulated = []
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                lines_iter = response.aiter_lines().__aiter__()
                while True:
                    try:
                        line = await asyncio.wait_for(lines_iter.__anext__(), timeout=idle_timeout)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Stream idle watchdog triggered: no response received for {int(idle_timeout)}s"
                        )

                    if not line.startswith("data: "):
                        continue
                    try:
                        event_data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

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
