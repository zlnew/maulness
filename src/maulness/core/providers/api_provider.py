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
        endpoint = self.BASE_URLS.get(self.profile.provider.lower().strip().replace("-", "_"))
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
        provider_type = self.profile.provider.lower().strip().replace("-", "_")

        # Ollama usually does not require an API key
        if not api_key and provider_type != "ollama":
            key_name = self.profile.api_key_env or f"{self.profile.provider.upper().replace('-', '_')}_API_KEY"
            raise ValueError(
                f"Missing API key for profile '{self.profile.name}'. "
                f"Set {key_name} in ~/.config/maulness/env or profile .env"
            )

        if provider_type == "anthropic":
            return await self._stream_anthropic(session_id, prompt, api_key or "", on_message, on_thought)
        else:
            return await self._stream_openai_compatible(
                provider_type, session_id, prompt, api_key or "", on_message, on_thought, on_tool_call
            )

    async def _stream_openai_compatible(
        self,
        provider_type: str,
        session_id: str,
        prompt: str,
        api_key: str,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]],
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None,
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
        if provider_type == "ollama":
            default_model = "llama3"
        elif provider_type.startswith("opencode") or provider_type == "deepseek":
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
        effort = self.profile.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort

        idle_timeout = float(config.stream_idle_timeout_seconds)
        accumulated: list[str] = []
        captured_tool_calls: list[dict[str, Any]] = []
        is_first_chunk = True

        timeout = httpx.Timeout(120.0, connect=20.0, read=idle_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                stream_ctx = client.stream("POST", url, headers=headers, json=payload)
            except Exception as e:
                raise RuntimeError(f"Failed to initiate stream with {provider_type} ({url}): {e}")

            async with stream_ctx as response:
                if response.is_error:
                    err_body = await response.aread()
                    err_msg = err_body.decode(errors="replace")
                    raise RuntimeError(
                        f"Provider '{provider_type}' ({self.profile.model}) returned HTTP {response.status_code}: {err_msg[:500]}"
                    )

                lines_iter = response.aiter_lines().__aiter__()
                while True:
                    try:
                        chunk_timeout = 60.0 if (is_first_chunk and provider_type == "ollama") else (30.0 if is_first_chunk else idle_timeout)
                        line = await asyncio.wait_for(lines_iter.__anext__(), timeout=chunk_timeout)
                        is_first_chunk = False
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        if is_first_chunk:
                            raise TimeoutError(
                                f"Stream from {provider_type} ({self.profile.model}) timed out waiting for first token after {int(chunk_timeout)}s"
                            )
                        raise TimeoutError(
                            f"Stream idle watchdog triggered: no response received for {int(idle_timeout)}s"
                        )

                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    # 1. Capture reasoning / thoughts (Ollama, DeepSeek, OpenCode)
                    reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning_delta and on_thought:
                        await on_thought(
                            AgentThoughtEvent(delta=reasoning_delta, session_id=session_id)
                        )

                    # 2. Capture tool calls (OpenAI format)
                    tool_calls_delta = delta.get("tool_calls")
                    if tool_calls_delta:
                        for tc in tool_calls_delta:
                            idx = tc.get("index", 0)
                            while len(captured_tool_calls) <= idx:
                                captured_tool_calls.append({"name": "", "arguments": ""})
                            fn = tc.get("function", {})
                            if "name" in fn:
                                captured_tool_calls[idx]["name"] += fn["name"]
                            if "arguments" in fn:
                                captured_tool_calls[idx]["arguments"] += fn["arguments"]

                    # 3. Capture visible text content
                    text_delta = delta.get("content", "")
                    if text_delta:
                        accumulated.append(text_delta)
                        if on_message:
                            await on_message(
                                AgentMessageEvent(delta=text_delta, session_id=session_id)
                            )

        # If model only generated tool calls and no content, surface tool invocation
        if not accumulated and captured_tool_calls:
            for tc in captured_tool_calls:
                call_name = tc.get("name", "unknown_tool")
                call_args = tc.get("arguments", "")
                tool_msg = f"[Tool Request: {call_name}({call_args})]"
                accumulated.append(tool_msg)
                if on_tool_call:
                    await on_tool_call(
                        AgentToolCallEvent(
                            call_id=f"call_{call_name}",
                            tool_name=call_name,
                            args={"raw_arguments": call_args},
                            session_id=session_id,
                        )
                    )
                if on_message:
                    await on_message(
                        AgentMessageEvent(delta=tool_msg, session_id=session_id)
                    )

        result_text = "".join(accumulated).strip()
        if not result_text:
            raise RuntimeError(
                f"Provider '{provider_type}' ({self.profile.model}) returned an empty response"
            )

        return result_text

    async def _stream_anthropic(
        self,
        session_id: str,
        prompt: str,
        api_key: str,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]],
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
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
        _EFFORT_BUDGET = {"low": 1024, "medium": 8192, "high": 24576}
        effort = self.profile.reasoning_effort
        if effort:
            payload["thinking"] = {"type": "enabled", "budget_tokens": _EFFORT_BUDGET.get(effort.lower(), 8192)}
            payload["temperature"] = 1  # Anthropic requires temperature=1 when thinking is enabled

        idle_timeout = float(config.stream_idle_timeout_seconds)
        accumulated: list[str] = []
        is_first_chunk = True

        timeout = httpx.Timeout(120.0, connect=20.0, read=idle_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                stream_ctx = client.stream("POST", url, headers=headers, json=payload)
            except Exception as e:
                raise RuntimeError(f"Failed to initiate stream with Anthropic ({url}): {e}")

            async with stream_ctx as response:
                if response.is_error:
                    err_body = await response.aread()
                    err_msg = err_body.decode(errors="replace")
                    raise RuntimeError(
                        f"Anthropic ({self.profile.model}) returned HTTP {response.status_code}: {err_msg[:500]}"
                    )

                lines_iter = response.aiter_lines().__aiter__()
                while True:
                    try:
                        chunk_timeout = 30.0 if is_first_chunk else idle_timeout
                        line = await asyncio.wait_for(lines_iter.__anext__(), timeout=chunk_timeout)
                        is_first_chunk = False
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        if is_first_chunk:
                            raise TimeoutError(
                                f"Stream from Anthropic ({self.profile.model}) timed out waiting for first token after {int(chunk_timeout)}s"
                            )
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
                        delta_obj = event_data.get("delta", {})
                        # Handle reasoning/thinking delta
                        if delta_obj.get("type") == "thinking_delta":
                            th_text = delta_obj.get("thinking", "")
                            if th_text and on_thought:
                                await on_thought(
                                    AgentThoughtEvent(delta=th_text, session_id=session_id)
                                )
                        # Handle text delta
                        elif delta_obj.get("type") == "text_delta" or "text" in delta_obj:
                            delta = delta_obj.get("text", "")
                            if delta:
                                accumulated.append(delta)
                                if on_message:
                                    await on_message(
                                        AgentMessageEvent(delta=delta, session_id=session_id)
                                    )

        result_text = "".join(accumulated).strip()
        if not result_text:
            raise RuntimeError(f"Anthropic model '{self.profile.model}' returned an empty response")

        return result_text
