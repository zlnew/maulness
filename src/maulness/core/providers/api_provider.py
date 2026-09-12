import asyncio
import json
import logging
import uuid
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
from maulness.core.kernel import DurableAgentKernel, ModelTurnOutput
from maulness.core.kernel.loop import compact_in_flight_tool_messages
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.providers.api")

__all__ = ["UnifiedApiProvider", "compact_in_flight_tool_messages"]


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

    def __init__(self, profile: Profile, storage: Optional[StorageManager] = None):
        super().__init__(profile)
        self.storage = storage or StorageManager()
        self._last_conv_id: Optional[str] = None

    @property
    def last_conversation_id(self) -> Optional[str]:
        return self._last_conv_id

    @property
    def base_url(self) -> Optional[str]:
        if self.profile.base_url:
            return self.profile.base_url
        endpoint = self.BASE_URLS.get(
            self.profile.provider.lower().strip().replace("-", "_")
        )
        if endpoint:
            for suffix in ("/chat/completions", "/messages"):
                if endpoint.endswith(suffix):
                    return endpoint[: -len(suffix)]
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
        **kwargs: Any,
    ) -> str:
        api_key = self.profile.get_api_key()
        provider_type = self.profile.provider.lower().strip().replace("-", "_")

        # Ollama usually does not require an API key
        if not api_key and provider_type != "ollama":
            key_name = (
                self.profile.api_key_env
                or f"{self.profile.provider.upper().replace('-', '_')}_API_KEY"
            )
            raise ValueError(
                f"Missing API key for profile '{self.profile.name}'. "
                f"Set {key_name} in ~/.config/maulness/env or profile .env"
            )

        conv_id = conversation_id or str(uuid.uuid4())
        self._last_conv_id = conv_id
        if on_init:
            await on_init(conv_id)

        kernel = DurableAgentKernel(self.profile, self.storage)
        generator_fn = (
            self._generate_anthropic_turn
            if provider_type == "anthropic"
            else self._generate_openai_turn
        )

        return await kernel.run(
            turn_generator_fn=generator_fn,
            session_id=session_id,
            prompt=prompt,
            workspace_path=workspace_path,
            conversation_id=conv_id,
            on_init=None,
            on_thought=on_thought,
            on_message=on_message,
            on_tool_call=on_tool_call,
            on_approval=on_approval,
            **kwargs,
        )

    async def _generate_openai_turn(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        session_id: str,
        on_thought: Optional[
            Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_message: Optional[
            Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]
        ] = None,
        **kwargs: Any,
    ) -> ModelTurnOutput:
        provider_type = self.profile.provider.lower().strip().replace("-", "_")
        url = self.profile.base_url or self.BASE_URLS.get(
            provider_type, self.BASE_URLS["openai"]
        )
        if not url.endswith("/chat/completions"):
            url = f"{url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        api_key = self.profile.get_api_key()
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

        payload: dict[str, Any] = {
            "model": self.profile.model or default_model,
            "messages": messages,
            "temperature": self.profile.temperature,
            "max_tokens": self.profile.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        effort = self.profile.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort

        idle_timeout = float(config.stream_idle_timeout_seconds)
        captured_tool_calls: list[dict[str, Any]] = []
        streamed_text_chunks: list[str] = []
        thought_chunks: list[str] = []
        is_first_chunk = True
        finish_reason: Optional[str] = None

        timeout = httpx.Timeout(120.0, connect=20.0, read=idle_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                stream_ctx = client.stream("POST", url, headers=headers, json=payload)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initiate stream with {provider_type} ({url}): {e}"
                )

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
                        chunk_timeout = (
                            60.0
                            if (is_first_chunk and provider_type == "ollama")
                            else (30.0 if is_first_chunk else idle_timeout)
                        )
                        line = await asyncio.wait_for(
                            lines_iter.__anext__(), timeout=chunk_timeout
                        )
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
                    finish_reason = choice.get("finish_reason") or finish_reason

                    # 1. Capture reasoning / thoughts (Ollama, DeepSeek, OpenCode)
                    reasoning_delta = delta.get("reasoning") or delta.get(
                        "reasoning_content"
                    )
                    if reasoning_delta:
                        thought_chunks.append(reasoning_delta)
                        if on_thought:
                            await on_thought(
                                AgentThoughtEvent(
                                    delta=reasoning_delta, session_id=session_id
                                )
                            )

                    # 2. Capture tool calls (OpenAI format)
                    tool_calls_delta = delta.get("tool_calls")
                    if tool_calls_delta and tools:
                        for tc in tool_calls_delta:
                            idx = tc.get("index", 0)
                            while len(captured_tool_calls) <= idx:
                                captured_tool_calls.append(
                                    {"id": "", "name": "", "arguments": ""}
                                )
                            if tc.get("id"):
                                captured_tool_calls[idx]["id"] += tc["id"]
                            fn = tc.get("function", {})
                            if "name" in fn:
                                captured_tool_calls[idx]["name"] += fn["name"]
                            if "arguments" in fn:
                                captured_tool_calls[idx]["arguments"] += fn["arguments"]

                    # 3. Capture visible text content
                    text_delta = delta.get("content", "")
                    if text_delta:
                        streamed_text_chunks.append(text_delta)
                        if on_message:
                            await on_message(
                                AgentMessageEvent(
                                    delta=text_delta, session_id=session_id
                                )
                            )

        formatted_tool_calls = []
        for i, tc in enumerate(captured_tool_calls):
            call_name = tc.get("name", "").strip()
            call_args_str = tc.get("arguments", "").strip()
            if not call_name:
                continue
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}_{call_name}"
            try:
                call_args = json.loads(call_args_str) if call_args_str else {}
            except Exception:
                call_args = {"raw": call_args_str}

            formatted_tool_calls.append(
                {
                    "id": call_id,
                    "name": call_name,
                    "arguments": call_args,
                }
            )

        return ModelTurnOutput(
            content="".join(streamed_text_chunks),
            thought="".join(thought_chunks) if thought_chunks else None,
            tool_calls=formatted_tool_calls,
            finish_reason=finish_reason,
        )

    async def _generate_anthropic_turn(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        session_id: str,
        on_thought: Optional[
            Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_message: Optional[
            Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]
        ] = None,
        **kwargs: Any,
    ) -> ModelTurnOutput:
        api_key = self.profile.get_api_key() or ""
        url = self.profile.base_url or self.BASE_URLS["anthropic"]
        if not url.endswith("/messages"):
            url = f"{url.rstrip('/')}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        anthropic_messages: list[dict[str, Any]] = []
        for turn in messages:
            if turn.get("role") == "system":
                continue
            anthropic_messages.append(
                {
                    "role": "assistant" if turn["role"] == "assistant" else "user",
                    "content": turn.get("content", ""),
                }
            )

        payload: dict[str, Any] = {
            "model": self.profile.model or "claude-3-7-sonnet-20250219",
            "system": self.profile.effective_system_prompt(),
            "messages": anthropic_messages,
            "max_tokens": self.profile.max_tokens,
            "temperature": self.profile.temperature,
            "stream": True,
        }
        _EFFORT_BUDGET = {"low": 1024, "medium": 8192, "high": 24576}
        effort = self.profile.reasoning_effort
        if effort:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": _EFFORT_BUDGET.get(effort.lower(), 8192),
            }
            payload["temperature"] = 1

        idle_timeout = float(config.stream_idle_timeout_seconds)
        streamed_text_chunks: list[str] = []
        thought_chunks: list[str] = []
        is_first_chunk = True

        timeout = httpx.Timeout(120.0, connect=20.0, read=idle_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                stream_ctx = client.stream("POST", url, headers=headers, json=payload)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initiate stream with Anthropic ({url}): {e}"
                )

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
                        line = await asyncio.wait_for(
                            lines_iter.__anext__(), timeout=chunk_timeout
                        )
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
                        if delta_obj.get("type") == "thinking_delta":
                            th_text = delta_obj.get("thinking", "")
                            if th_text:
                                thought_chunks.append(th_text)
                                if on_thought:
                                    await on_thought(
                                        AgentThoughtEvent(
                                            delta=th_text, session_id=session_id
                                        )
                                    )
                        elif (
                            delta_obj.get("type") == "text_delta" or "text" in delta_obj
                        ):
                            delta = delta_obj.get("text", "")
                            if delta:
                                streamed_text_chunks.append(delta)
                                if on_message:
                                    await on_message(
                                        AgentMessageEvent(
                                            delta=delta, session_id=session_id
                                        )
                                    )

        return ModelTurnOutput(
            content="".join(streamed_text_chunks),
            thought="".join(thought_chunks) if thought_chunks else None,
            tool_calls=[],
        )
