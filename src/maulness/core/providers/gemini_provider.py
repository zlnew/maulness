import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from google import genai
from google.genai import types

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.kernel import DurableAgentKernel, ModelTurnOutput
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider
from maulness.core.tools import TOOL_DEFINITIONS, tombstone_tool_output
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.providers.gemini")


def compact_gemini_in_flight_contents(contents: list[Any], keep_recent: int = 3) -> None:
    """Compact older Gemini function_response parts in place to protect context window."""
    resp_entries = []
    for c_idx, content in enumerate(contents):
        parts = getattr(content, "parts", None)
        if parts:
            for p_idx, part in enumerate(parts):
                fn_resp = getattr(part, "function_response", None)
                if fn_resp:
                    resp_entries.append((c_idx, p_idx, fn_resp))

    if len(resp_entries) <= keep_recent:
        return

    to_compact = resp_entries[:-keep_recent]
    for c_idx, p_idx, fn_resp in to_compact:
        resp_dict = getattr(fn_resp, "response", None)
        if isinstance(resp_dict, dict) and "result" in resp_dict:
            raw_res = str(resp_dict["result"])
            if len(raw_res) > 200 and not raw_res.startswith("[Tool result for "):
                fn_name = getattr(fn_resp, "name", "tool")
                tombstone = tombstone_tool_output(fn_name, {}, raw_res)
                resp_dict["result"] = tombstone


def _find_tool_name(messages: list[dict[str, Any]], msg_idx: int, call_id: str) -> str:
    for prev_idx in range(msg_idx - 1, -1, -1):
        prev = messages[prev_idx]
        if prev.get("role") == "assistant":
            for tc in prev.get("tool_calls", []):
                if tc.get("id") == call_id:
                    fn = tc.get("function", {})
                    return fn.get("name") or tc.get("name") or "tool"
    if call_id.startswith("call_"):
        parts = call_id.split("_")
        if len(parts) >= 2:
            return parts[-1]
    return "tool"


def messages_to_gemini_contents(messages: list[dict[str, Any]]) -> list[types.Content]:
    """Convert standard conversation messages into Gemini SDK Content objects."""
    contents: list[types.Content] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        if role == "system":
            continue

        if role == "user":
            content_text = msg.get("content", "")
            if content_text:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_text)],
                    )
                )
        elif role == "assistant":
            parts: list[types.Part] = []
            content_text = msg.get("content", "")
            if content_text:
                parts.append(types.Part.from_text(text=content_text))
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                call_name = fn.get("name") or tc.get("name", "tool")
                call_args = fn.get("arguments") or tc.get("arguments", {})
                if isinstance(call_args, str):
                    try:
                        call_args = json.loads(call_args)
                    except Exception:
                        call_args = {}
                parts.append(types.Part.from_function_call(name=call_name, args=call_args))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            tool_name = _find_tool_name(messages, idx, call_id)
            tool_result = msg.get("content", "")
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": tool_result},
                        )
                    ],
                )
            )

    return contents


class GeminiProvider(BaseProvider):
    """Direct Google Gemini API provider using the official google-genai SDK."""

    def __init__(self, profile: Profile, storage: Optional[StorageManager] = None):
        super().__init__(profile)
        self.storage = storage or StorageManager()
        self._last_conversation_id: Optional[str] = None

        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        if self._client is not None:
            return self._client
        if self.profile.vertex:
            project = self.profile.project or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = self.profile.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            self._client = genai.Client(vertexai=True, project=project, location=location)
            return self._client
        else:
            api_key = self.profile.get_api_key() or config.gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                import shutil
                if self.profile.command and shutil.which(self.profile.command.split()[0]):
                    return None
                raise ValueError(
                    f"Missing API key for profile '{self.profile.name}'. "
                    f"Set {self.profile.api_key_env or 'GEMINI_API_KEY or GOOGLE_API_KEY'} in ~/.config/maulness/env or profile .env"
                )
            self._client = genai.Client(api_key=api_key)
            return self._client

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
        conv_id = conversation_id or str(uuid.uuid4())
        self._last_conversation_id = conv_id
        if on_init:
            await on_init(conv_id)

        client = self._get_client()
        if client is None:
            from maulness.core.providers.acp_provider import AcpProvider
            fallback_provider = AcpProvider(self.profile)
            return await fallback_provider.run(
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

        kernel = DurableAgentKernel(self.profile, self.storage)
        return await kernel.run(
            turn_generator_fn=self.generate_turn,
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

    async def generate_turn(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        session_id: str,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None,
        **kwargs: Any,
    ) -> ModelTurnOutput:
        client = self._get_client()
        if client is None:
            raise RuntimeError("Gemini client not initialized")

        model_name = self.profile.model or "gemini-3.6-flash"
        _EFFORT_BUDGET = {"low": 1024, "medium": 8192, "high": 24576}

        max_tokens = self.profile.max_tokens or 4096
        effort = self.profile.reasoning_effort
        gen_config_kwargs: dict = {
            "system_instruction": self.profile.effective_system_prompt(),
            "temperature": self.profile.temperature,
        }
        if effort:
            budget = _EFFORT_BUDGET.get(effort.lower(), 8192)
            gen_config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            max_tokens = max(max_tokens, budget + 4096)

        gen_config_kwargs["max_output_tokens"] = max_tokens

        if tools:
            tool_declarations = []
            for td in tools:
                fn = td["function"]
                tool_declarations.append(
                    types.FunctionDeclaration(
                        name=fn["name"],
                        description=fn["description"],
                        parameters=fn.get("parameters"),
                    )
                )
            if tool_declarations:
                gen_config_kwargs["tools"] = [types.Tool(function_declarations=tool_declarations)]

        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
        current_contents = messages_to_gemini_contents(messages)

        init_timeout = max(float(config.stream_idle_timeout_seconds), 60.0)
        idle_timeout = float(config.stream_idle_timeout_seconds)
        first_chunk_timeout = 60.0 if effort else 30.0

        try:
            response_stream = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=current_contents,
                    config=gen_config,
                ),
                timeout=init_timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Gemini API connection handshake timed out after {int(init_timeout)}s (model '{model_name}' overloaded)"
            )

        stream_iter = response_stream.__aiter__()
        is_first_chunk = True
        tool_calls: list[dict[str, Any]] = []
        executed_in_turn: set[str] = set()
        streamed_turn_chunks: list[str] = []
        thought_chunks: list[str] = []

        while True:
            try:
                chunk_timeout = first_chunk_timeout if is_first_chunk else idle_timeout
                chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
                is_first_chunk = False
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if is_first_chunk:
                    raise TimeoutError(
                        f"Gemini model '{model_name}' timed out waiting for first token response after {int(first_chunk_timeout)}s"
                    )
                raise TimeoutError(
                    f"Gemini stream idle watchdog triggered: no response received for {int(idle_timeout)}s"
                )

            has_parts = False
            if hasattr(chunk, "candidates") and chunk.candidates:
                for cand in chunk.candidates:
                    if hasattr(cand, "content") and cand.content:
                        for part in cand.content.parts:
                            if getattr(part, "function_call", None) and tools:
                                fn = part.function_call
                                call_name = fn.name
                                call_args = dict(fn.args) if fn.args else {}
                                call_sig = f"{call_name}::{json.dumps(call_args, sort_keys=True)}"
                                if call_sig not in executed_in_turn:
                                    executed_in_turn.add(call_sig)
                                    tool_calls.append({
                                        "id": f"call_{call_name}",
                                        "name": call_name,
                                        "arguments": call_args,
                                    })
                                has_parts = True

                            part_text = getattr(part, "text", None)
                            if not part_text:
                                continue

                            is_thought = getattr(part, "thought", False)
                            if is_thought:
                                has_parts = True
                                thought_chunks.append(part_text)
                                if on_thought:
                                    await on_thought(AgentThoughtEvent(delta=part_text, session_id=session_id))
                            else:
                                has_parts = True
                                streamed_turn_chunks.append(part_text)
                                if on_message:
                                    await on_message(AgentMessageEvent(delta=part_text, session_id=session_id))

            if not has_parts and hasattr(chunk, "text") and chunk.text:
                text_cand = chunk.text
                streamed_turn_chunks.append(text_cand)
                if on_message:
                    await on_message(AgentMessageEvent(delta=text_cand, session_id=session_id))

        return ModelTurnOutput(
            content="".join(streamed_turn_chunks),
            thought="".join(thought_chunks) if thought_chunks else None,
            tool_calls=tool_calls,
        )
