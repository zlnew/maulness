import asyncio
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
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider
from maulness.core.tools import TOOL_DEFINITIONS, execute_tool_call, format_lean_tool_breadcrumb
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.providers.gemini")


class GeminiProvider(BaseProvider):
    """Direct Google Gemini API provider using the official google-genai SDK."""

    def __init__(self, profile: Profile, storage: Optional[StorageManager] = None):
        super().__init__(profile)
        self.storage = storage or StorageManager()
        self._last_conversation_id: Optional[str] = None

    @property
    def last_conversation_id(self) -> Optional[str]:
        return self._last_conversation_id

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
        conv_id = conversation_id or str(uuid.uuid4())
        self._last_conversation_id = conv_id
        if on_init:
            await on_init(conv_id)

        if self.profile.vertex:
            project = self.profile.project or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = self.profile.location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = self.profile.get_api_key() or config.gemini_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                import shutil
                # Only fall back to ACP if this profile actually has command configured
                if self.profile.command and shutil.which(self.profile.command.split()[0]):
                    from maulness.core.providers.acp_provider import AcpProvider
                    fallback_provider = AcpProvider(self.profile)
                    return await fallback_provider.run(
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
                raise ValueError(
                    f"Missing API key for profile '{self.profile.name}'. "
                    f"Set {self.profile.api_key_env or 'GEMINI_API_KEY or GOOGLE_API_KEY'} in ~/.config/maulness/env or profile .env"
                )

            client = genai.Client(api_key=api_key)

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
            # Ensure output budget has room for both reasoning tokens and visible output text
            max_tokens = max(max_tokens, budget + 4096)

        gen_config_kwargs["max_output_tokens"] = max_tokens

        # Direct provider tool declarations
        tool_declarations = []
        for td in TOOL_DEFINITIONS:
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

        # Retrieve conversation history
        history_turns = await self.storage.get_conversation_messages(conv_id, limit=20)
        contents: list[types.Content] = []
        for turn in history_turns:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn["content"])],
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        accumulated = []
        init_timeout = min(float(config.stream_idle_timeout_seconds), 20.0)
        idle_timeout = config.stream_idle_timeout_seconds

        max_tool_turns = 5
        turn_count = 0

        while turn_count < max_tool_turns:
            turn_count += 1
            has_tool_call = False
            model_parts: list[types.Part] = []
            tool_response_parts: list[types.Part] = []

            try:
                response_stream = await asyncio.wait_for(
                    client.aio.models.generate_content_stream(
                        model=model_name,
                        contents=contents,
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

            while True:
                try:
                    chunk_timeout = 25.0 if is_first_chunk else idle_timeout
                    chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=chunk_timeout)
                    is_first_chunk = False
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    if is_first_chunk:
                        raise TimeoutError(
                            f"Gemini model '{model_name}' timed out waiting for first token response after 25s"
                        )
                    raise TimeoutError(
                        f"Gemini stream idle watchdog triggered: no response received for {int(idle_timeout)}s"
                    )

                # Process candidate parts for thought, tool calls, and text content
                has_parts = False
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for cand in chunk.candidates:
                        if hasattr(cand, "content") and cand.content:
                            for part in cand.content.parts:
                                # Handle tool/function calls if returned
                                if getattr(part, "function_call", None):
                                    fn = part.function_call
                                    has_parts = True
                                    has_tool_call = True
                                    model_parts.append(part)
                                    call_name = fn.name
                                    call_args = dict(fn.args) if fn.args else {}
                                    if on_tool_call:
                                        await on_tool_call(
                                            AgentToolCallEvent(
                                                call_id=f"call_{call_name}",
                                                tool_name=call_name,
                                                args=call_args,
                                                session_id=session_id,
                                            )
                                        )
                                    # Execute the tool with HITL gating
                                    tool_result = await execute_tool_call(
                                        name=call_name,
                                        args=call_args,
                                        workspace_path=workspace_path,
                                        session_id=session_id,
                                        on_approval=on_approval,
                                    )
                                    tool_desc = format_lean_tool_breadcrumb(call_name, call_args, tool_result)
                                    accumulated.append(tool_desc)
                                    if on_message:
                                        await on_message(
                                            AgentMessageEvent(delta=tool_desc, session_id=session_id)
                                        )
                                    tool_response_parts.append(
                                        types.Part.from_function_response(
                                            name=call_name,
                                            response={"result": tool_result},
                                        )
                                    )

                                part_text = getattr(part, "text", None)
                                if not part_text:
                                    continue
                                has_parts = True
                                model_parts.append(part)
                                if getattr(part, "thought", None):
                                    if on_thought:
                                        await on_thought(
                                            AgentThoughtEvent(delta=part_text, session_id=session_id)
                                        )
                                else:
                                    accumulated.append(part_text)
                                    if on_message:
                                        await on_message(
                                            AgentMessageEvent(delta=part_text, session_id=session_id)
                                        )

                if not has_parts and getattr(chunk, "text", None):
                    accumulated.append(chunk.text)
                    model_parts.append(types.Part.from_text(text=chunk.text))
                    if on_message:
                        await on_message(
                            AgentMessageEvent(delta=chunk.text, session_id=session_id)
                        )

            # If tool calls were made during this turn, feed response back to model for synthesis
            if has_tool_call and tool_response_parts:
                contents.append(types.Content(role="model", parts=model_parts))
                contents.append(types.Content(role="user", parts=tool_response_parts))
                logger.info("[%s] Tool executed; continuing turn loop for model answer synthesis (step %d)", self.profile.name, turn_count)
                continue
            else:
                break

        result_text = "".join(accumulated).strip()
        if not result_text:
            raise RuntimeError(f"Gemini model '{model_name}' completed stream but returned an empty response")

        # Persist conversation turn
        try:
            await self.storage.add_conversation_message(conv_id, "user", prompt)
            await self.storage.add_conversation_message(conv_id, "assistant", result_text)
        except Exception as e:
            logger.debug("Failed to persist conversation turn: %s", e)

        return result_text
