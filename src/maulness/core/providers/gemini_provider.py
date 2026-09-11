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
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider
from maulness.core.tools import (
    TOOL_DEFINITIONS,
    clean_history_message,
    clean_relay_completion_tags,
    execute_tool_call,
    extract_checkpoint_info,
    format_lean_tool_breadcrumb,
)
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

        # Retrieve conversation history (sanitized)
        history_turns = await self.storage.get_conversation_messages(conv_id, limit=20)
        base_contents: list[types.Content] = []
        for turn in history_turns:
            role = "model" if turn["role"] == "assistant" else "user"
            cleaned_content = clean_history_message(turn["content"])
            base_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=cleaned_content)],
                )
            )
        base_contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        max_relays = getattr(self.profile, "max_relays", config.max_relays)
        max_tool_turns = getattr(self.profile, "max_tool_turns", config.max_tool_turns)
        init_timeout = max(float(config.stream_idle_timeout_seconds), 60.0)
        idle_timeout = float(config.stream_idle_timeout_seconds)
        first_chunk_timeout = 60.0 if effort else 30.0

        accumulated = []
        model_produced_text = False
        relay_count = 0
        relay_checkpoints: list[str] = []
        current_contents = list(base_contents)

        while relay_count < max_relays:
            relay_count += 1
            turn_count = 0
            forcing_synthesis = False
            last_streamed_turn_text = ""

            # Ensure tools are active for new relay
            gen_config = types.GenerateContentConfig(**gen_config_kwargs)

            while turn_count <= max_tool_turns:
                turn_count += 1
                has_tool_call = False
                model_parts: list[types.Part] = []
                tool_response_parts: list[types.Part] = []
                executed_in_turn: set[str] = set()
                streamed_turn_chunks: list[str] = []

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

                    # Process candidate parts for thought, tool calls, and text content
                    has_parts = False
                    if hasattr(chunk, "candidates") and chunk.candidates:
                        for cand in chunk.candidates:
                            if hasattr(cand, "content") and cand.content:
                                for part in cand.content.parts:
                                    # Handle tool/function calls if returned
                                    if getattr(part, "function_call", None) and not forcing_synthesis:
                                        fn = part.function_call
                                        call_name = fn.name
                                        call_args = dict(fn.args) if fn.args else {}
                                        call_sig = f"{call_name}::{json.dumps(call_args, sort_keys=True)}"
                                        if call_sig in executed_in_turn:
                                            continue
                                        executed_in_turn.add(call_sig)

                                        has_parts = True
                                        has_tool_call = True
                                        model_parts.append(part)
                                        if on_tool_call:
                                            await on_tool_call(
                                                AgentToolCallEvent(
                                                    call_id=f"call_{call_name}",
                                                    tool_name=call_name,
                                                    args=call_args,
                                                    session_id=session_id,
                                                )
                                            )
                                        # Execute the tool with execution rules & HITL gating
                                        tool_result = await execute_tool_call(
                                            name=call_name,
                                            args=call_args,
                                            workspace_path=workspace_path,
                                            session_id=session_id,
                                            on_approval=on_approval,
                                            yolo=self.profile.execution.yolo,
                                            rule_engine=self.profile.get_rule_engine(),
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
                                        if part_text.strip():
                                            model_produced_text = True
                                        streamed_turn_chunks.append(part_text)
                                        accumulated.append(part_text)
                                        if on_message:
                                            await on_message(
                                                AgentMessageEvent(delta=part_text, session_id=session_id)
                                            )

                    if not has_parts and getattr(chunk, "text", None):
                        text_cand = chunk.text
                        if text_cand.strip():
                            model_produced_text = True
                        streamed_turn_chunks.append(text_cand)
                        accumulated.append(text_cand)
                        model_parts.append(types.Part.from_text(text=text_cand))
                        if on_message:
                            await on_message(
                                AgentMessageEvent(delta=text_cand, session_id=session_id)
                            )

                last_streamed_turn_text = "".join(streamed_turn_chunks).strip()

                # If tool calls were made during this turn, feed response back to model for synthesis
                if has_tool_call and tool_response_parts:
                    if forcing_synthesis:
                        logger.warning("[%s] Model emitted tool call during forced synthesis; halting tool loop", self.profile.name)
                        break

                    current_contents.append(types.Content(role="model", parts=model_parts))
                    current_contents.append(types.Content(role="user", parts=tool_response_parts))

                    if turn_count >= max_tool_turns:
                        forcing_synthesis = True
                        no_tools_kwargs = {
                            "temperature": self.profile.temperature,
                            "max_output_tokens": self.profile.max_tokens,
                            "system_instruction": self.profile.effective_system_prompt(),
                        }
                        if effort:
                            no_tools_kwargs["thinking_config"] = types.ThinkingConfig(
                                thinking_budget=int(self.EFFORT_BUDGET.get(effort.lower(), 4096))
                            )
                        gen_config = types.GenerateContentConfig(**no_tools_kwargs)

                        if relay_count < max_relays:
                            logger.info(
                                "[%s] Reached burst turn limit (%d) in relay %d/%d; requesting status checkpoint",
                                self.profile.name,
                                max_tool_turns,
                                relay_count,
                                max_relays,
                            )
                            eval_prompt = (
                                f"[SYSTEM NOTE: Execution burst limit reached ({max_tool_turns} tool actions in Relay {relay_count}/{max_relays}). "
                                "Maximum allowed tool actions for this burst reached. "
                                "If the task is fully finished, output your complete final answer to the user now. "
                                "If the task is STILL IN PROGRESS, output exactly:\n"
                                "[STATUS: IN_PROGRESS]\n"
                                "Accomplished: <1-2 sentences on what was completed in this burst>\n"
                                "Key Findings: <key facts, file paths, or test results discovered>\n"
                                "Next Step: <exact action to take in the next burst>\n"
                                "Do not attempt to call any tools.]"
                            )
                        else:
                            logger.info(
                                "[%s] Reached burst turn limit (%d) in final relay %d/%d; requesting final answer synthesis",
                                self.profile.name,
                                max_tool_turns,
                                relay_count,
                                max_relays,
                            )
                            eval_prompt = (
                                f"[SYSTEM NOTE: You have reached the maximum allowed execution budget across all relays (Relay {relay_count}/{max_relays}). "
                                "Maximum allowed tool actions reached. "
                                "Formulate and output your final, comprehensive answer to the user now based on all information and tool results above. "
                                "Do not attempt to call any tools.]"
                            )
                        current_contents.append(types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=eval_prompt)],
                        ))
                        continue

                    logger.info("[%s] Tool executed; continuing turn loop for model answer synthesis (step %d)", self.profile.name, turn_count)
                    continue
                else:
                    break

            # Burst completed. Check if model requested autonomous relay continuation
            if forcing_synthesis and relay_count < max_relays:
                is_in_progress, checkpoint_body, next_step = extract_checkpoint_info(last_streamed_turn_text)
                if is_in_progress:
                    logger.info(
                        "[%s] Relay %d/%d produced IN_PROGRESS checkpoint: %s. Auto-advancing to next relay.",
                        self.profile.name,
                        relay_count,
                        max_relays,
                        next_step,
                    )
                    checkpoint_notice = (
                        f"\n\n> **[Relay Checkpoint {relay_count}/{max_relays}]** "
                        f"Auto-advancing with: *{next_step}*\n\n"
                    )
                    accumulated.append(checkpoint_notice)
                    if on_message:
                        await on_message(AgentMessageEvent(delta=checkpoint_notice, session_id=session_id))
                    if on_thought:
                        await on_thought(AgentThoughtEvent(
                            delta=f"Relay Checkpoint {relay_count}/{max_relays}: {next_step}",
                            session_id=session_id,
                        ))

                    relay_checkpoints.append(
                        f"### Relay {relay_count} Checkpoint\n{checkpoint_body}"
                    )
                    checkpoints_summary = "\n\n".join(relay_checkpoints)

                    # Compact context: prune bulky prior tool payloads, preserve doctrine, history, and checkpoint ledger
                    current_contents = list(base_contents)
                    current_contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=f"[AUTONOMOUS RELAY CHECKPOINTS]\n{checkpoints_summary}")],
                    ))
                    current_contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=(
                                f"[SYSTEM NOTE: Autonomous relay {relay_count + 1} of {max_relays} initiated.\n"
                                f"Original user task: {prompt}\n"
                                f"Target for this burst: {next_step}\n"
                                "Prior bulky tool outputs have been compacted to retain focus. "
                                "Tools are re-enabled. Continue executing the task now.]"
                            )
                        )],
                    ))
                    continue

            # Task completed or max relays ceiling reached
            break

        if relay_count >= max_relays and forcing_synthesis:
            is_in_progress, _, _ = extract_checkpoint_info(last_streamed_turn_text)
            if is_in_progress:
                pause_note = f"\n\n*(Maximum relay budget of {max_relays} relays reached. Task paused at checkpoint.)*"
                accumulated.append(pause_note)
                if on_message:
                    await on_message(AgentMessageEvent(delta=pause_note, session_id=session_id))

        result_text = "".join(accumulated).strip()
        result_text = clean_relay_completion_tags(result_text)
        if not result_text:
            raise RuntimeError(f"Gemini model '{model_name}' completed stream but returned an empty response")

        if not model_produced_text:
            fallback_note = "\n\n*(Agent completed tool executions but did not produce a final textual summary.)*"
            result_text += fallback_note
            if on_message:
                await on_message(AgentMessageEvent(delta=fallback_note, session_id=session_id))

        # Persist conversation turn
        try:
            await self.storage.add_conversation_message(conv_id, "user", prompt)
            await self.storage.add_conversation_message(conv_id, "assistant", result_text)
        except Exception as e:
            logger.debug("Failed to persist conversation turn: %s", e)

        return result_text
