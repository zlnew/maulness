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
from maulness.core.tools import (
    TOOL_DEFINITIONS,
    clean_history_message,
    clean_relay_completion_tags,
    detect_simulated_tool_call,
    execute_tool_call,
    extract_checkpoint_info,
    format_lean_tool_breadcrumb,
    tombstone_tool_output,
)
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.providers.api")


def compact_in_flight_tool_messages(messages: list[dict[str, Any]], keep_recent: int = 3) -> None:
    """Compact older tool messages in place to preserve context window and attention fidelity."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return

    to_compact = tool_indices[:-keep_recent]
    for idx in to_compact:
        msg = messages[idx]
        raw_content = msg.get("content", "")
        # Only compact substantial content that has not already been compacted
        if len(raw_content) > 200 and not raw_content.startswith("[Tool result for "):
            call_id = msg.get("tool_call_id")
            name = "tool"
            args: dict[str, Any] = {}
            for prev_idx in range(idx - 1, -1, -1):
                prev_msg = messages[prev_idx]
                if prev_msg.get("role") == "assistant":
                    for tc in prev_msg.get("tool_calls", []):
                        if tc.get("id") == call_id:
                            name = tc.get("function", {}).get("name", "tool")
                            try:
                                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                            except Exception:
                                pass
                            break
                    break
            msg["content"] = tombstone_tool_output(name, args, raw_content)


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

        conv_id = conversation_id or str(uuid.uuid4())
        self._last_conv_id = conv_id
        if on_init:
            await on_init(conv_id)

        if provider_type == "anthropic":
            return await self._stream_anthropic(
                session_id=session_id,
                conv_id=conv_id,
                prompt=prompt,
                api_key=api_key or "",
                on_message=on_message,
                on_thought=on_thought,
            )
        else:
            return await self._stream_openai_compatible(
                provider_type=provider_type,
                session_id=session_id,
                conv_id=conv_id,
                prompt=prompt,
                api_key=api_key or "",
                workspace_path=workspace_path,
                on_message=on_message,
                on_thought=on_thought,
                on_tool_call=on_tool_call,
                on_approval=on_approval,
            )

    async def _stream_openai_compatible(
        self,
        provider_type: str,
        session_id: str,
        conv_id: str,
        prompt: str,
        api_key: str,
        workspace_path: Optional[Path] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]] = None,
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

        # Multi-turn history retrieval (sanitized to remove simulated tool syntax)
        history = await self.storage.get_conversation_messages(conv_id, limit=20)
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.profile.effective_system_prompt()}
        ]
        for turn in history:
            base_messages.append({
                "role": turn["role"],
                "content": clean_history_message(turn["content"]),
            })
        base_messages.append({"role": "user", "content": prompt})

        max_relays = getattr(self.profile, "max_relays", config.max_relays)
        max_tool_turns = getattr(self.profile, "max_tool_turns", config.max_tool_turns)
        idle_timeout = float(config.stream_idle_timeout_seconds)

        accumulated: list[str] = []
        model_produced_text = False
        relay_count = 0
        relay_checkpoints: list[str] = []
        current_messages = list(base_messages)

        while relay_count < max_relays:
            relay_count += 1
            turn_count = 0
            forcing_synthesis = False
            last_streamed_turn_text = ""

            payload: dict[str, Any] = {
                "model": self.profile.model or default_model,
                "messages": current_messages,
                "temperature": self.profile.temperature,
                "max_tokens": self.profile.max_tokens,
                "stream": True,
                "tools": TOOL_DEFINITIONS,
            }
            effort = self.profile.reasoning_effort
            if effort:
                payload["reasoning_effort"] = effort

            while max_tool_turns <= 0 or turn_count <= max_tool_turns:
                turn_count += 1
                captured_tool_calls: list[dict[str, Any]] = []
                streamed_text_chunks: list[str] = []
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
                            if tool_calls_delta and not forcing_synthesis:
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
                                streamed_text_chunks.append(text_delta)
                                accumulated.append(text_delta)
                                if text_delta.strip():
                                    model_produced_text = True
                                if on_message:
                                    await on_message(
                                        AgentMessageEvent(delta=text_delta, session_id=session_id)
                                    )

                last_streamed_turn_text = "".join(streamed_text_chunks).strip()

                # Process captured tool calls if any and loop back for synthesized response
                if captured_tool_calls:
                    if forcing_synthesis:
                        logger.warning("[%s] Model emitted tool call during forced synthesis; halting tool loop", provider_type)
                        break

                    asst_tool_calls = []
                    tool_results = []
                    for i, tc in enumerate(captured_tool_calls):
                        call_name = tc.get("name", "").strip()
                        call_args_str = tc.get("arguments", "").strip()
                        if not call_name:
                            continue
                        try:
                            call_args = json.loads(call_args_str) if call_args_str else {}
                        except Exception:
                            call_args = {"raw": call_args_str}

                        call_id = f"call_{turn_count}_{i}_{call_name}"
                        asst_tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call_name, "arguments": call_args_str or "{}"},
                        })

                        if on_tool_call:
                            await on_tool_call(
                                AgentToolCallEvent(
                                    call_id=call_id,
                                    tool_name=call_name,
                                    args=call_args,
                                    session_id=session_id,
                                )
                            )

                        tool_result = await execute_tool_call(
                            name=call_name,
                            args=call_args,
                            workspace_path=workspace_path,
                            session_id=session_id,
                            on_approval=on_approval,
                            yolo=self.profile.execution.yolo,
                            rule_engine=self.profile.get_rule_engine(),
                            sandbox_mode=self.profile.sandbox_mode,
                        )
                        breadcrumb = format_lean_tool_breadcrumb(call_name, call_args, tool_result)
                        accumulated.append(breadcrumb)
                        if on_message:
                            await on_message(
                                AgentMessageEvent(delta=breadcrumb, session_id=session_id)
                            )

                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result,
                        })

                    if asst_tool_calls:
                        current_messages.append({"role": "assistant", "tool_calls": asst_tool_calls})
                        current_messages.extend(tool_results)
                        compact_in_flight_tool_messages(current_messages, keep_recent=3)
                        payload["messages"] = current_messages

                        if max_tool_turns > 0 and turn_count >= max_tool_turns:
                            forcing_synthesis = True
                            payload.pop("tools", None)
                            if relay_count < max_relays:
                                logger.info(
                                    "[%s] Reached burst turn limit (%d) in relay %d/%d; requesting status checkpoint",
                                    provider_type,
                                    max_tool_turns,
                                    relay_count,
                                    max_relays,
                                )
                                eval_prompt = (
                                    f"[SYSTEM NOTE: Execution burst limit reached ({max_tool_turns} tool actions in Relay {relay_count}/{max_relays}). "
                                    "Maximum allowed tool actions for this burst reached. "
                                    "If the task is fully finished, output your complete final answer to the user now. "
                                    "If the task is STILL IN PROGRESS, output your current progress and Next Step: <exact action to take in the next burst>. "
                                    "Tools will be automatically re-enabled for the next burst. Do not attempt to call any tools in this turn.]"
                                )
                            else:
                                logger.info(
                                    "[%s] Reached burst turn limit (%d) in final relay %d/%d; requesting final answer synthesis",
                                    provider_type,
                                    max_tool_turns,
                                    relay_count,
                                    max_relays,
                                )
                                eval_prompt = (
                                    f"[SYSTEM NOTE: You have reached the maximum allowed execution budget across all relays (Relay {relay_count}/{max_relays}). "
                                    "Maximum allowed tool actions reached. "
                                    "Formulate and output your final, comprehensive answer to the user now based on all information and tool results above. "
                                    "Do not attempt to call any more tools.]"
                                )
                            current_messages.append({
                                "role": "user",
                                "content": eval_prompt,
                            })
                            payload["messages"] = current_messages
                            continue

                        logger.info("[%s] Tool executed; continuing turn loop for model answer synthesis (step %d)", provider_type, turn_count)
                        continue
                    else:
                        break
                else:
                    # Fail-safe: Detect if model simulated tool breadcrumbs in its own streamed text instead of tool calling
                    current_turn_text = "".join(streamed_text_chunks).strip()
                    simulated = detect_simulated_tool_call(current_turn_text)
                    if simulated and not forcing_synthesis:
                        call_name, call_args = simulated
                        logger.warning(
                            "[%s] Intercepted simulated tool call in model text: %s with args %s",
                            provider_type,
                            call_name,
                            call_args,
                        )
                        call_id = f"call_intercepted_{turn_count}_{call_name}"
                        asst_tool_calls = [{
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call_name, "arguments": json.dumps(call_args)},
                        }]
                        if on_tool_call:
                            await on_tool_call(
                                AgentToolCallEvent(
                                    call_id=call_id,
                                    tool_name=call_name,
                                    args=call_args,
                                    session_id=session_id,
                                )
                            )
                        tool_result = await execute_tool_call(
                            name=call_name,
                            args=call_args,
                            workspace_path=workspace_path,
                            session_id=session_id,
                            on_approval=on_approval,
                            yolo=self.profile.execution.yolo,
                            rule_engine=self.profile.get_rule_engine(),
                            sandbox_mode=self.profile.sandbox_mode,
                        )
                        real_breadcrumb = format_lean_tool_breadcrumb(call_name, call_args, tool_result)
                        if streamed_text_chunks:
                            accumulated = accumulated[:-len(streamed_text_chunks)]
                        accumulated.append(real_breadcrumb)
                        if on_message:
                            await on_message(
                                AgentMessageEvent(delta=real_breadcrumb, session_id=session_id)
                            )
                        current_messages.append({"role": "assistant", "tool_calls": asst_tool_calls})
                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result,
                        })
                        compact_in_flight_tool_messages(current_messages, keep_recent=3)
                        payload["messages"] = current_messages

                        if max_tool_turns > 0 and turn_count >= max_tool_turns:
                            forcing_synthesis = True
                            payload.pop("tools", None)
                            if relay_count < max_relays:
                                logger.info(
                                    "[%s] Reached burst turn limit (%d) in relay %d/%d; requesting status checkpoint",
                                    provider_type,
                                    max_tool_turns,
                                    relay_count,
                                    max_relays,
                                )
                                eval_prompt = (
                                    f"[SYSTEM NOTE: Execution burst limit reached ({max_tool_turns} tool actions in Relay {relay_count}/{max_relays}). "
                                    "Maximum allowed tool actions for this burst reached. "
                                    "If the task is fully finished, output your complete final answer to the user now. "
                                    "If the task is STILL IN PROGRESS, output your current progress and Next Step: <exact action to take in the next burst>. "
                                    "Tools will be automatically re-enabled for the next burst. Do not attempt to call any tools in this turn.]"
                                )
                            else:
                                logger.info(
                                    "[%s] Reached burst turn limit (%d) in final relay %d/%d; requesting final answer synthesis",
                                    provider_type,
                                    max_tool_turns,
                                    relay_count,
                                    max_relays,
                                )
                                eval_prompt = (
                                    f"[SYSTEM NOTE: You have reached the maximum allowed execution budget across all relays (Relay {relay_count}/{max_relays}). "
                                    "Maximum allowed tool actions reached. "
                                    "Formulate and output your final, comprehensive answer to the user now based on all information and tool results above. "
                                    "Do not attempt to call any more tools.]"
                                )
                            current_messages.append({
                                "role": "user",
                                "content": eval_prompt,
                            })
                            payload["messages"] = current_messages
                            continue

                        logger.info("[%s] Intercepted tool executed; continuing turn loop for model answer synthesis (step %d)", provider_type, turn_count)
                        continue
                    break

            # Burst completed. Check if model requested autonomous relay continuation
            if forcing_synthesis and relay_count < max_relays:
                is_in_progress, checkpoint_body, next_step = extract_checkpoint_info(last_streamed_turn_text)
                if is_in_progress:
                    logger.info(
                        "[%s] Relay %d/%d produced IN_PROGRESS checkpoint: %s. Auto-advancing to next relay.",
                        provider_type,
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
                    current_messages = list(base_messages)
                    current_messages.append({
                        "role": "assistant",
                        "content": f"[AUTONOMOUS RELAY CHECKPOINTS]\n{checkpoints_summary}",
                    })
                    current_messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM NOTE: Autonomous relay {relay_count + 1} of {max_relays} initiated.\n"
                            f"Original user task: {prompt}\n"
                            f"Target for this burst: {next_step}\n"
                            "Prior bulky tool outputs have been compacted to retain focus. "
                            "Tools are re-enabled. Continue executing the task now.]"
                        ),
                    })
                    continue

            # Task finished or maximum relay ceiling reached
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
            raise RuntimeError(
                f"Provider '{provider_type}' ({self.profile.model}) returned an empty response"
            )

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

    async def _stream_anthropic(
        self,
        session_id: str,
        conv_id: str,
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

        # Multi-turn history retrieval (sanitized)
        history = await self.storage.get_conversation_messages(conv_id, limit=20)
        messages: list[dict[str, Any]] = []
        for turn in history:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": clean_history_message(turn["content"])})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.profile.model or "claude-3-7-sonnet-20250219",
            "system": self.profile.effective_system_prompt(),
            "messages": messages,
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

        # Persist conversation turn
        try:
            await self.storage.add_conversation_message(conv_id, "user", prompt)
            await self.storage.add_conversation_message(conv_id, "assistant", result_text)
        except Exception as e:
            logger.debug("Failed to persist conversation turn: %s", e)

        return result_text
