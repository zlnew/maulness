import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.config import config
from maulness.core.kernel.models import KernelEventType, ModelTurnOutput
from maulness.core.kernel.step_runner import DurableStepRunner
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.tools import (
    ActionLoopDetector,
    clean_history_message,
    clean_relay_completion_tags,
    clear_turn_tools,
    detect_simulated_tool_call,
    execute_tool_call,
    extract_checkpoint_info,
    format_lean_tool_breadcrumb,
    tombstone_tool_output,
    truncate_observation,
)
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.kernel.loop")


def parse_session_scope(session_id: str) -> tuple[str, str]:
    """Parse task_id and stage name from session_id (e.g. task_abc123_building -> task_abc123, building)."""
    if "_" in session_id:
        parts = session_id.split("_")
        if len(parts) >= 2 and parts[0] == "task":
            task_id = f"{parts[0]}_{parts[1]}"
            stage = "_".join(parts[2:]) if len(parts) > 2 else "default"
            return task_id, stage
        return parts[0], "_".join(parts[1:])
    return session_id, "default"


def compact_in_flight_tool_messages(
    messages: list[dict[str, Any]], keep_recent: int = 3
) -> None:
    """Compact older tool observation messages in-place while preserving API schema envelopes."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return

    to_compact = tool_indices[:-keep_recent]
    for idx in to_compact:
        msg = messages[idx]
        raw_content = str(msg.get("content", ""))
        if len(raw_content) > 200 and not raw_content.startswith("[Tool result for "):
            call_id = msg.get("tool_call_id", "")
            name = "tool"
            args: dict[str, Any] = {}
            for prev_idx in range(idx - 1, -1, -1):
                prev_msg = messages[prev_idx]
                if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                    for tc in prev_msg["tool_calls"]:
                        if tc.get("id") == call_id:
                            fn = tc.get("function", {})
                            name = fn.get("name", "tool")
                            try:
                                args = json.loads(fn.get("arguments", "{}"))
                            except Exception:
                                args = {}
                            break
                    break
            msg["content"] = tombstone_tool_output(name, args, raw_content)


class DurableAgentKernel:
    """Durable agent execution kernel with event sourcing, step memoization, and crash resumption."""

    def __init__(self, profile: Profile, storage: Optional[StorageManager] = None):
        self.profile = profile
        self.storage = storage or StorageManager()
        self.step_runner = DurableStepRunner(storage=self.storage)
        self.rule_engine = profile.get_rule_engine()
        self.loop_detector = ActionLoopDetector()

    async def run(
        self,
        turn_generator_fn: Callable[..., Coroutine[Any, Any, ModelTurnOutput]],
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
        """Drive multi-turn execution loop with durable step memoization."""
        conv_id = conversation_id or str(uuid.uuid4())
        if on_init:
            await on_init(conv_id)

        # 1. Load multi-turn history from storage
        history = await self.storage.get_conversation_messages(conv_id, limit=20)

        task_id, stage = parse_session_scope(session_id)
        if task_id == "chat" or task_id.startswith("chat"):
            if conv_id:
                task_id = f"chat_{conv_id[:12]}"
            # Scope stage by turn count in conversation to avoid cross-turn memoization collisions
            turn_idx = len(history)
            stage = f"{stage}_t{turn_idx}"
        effective_workspace = workspace_path or (
            Path(self.profile.workspace)
            if self.profile.workspace and self.profile.workspace != "inherit"
            else config.workspace_root
        )
        sandbox_mode = kwargs.get("sandbox_mode", self.profile.sandbox_mode)
        yolo = kwargs.get("yolo", self.profile.yolo)
        max_relays = kwargs.get(
            "max_relays", getattr(self.profile, "max_relays", config.max_relays)
        )
        max_tool_turns = kwargs.get(
            "max_tool_turns",
            getattr(self.profile, "max_tool_turns", config.max_tool_turns),
        )

        system_content = self.profile.effective_system_prompt(
            workspace_path=effective_workspace
        )
        if kwargs.get("extra_system_prompt"):
            system_content += "\n\n" + str(kwargs["extra_system_prompt"]).strip()
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]
        for turn in history:
            base_messages.append(
                {
                    "role": turn["role"],
                    "content": clean_history_message(turn["content"]),
                }
            )
        base_messages.append({"role": "user", "content": prompt})

        # Memoize initial prompt event if not present
        await self.step_runner.execute_step(
            task_id=task_id,
            stage=stage,
            step_index=0,
            action_type=KernelEventType.PROMPT,
            action_fn=lambda: {"prompt": prompt, "conversation_id": conv_id},
            action_meta=prompt,
        )

        accumulated: list[str] = []
        model_produced_text = False
        relay_count = 0
        relay_checkpoints: list[str] = []
        current_messages = list(base_messages)
        loop_intervention_active: Optional[str] = None
        step_counter = 0

        # Fetch effective tool definitions once: static tools + any active MCP tools
        from maulness.core.tools import get_effective_tool_definitions

        effective_tool_definitions = await get_effective_tool_definitions(
            effective_workspace
        )

        turn_output: Optional[ModelTurnOutput] = None
        while relay_count < max_relays:
            relay_count += 1
            turn_count = 0
            forcing_synthesis = False
            synthesis_retried = False
            last_streamed_turn_text = ""

            while max_tool_turns <= 0 or turn_count <= max_tool_turns:
                turn_count += 1
                step_counter += 1

                # If forced synthesis triggered, strip tools
                active_tools = None if forcing_synthesis else effective_tool_definitions

                # If loop intervention was tripped, append system instruction
                if loop_intervention_active:
                    current_messages.append(
                        {
                            "role": "system",
                            "content": loop_intervention_active,
                        }
                    )
                    loop_intervention_active = None

                # Step 1: Model Generation Step (Memoized)
                turn_step_idx = step_counter
                model_step_meta = {
                    "turn_count": turn_count,
                    "relay_count": relay_count,
                    "forcing_synthesis": forcing_synthesis,
                    "message_count": len(current_messages),
                    "last_msg": current_messages[-1] if current_messages else {},
                }

                # Inner closure for execution
                async def _invoke_model_turn() -> dict[str, Any]:
                    output = await turn_generator_fn(
                        messages=current_messages,
                        tools=active_tools,
                        session_id=session_id,
                        on_thought=on_thought,
                        on_message=on_message,
                        **kwargs,
                    )
                    return output.to_dict()

                turn_res = await self.step_runner.execute_step(
                    task_id=task_id,
                    stage=stage,
                    step_index=turn_step_idx,
                    action_type=KernelEventType.MODEL_TURN,
                    action_fn=_invoke_model_turn,
                    action_meta=model_step_meta,
                )

                turn_output = ModelTurnOutput.from_dict(turn_res.value)
                last_streamed_turn_text = turn_output.content
                if turn_output.content:
                    accumulated.append(turn_output.content)
                    if turn_output.content.strip():
                        model_produced_text = True

                # If retrieved from cache, replay visible streaming events for UI
                if turn_res.from_cache:
                    if turn_output.thought and on_thought:
                        await on_thought(
                            AgentThoughtEvent(
                                delta=turn_output.thought, session_id=session_id
                            )
                        )
                    if turn_output.content and on_message:
                        await on_message(
                            AgentMessageEvent(
                                delta=turn_output.content, session_id=session_id
                            )
                        )

                # Step 2: Handle completion or simulated tools
                tool_calls = turn_output.tool_calls
                if not tool_calls and not forcing_synthesis and turn_output.content:
                    simulated = detect_simulated_tool_call(turn_output.content)
                    if simulated:
                        call_name, call_args = simulated
                        if accumulated and accumulated[-1] == turn_output.content:
                            accumulated.pop()
                        tool_calls = [
                            {
                                "id": f"call_intercepted_{turn_count}_{call_name}",
                                "name": call_name,
                                "arguments": call_args,
                            }
                        ]

                # If no tool calls produced, turn cycle completed
                if not tool_calls or forcing_synthesis:
                    if (
                        not (turn_output.content and turn_output.content.strip())
                        and not synthesis_retried
                    ):
                        synthesis_retried = True
                        forcing_synthesis = True  # strip tools so model must synthesize final text
                        logger.info(
                            "[kernel] Model completed tool execution but emitted empty visible content; retrying for final textual answer"
                        )
                        current_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM NOTE: You analyzed the data in internal reasoning, but did not generate a final textual response to the user. "
                                    "Formulate and output your final textual answer to the user now based on your reasoning and the tool results above. "
                                    "Do not call any tools or output only internal thought.]"
                                ),
                            }
                        )
                        continue
                    break

                # Step 3: Execute Tool Calls (Memoized per call)
                assistant_msg = {
                    "role": "assistant",
                    "content": turn_output.content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"]
                                if isinstance(tc["arguments"], str)
                                else json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                current_messages.append(assistant_msg)

                for tc in tool_calls:
                    step_counter += 1
                    call_id = tc["id"]
                    call_name = tc["name"]
                    call_args = tc["arguments"]
                    if isinstance(call_args, str):
                        try:
                            call_args_dict = json.loads(call_args)
                        except Exception:
                            call_args_dict = {"raw": call_args}
                    else:
                        call_args_dict = call_args

                    if on_tool_call:
                        await on_tool_call(
                            AgentToolCallEvent(
                                call_id=call_id,
                                tool_name=call_name,
                                args=call_args_dict,
                                session_id=session_id,
                            )
                        )

                    # Tool execution wrapper
                    async def _run_tool() -> str:
                        res = await execute_tool_call(
                            name=call_name,
                            args=call_args_dict,
                            workspace_path=effective_workspace,
                            session_id=session_id,
                            on_approval=on_approval,
                            yolo=yolo,
                            rule_engine=self.rule_engine,
                            sandbox_mode=sandbox_mode,
                        )
                        return truncate_observation(res)

                    tool_step_res = await self.step_runner.execute_step(
                        task_id=task_id,
                        stage=stage,
                        step_index=step_counter,
                        action_type=KernelEventType.TOOL_RESULT,
                        action_fn=_run_tool,
                        action_meta={
                            "call_id": call_id,
                            "name": call_name,
                            "args": call_args_dict,
                        },
                    )

                    result_str = str(tool_step_res.value)
                    breadcrumb = format_lean_tool_breadcrumb(
                        call_name, call_args_dict, result_str
                    )
                    accumulated.append(breadcrumb)
                    if on_message:
                        await on_message(
                            AgentMessageEvent(delta=breadcrumb, session_id=session_id)
                        )

                    # Check for action loop intervention
                    if "[LOOP INTERVENTION:" in result_str:
                        loop_intervention_active = result_str

                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_str,
                        }
                    )

                # In-flight microcompaction of historical observations
                compact_in_flight_tool_messages(current_messages, keep_recent=3)

                # Check if tool turns budget is reached
                if max_tool_turns > 0 and turn_count >= max_tool_turns:
                    forcing_synthesis = True
                    if relay_count < max_relays:
                        logger.info(
                            "[kernel] Reached burst turn limit (%d) in relay %d/%d; requesting status checkpoint",
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
                            "[kernel] Reached burst turn limit (%d) in final relay %d/%d; requesting final answer synthesis",
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
                    current_messages.append(
                        {
                            "role": "user",
                            "content": eval_prompt,
                        }
                    )
                    continue

            # Check for Natural Progress Checkpoints / Relays
            if forcing_synthesis and relay_count < max_relays:
                is_in_progress, checkpoint_body, next_step = extract_checkpoint_info(
                    last_streamed_turn_text
                )
                if is_in_progress and next_step:
                    logger.info(
                        "[kernel] Relay %d/%d produced IN_PROGRESS checkpoint: %s. Auto-advancing to next relay.",
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
                        await on_message(
                            AgentMessageEvent(
                                delta=checkpoint_notice, session_id=session_id
                            )
                        )
                    if on_thought:
                        await on_thought(
                            AgentThoughtEvent(
                                delta=f"Relay Checkpoint {relay_count}/{max_relays}: {next_step}",
                                session_id=session_id,
                            )
                        )

                    relay_checkpoints.append(
                        f"### Relay {relay_count} Checkpoint\n{checkpoint_body}"
                    )
                    checkpoints_summary = "\n\n".join(relay_checkpoints)

                    # Compact context: prune bulky prior tool payloads, preserve doctrine, history, and checkpoint ledger
                    current_messages = list(base_messages)
                    current_messages.append(
                        {
                            "role": "assistant",
                            "content": f"[AUTONOMOUS RELAY CHECKPOINTS]\n{checkpoints_summary}",
                        }
                    )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[SYSTEM NOTE: Autonomous relay {relay_count + 1} of {max_relays} initiated.\n"
                                f"Original user task: {prompt}\n"
                                f"Target for this burst: {next_step}\n"
                                "Prior bulky tool outputs have been compacted to retain focus. "
                                "Tools are re-enabled. Continue executing the task now.]"
                            ),
                        }
                    )
                    continue

            # Task finished or maximum relay ceiling reached
            break

        if relay_count >= max_relays and forcing_synthesis:
            is_in_progress, _, _ = extract_checkpoint_info(last_streamed_turn_text)
            if is_in_progress:
                pause_note = f"\n\n*(Maximum relay budget of {max_relays} relays reached. Task paused at checkpoint.)*"
                accumulated.append(pause_note)
                if on_message:
                    await on_message(
                        AgentMessageEvent(delta=pause_note, session_id=session_id)
                    )

        final_text = "".join(accumulated).strip()
        final_text = clean_relay_completion_tags(final_text)

        if not model_produced_text:
            if turn_output and turn_output.thought and turn_output.thought.strip():
                clean_thought = turn_output.thought.strip()
                fallback_note = f"\n\n*(Summary derived from agent analysis:)*\n{clean_thought[:1200]}"
            else:
                fallback_note = "\n\n*(Agent completed tool executions but did not produce a final textual summary.)*"
            final_text += fallback_note
            if on_message:
                await on_message(
                    AgentMessageEvent(delta=fallback_note, session_id=session_id)
                )

        if not final_text:
            raise RuntimeError(
                f"Provider '{self.profile.provider}' ({self.profile.model}) returned an empty response"
            )

        # Memoize final response
        await self.step_runner.execute_step(
            task_id=task_id,
            stage=stage,
            step_index=step_counter + 1,
            action_type=KernelEventType.FINAL_RESPONSE,
            action_fn=lambda: final_text,
            action_meta={"session_id": session_id, "length": len(final_text)},
        )

        # Persist conversation turns to storage
        try:
            await self.storage.add_conversation_message(conv_id, "user", prompt)
            await self.storage.add_conversation_message(
                conv_id, "assistant", final_text
            )
        except Exception as e:
            logger.debug("Failed to persist conversation turn: %s", e)

        return final_text
