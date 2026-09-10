import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.config import config
from maulness.core.acp import AcpClient
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider

logger = logging.getLogger("maulness.providers.acp")


class AcpProvider(BaseProvider):
    """Antigravity subprocess provider supporting native stream-json and standard ACP JSON-RPC."""

    def __init__(self, profile: Profile):
        super().__init__(profile)
        self.last_conversation_id: Optional[str] = None

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
        if not self.profile.command or not self.profile.command.strip():
            raise ValueError(f"Profile '{self.profile.name}' specifies provider 'acp' but has no 'command' configured in config.yaml")
        cmd_tokens = self.profile.command.split()
        cwd = str(workspace_path) if workspace_path else str(config.workspace_root)

        binary = cmd_tokens[0]

        # If using Antigravity CLI (agy), use native stream-json engine
        if "agy" in binary:
            return await self._run_agy_stream(
                binary=binary,
                session_id=session_id,
                prompt=prompt,
                cwd=cwd,
                conversation_id=conversation_id,
                on_init=on_init,
                on_thought=on_thought,
                on_message=on_message,
                on_tool_call=on_tool_call,
                on_approval=on_approval,
            )

        # Standard ACP JSON-RPC 2.0 client
        return await self._run_acp_jsonrpc(
            cmd=cmd_tokens,
            session_id=session_id,
            prompt=prompt,
            cwd=cwd,
            conversation_id=conversation_id,
            on_thought=on_thought,
            on_message=on_message,
            on_tool_call=on_tool_call,
            on_approval=on_approval,
        )

    async def _run_agy_stream(
        self,
        binary: str,
        session_id: str,
        prompt: str,
        cwd: str,
        conversation_id: Optional[str] = None,
        on_init: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        on_thought=None,
        on_message=None,
        on_tool_call=None,
        on_approval=None,
    ) -> str:
        # If conversation_id is provided, resume existing conversation
        effective_prompt = prompt
        cmd = [binary]
        if conversation_id and not str(conversation_id).startswith("chat_"):
            cmd.extend(["--conversation", str(conversation_id)])
        else:
            # If starting a fresh conversation, inject SOUL doctrine & system prompt
            system_doctrine = self.profile.effective_system_prompt()
            if system_doctrine and system_doctrine.strip():
                effective_prompt = (
                    f"[SYSTEM OPERATING DOCTRINE & IDENTITY]\n"
                    f"{system_doctrine.strip()}\n"
                    f"[/SYSTEM OPERATING DOCTRINE]\n\n"
                    f"{prompt}"
                )

        # Append --model from profile.model if not already specified in command
        if self.profile.model and "--model" not in self.profile.command:
            cmd.extend(["--model", self.profile.model])

        # Append --effort from profile.reasoning_effort if not already specified in command
        effort = self.profile.reasoning_effort
        if effort and "--effort" not in self.profile.command:
            cmd.extend(["--effort", effort])

        cmd.extend(["--output-format", "stream-json", "-p", effective_prompt])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        accumulated: list[str] = []
        assert proc.stdout is not None

        idle_timeout = config.stream_idle_timeout_seconds
        try:
            while True:
                try:
                    line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=idle_timeout)
                except asyncio.TimeoutError:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                    raise TimeoutError(
                        f"Antigravity stream idle watchdog triggered: no output received for {int(idle_timeout)}s"
                    )

                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event = data.get("event")
                if event == "init":
                    conv_id = data.get("conversation_id")
                    if conv_id:
                        self.last_conversation_id = conv_id
                        if on_init:
                            await on_init(conv_id)

                elif event == "step_update":
                    su = data.get("step_update", {})
                    step_type = su.get("step_type")
                    text_delta = su.get("text_delta")

                    if step_type == "agent_thought" and text_delta:
                        if on_thought:
                            await on_thought(AgentThoughtEvent(delta=text_delta, session_id=session_id))

                    elif step_type in ("agent_response", None) and text_delta:
                        accumulated.append(text_delta)
                        if on_message:
                            await on_message(AgentMessageEvent(delta=text_delta, session_id=session_id))

                    elif step_type == "tool_call" and on_tool_call:
                        await on_tool_call(
                            AgentToolCallEvent(
                                call_id=str(su.get("step_index", "")),
                                tool_name=su.get("tool_name", "tool"),
                                args=su.get("args", {}),
                                session_id=session_id,
                            )
                        )

                elif event == "result":
                    res = data.get("result", {})
                    conv_id = res.get("conversation_id")
                    if conv_id:
                        self.last_conversation_id = conv_id
                    resp_text = res.get("response")
                    if resp_text and not accumulated:
                        accumulated.append(resp_text)
                        if on_message:
                            await on_message(AgentMessageEvent(delta=resp_text, session_id=session_id))

            await proc.wait()
            if proc.returncode != 0:
                stderr_text = ""
                if proc.stderr:
                    stderr_text = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
                logger.warning("agy exited with code %s: %s", proc.returncode, stderr_text)
                if not accumulated:
                    raise RuntimeError(f"agy execution failed (code {proc.returncode}): {stderr_text or 'No output'}")

        except asyncio.CancelledError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            raise

        return "".join(accumulated)

    async def _run_acp_jsonrpc(
        self,
        cmd: list[str],
        session_id: str,
        prompt: str,
        cwd: str,
        conversation_id: Optional[str] = None,
        on_thought=None,
        on_message=None,
        on_tool_call=None,
        on_approval=None,
    ) -> str:
        """Run standard ACP JSON-RPC 2.0 handshake and session."""
        client = await AcpClient.spawn(command=cmd, cwd=cwd)
        client.on_thought = on_thought
        client.on_tool_call = on_tool_call
        client.on_approval_request = on_approval

        accumulated_text: list[str] = []

        async def internal_msg_handler(event: AgentMessageEvent):
            accumulated_text.append(event.delta)
            if on_message:
                await on_message(event)

        client.on_message = internal_msg_handler

        effective_prompt = prompt
        if not conversation_id:
            system_doctrine = self.profile.effective_system_prompt()
            if system_doctrine and system_doctrine.strip():
                effective_prompt = (
                    f"[SYSTEM OPERATING DOCTRINE & IDENTITY]\n"
                    f"{system_doctrine.strip()}\n"
                    f"[/SYSTEM OPERATING DOCTRINE]\n\n"
                    f"{prompt}"
                )

        try:
            workspace_uri = Path(cwd).as_uri()
            await client.initialize(client_name="maulness", workspace_uri=workspace_uri)
            await client.prompt(session_id=session_id, prompt_text=effective_prompt)
        finally:
            await client.stop()

        return "".join(accumulated_text)
