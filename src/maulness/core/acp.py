import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Optional

from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)

logger = logging.getLogger("maulness.acp")


class AcpClient:
    """Agent Client Protocol (ACP) JSON-RPC 2.0 client over stdio streams."""

    def __init__(
        self,
        reader: Optional[asyncio.StreamReader] = None,
        writer: Optional[asyncio.StreamWriter] = None,
        process: Optional[asyncio.subprocess.Process] = None,
    ):
        self.reader = reader
        self.writer = writer
        self.process = process
        self._next_id = 1
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Event Callbacks
        self.on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None
        self.on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None
        self.on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None
        self.on_approval_request: Optional[
            Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]
        ] = None

    @classmethod
    async def spawn(
        cls,
        command: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> "AcpClient":
        """Spawn an agent subprocess and initialize ACP client over its stdin/stdout."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        if process.stdout is None or process.stdin is None:
            raise RuntimeError("Failed to open stdin/stdout pipes for agent process")

        client = cls(reader=process.stdout, writer=process.stdin, process=process)
        client.start_listening()
        return client

    def start_listening(self):
        """Start the background JSON-RPC line reader task."""
        if self._reader_task is None or self._reader_task.done():
            self._is_running = True
            self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self):
        """Cleanly stop the client and terminate the underlying subprocess."""
        self._is_running = False
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()

        # Reject all pending requests
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.set_exception(ConnectionResetError("ACP connection closed"))
        self._pending_requests.clear()

    async def send_request(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Send a JSON-RPC request and wait for the matching result."""
        if not self.writer or self.writer.is_closing():
            raise ConnectionError("ACP writer stream is closed")

        req_id = self._next_id
        self._next_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[req_id] = future

        line = json.dumps(payload) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

        return await future

    async def send_response(
        self,
        request_id: int,
        result: Optional[Any] = None,
        error: Optional[dict[str, Any]] = None,
    ):
        """Send a JSON-RPC response back to an incoming agent request."""
        if not self.writer or self.writer.is_closing():
            raise ConnectionError("ACP writer stream is closed")

        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result if result is not None else {}

        line = json.dumps(payload) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

    async def initialize(self, client_name: str, workspace_uri: str) -> dict[str, Any]:
        """Perform the ACP protocol handshake."""
        return await self.send_request(
            "initialize",
            {
                "protocolVersion": "1.0",
                "clientInfo": {"name": client_name, "version": "0.1.0"},
                "workspace": {"rootUri": workspace_uri},
                "capabilities": {"prompts": True, "tools": True},
            },
        )

    async def prompt(self, session_id: str, prompt_text: str) -> Any:
        """Dispatch a user prompt to an active ACP session."""
        return await self.send_request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": prompt_text,
            },
        )

    async def _read_loop(self):
        """Read and dispatch JSON-RPC messages from stdout."""
        assert self.reader is not None
        while self._is_running:
            try:
                line_bytes = await self.reader.readline()
                if not line_bytes:
                    break  # EOF reached

                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON from ACP agent: %s", line)
                    continue

                await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error reading from ACP stream: %s", e)
                break

    async def _handle_message(self, message: dict[str, Any]):
        """Route an incoming message as request response, notification, or approval request."""
        msg_id = message.get("id")
        method = message.get("method")

        # 1. Response to our client-initiated request
        if msg_id is not None and method is None:
            fut = self._pending_requests.pop(msg_id, None)
            if fut and not fut.done():
                if "error" in message:
                    fut.set_exception(RuntimeError(message["error"].get("message", "ACP RPC Error")))
                else:
                    fut.set_result(message.get("result"))
            return

        # 2. Incoming request from Agent to Client (e.g. session/requestApproval)
        if msg_id is not None and method is not None:
            if method == "session/requestApproval":
                params = message.get("params", {})
                event = ApprovalRequestEvent(
                    request_id=msg_id,
                    call_id=params.get("callId", ""),
                    tool_name=params.get("tool", ""),
                    args=params.get("args", {}),
                    session_id=params.get("sessionId", ""),
                )
                approved = False
                if self.on_approval_request:
                    try:
                        approved = await self.on_approval_request(event)
                    except Exception as e:
                        logger.error("Error in approval handler: %s", e)
                        approved = False

                await self.send_response(msg_id, result={"approved": approved})
            else:
                # Unknown agent request, return method not found
                await self.send_response(
                    msg_id, error={"code": -32601, "message": f"Method {method} not found"}
                )
            return

        # 3. Unprompted streaming notifications
        if method:
            params = message.get("params", {})
            session_id = params.get("sessionId", "")

            if method == "session/thought" and self.on_thought:
                delta = params.get("delta", "")
                await self.on_thought(AgentThoughtEvent(delta=delta, session_id=session_id))

            elif method == "session/message" and self.on_message:
                delta = params.get("delta", "")
                await self.on_message(AgentMessageEvent(delta=delta, session_id=session_id))

            elif method == "session/toolCall" and self.on_tool_call:
                event = AgentToolCallEvent(
                    call_id=params.get("callId", ""),
                    tool_name=params.get("tool", ""),
                    args=params.get("args", {}),
                    session_id=session_id,
                )
                await self.on_tool_call(event)
