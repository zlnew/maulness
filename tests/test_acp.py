import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from maulness.core.acp import AcpClient
from maulness.core.models import AgentThoughtEvent, ApprovalRequestEvent


class MockStreamWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data: bytes):
        self.data.extend(data)

    async def drain(self):
        pass

    def is_closing(self):
        return self.closed


@pytest.mark.asyncio
async def test_acp_initialize():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    client = AcpClient(reader=reader, writer=writer)
    client.start_listening()

    # Schedule simulated agent response on stdout
    async def respond():
        await asyncio.sleep(0.01)
        response = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "1.0", "agentInfo": {"name": "Antigravity"}}}
        reader.feed_data(json.dumps(response).encode("utf-8") + b"\n")

    asyncio.create_task(respond())

    result = await client.initialize("test-client", "file:///test")
    assert result["agentInfo"]["name"] == "Antigravity"

    # Verify what client sent to stdin
    sent_request = json.loads(writer.data.decode("utf-8").strip())
    assert sent_request["method"] == "initialize"
    assert sent_request["params"]["clientInfo"]["name"] == "test-client"

    await client.stop()


@pytest.mark.asyncio
async def test_acp_thought_notification():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    client = AcpClient(reader=reader, writer=writer)

    received_thoughts = []

    async def on_thought(event: AgentThoughtEvent):
        received_thoughts.append(event.delta)

    client.on_thought = on_thought
    client.start_listening()

    # Feed thought notification
    notification = {
        "jsonrpc": "2.0",
        "method": "session/thought",
        "params": {"sessionId": "sess_1", "delta": "Thinking about the fix..."},
    }
    reader.feed_data(json.dumps(notification).encode("utf-8") + b"\n")

    await asyncio.sleep(0.02)
    assert len(received_thoughts) == 1
    assert received_thoughts[0] == "Thinking about the fix..."

    await client.stop()


@pytest.mark.asyncio
async def test_acp_approval_request_flow():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    client = AcpClient(reader=reader, writer=writer)

    async def on_approval(event: ApprovalRequestEvent) -> bool:
        assert event.tool_name == "run_command"
        assert event.args["command"] == "pytest"
        return True  # Approved by user

    client.on_approval_request = on_approval
    client.start_listening()

    # Agent sends approval request to client
    req = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "session/requestApproval",
        "params": {
            "sessionId": "sess_1",
            "callId": "call_1",
            "tool": "run_command",
            "args": {"command": "pytest"},
        },
    }
    reader.feed_data(json.dumps(req).encode("utf-8") + b"\n")

    await asyncio.sleep(0.02)

    # Verify client responded with approved: true
    sent_response = json.loads(writer.data.decode("utf-8").strip())
    assert sent_response["id"] == 99
    assert sent_response["result"]["approved"] is True

    await client.stop()


@pytest.mark.asyncio
async def test_agy_stream_conversation_persistence(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from maulness.core.profiles import Profile
    from maulness.core.providers.acp_provider import AcpProvider

    profile = Profile(
        identity={"name": "builder"},
        agent={"provider": "acp", "command": "agy --output-format stream-json"},
    )
    provider = AcpProvider(profile)

    captured_cmds = []

    class MockProcess:
        def __init__(self, cmd):
            captured_cmds.append(cmd)
            self.returncode = 0
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()

            # Feed test events
            self.stdout.feed_data(
                b'{"event":"init","conversation_id":"conv-abc-123"}\n'
                b'{"event":"step_update","step_update":{"step_type":"agent_response","text_delta":"Sure!"}}\n'
                b'{"event":"result","result":{"conversation_id":"conv-abc-123","response":"Sure!"}}\n'
            )
            self.stdout.feed_eof()

        async def wait(self):
            return 0

    def mock_exec(*cmd, **kwargs):
        return MockProcess(list(cmd))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(side_effect=mock_exec))

    init_convs = []

    async def on_init(cid: str):
        init_convs.append(cid)

    # Turn 1: Fresh conversation
    res1 = await provider.run(
        session_id="task_1",
        prompt="First prompt",
        on_init=on_init,
    )
    assert res1 == "Sure!"
    assert init_convs == ["conv-abc-123"]
    assert provider.last_conversation_id == "conv-abc-123"
    assert "--conversation" not in captured_cmds[0]

    # Turn 2: Resumed conversation
    res2 = await provider.run(
        session_id="task_2",
        prompt="Second prompt",
        conversation_id="conv-abc-123",
    )
    assert res2 == "Sure!"
    assert "--conversation" in captured_cmds[1]
    conv_idx = captured_cmds[1].index("--conversation")
    assert captured_cmds[1][conv_idx + 1] == "conv-abc-123"


@pytest.mark.asyncio
async def test_acp_client_closed_writer_and_pending_cleanup():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    writer.closed = True
    client = AcpClient(reader=reader, writer=writer)

    # 1. Closed writer raises ConnectionError
    with pytest.raises(ConnectionError, match="ACP writer stream is closed"):
        await client.send_request("test_method")

    # 2. Stop rejects pending requests
    writer.closed = False
    client.start_listening()
    req_task = asyncio.create_task(client.send_request("pending_call"))
    await asyncio.sleep(0.01)
    await client.stop()

    with pytest.raises(ConnectionResetError):
        await req_task


@pytest.mark.asyncio
async def test_acp_client_messages_and_unknown_method():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    client = AcpClient(reader=reader, writer=writer)

    received_tools = []
    received_msgs = []

    async def on_tool(event):
        received_tools.append(event)

    async def on_msg(event):
        received_msgs.append(event)

    client.on_tool_call = on_tool
    client.on_message = on_msg
    client.start_listening()

    # 1. Malformed JSON line (logged, ignored, loop continues)
    reader.feed_data(b"not a valid json line\n")

    # 2. Tool call notification
    reader.feed_data(
        json.dumps({
            "jsonrpc": "2.0",
            "method": "session/toolCall",
            "params": {"sessionId": "s1", "callId": "c1", "tool": "run_command", "args": {"command": "ls"}},
        }).encode("utf-8") + b"\n"
    )

    # 3. Message delta notification
    reader.feed_data(
        json.dumps({
            "jsonrpc": "2.0",
            "method": "session/message",
            "params": {"sessionId": "s1", "delta": "Chunk 1"},
        }).encode("utf-8") + b"\n"
    )

    # 4. Unknown agent method request
    reader.feed_data(
        json.dumps({
            "jsonrpc": "2.0",
            "id": 55,
            "method": "custom/unknownMethod",
            "params": {},
        }).encode("utf-8") + b"\n"
    )

    await asyncio.sleep(0.05)
    assert len(received_tools) == 1
    assert received_tools[0].tool_name == "run_command"
    assert len(received_msgs) == 1
    assert received_msgs[0].delta == "Chunk 1"

    # Verify response to unknown method was -32601 error
    responses = [json.loads(line) for line in writer.data.decode("utf-8").strip().splitlines() if line.strip()]
    err_resp = next(r for r in responses if r.get("id") == 55)
    assert err_resp["error"]["code"] == -32601

    await client.stop()


@pytest.mark.asyncio
async def test_acp_provider_run_acp_jsonrpc(tmp_path: Path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from maulness.core.profiles import Profile
    from maulness.core.providers.acp_provider import AcpProvider
    from maulness.core.models import AgentMessageEvent

    profile = Profile(
        identity={"name": "builder"},
        agent={"provider": "acp", "command": "mcp-agent"},
    )
    provider = AcpProvider(profile)

    mock_client = MagicMock()
    mock_client.initialize = AsyncMock(return_value={"agent": "mcp"})
    
    async def mock_prompt(session_id, prompt_text):
        if mock_client.on_message:
            await mock_client.on_message(AgentMessageEvent(delta="JSONRPC Response", session_id=session_id))
        return "JSONRPC Response"

    mock_client.prompt = AsyncMock(side_effect=mock_prompt)
    mock_client.stop = AsyncMock()

    monkeypatch.setattr(AcpClient, "spawn", AsyncMock(return_value=mock_client))

    messages = []
    async def on_message(event):
        messages.append(event.delta)

    result = await provider._run_acp_jsonrpc(
        cmd=["mcp-agent"],
        session_id="sess_jsonrpc",
        prompt="Build the bridge",
        cwd=str(tmp_path),
        on_message=on_message,
    )

    assert result == "JSONRPC Response"
    assert messages == ["JSONRPC Response"]
    mock_client.initialize.assert_awaited_once()
    mock_client.prompt.assert_awaited_once()
    mock_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_acp_client_spawn_and_lifecycle():
    # 1. AcpClient.spawn pipes None raises RuntimeError (lines 57-58)
    mock_bad_proc = MagicMock(stdout=None, stdin=None)
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_bad_proc)):
        with pytest.raises(RuntimeError, match="Failed to open stdin/stdout pipes"):
            await AcpClient.spawn(command=["agy"])

    # 2. AcpClient.spawn success (lines 49-62)
    mock_proc = MagicMock(
        stdout=asyncio.StreamReader(),
        stdin=MockStreamWriter(),
        stderr=asyncio.StreamReader(),
        returncode=None,
        wait=AsyncMock(return_value=0),
        terminate=MagicMock(),
        kill=MagicMock(),
    )
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        client = await AcpClient.spawn(command=["agy"])
        assert client.process == mock_proc
        assert client._is_running is True

        # Stop lifecycle with process timeout (lines 83-87) and pending request rejection (lines 91-92)
        fut_stop = asyncio.get_running_loop().create_future()
        client._pending_requests[101] = fut_stop
        mock_proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), 0])
        await client.stop()
        mock_proc.terminate.assert_called()
        mock_proc.kill.assert_called()
        with pytest.raises(ConnectionResetError, match="ACP connection closed"):
            await fut_stop


@pytest.mark.asyncio
async def test_acp_client_send_response_closed_and_pending_exceptions():
    reader = asyncio.StreamReader()
    writer = MockStreamWriter()
    client = AcpClient(reader=reader, writer=writer)

    # 1. send_response when writer is closed raises ConnectionError (line 128)
    writer.closed = True
    with pytest.raises(ConnectionError, match="ACP writer stream is closed"):
        await client.send_response(1, result="ok")

    # 2. send_request when writer is closed raises ConnectionError (line 98)
    with pytest.raises(ConnectionError, match="ACP writer stream is closed"):
        await client.send_request("method")

    # 3. prompt helper (lines 154-159)
    writer.closed = False
    loop = asyncio.get_running_loop()

    async def respond_prompt():
        await asyncio.sleep(0.01)
        # Verify prompt payload sent to writer
        sent = json.loads(writer.data.decode("utf-8").strip().splitlines()[-1])
        assert sent["method"] == "session/prompt"
        assert sent["params"]["prompt"] == "hello test"
        await client._handle_message({"jsonrpc": "2.0", "id": sent["id"], "result": "prompt result"})

    asyncio.create_task(respond_prompt())
    client.start_listening()
    res = await client.prompt("s1", "hello test")
    assert res == "prompt result"

    # 4. _handle_message response with error (line 208)
    fut = loop.create_future()
    client._pending_requests[999] = fut
    await client._handle_message({"jsonrpc": "2.0", "id": 999, "error": {"message": "Custom agent error"}})
    with pytest.raises(RuntimeError, match="Custom agent error"):
        await fut

    # 5. _handle_message approval request handler exception (lines 228-230)
    async def faulty_approval(event):
        raise ValueError("Crash in approval")

    client.on_approval_request = faulty_approval
    await client._handle_message({
        "jsonrpc": "2.0",
        "id": 888,
        "method": "session/requestApproval",
        "params": {"sessionId": "s", "tool": "rm", "args": {}},
    })
    # Check that it replied with approved=False
    last_line = json.loads(writer.data.decode("utf-8").strip().splitlines()[-1])
    assert last_line["id"] == 888
    assert last_line["result"] == {"approved": False}

    await client.stop()

    # 6. Read loop stream exception (lines 185-187) and empty line/EOF (lines 169, 173)
    reader_eof = asyncio.StreamReader()
    reader_eof.feed_data(b"\n") # line 173
    reader_eof.feed_eof()       # line 169
    client_eof = AcpClient(reader=reader_eof, writer=writer)
    client_eof._is_running = True
    await client_eof._read_loop()
    assert client_eof._is_running is True

    client2 = AcpClient(reader=reader, writer=writer)
    fut2 = loop.create_future()
    client2._pending_requests[77] = fut2
    client2._is_running = True
    with patch.object(reader, "readline", side_effect=OSError("Read error")):
        await client2._read_loop()
        with pytest.raises(ConnectionResetError, match="ACP stream closed unexpectedly"):
            await fut2


@pytest.mark.asyncio
async def test_acp_provider_agy_stream_full_coverage(tmp_path: Path):
    from maulness.core.profiles import Profile
    from maulness.core.providers.acp_provider import AcpProvider
    from maulness.core.models import AgentThoughtEvent, AgentMessageEvent, AgentToolCallEvent

    # 1. Profile missing command raises ValueError (lines 41-42)
    p_no_cmd = Profile(identity={"name": "no_cmd"}, agent={"provider": "acp", "command": "agy"})
    p_no_cmd.agent_cfg.command = "   "
    prov_no_cmd = AcpProvider(p_no_cmd)
    with pytest.raises(ValueError, match="specifies provider 'acp' but has no 'command'"):
        await prov_no_cmd.run("s", "prompt")

    # 2. Command with model and effort appended (lines 107, 112)
    p_agy = Profile(
        identity={"name": "builder"},
        agent={"provider": "acp", "command": "agy", "model": "gemini-3.6-flash", "reasoning_effort": "high"},
    )
    prov_agy = AcpProvider(p_agy)

    # 3. Idle watchdog timeout with kill (lines 131-138)
    class TimeoutProc:
        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = None
            self.terminate = MagicMock()
            self.kill = MagicMock()

        async def wait(self):
            return 0

    mock_timeout_proc = TimeoutProc()
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_timeout_proc)):
        with patch.object(mock_timeout_proc, "wait", side_effect=[asyncio.TimeoutError(), 0]):
            with patch.object(mock_timeout_proc.stdout, "readline", side_effect=asyncio.TimeoutError):
                with pytest.raises(TimeoutError, match="Antigravity stream idle watchdog triggered"):
                    await prov_agy.run("s", "prompt")
                mock_timeout_proc.terminate.assert_called()
                mock_timeout_proc.kill.assert_called()

    # 4. CancelledError handling (lines 206-213)
    mock_cancel_proc = TimeoutProc()
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_cancel_proc)):
        with patch.object(mock_cancel_proc, "wait", side_effect=[asyncio.TimeoutError(), 0]):
            with patch.object(mock_cancel_proc.stdout, "readline", side_effect=asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await prov_agy.run("s", "prompt")
                mock_cancel_proc.terminate.assert_called()
                mock_cancel_proc.kill.assert_called()

    # 5. Return code non-zero with no output raises RuntimeError (lines 199-205)
    class ErrorProc:
        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = 1
        async def wait(self): return 1

    mock_err_proc = ErrorProc()
    mock_err_proc.stdout.feed_eof()
    mock_err_proc.stderr.feed_data(b"CLI crash reason\n")
    mock_err_proc.stderr.feed_eof()
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_err_proc)):
        with pytest.raises(RuntimeError, match="agy execution failed \\(code 1\\): CLI crash reason"):
            await prov_agy.run("s", "prompt")

    # 6. Stream with:
    # - empty line (line 147)
    # - invalid json line (lines 151-152)
    # - init event with on_init (lines 155-161)
    # - step_update agent_thought (lines 167-169)
    # - step_update agent_response and None (lines 171-175)
    # - step_update tool_call (lines 176-184)
    # - result event with response fallback (lines 191-196)
    class SuccessProc:
        def __init__(self):
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.returncode = 0
        async def wait(self): return 0

    mock_succ_proc = SuccessProc()
    mock_succ_proc.stdout.feed_data(b"\n   \n") # empty lines
    mock_succ_proc.stdout.feed_data(b"not json\n") # invalid json
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "init", "conversation_id": "conv-123"}).encode() + b"\n")
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_thought", "text_delta": "Thinking..."}}).encode() + b"\n")
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "step_update", "step_update": {"step_type": "agent_response", "text_delta": "Hello "}}).encode() + b"\n")
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "step_update", "step_update": {"step_type": None, "text_delta": "World"}}).encode() + b"\n")
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "step_update", "step_update": {"step_type": "tool_call", "step_index": 1, "tool_name": "run_command", "args": {"cmd": "ls"}}}).encode() + b"\n")
    mock_succ_proc.stdout.feed_data(json.dumps({"event": "result", "result": {"conversation_id": "conv-123", "response": "Fallback"}}).encode() + b"\n")
    mock_succ_proc.stdout.feed_eof()
    mock_succ_proc.stderr.feed_eof()

    init_calls = []
    thoughts = []
    messages = []
    tool_calls = []
    async def on_init(cid): init_calls.append(cid)
    async def on_th(ev): thoughts.append(ev.delta)
    async def on_msg(ev): messages.append(ev.delta)
    async def on_tc(ev): tool_calls.append(ev.tool_name)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_succ_proc)):
        res = await prov_agy.run(
            session_id="s",
            prompt="do work",
            on_init=on_init,
            on_thought=on_th,
            on_message=on_msg,
            on_tool_call=on_tc,
        )
        assert res == "Hello World"
        assert init_calls == ["conv-123"]
        assert thoughts == ["Thinking..."]
        assert messages == ["Hello ", "World"]
        assert tool_calls == ["run_command"]
        assert prov_agy.last_conversation_id == "conv-123"

    # 7. Result response fallback when not accumulated (lines 193-195)
    mock_result_proc = SuccessProc()
    mock_result_proc.stdout.feed_data(json.dumps({"event": "result", "result": {"conversation_id": "conv-456", "response": "Only in result"}}).encode() + b"\n")
    mock_result_proc.stdout.feed_eof()
    mock_result_proc.stderr.feed_eof()
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_result_proc)):
        res_direct = await prov_agy.run("s", "work", on_message=on_msg)
        assert res_direct == "Only in result"


@pytest.mark.asyncio
async def test_acp_provider_run_acp_jsonrpc_gated_approval_all_policies(tmp_path: Path):
    from maulness.core.profiles import Profile
    from maulness.core.providers.acp_provider import AcpProvider
    from maulness.core.rules import PolicyAction, RuleEngine

    profile = Profile(
        identity={"name": "custom_acp"},
        agent={"provider": "acp", "command": "custom-agent"},
    )
    provider = AcpProvider(profile)

    mock_client = MagicMock()
    mock_client.initialize = AsyncMock(return_value={})
    mock_client.prompt = AsyncMock(return_value="OK")
    mock_client.stop = AsyncMock()

    with patch.object(AcpClient, "spawn", AsyncMock(return_value=mock_client)):
        approval_cb = AsyncMock(return_value=True)
        # Call provider.run (covers line 64: standard ACP JSON-RPC client)
        await provider.run(
            session_id="s",
            prompt="test",
            workspace_path=tmp_path,
            on_approval=approval_cb,
        )

        gated_approval = mock_client.on_approval_request
        assert gated_approval is not None

        # Test gated_approval with DENY, ALLOW, ASK (lines 237-250)
        with patch.object(RuleEngine, "evaluate") as mock_eval:
            # 1. Policy DENY
            mock_eval.return_value = (PolicyAction.DENY, "forbidden")
            ev_deny = ApprovalRequestEvent(request_id=1, call_id="c1", tool_name="bad", args={}, session_id="s")
            assert await gated_approval(ev_deny) is False

            # 2. Policy ALLOW
            mock_eval.return_value = (PolicyAction.ALLOW, "safe")
            ev_allow = ApprovalRequestEvent(request_id=2, call_id="c2", tool_name="safe", args={}, session_id="s")
            assert await gated_approval(ev_allow) is True

            # 3. Policy ASK delegates to on_approval
            mock_eval.return_value = (PolicyAction.ASK, "check user")
            ev_ask = ApprovalRequestEvent(request_id=3, call_id="c3", tool_name="ask", args={}, session_id="s")
            assert await gated_approval(ev_ask) is True
            approval_cb.assert_awaited_with(ev_ask)

        # 4. Policy ASK with on_approval=None returns False (line 250)
        await provider._run_acp_jsonrpc(
            cmd=["custom-agent"],
            session_id="s",
            prompt="test",
            cwd=str(tmp_path),
            on_approval=None,
        )
        gated_no_approval = mock_client.on_approval_request
        with patch.object(RuleEngine, "evaluate", return_value=(PolicyAction.ASK, "check")):
            assert await gated_no_approval(ev_ask) is False



