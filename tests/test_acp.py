import asyncio
import json
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

    profile = Profile(name="builder", command="agy --output-format stream-json")
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

