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
