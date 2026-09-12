import json
import sys
from pathlib import Path

import pytest

from maulness.core.mcp_client import MCPClientManager, MCPServerConnection
from maulness.core.tools import get_effective_tool_definitions


def test_discover_config_workspace(tmp_path: Path):
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mock_server": {
                        "command": "python",
                        "args": ["-m", "mock_server"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPClientManager()
    configs = manager.discover_config(workspace_path=tmp_path)
    assert "mock_server" in configs
    assert configs["mock_server"]["command"] == "python"


@pytest.mark.asyncio
async def test_mcp_stdio_server_lifecycle(tmp_path: Path):
    # A tiny inline Python script acting as an MCP stdio server
    server_script = (
        "import sys, json\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line: break\n"
        "    req = json.loads(line)\n"
        "    method = req.get('method')\n"
        "    req_id = req.get('id')\n"
        "    if method == 'initialize':\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'protocolVersion': '2024-11-05', 'capabilities': {}, 'serverInfo': {'name': 'mock-srv'}}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
        "    elif method == 'tools/list':\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': [{'name': 'echo', 'description': 'Echo input', 'inputSchema': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}}]}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
        "    elif method == 'tools/call':\n"
        "        args = req.get('params', {}).get('arguments', {})\n"
        "        msg = args.get('msg', 'hello')\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'content': [{'type': 'text', 'text': f'Echoed: {msg}'}]}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
    )

    config = {
        "command": sys.executable,
        "args": ["-c", server_script],
    }

    conn = MCPServerConnection(name="test_echo", config=config, workspace_path=tmp_path)
    connected = await conn.connect()
    assert connected is True
    assert len(conn.tools) == 1
    assert conn.tools[0]["name"] == "echo"

    # Call tool
    output = await conn.call_tool("echo", {"msg": "Maulness testing"})
    assert "Echoed: Maulness testing" in output

    await conn.close()
    assert conn.process is None


@pytest.mark.asyncio
async def test_mcp_client_manager_integration(tmp_path: Path):
    server_script = (
        "import sys, json\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line: break\n"
        "    req = json.loads(line)\n"
        "    method = req.get('method')\n"
        "    req_id = req.get('id')\n"
        "    if method == 'initialize':\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'protocolVersion': '2024-11-05'}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
        "    elif method == 'tools/list':\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'tools': [{'name': 'add', 'description': 'Add numbers', 'inputSchema': {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}}}]}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
        "    elif method == 'tools/call':\n"
        "        args = req.get('params', {}).get('arguments', {})\n"
        "        res_val = args.get('a', 0) + args.get('b', 0)\n"
        "        res = {'jsonrpc': '2.0', 'id': req_id, 'result': {'content': [{'type': 'text', 'text': str(res_val)}]}}\n"
        "        sys.stdout.write(json.dumps(res) + '\\n'); sys.stdout.flush()\n"
    )

    mcp_config = tmp_path / ".mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "calc": {
                        "command": sys.executable,
                        "args": ["-c", server_script],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPClientManager.get_instance()
    await manager.shutdown()  # clean slate

    defs = await manager.get_tool_definitions(workspace_path=tmp_path)
    tool_names = [d["function"]["name"] for d in defs]
    assert "mcp__calc__add" in tool_names
    assert "add" in tool_names

    assert manager.is_mcp_tool("add") is True
    assert manager.is_mcp_tool("mcp__calc__add") is True

    # Call via manager
    result = await manager.call_tool("add", {"a": 40, "b": 2}, workspace_path=tmp_path)
    assert result == "42"

    # Test tools.py get_effective_tool_definitions
    effective_defs = await get_effective_tool_definitions(workspace_path=tmp_path)
    eff_names = [d["function"]["name"] for d in effective_defs]
    assert "run_command" in eff_names
    assert "mcp__calc__add" in eff_names

    await manager.shutdown()
