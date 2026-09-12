"""Model Context Protocol (MCP) Client Manager for Maulness.

Discovers external MCP servers (stdio and HTTP/SSE), executes handshakes,
translates external tool schemas into OpenAI/Antigravity function definitions,
and transparently proxies tool calls between LLM agents and MCP servers.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("maulness.core.mcp")


class MCPServerConnection:
    """Connection to a single MCP server (stdio subprocess or HTTP/SSE)."""

    def __init__(
        self, name: str, config: dict[str, Any], workspace_path: Optional[Path] = None
    ):
        self.name = name
        self.config = config
        self.workspace_path = workspace_path or Path.cwd()
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: list[dict[str, Any]] = []
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._is_initialized = False

    async def connect(self) -> bool:
        """Spawn process or test connection and perform protocol handshake."""
        cmd = self.config.get("command")
        url = self.config.get("url")

        if url:
            # HTTP/SSE MCP Server
            return await self._init_http(url)

        if not cmd:
            logger.warning(
                "MCP server '%s' has neither 'command' nor 'url'.", self.name
            )
            return False

        # Stdio Subprocess MCP Server
        args = self.config.get("args", [])
        env = os.environ.copy()
        if "env" in self.config:
            env.update({k: str(v) for k, v in self.config["env"].items()})

        try:
            self.process = await asyncio.create_subprocess_exec(
                cmd,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path),
                env=env,
            )
            self._reader_task = asyncio.create_task(self._read_stdio_loop())

            # 1. Initialize
            init_res = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "maulness", "version": "0.1.0"},
                },
            )
            if not init_res:
                logger.warning(
                    "MCP server '%s' failed initialization handshake.", self.name
                )
                return False

            # 2. Initialized notification
            await self._send_notification("notifications/initialized", {})

            # 3. List tools
            tools_res = await self._send_request("tools/list", {})
            if tools_res and "result" in tools_res and "tools" in tools_res["result"]:
                self.tools = tools_res["result"]["tools"]
                logger.info(
                    "MCP server '%s' initialized with %d tools.",
                    self.name,
                    len(self.tools),
                )
            self._is_initialized = True
            return True
        except Exception as e:
            logger.warning("Failed starting MCP server '%s': %s", self.name, e)
            await self.close()
            return False

    async def _init_http(self, url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    self.tools = data.get("result", {}).get("tools", [])
                    self._is_initialized = True
                    return True
        except Exception as e:
            logger.debug("HTTP MCP server '%s' connection failed: %s", self.name, e)
        return False

    async def _send_request(
        self, method: str, params: dict[str, Any], timeout: float = 20.0
    ) -> Optional[dict[str, Any]]:
        if not self.process or not self.process.stdin:
            return None

        self._request_id += 1
        req_id = self._request_id
        req_payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        fut = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = fut

        try:
            line = json.dumps(req_payload) + "\n"
            self.process.stdin.write(line.encode("utf-8"))
            await self.process.stdin.drain()
            return await asyncio.wait_for(fut, timeout=timeout)
        except Exception as e:
            logger.debug("MCP request '%s' (id=%d) error: %s", method, req_id, e)
            return None
        finally:
            self._pending_requests.pop(req_id, None)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            return
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            line = json.dumps(payload) + "\n"
            self.process.stdin.write(line.encode("utf-8"))
            await self.process.stdin.drain()
        except Exception:
            pass

    async def _read_stdio_loop(self) -> None:
        """Continuously read newline-delimited JSON-RPC responses from the MCP server stdout."""
        if not self.process or not self.process.stdout:
            return
        try:
            while True:
                line_bytes = await self.process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    req_id = data.get("id")
                    if req_id is not None and req_id in self._pending_requests:
                        fut = self._pending_requests[req_id]
                        if not fut.done():
                            fut.set_result(data)
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Error in MCP read loop for '%s': %s", self.name, e)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on this MCP server and format the response."""
        url = self.config.get("url")
        if url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": tool_name, "arguments": arguments},
                        },
                    )
                    data = resp.json()
                    return self._format_tool_result(data)
            except Exception as e:
                return f"Error executing HTTP MCP tool '{tool_name}': {e}"

        res = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=45.0,
        )
        if not res:
            return f"Error: MCP server '{self.name}' timed out or failed to execute '{tool_name}'."
        return self._format_tool_result(res)

    def _format_tool_result(self, response: dict[str, Any]) -> str:
        if "error" in response:
            err = response["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return f"MCP Tool Error: {msg}"

        result = response.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(json.dumps(item))
                else:
                    parts.append(str(item))
            output = "\n".join(parts).strip()
            return output or "(MCP tool completed with empty output)"
        elif isinstance(content, str):
            return content.strip()
        return json.dumps(result)

    async def close(self) -> None:
        """Terminate process and close stream resources."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None
        self._is_initialized = False


class MCPClientManager:
    """Manages active MCP servers and translates tools into function schemas."""

    _instance: Optional["MCPClientManager"] = None

    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self.tool_to_server: dict[str, tuple[MCPServerConnection, str]] = {}
        self.last_workspace: Optional[Path] = None

    @classmethod
    def get_instance(cls) -> "MCPClientManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def discover_config(self, workspace_path: Optional[Path] = None) -> dict[str, Any]:
        """Locate and merge MCP configs across workspace and global config."""
        configs: dict[str, Any] = {}
        ws = Path(workspace_path or Path.cwd()).resolve()

        search_locations = [
            Path.home() / ".config" / "maulness" / "mcp.json",
            ws / ".mcp.json",
            ws / ".agents" / "mcp.json",
            ws / ".maulness" / "mcp.json",
        ]

        for p in search_locations:
            if p.exists() and p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    servers = data.get("mcpServers", {})
                    configs.update(servers)
                except Exception as e:
                    logger.warning("Failed parsing MCP config at '%s': %s", p, e)

        return configs

    async def initialize_servers(self, workspace_path: Optional[Path] = None) -> None:
        """Discover and connect to all declared MCP servers."""
        ws = Path(workspace_path or Path.cwd()).resolve()
        # Only cache when an explicit workspace_path is provided; otherwise always
        # re-discover to avoid cross-task server leakage in daemon mode.
        if workspace_path is not None and self.last_workspace == ws and self.servers:
            return

        await self.shutdown()
        self.last_workspace = ws

        server_configs = self.discover_config(ws)
        for s_name, s_cfg in server_configs.items():
            conn = MCPServerConnection(s_name, s_cfg, workspace_path=ws)
            ok = await conn.connect()
            if ok:
                self.servers[s_name] = conn
                for t in conn.tools:
                    raw_name = t.get("name", "")
                    namespaced_name = f"mcp__{s_name}__{raw_name}"
                    s_name_underscore = s_name.replace("-", "_")
                    s_name_hyphen = s_name.replace("_", "-")

                    # Allow both namespaced (with hyphens & underscores) and bare name
                    self.tool_to_server[namespaced_name] = (conn, raw_name)
                    self.tool_to_server[f"mcp__{s_name_underscore}__{raw_name}"] = (
                        conn,
                        raw_name,
                    )
                    self.tool_to_server[f"mcp__{s_name_hyphen}__{raw_name}"] = (
                        conn,
                        raw_name,
                    )
                    self.tool_to_server[f"{s_name}__{raw_name}"] = (conn, raw_name)
                    self.tool_to_server[f"{s_name_underscore}__{raw_name}"] = (
                        conn,
                        raw_name,
                    )
                    self.tool_to_server[f"{s_name_hyphen}__{raw_name}"] = (
                        conn,
                        raw_name,
                    )

                    if raw_name not in self.tool_to_server:
                        self.tool_to_server[raw_name] = (conn, raw_name)

    async def get_tool_definitions(
        self, workspace_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Return registered MCP tools as OpenAI/Antigravity function schemas."""
        await self.initialize_servers(workspace_path)
        tool_defs: list[dict[str, Any]] = []

        seen_names = set()
        for s_name, conn in self.servers.items():
            for t in conn.tools:
                raw_name = t.get("name", "")
                namespaced_name = f"mcp__{s_name}__{raw_name}"
                desc = t.get(
                    "description", f"MCP tool '{raw_name}' from server '{s_name}'"
                )
                schema = t.get("inputSchema", {"type": "object", "properties": {}})

                # Register namespaced tool
                tool_defs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": namespaced_name,
                            "description": desc,
                            "parameters": schema,
                        },
                    }
                )

                # Register bare tool if not colliding
                if raw_name not in seen_names:
                    seen_names.add(raw_name)
                    tool_defs.append(
                        {
                            "type": "function",
                            "function": {
                                "name": raw_name,
                                "description": desc,
                                "parameters": schema,
                            },
                        }
                    )

        return tool_defs

    def _resolve_tool(self, name: str) -> Optional[tuple[MCPServerConnection, str]]:
        """Resolve a tool name to (MCPServerConnection, original_name) handling aliases, hyphens/underscores, and namespaces."""
        # 1. Direct match
        if name in self.tool_to_server:
            return self.tool_to_server[name]

        # 2. Hyphen / Underscore normalization
        norm_name = name.replace("-", "_")
        for registered_name, mapping in self.tool_to_server.items():
            if registered_name.replace("-", "_") == norm_name:
                return mapping

        # 3. Namespace parsing for mcp__<server>__<tool> or <server>__<tool>
        if "__" in name:
            parts = name.split("__")
            candidate_server = parts[-2].replace("-", "_")
            candidate_tool = parts[-1]
            for s_name, conn in self.servers.items():
                if s_name.replace("-", "_") == candidate_server:
                    for t in conn.tools:
                        raw_tname = t.get("name", "")
                        if raw_tname == candidate_tool or raw_tname.replace(
                            "-", "_"
                        ) == candidate_tool.replace("-", "_"):
                            return (conn, raw_tname)

            # If server match failed, check candidate_tool directly
            if candidate_tool in self.tool_to_server:
                return self.tool_to_server[candidate_tool]

            cand_norm = candidate_tool.replace("-", "_")
            for registered_name, mapping in self.tool_to_server.items():
                if registered_name.replace("-", "_") == cand_norm:
                    return mapping

        return None

    def is_mcp_tool(self, name: str) -> bool:
        """Check if tool name corresponds to an active MCP tool."""
        return self._resolve_tool(name) is not None

    async def call_tool(
        self, name: str, args: dict[str, Any], workspace_path: Optional[Path] = None
    ) -> str:
        """Route tool call to appropriate MCP server."""
        resolved = self._resolve_tool(name)
        if not resolved:
            return f"Error: MCP tool '{name}' is not registered."

        conn, original_name = resolved
        return await conn.call_tool(original_name, args)

    async def shutdown(self) -> None:
        """Close all active MCP server connections."""
        for conn in self.servers.values():
            await conn.close()
        self.servers.clear()
        self.tool_to_server.clear()
        self.last_workspace = None
