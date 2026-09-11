import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.core.models import ApprovalRequestEvent
from maulness.core.rules import PolicyAction, RuleEngine

logger = logging.getLogger("maulness.tools")

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command inside the workspace directory. Use for running tests, checking status, or invoking CLI tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command line to run",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file relative to the workspace or by absolute path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite text content to a file. Requires user approval in HITL mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write into the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories within a given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (defaults to current workspace if empty)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working tree status (git status --short) of the current repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Optional repository path (defaults to current workspace)",
                    }
                },
            },
        },
    },
]


def resolve_path(target_path: str, workspace_path: Optional[Path]) -> Path:
    """Resolve relative or absolute path against the workspace."""
    p = Path(target_path).expanduser()
    if not p.is_absolute():
        base = workspace_path or Path.cwd()
        p = (base / p).resolve()
    return p


_ACTIVE_TURN_CACHE: dict[str, dict[str, str]] = {}
_ACTIVE_TURN_HISTORY: dict[str, list[dict[str, Any]]] = {}


def get_turn_executed_tools(session_id: str) -> list[dict[str, Any]]:
    """Return list of executed tools in the current prompt turn."""
    return list(_ACTIVE_TURN_HISTORY.get(session_id, []))


def clear_turn_tools(session_id: str) -> None:
    """Clear executed tool cache for the given session turn."""
    _ACTIVE_TURN_CACHE.pop(session_id, None)
    _ACTIVE_TURN_HISTORY.pop(session_id, None)


async def execute_tool_call(
    name: str,
    args: dict[str, Any],
    workspace_path: Optional[Path] = None,
    session_id: str = "default",
    on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]] = None,
    yolo: bool = False,
    rule_engine: Optional[RuleEngine] = None,
) -> str:
    """Execute a supported tool action with execution rules and HITL approval gating."""
    cwd = workspace_path or Path.cwd()
    logger.info("Executing tool '%s' with args: %s (cwd: %s)", name, args, cwd)

    # Check turn-scoped tool cache to avoid duplicate execution on provider failover
    canonical_args = json.dumps(args, sort_keys=True)
    cache_key = f"{name}::{canonical_args}"
    if session_id and session_id in _ACTIVE_TURN_CACHE and cache_key in _ACTIVE_TURN_CACHE[session_id]:
        cached_res = _ACTIVE_TURN_CACHE[session_id][cache_key]
        logger.info("Tool '%s' already executed during this turn for session '%s', reusing result", name, session_id)
        return cached_res

    engine = rule_engine or RuleEngine()
    policy, reason = engine.evaluate(name, args, cwd=cwd, yolo=yolo)

    if policy == PolicyAction.DENY:
        logger.warning("Tool execution blocked by security policy: %s", reason)
        return f"Execution blocked by security policy: {reason}"

    # Generic HITL Gate for any tool explicitly configured to ASK
    if policy == PolicyAction.ASK and name not in ("run_command", "write_file"):
        if on_approval and not yolo:
            req = ApprovalRequestEvent(
                request_id=1,
                call_id=f"call_{name}",
                tool_name=name,
                args=args,
                session_id=session_id,
            )
            approved = await on_approval(req)
            if not approved:
                return f"Execution cancelled: User rejected tool '{name}'."

    res = await _execute_tool_action(
        name=name,
        args=args,
        cwd=cwd,
        session_id=session_id,
        policy=policy,
        on_approval=on_approval,
        yolo=yolo,
    )

    if session_id and not res.startswith("Execution cancelled"):
        if session_id not in _ACTIVE_TURN_CACHE:
            _ACTIVE_TURN_CACHE[session_id] = {}
            _ACTIVE_TURN_HISTORY[session_id] = []
        _ACTIVE_TURN_CACHE[session_id][cache_key] = res
        _ACTIVE_TURN_HISTORY[session_id].append({
            "name": name,
            "args": args,
            "result": res,
        })

    return res


async def _execute_tool_action(
    name: str,
    args: dict[str, Any],
    cwd: Path,
    session_id: str,
    policy: PolicyAction,
    on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]],
    yolo: bool,
) -> str:
    """Internal tool action dispatcher."""

    # 1. run_command
    if name == "run_command":
        cmd = args.get("command", "").strip()
        if not cmd:
            return "Error: No command provided."

        # HITL Gate for command execution if policy requires approval
        if policy == PolicyAction.ASK and on_approval and not yolo:
            req = ApprovalRequestEvent(
                request_id=1,
                call_id=f"call_{name}",
                tool_name="run_command",
                args={"command": cmd, "cwd": str(cwd)},
                session_id=session_id,
            )
            approved = await on_approval(req)
            if not approved:
                return f"Execution cancelled: User rejected command '{cmd}'."

        try:
            loop = asyncio.get_running_loop()
            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=60,
                ),
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            return output.strip() or "(Command completed with empty output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60s."
        except Exception as e:
            return f"Error executing command: {e}"

    # 2. read_file
    elif name == "read_file":
        target = resolve_path(args.get("path", ""), cwd)
        try:
            if not target.exists():
                return f"Error: File '{target}' does not exist."
            if target.is_dir():
                return f"Error: '{target}' is a directory, not a file."
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > 50000:
                return content[:50000] + "\n\n... (truncated 50,000 chars)"
            return content
        except Exception as e:
            return f"Error reading file '{target}': {e}"

    # 3. write_file
    elif name == "write_file":
        target = resolve_path(args.get("path", ""), cwd)
        content = args.get("content", "")

        if policy == PolicyAction.ASK and on_approval and not yolo:
            req = ApprovalRequestEvent(
                request_id=1,
                call_id=f"call_{name}",
                tool_name="write_file",
                args={"path": str(target), "bytes": len(content)},
                session_id=session_id,
            )
            approved = await on_approval(req)
            if not approved:
                return f"Write cancelled: User rejected writing to '{target}'."

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{target}'."
        except Exception as e:
            return f"Error writing file '{target}': {e}"

    # 4. list_dir
    elif name == "list_dir":
        raw_path = args.get("path", "")
        target = resolve_path(raw_path, cwd) if raw_path else cwd
        try:
            if not target.exists() or not target.is_dir():
                return f"Error: Directory '{target}' not found."
            entries = []
            for item in sorted(target.iterdir()):
                prefix = "[dir] " if item.is_dir() else "[file] "
                entries.append(f"{prefix}{item.name}")
            return "\n".join(entries) or "(Directory is empty)"
        except Exception as e:
            return f"Error listing directory '{target}': {e}"

    # 5. git_status
    elif name == "git_status":
        raw_repo = args.get("repo_path", "")
        target = resolve_path(raw_repo, cwd) if raw_repo else cwd
        try:
            res = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(target),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return res.stdout.strip() or "Working tree clean (no changes)."
        except Exception as e:
            return f"Error running git status in '{target}': {e}"

    return f"Error: Unknown tool '{name}'."


def format_lean_tool_breadcrumb(name: str, args: dict[str, Any], result: str) -> str:
    """Format a compact, elegant breadcrumb for tool execution in chat streams."""
    clean_res = result.strip()
    summary_arg = ""
    if name == "run_command":
        summary_arg = args.get("command", "")
    elif name in ("read_file", "write_file"):
        summary_arg = args.get("path", "")
    elif name == "list_dir":
        summary_arg = args.get("path", "") or "."
    elif name == "git_status":
        summary_arg = args.get("repo_path", "") or "status"

    label = f"{name}: {summary_arg}" if summary_arg else name

    # If single-line or brief output (<= 120 chars, no newlines)
    if "\n" not in clean_res and len(clean_res) <= 120:
        return f"> **`{label}`** -> `{clean_res}`\n\n"

    # Multiline output: compact preview
    preview_lines = clean_res.splitlines()
    if len(preview_lines) > 8:
        preview = "\n".join(preview_lines[:6]) + f"\n... (+{len(preview_lines)-6} more lines)"
    else:
        preview = clean_res
    return f"> **`{label}`**:\n```\n{preview}\n```\n\n"


_TOOL_SIMULATION_PATTERN = re.compile(
    r"^>\s*(?:⚡\s*)?\*\*`?([a-zA-Z_]+)(?::\s*([^`*\n]+))?`?\*\*",
    re.MULTILINE,
)


def detect_simulated_tool_call(text: str) -> Optional[tuple[str, dict[str, Any]]]:
    """Detect if a model generated markdown simulating a tool call in text instead of function calling."""
    if not text:
        return None
    match = _TOOL_SIMULATION_PATTERN.search(text)
    if not match:
        return None
    tool_name = match.group(1).strip()
    raw_arg = (match.group(2) or "").strip()

    known_tools = {"run_command", "read_file", "write_file", "list_dir", "git_status"}
    if tool_name not in known_tools:
        return None

    args: dict[str, Any] = {}
    if tool_name == "run_command":
        args = {"command": raw_arg}
    elif tool_name in ("list_dir", "read_file"):
        args = {"path": raw_arg}
    elif tool_name == "git_status":
        args = {"repo_path": raw_arg}
    elif tool_name == "write_file":
        args = {"path": raw_arg}

    return tool_name, args


def clean_history_message(content: str) -> str:
    """Sanitize message history to prevent LLMs from mimicking markdown tool breadcrumbs as plain text."""
    if not content:
        return ""
    # Strip multiline tool breadcrumbs: > **...**: or > **...** ➔ followed by code block
    cleaned = re.sub(
        r">\s*(?:⚡\s*)?\*\*`?[a-zA-Z_]+(?::\s*[^`*\n]+)?`?\*\*\s*(?::|->|➔)?\s*\n```[\s\S]*?```\s*",
        "",
        content,
    )
    # Strip single-line tool breadcrumbs: > **`name: arg`** -> `...`\n\n
    cleaned = re.sub(
        r">\s*(?:⚡\s*)?\*\*`?[a-zA-Z_]+(?::\s*[^`*\n]+)?`?\*\*\s*(?:->|➔)\s*`?[^\n]+`?\s*",
        "",
        cleaned,
    )
    # Strip legacy "⚡ **Tool Result (`name`):**\n```...```"
    cleaned = re.sub(
        r"⚡\s*\*\*Tool Result\s*\(`[a-zA-Z_]+`\):\*\*\s*\n```[\s\S]*?```\s*",
        "",
        cleaned,
    )
    cleaned = cleaned.strip()
    if not cleaned:
        first_line = content.strip().splitlines()[0]
        sanitized = re.sub(r"[>⚡*`➔:]", "", first_line).strip()
        return f"[{sanitized}]" if sanitized else "[Tool execution]"
    return cleaned

