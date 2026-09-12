import asyncio
from collections import deque
import fnmatch
import hashlib
import html
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import httpx

from maulness.core.indexing.repo_map import (
    find_symbol as query_symbol,
    get_outline as extract_outline,
)
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
            "description": "Read the text contents of a file relative to the workspace or by absolute path. Optional start_line and end_line allow reading specific slices with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed starting line number",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed ending line number (inclusive)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_file_content",
            "description": "Surgically replace an exact target block of text within a file. Preferred over rewriting the whole file to prevent regressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to modify",
                    },
                    "target_content": {
                        "type": "string",
                        "description": "The exact string or lines of code to replace (must match exactly)",
                    },
                    "replacement_content": {
                        "type": "string",
                        "description": "The replacement string or lines of code",
                    },
                    "allow_multiple": {
                        "type": "boolean",
                        "description": "Whether to replace multiple occurrences if found (defaults to false)",
                    },
                },
                "required": ["path", "target_content", "replacement_content"],
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
            "name": "search_files",
            "description": "Search for a regex or string pattern across workspace files using ripgrep or fallback search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory or file path to search within (defaults to workspace root)",
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": "Optional glob filter (e.g. '*.py' or '!vendor/*')",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Whether search is case-insensitive (defaults to true)",
                    },
                },
                "required": ["pattern"],
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
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": "Apply a standard unified diff patch to a target file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to patch",
                    },
                    "patch": {
                        "type": "string",
                        "description": "Unified diff patch content (e.g. diff starting with @@ or unified chunk lines)",
                    },
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_outline",
            "description": "Inspect the structural outline of a source file (classes, methods, functions, docstrings) without full implementation bodies (~100 tokens vs thousands).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file to inspect",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Search for symbol definitions (classes, functions, methods, types) across the workspace using AST indexing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name or substring of the symbol to find",
                    },
                    "kind": {
                        "type": "string",
                        "description": "Optional symbol kind: 'class', 'function', 'method', 'type', or 'all'",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional file path or substring to filter symbols",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date documentation, API signatures, error explanations, or library usage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_doc_markdown",
            "description": "Fetch web documentation or an external web page and convert it to clean, readable Markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the documentation or page to fetch",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters of markdown to return (default 10000)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load the full detailed playbook instructions for an available skill (progressive disclosure).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the skill to load (from available skills)",
                    },
                },
                "required": ["name"],
            },
        },
    },
]


def resolve_path(target_path: str, workspace_path: Optional[Path]) -> Path:
    """Resolve relative or absolute path against the workspace."""
    p = Path(target_path).expanduser()
    if p.is_absolute():
        return p

    base = (workspace_path or Path.cwd()).resolve()
    candidate = (base / p).resolve()
    if candidate.exists():
        return candidate

    # Resilient resolution for overlapping path queries (e.g. cwd=/home/.../www, target=www/personal)
    if p.parts and p.parts[0] == base.name:
        sub_candidate = (base / Path(*p.parts[1:])).resolve()
        if sub_candidate.exists():
            return sub_candidate

    # Check relative to base.parent (e.g. target includes base folder name)
    parent_candidate = (base.parent / p).resolve()
    if parent_candidate.exists():
        return parent_candidate

    # Check relative to home
    home_candidate = (Path.home() / p).resolve()
    if home_candidate.exists():
        return home_candidate

    return candidate


def truncate_observation(
    output: str,
    max_lines: int = 300,
    max_chars: int = 15000,
    head_lines: int = 100,
    tail_lines: int = 100,
) -> str:
    """Truncate verbose command or tool outputs preserving informative head and tail context."""
    if not output:
        return output

    lines = output.splitlines()
    if len(lines) > max_lines:
        omitted = len(lines) - (head_lines + tail_lines)
        if omitted > 10:
            output = (
                "\n".join(lines[:head_lines])
                + f"\n\n[... {omitted} lines omitted for brevity ...]\n\n"
                + "\n".join(lines[-tail_lines:])
            )

    if len(output) > max_chars:
        head_chars = max_chars // 2 - 200
        tail_chars = max_chars // 2 - 200
        omitted_chars = len(output) - (head_chars + tail_chars)
        output = (
            output[:head_chars]
            + f"\n\n[... {omitted_chars} characters omitted for brevity ...]\n\n"
            + output[-tail_chars:]
        )

    return output


def tombstone_tool_output(name: str, args: dict[str, Any], result: str) -> str:
    """Produce a concise single-line tombstone summary for an older tool result."""
    clean_res = result.strip()
    lines = clean_res.splitlines()
    if len(clean_res) <= 200 and len(lines) <= 4:
        return clean_res

    arg_summary = ""
    if name == "run_command":
        cmd = args.get("command", "")
        arg_summary = f"`{cmd[:60]}`" if len(cmd) > 60 else f"`{cmd}`"
    elif name in ("read_file", "write_file", "replace_file_content"):
        arg_summary = f"`{args.get('path', '')}`"
    elif name == "search_files":
        arg_summary = f"pattern: `{args.get('pattern', '')}`"
    elif name == "list_dir":
        arg_summary = f"`{args.get('path', '.')}`"
    elif name == "git_status":
        arg_summary = f"`{args.get('repo_path', 'status')}`"

    summary_part = f" ({arg_summary})" if arg_summary else ""
    return (
        f"[Tool result for '{name}'{summary_part} compacted: "
        f"{len(lines)} lines / {len(clean_res)} characters originally returned]"
    )


class ActionLoopDetector:
    """Sliding-window loop and action thrashing detector to protect long-running agent sessions."""

    def __init__(self, window_size: int = 6, repetition_threshold: int = 3):
        self.window_size = window_size
        self.repetition_threshold = repetition_threshold
        # session_id -> deque of (tool_name, args_hash, is_modifying)
        self._history: dict[str, deque[tuple[str, str, bool]]] = {}

    def _hash_args(self, args: dict[str, Any]) -> str:
        try:
            canonical = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            canonical = str(args)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _is_modifying(self, name: str) -> bool:
        return name in ("write_file", "replace_file_content")

    def record_and_check(
        self, session_id: str, name: str, args: dict[str, Any]
    ) -> tuple[bool, str]:
        """Record an action and return (is_loop_detected, intervention_message)."""
        if not session_id:
            return False, ""

        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=self.window_size)

        win = self._history[session_id]
        args_hash = self._hash_args(args)
        is_mod = self._is_modifying(name)

        # Count occurrences of exact same (name, args_hash) in current sliding window
        identical_count = sum(1 for (n, h, _) in win if n == name and h == args_hash)

        if identical_count >= self.repetition_threshold - 1:
            recent_modifying = any(m for (_, _, m) in win)
            # If this action is modifying or if no modifying actions occurred between repetitions:
            if not recent_modifying or is_mod:
                intervention = (
                    f"[LOOP INTERVENTION] Tool '{name}' has been executed with identical arguments "
                    f"{identical_count + 1} times recently without progress. "
                    "Further identical calls are blocked to prevent an infinite loop. "
                    "Analyze why the prior attempts did not yield the expected result, "
                    "re-read the previous output, or adopt an alternative approach."
                )
                logger.warning(
                    "[%s] Loop detector tripped on tool '%s' (hash: %s)",
                    session_id,
                    name,
                    args_hash,
                )
                return True, intervention

        win.append((name, args_hash, is_mod))
        return False, ""

    def clear(self, session_id: Optional[str] = None) -> None:
        if session_id:
            self._history.pop(session_id, None)
        else:
            self._history.clear()


_LOOP_DETECTOR = ActionLoopDetector()


def get_loop_detector() -> ActionLoopDetector:
    """Return the global ActionLoopDetector instance."""
    return _LOOP_DETECTOR


_ACTIVE_TURN_CACHE: dict[str, dict[str, str]] = {}
_ACTIVE_TURN_HISTORY: dict[str, list[dict[str, Any]]] = {}


def get_turn_executed_tools(session_id: str) -> list[dict[str, Any]]:
    """Return list of executed tools in the current prompt turn."""
    return list(_ACTIVE_TURN_HISTORY.get(session_id, []))


def clear_turn_tools(session_id: str) -> None:
    """Clear executed tool cache for the given session turn."""
    _ACTIVE_TURN_CACHE.pop(session_id, None)
    _ACTIVE_TURN_HISTORY.pop(session_id, None)


def build_sandboxed_command(
    cmd: str,
    cwd: Path,
    sandbox_mode: Optional[str] = None,
) -> tuple[list[str] | str, bool]:
    """Wrap shell command in unprivileged Bubblewrap (bwrap) sandbox if available.

    Returns:
        (command_or_args, is_shell)
    """
    from maulness.config import config

    mode = (sandbox_mode or getattr(config, "sandbox_mode", "auto")).lower().strip()
    if mode in ("none", "false", "0", "disabled"):
        return cmd, True

    bwrap_path = shutil.which("bwrap")
    if not bwrap_path:
        if mode == "bwrap":
            logger.warning(
                "bwrap requested but binary not found; falling back to direct host execution"
            )
        return cmd, True

    workspace = cwd.resolve()
    home = Path.home()
    ssh_path = home / ".ssh"
    gnupg_path = home / ".gnupg"

    bwrap_args = [
        bwrap_path,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
    ]

    # Mask sensitive credentials
    if ssh_path.exists():
        bwrap_args.extend(["--tmpfs", str(ssh_path)])
    if gnupg_path.exists():
        bwrap_args.extend(["--tmpfs", str(gnupg_path)])

    # Bind workspace read-write and set working directory & context
    bwrap_args.extend(
        [
            "--bind",
            str(workspace),
            str(workspace),
            "--chdir",
            str(workspace),
            "--unshare-all",
            "--share-net",
            "--setenv",
            "HOME",
            str(workspace),
            "--",
            "/bin/bash",
            "-c",
            cmd,
        ]
    )

    return bwrap_args, False


async def execute_tool_call(
    name: str,
    args: dict[str, Any],
    workspace_path: Optional[Path] = None,
    session_id: str = "default",
    on_approval: Optional[
        Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]
    ] = None,
    yolo: bool = False,
    rule_engine: Optional[RuleEngine] = None,
    sandbox_mode: Optional[str] = None,
) -> str:
    """Execute a supported tool action with execution rules and HITL approval gating."""
    cwd = workspace_path or Path.cwd()
    logger.info("Executing tool '%s' with args: %s (cwd: %s)", name, args, cwd)

    # Check turn-scoped tool cache to avoid duplicate execution on provider failover
    canonical_args = json.dumps(args, sort_keys=True)
    cache_key = f"{name}::{canonical_args}"
    if (
        session_id
        and session_id in _ACTIVE_TURN_CACHE
        and cache_key in _ACTIVE_TURN_CACHE[session_id]
    ):
        cached_res = _ACTIVE_TURN_CACHE[session_id][cache_key]
        logger.info(
            "Tool '%s' already executed during this turn for session '%s', reusing result",
            name,
            session_id,
        )
        return cached_res

    engine = rule_engine or RuleEngine()
    policy, reason = engine.evaluate(name, args, cwd=cwd, yolo=yolo)

    if policy == PolicyAction.DENY:
        logger.warning("Tool execution blocked by security policy: %s", reason)
        return f"Execution blocked by security policy: {reason}"

    # Anti-thrashing loop check
    is_loop, intervention = _LOOP_DETECTOR.record_and_check(session_id, name, args)
    if is_loop:
        return intervention

    # Generic HITL Gate for any tool explicitly configured to ASK
    if policy == PolicyAction.ASK and name not in (
        "run_command",
        "write_file",
        "replace_file_content",
    ):
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
        sandbox_mode=sandbox_mode,
    )

    if session_id and not res.startswith("Execution cancelled"):
        if session_id not in _ACTIVE_TURN_CACHE:
            _ACTIVE_TURN_CACHE[session_id] = {}
            _ACTIVE_TURN_HISTORY[session_id] = []
        _ACTIVE_TURN_CACHE[session_id][cache_key] = res
        _ACTIVE_TURN_HISTORY[session_id].append(
            {
                "name": name,
                "args": args,
                "result": res,
            }
        )

    return res


def _execute_replace_file_content(
    target: Path,
    target_content: str,
    replacement_content: str,
    allow_multiple: bool = False,
) -> str:
    """Execute precision chunk replacement within a target file with fuzzy fallback."""
    if not target.exists():
        return f"Error: File '{target}' does not exist."
    if target.is_dir():
        return f"Error: '{target}' is a directory, not a file."
    if not target_content:
        return "Error: target_content must not be empty."

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file '{target}': {e}"

    count = content.count(target_content)

    # If exact count is 0, attempt indentation-agnostic fuzzy matching
    if count == 0:
        lines = content.splitlines(keepends=True)
        target_lines = [tl.strip() for tl in target_content.splitlines() if tl.strip()]
        if target_lines:
            matched_start = -1
            matched_len = -1
            for i in range(len(lines)):
                match = True
                curr_t_idx = 0
                for j in range(i, len(lines)):
                    if not lines[j].strip():
                        continue
                    if lines[j].strip() == target_lines[curr_t_idx]:
                        curr_t_idx += 1
                        if curr_t_idx == len(target_lines):
                            matched_start = i
                            matched_len = j - i + 1
                            break
                    else:
                        match = False
                        break
                if match and matched_start != -1:
                    break

            if matched_start != -1:
                orig_block = "".join(lines[matched_start : matched_start + matched_len])
                new_content = (
                    content[: content.find(orig_block)]
                    + replacement_content
                    + content[content.find(orig_block) + len(orig_block) :]
                )
                try:
                    target.write_text(new_content, encoding="utf-8")
                    lines_add = len(replacement_content.splitlines())
                    lines_sub = matched_len
                    return (
                        f"Successfully replaced block in '{target}' via fuzzy matching. "
                        f"(+{lines_add} / -{lines_sub} lines)"
                    )
                except Exception as e:
                    return f"Error writing file '{target}': {e}"

        return (
            f"Error: target_content not found in '{target}'. "
            "Ensure matching including surrounding lines or try applying a unified diff via patch_file."
        )

    if count > 1 and not allow_multiple:
        return (
            f"Error: target_content found {count} times in '{target}'. "
            "Provide more surrounding lines of context to uniquely identify the block to replace, "
            "or set allow_multiple=true if you intend to replace all occurrences."
        )

    new_content = content.replace(
        target_content, replacement_content, -1 if allow_multiple else 1
    )
    try:
        target.write_text(new_content, encoding="utf-8")
        replaced_count = count if allow_multiple else 1
        lines_add = len(replacement_content.splitlines())
        lines_sub = len(target_content.splitlines())
        return (
            f"Successfully replaced {replaced_count} occurrence(s) in '{target}'. "
            f"(+{lines_add} / -{lines_sub} lines)"
        )
    except Exception as e:
        return f"Error writing file '{target}': {e}"


def _execute_patch_file(target: Path, patch_text: str) -> str:
    """Apply a unified diff patch to a target file."""
    if not target.exists():
        return f"Error: File '{target}' does not exist."
    if target.is_dir():
        return f"Error: '{target}' is a directory, not a file."
    if not patch_text.strip():
        return "Error: patch must not be empty."

    git_bin = shutil.which("git")
    patch_bin = shutil.which("patch")

    # Try applying via git apply or patch command
    try:
        target.read_text(encoding="utf-8", errors="replace")
        patch_input = patch_text.strip()
        # If header missing, synthesize a minimal unified diff header
        if not patch_input.startswith("---"):
            header = f"--- a/{target.name}\n+++ b/{target.name}\n"
            patch_input = header + patch_input

        if git_bin:
            res = subprocess.run(
                [git_bin, "apply", "--recount", "--whitespace=fix", "-"],
                input=patch_input,
                cwd=str(target.parent),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                return f"Successfully applied unified patch to '{target}'."

        if patch_bin:
            res = subprocess.run(
                [patch_bin, "-u", str(target)],
                input=patch_text,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                return f"Successfully applied patch to '{target}'."

        # Fallback: simple line replacement if patch contains + and - markers
        return f"Error applying patch to '{target}'. Check patch format."
    except Exception as e:
        return f"Error applying patch to '{target}': {e}"


async def _execute_search_files(
    pattern: str,
    target_dir: Path,
    glob_pattern: Optional[str] = None,
    case_insensitive: bool = True,
    max_results: int = 50,
) -> str:
    """Search for string or regex pattern across workspace files."""
    if not target_dir.exists():
        return f"Error: Search directory '{target_dir}' does not exist."

    rg_path = shutil.which("rg")
    if rg_path:
        cmd = [
            rg_path,
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(max_results),
        ]
        if case_insensitive:
            cmd.append("-i")
        if glob_pattern:
            cmd.extend(["-g", glob_pattern])
        cmd.extend(["-e", pattern, "."])
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True,
                    timeout=15,
                ),
            )
            out = res.stdout.strip()
            if not out:
                return f"No matches found for pattern '{pattern}'."
            lines = out.splitlines()
            if len(lines) >= max_results:
                return (
                    "\n".join(lines[:max_results])
                    + f"\n... (Results capped at {max_results} matches)"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.debug("ripgrep search failed, falling back to python: %s", e)

    matches = []
    regex_flags = re.IGNORECASE if case_insensitive else 0
    try:
        rx = re.compile(pattern, regex_flags)
    except re.error as e:
        return f"Error: Invalid regex pattern '{pattern}': {e}"

    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [
            d
            for d in dirs
            if d not in (".git", ".venv", "__pycache__", "node_modules", ".worktrees")
        ]
        for file in sorted(files):
            if glob_pattern and not fnmatch.fnmatch(file, glob_pattern):
                continue
            file_path = Path(root) / file
            try:
                rel_path = file_path.relative_to(target_dir)
                content = file_path.read_text(encoding="utf-8", errors="replace")
                for line_no, line in enumerate(content.splitlines(), 1):
                    if rx.search(line):
                        matches.append(f"{rel_path}:{line_no}: {line.strip()[:200]}")
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches found for pattern '{pattern}'."
    res = "\n".join(matches)
    if len(matches) >= max_results:
        res += f"\n... (Results capped at {max_results} matches)"
    return res


async def _execute_tool_action(
    name: str,
    args: dict[str, Any],
    cwd: Path,
    session_id: str,
    policy: PolicyAction,
    on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]],
    yolo: bool,
    sandbox_mode: Optional[str] = None,
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
            cmd_target, is_shell = build_sandboxed_command(
                cmd, cwd, sandbox_mode=sandbox_mode
            )

            def _run_with_pgroup():
                p = subprocess.Popen(
                    cmd_target,
                    shell=is_shell,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                    env=os.environ.copy(),
                )
                try:
                    stdout, stderr = p.communicate(timeout=60)
                    return p.returncode, stdout, stderr
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        p.communicate(timeout=2)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    raise

            loop = asyncio.get_running_loop()
            returncode, stdout, stderr = await loop.run_in_executor(
                None, _run_with_pgroup
            )
            output = stdout
            if stderr:
                output += f"\n[stderr]\n{stderr}"
            return truncate_observation(
                output.strip() or "(Command completed with empty output)"
            )
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60s."
        except Exception as e:
            return f"Error executing command: {e}"

    # 2. read_file
    elif name == "read_file":
        target = resolve_path(args.get("path", ""), cwd)
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        try:
            if not target.exists():
                return f"Error: File '{target}' does not exist."
            if target.is_dir():
                return f"Error: '{target}' is a directory, not a file."
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            if start_line is not None or end_line is not None:
                total_lines = len(lines)
                s = max(1, int(start_line)) if start_line is not None else 1
                e = (
                    min(total_lines, int(end_line))
                    if end_line is not None
                    else total_lines
                )
                if s > total_lines:
                    return f"Error: start_line {s} exceeds total lines ({total_lines}) in '{target}'."
                if s > e:
                    return f"Error: start_line {s} is greater than end_line {e}."
                sliced = lines[s - 1 : e]
                formatted = [f"L{s + idx}: {line}" for idx, line in enumerate(sliced)]
                header = f"[{target.name} lines {s}-{e} of {total_lines}]\n"
                return header + "\n".join(formatted)

            if len(content) > 50000:
                return content[:50000] + "\n\n... (truncated 50,000 chars)"
            return content
        except Exception as e:
            return f"Error reading file '{target}': {e}"

    # 3. replace_file_content
    elif name == "replace_file_content":
        target = resolve_path(args.get("path", ""), cwd)
        target_content = args.get("target_content", "")
        replacement_content = args.get("replacement_content", "")
        allow_multiple = bool(args.get("allow_multiple", False))

        if policy == PolicyAction.ASK and on_approval and not yolo:
            req = ApprovalRequestEvent(
                request_id=1,
                call_id=f"call_{name}",
                tool_name="replace_file_content",
                args={
                    "path": str(target),
                    "target_len": len(target_content),
                    "replacement_len": len(replacement_content),
                    "allow_multiple": allow_multiple,
                },
                session_id=session_id,
            )
            approved = await on_approval(req)
            if not approved:
                return f"Execution cancelled: User rejected replace in '{target}'."

        return _execute_replace_file_content(
            target, target_content, replacement_content, allow_multiple
        )

    # 4. write_file
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

    # 5. search_files
    elif name == "search_files":
        pattern = args.get("pattern", "").strip()
        if not pattern:
            return "Error: No pattern provided."
        raw_path = args.get("path", "")
        target_dir = resolve_path(raw_path, cwd) if raw_path else cwd
        glob_pat = args.get("glob_pattern")
        case_ins = bool(args.get("case_insensitive", True))
        return await _execute_search_files(
            pattern, target_dir, glob_pattern=glob_pat, case_insensitive=case_ins
        )

    # 6. list_dir
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

    # 7. git_status
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

    # 8. patch_file
    elif name == "patch_file":
        target = resolve_path(args.get("path", ""), cwd)
        patch_text = args.get("patch", "")

        if policy == PolicyAction.ASK and on_approval and not yolo:
            req = ApprovalRequestEvent(
                request_id=1,
                call_id=f"call_{name}",
                tool_name="patch_file",
                args={
                    "path": str(target),
                    "patch_len": len(patch_text),
                },
                session_id=session_id,
            )
            approved = await on_approval(req)
            if not approved:
                return f"Execution cancelled: User rejected patch on '{target}'."

        return _execute_patch_file(target, patch_text)

    # 9. get_outline
    elif name == "get_outline":
        raw_path = args.get("path", "")
        if not raw_path:
            return "Error: Path is required for get_outline."
        target = resolve_path(raw_path, cwd)
        try:
            return extract_outline(cwd, str(target))
        except Exception as e:
            return f"Error extracting outline for '{raw_path}': {e}"

    # 10. find_symbol
    elif name == "find_symbol":
        sym_name = args.get("name", "").strip()
        if not sym_name:
            return "Error: Symbol name is required for find_symbol."
        kind = args.get("kind")
        file_filter = args.get("path")
        try:
            matches = query_symbol(cwd, sym_name, kind=kind, file_filter=file_filter)
            if not matches:
                return f"No symbols matching '{sym_name}' found in workspace."
            lines = [f"Found {len(matches)} matching symbol(s) for '{sym_name}':"]
            for s in matches[:30]:
                doc_str = f' -- "{s.docstring}"' if s.docstring else ""
                lines.append(
                    f"- [{s.kind}] {s.file_path}:L{s.line_number} -> {s.signature}{doc_str}"
                )
            if len(matches) > 30:
                lines.append(f"... ({len(matches) - 30} additional symbols omitted)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error finding symbol '{sym_name}': {e}"

    # 11. web_search
    elif name == "web_search":
        query = args.get("query", "").strip()
        if not query:
            return "Error: Search query is required for web_search."
        max_res = int(args.get("max_results", 5))
        return await _execute_web_search(query, max_res)

    # 12. fetch_doc_markdown
    elif name == "fetch_doc_markdown":
        url = args.get("url", "").strip()
        if not url:
            return "Error: URL is required for fetch_doc_markdown."
        max_chars = int(args.get("max_chars", 10000))
        return await _execute_fetch_doc_markdown(url, max_chars)

    # 13. load_skill
    elif name == "load_skill":
        skill_name = args.get("name", "").strip()
        if not skill_name:
            return "Error: Skill name is required."
        from maulness.core.skills import SkillManager

        manager = SkillManager()
        return manager.get_skill_instruction(skill_name, workspace_path=cwd)

    return f"Error: Unknown tool '{name}'."


async def _execute_web_search(query: str, max_results: int = 5) -> str:
    """Execute web search across DuckDuckGo HTML, GitHub API, or search providers."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": max_results,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("results", [])
                    if results:
                        lines = [f'### Web Search Results for "{query}":']
                        for idx, r in enumerate(results[:max_results], 1):
                            lines.append(
                                f"{idx}. **{r.get('title')}** - {r.get('url')}\n   {r.get('content')}"
                            )
                        return "\n\n".join(lines)
        except Exception as e:
            logger.debug("Tavily search failed: %s", e)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
        async with httpx.AsyncClient(
            headers=headers, timeout=8.0, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                raw_matches = re.findall(
                    r'<a[^>]+class=[\'"]result__snippet[\'"][^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',
                    resp.text,
                    re.DOTALL,
                )
                if not raw_matches:
                    raw_matches = re.findall(
                        r'<a[^>]+class=[\'"]result__url[\'"][^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',
                        resp.text,
                        re.DOTALL,
                    )
                if raw_matches:
                    lines = [f'### Web Search Results for "{query}":']
                    for idx, (link, snippet) in enumerate(raw_matches[:max_results], 1):
                        clean_snip = html.unescape(
                            re.sub(r"<[^>]+>", "", snippet)
                        ).strip()
                        m_uddg = re.search(r"uddg=([^&]+)", link)
                        actual_url = (
                            urllib.parse.unquote(m_uddg.group(1)) if m_uddg else link
                        )
                        lines.append(
                            f"{idx}. **Result {idx}** - {actual_url}\n   {clean_snip}"
                        )
                    return "\n\n".join(lines)
    except Exception as e:
        logger.debug("DuckDuckGo HTML search failed: %s", e)

    try:
        gh_url = f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(query)}&per_page={max_results}"
        async with httpx.AsyncClient(
            headers={"User-Agent": "maulness-agent"}, timeout=6.0
        ) as client:
            gh_resp = await client.get(gh_url)
            if gh_resp.status_code == 200:
                items = gh_resp.json().get("items", [])
                if items:
                    lines = [f'### Web / GitHub Documentation Results for "{query}":']
                    for idx, item in enumerate(items[:max_results], 1):
                        lines.append(
                            f"{idx}. **{item.get('full_name')}** - {item.get('html_url')}\n"
                            f"   {item.get('description') or 'No description'}"
                        )
                    return "\n\n".join(lines)
    except Exception as e:
        logger.debug("GitHub search fallback failed: %s", e)

    return f'Web search results for "{query}": No results found or web search providers currently unreachable.'


async def _execute_fetch_doc_markdown(url: str, max_chars: int = 10000) -> str:
    """Fetch URL and convert HTML to token-dense clean Markdown."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return (
            "Error: Invalid URL scheme. Only http:// and https:// URLs are supported."
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }
    try:
        async with httpx.AsyncClient(
            headers=headers, timeout=15.0, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return f"Error fetching '{url}': HTTP {resp.status_code} ({resp.reason_phrase})"

            content_type = resp.headers.get("content-type", "").lower()
            text = resp.text

            if "text/markdown" in content_type or "text/plain" in content_type:
                if len(text) > max_chars:
                    return (
                        text[:max_chars]
                        + f"\n\n[... content truncated to {max_chars} chars ...]"
                    )
                return text

            clean = re.sub(
                r"<(script|style|nav|footer|header|svg|noscript)[^>]*>.*?</\1>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            title_m = re.search(
                r"<title[^>]*>(.*?)</title>", clean, flags=re.IGNORECASE | re.DOTALL
            )
            title = title_m.group(1).strip() if title_m else ""

            for i in range(6, 0, -1):
                clean = re.sub(
                    rf"<h{i}[^>]*>(.*?)</h{i}>",
                    rf"\n\n{'#' * i} \1\n\n",
                    clean,
                    flags=re.DOTALL | re.IGNORECASE,
                )

            clean = re.sub(
                r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
                r"\n```\n\1\n```\n",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean = re.sub(
                r"<pre[^>]*>(.*?)</pre>",
                r"\n```\n\1\n```\n",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean = re.sub(
                r"<code[^>]*>(.*?)</code>",
                r"`\1`",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean = re.sub(
                r'<a[^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>',
                r"[\2](\1)",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean = re.sub(
                r"<li[^>]*>(.*?)</li>",
                r"\n- \1",
                clean,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean = re.sub(r"<p[^>]*>", r"\n\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"</p>", r"\n\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<br\s*/?>", r"\n", clean, flags=re.IGNORECASE)
            clean = re.sub(r"<[^>]+>", "", clean)
            clean = html.unescape(clean)

            lines = [line.strip() for line in clean.splitlines()]
            result = "\n".join(lines)
            result = re.sub(r"\n{3,}", "\n\n", result).strip()

            if title and not result.startswith("#"):
                result = f"# {title}\n\n" + result

            if len(result) > max_chars:
                result = (
                    result[:max_chars]
                    + f"\n\n[... content truncated to {max_chars} chars ...]"
                )

            return result or "(Fetched document produced empty content)"
    except Exception as e:
        return f"Error fetching document from '{url}': {e}"


def format_lean_tool_breadcrumb(name: str, args: dict[str, Any], result: str) -> str:
    """Format a compact, elegant breadcrumb for tool execution in chat streams."""
    clean_res = result.strip()
    summary_arg = ""
    if name == "run_command":
        summary_arg = args.get("command", "")
    elif name in ("read_file", "write_file", "replace_file_content"):
        summary_arg = args.get("path", "")
    elif name == "search_files":
        pattern = args.get("pattern", "")
        summary_arg = f"'{pattern[:40]}'" if len(pattern) > 40 else f"'{pattern}'"
    elif name == "list_dir":
        summary_arg = args.get("path", "") or "."
    elif name == "git_status":
        summary_arg = args.get("repo_path", "") or "status"
    elif name == "get_outline":
        summary_arg = args.get("path", "")
    elif name == "find_symbol":
        summary_arg = args.get("name", "")
    elif name == "web_search":
        q = args.get("query", "")
        summary_arg = f"'{q[:40]}'" if len(q) > 40 else f"'{q}'"
    elif name == "fetch_doc_markdown":
        summary_arg = args.get("url", "")

    label = f"{name}: {summary_arg}" if summary_arg else name

    # If single-line or brief output (<= 120 chars, no newlines)
    if "\n" not in clean_res and len(clean_res) <= 120:
        return f"> **`{label}`** -> `{clean_res}`\n\n"

    # Multiline output: compact preview
    preview_lines = clean_res.splitlines()
    if len(preview_lines) > 8:
        preview = (
            "\n".join(preview_lines[:6])
            + f"\n... (+{len(preview_lines) - 6} more lines)"
        )
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

    known_tools = {
        "run_command",
        "read_file",
        "write_file",
        "list_dir",
        "git_status",
        "replace_file_content",
        "search_files",
    }
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
    elif tool_name == "replace_file_content":
        args = {"path": raw_arg}
    elif tool_name == "search_files":
        args = {"pattern": raw_arg}

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


def extract_checkpoint_info(text: str) -> tuple[bool, str, str]:
    """Check if model response indicates an in-progress long-horizon task.

    Returns:
        (is_in_progress, checkpoint_body, next_step)
    """
    if not text:
        return False, "", ""

    # Explicit completion markers take precedence
    if re.search(r"\[STATUS:\s*(?:COMPLETE|FINISHED)\]", text, re.IGNORECASE):
        return False, "", ""

    upper = text.upper()

    # Positive signals that a task is still in progress
    explicit_in_progress = (
        "[STATUS: IN_PROGRESS]" in upper
        or "STATUS: IN_PROGRESS" in upper
        or "STATUS: IN PROGRESS" in upper
    )

    has_unchecked_checkboxes = bool(re.search(r"(?:^|\n)\s*[-*]\s*\[\s*\]", text))
    has_next_markers = bool(
        re.search(
            r"\b(?:NEXT\s+STEP|NEXT|CONTINUING\s+WITH|MOVING\s+TO|DIVING\s+INTO)\s*:",
            text,
            re.IGNORECASE,
        )
    )
    has_pending_items = bool(
        re.search(r"->\s*(?:NEXT|PENDING|IN_PROGRESS)", text, re.IGNORECASE)
    )

    is_in_progress = (
        explicit_in_progress
        or has_unchecked_checkboxes
        or has_next_markers
        or has_pending_items
    )

    checkpoint_body = text.strip()
    next_step = "Continuing task execution"

    for line in checkpoint_body.splitlines():
        clean = line.strip()
        clean_no_bullets = re.sub(r"^[-*\d\.]+\s*", "", clean)
        clean_normalized = re.sub(r"[\*`]", "", clean_no_bullets).strip()
        lower = clean_normalized.lower()

        if lower.startswith("next step:"):
            val = clean_normalized[10:].strip()
            if val:
                next_step = val
                is_in_progress = True
                break
        elif lower.startswith("next:"):
            val = clean_normalized[5:].strip()
            if val:
                next_step = val
                is_in_progress = True
                break

    # Second pass: If next_step is still fallback, check for items marked "-> next"
    if next_step == "Continuing task execution":
        for line in checkpoint_body.splitlines():
            clean = line.strip()
            clean_no_bullets = re.sub(r"^[-*\d\.]+\s*", "", clean)
            clean_normalized = re.sub(r"[\*`]", "", clean_no_bullets).strip()
            if "-> next" in clean_normalized.lower():
                next_step = clean_normalized
                is_in_progress = True
                break

    return is_in_progress, checkpoint_body, next_step


def clean_relay_completion_tags(text: str) -> str:
    """Clean internal relay protocol markers from final user output."""
    if not text:
        return ""
    cleaned = re.sub(
        r"\[STATUS:\s*(?:COMPLETE|FINISHED)\]", "", text, flags=re.IGNORECASE
    )
    return cleaned.strip()
