"""Slash command router and context tag expander for Maulness CLI and TUI.

Supports ergonomics features:
- Context tag expansion: @file:<path>, @dir:<path>, @diff, bare @<path>
- Developer slash commands: /undo, /diff, /test, /commit, /skills, /outline, /repo-map, /help
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple

from maulness.core.indexing.repo_map import get_outline, get_repo_map
from maulness.core.skills import SkillManager
from maulness.core.stack import detect_stack
from maulness.core.worktree import WorktreeManager

logger = logging.getLogger(__name__)


def expand_context_tags(prompt: str, workspace_path: Path) -> str:
    """Expand @file:<path>, @dir:<path>, @diff, and bare @<path> in prompt.

    Args:
        prompt: User input prompt string.
        workspace_path: Root workspace path to resolve relative file references.

    Returns:
        Expanded prompt with file/diff/directory contents embedded as markdown blocks.
    """
    if not prompt or "@" not in prompt:
        return prompt

    expanded = prompt
    ws = Path(workspace_path).resolve()

    # 1. Expand @diff
    if "@diff" in expanded:
        try:
            diff_res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff_out = diff_res.stdout.strip()
            if not diff_out:
                status_res = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=str(ws),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                diff_out = status_res.stdout.strip() or "(No uncommitted git changes)"
            diff_block = f"\n[Context from @diff]:\n```diff\n{diff_out}\n```\n"
            expanded = expanded.replace("@diff", diff_block)
        except Exception as e:
            logger.warning("Failed to expand @diff: %s", e)
            expanded = expanded.replace("@diff", f"\n[Context from @diff: error {e}]\n")

    # 2. Expand @file:<path>
    file_pattern = re.compile(r"@file:([^\s]+)")
    for match in file_pattern.finditer(expanded):
        full_match = match.group(0)
        rel_path = match.group(1).strip("'\"")
        target_file = (ws / rel_path).resolve()
        if target_file.exists() and target_file.is_file():
            try:
                content = target_file.read_text(encoding="utf-8", errors="replace")
                ext = target_file.suffix.lstrip(".") or "text"
                block = f"\n[Context from {full_match}]:\n```{ext}\n{content}\n```\n"
                expanded = expanded.replace(full_match, block)
            except Exception as e:
                expanded = expanded.replace(
                    full_match, f"\n[Error reading {rel_path}: {e}]\n"
                )
        else:
            expanded = expanded.replace(full_match, f"\n[File not found: {rel_path}]\n")

    # 3. Expand @dir:<path>
    dir_pattern = re.compile(r"@dir:([^\s]+)")
    for match in dir_pattern.finditer(expanded):
        full_match = match.group(0)
        rel_path = match.group(1).strip("'\"")
        target_dir = (ws / rel_path).resolve()
        if target_dir.exists() and target_dir.is_dir():
            try:
                entries = []
                for p in sorted(target_dir.iterdir()):
                    prefix = "📁 " if p.is_dir() else "📄 "
                    entries.append(f"{prefix}{p.name}")
                listing = "\n".join(entries[:100])
                block = f"\n[Context from {full_match}]:\n```text\n{listing}\n```\n"
                expanded = expanded.replace(full_match, block)
            except Exception as e:
                expanded = expanded.replace(
                    full_match, f"\n[Error listing {rel_path}: {e}]\n"
                )
        else:
            expanded = expanded.replace(
                full_match, f"\n[Directory not found: {rel_path}]\n"
            )

    # 4. Expand bare @<path> (only if path actually exists in workspace to avoid false positives)
    bare_pattern = re.compile(r"(?<![\w/])@([a-zA-Z0-9_\-\./]+)")
    for match in bare_pattern.finditer(prompt):
        full_match = match.group(0)
        rel_path = match.group(1).strip("'\"")
        # Ignore if it was already handled (@file, @dir, @diff)
        if full_match in ("@diff",) or full_match.startswith(("@file:", "@dir:")):
            continue
        candidate = (ws / rel_path).resolve()
        # Verify candidate is inside workspace and exists
        try:
            candidate.relative_to(ws)
        except ValueError:
            continue

        if candidate.exists():
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                    ext = candidate.suffix.lstrip(".") or "text"
                    block = (
                        f"\n[Context from {full_match}]:\n```{ext}\n{content}\n```\n"
                    )
                    expanded = expanded.replace(full_match, block, 1)
                except Exception as e:
                    logger.debug("Failed reading bare @ file: %s", e)
            elif candidate.is_dir():
                try:
                    entries = [
                        f"{'📁 ' if p.is_dir() else '📄 '}{p.name}"
                        for p in sorted(candidate.iterdir())
                    ]
                    listing = "\n".join(entries[:100])
                    block = f"\n[Context from {full_match}]:\n```text\n{listing}\n```\n"
                    expanded = expanded.replace(full_match, block, 1)
                except Exception as e:
                    logger.debug("Failed listing bare @ dir: %s", e)

    return expanded


def handle_slash_command(
    command_str: str,
    workspace_path: Path,
    current_profile: str = "default",
) -> Tuple[bool, str]:
    """Dispatch and execute developer slash commands.

    Args:
        command_str: Full slash command string (e.g. "/undo", "/test -k utils", "/diff").
        workspace_path: Working directory path.
        current_profile: Currently active agent profile.

    Returns:
        (is_handled, result_output_string)
    """
    raw = command_str.strip()
    if not raw.startswith("/"):
        return False, ""

    parts = raw.split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    ws = Path(workspace_path).resolve()

    # /undo
    if cmd == "/undo":
        wt = WorktreeManager()
        if not wt.is_git_repo(ws):
            return True, "Error: Workspace is not a valid Git repository."

        # Check for dirty changes
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status_res.stdout.strip():
            subprocess.run(
                ["git", "restore", "."], cwd=str(ws), capture_output=True, timeout=10
            )
            subprocess.run(
                ["git", "clean", "-fd"], cwd=str(ws), capture_output=True, timeout=10
            )
            return (
                True,
                "Undid uncommitted changes in working tree (`git restore . && git clean -fd`).",
            )

        # If clean, rollback previous milestone or commit
        success = wt.rollback_to_previous_milestone(ws)
        if success:
            return (
                True,
                "Rolled back working tree to previous milestone / commit (`git reset --hard HEAD~1`).",
            )
        return True, "Working tree is clean; no previous commit available to rollback."

    # /diff
    if cmd == "/diff":
        wt = WorktreeManager()
        if not wt.is_git_repo(ws):
            return True, "Error: Workspace is not a valid Git repository."

        diff_res = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_out = diff_res.stdout.strip()
        if not diff_out:
            st_res = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=5,
            )
            diff_out = (
                st_res.stdout.strip() or "(No uncommitted changes in working tree)"
            )
        return True, f"```diff\n{diff_out}\n```"

    # /test
    if cmd == "/test":
        stack = detect_stack(ws)
        test_cmd = stack.test_command
        if not test_cmd:
            return (
                True,
                f"No default test command detected for stack '{stack.name}' ({stack.language}).",
            )

        full_cmd = f"{test_cmd} {arg}".strip() if arg else test_cmd
        try:
            res = subprocess.run(
                full_cmd,
                shell=True,
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = res.stdout or res.stderr or "(No test output)"
            status = (
                "PASSED" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
            )
            return (
                True,
                f"### Test Suite Execution: {status}\nCommand: `{full_cmd}`\n\n```text\n{out}\n```",
            )
        except subprocess.TimeoutExpired:
            return True, f"Test execution timed out after 120s (`{full_cmd}`)."
        except Exception as e:
            return True, f"Failed executing test command '{full_cmd}': {e}"

    # /commit
    if cmd == "/commit":
        wt = WorktreeManager()
        if not wt.is_git_repo(ws):
            return True, "Error: Workspace is not a valid Git repository."

        msg = arg or "chore: update workspace"
        try:
            subprocess.run(
                ["git", "add", "-A"], cwd=str(ws), capture_output=True, timeout=10
            )
            res = subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=str(ws),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                rev = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(ws),
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                return True, f"Committed `{rev}`: {msg}"
            else:
                out = (res.stderr or res.stdout).strip()
                return True, f"Commit not created: {out}"
        except Exception as e:
            return True, f"Error creating commit: {e}"

    # /skills
    if cmd == "/skills":
        sm = SkillManager()
        skills = sm.list_skills(workspace_path=ws)
        if not skills:
            return True, "No skills discovered in workspace, user config, or templates."

        lines = ["### Discovered Skills\n"]
        for name, skill in sorted(skills.items()):
            lines.append(
                f"- **`{name}`**: {skill.description}\n  [dim]Path: {skill.path}[/dim]"
            )
        lines.append(
            "\nUse tool `load_skill(name)` or inspect instructions in the playbook file."
        )
        return True, "\n".join(lines)

    # /outline
    if cmd == "/outline":
        if not arg:
            return True, "Usage: `/outline <file_path>`"
        target = (ws / arg).resolve()
        if not target.exists():
            return True, f"File not found: {arg}"

        outline = get_outline(ws, arg)
        return True, f"### Outline for `{arg}`\n\n```text\n{outline}\n```"

    # /repo-map
    if cmd in ("/repo-map", "/repomap"):
        repo_map = get_repo_map(ws)
        return True, f"### Workspace Repo Map\n\n```text\n{repo_map}\n```"

    # /help
    if cmd in ("/help", "/?"):
        help_text = (
            "### Maulness Slash Command Reference\n\n"
            "| Command | Description |\n"
            "|---|---|\n"
            "| `/undo` | Discard dirty working changes or rollback previous milestone/commit |\n"
            "| `/diff` | Show git diff of current uncommitted changes |\n"
            "| `/test [args]` | Run auto-detected test suite for the workspace stack |\n"
            "| `/commit [msg]` | Stage and commit all changes with a commit message |\n"
            "| `/skills` | List all discovered skills across templates, user, and workspace |\n"
            "| `/outline <file>` | Inspect structural code outline (classes, functions, signatures) |\n"
            "| `/repo-map` | Generate dense PageRank AST repo map of workspace symbols |\n"
            "| `/compact` | Compact active chat history into persistent SQLite memory |\n"
            "| `/pipeline <name> <goal>` | Execute declarative multi-stage pipeline |\n"
            "| `/profile <name>` | Switch active agent persona/profile |\n"
            "| `/yolo` | Toggle autonomous tool execution without approval prompts |\n"
            "| `/worktree` | Toggle isolated Git worktree execution |\n"
            "| `!<command>` | Execute local shell command in workspace directly |\n"
            "| `/help` | Show this command reference table |\n\n"
            "**Context Tags (@):**\n"
            "- `@diff`: Embed current uncommitted git diff into prompt\n"
            "- `@file:<path>` or `@<path>`: Embed entire file contents into prompt\n"
            "- `@dir:<path>`: Embed directory file listing into prompt"
        )
        return True, help_text

    return False, ""
