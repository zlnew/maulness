import fnmatch
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("maulness.rules")


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


DEFAULT_ALLOW_COMMANDS = [
    "pwd",
    "whoami",
    "uname*",
    "which *",
    "ls*",
    "dir*",
    "git status*",
    "git diff*",
    "git log*",
    "git branch*",
    "git show*",
    "git rev-parse*",
    "git describe*",
    "cat *",
    "head *",
    "tail *",
    "grep *",
    "rg *",
    "find *",
    "pytest*",
    "*pytest*",
    "cargo check*",
    "npm test*",
    "uv run *",
]

DEFAULT_DENY_COMMANDS = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf / *",
    "rm -rf ~",
    "rm -rf ~*",
    "sudo *",
    "dd *",
    "mkfs*",
    "*--force*",
    "shutdown*",
    "reboot*",
    "init 0*",
    "init 6*",
    ":(){ :|:& };:",
]

DEFAULT_DENY_PATHS = [
    "**/.env*",
    ".env*",
    "**/id_rsa*",
    "id_rsa*",
    "**/.ssh/**",
    "**/*.pem",
    "**/*.key",
    "/etc/**",
    "/root/**",
]

DEFAULT_ALLOW_PATHS = [
    "**/.worktrees/**",
]

DEFAULT_TOOL_POLICIES: dict[str, PolicyAction] = {
    "view_file": PolicyAction.ALLOW,
    "read_file": PolicyAction.ALLOW,
    "list_dir": PolicyAction.ALLOW,
    "grep_search": PolicyAction.ALLOW,
    "find_by_name": PolicyAction.ALLOW,
    "read_url_content": PolicyAction.ALLOW,
    "search_web": PolicyAction.ALLOW,
    "search_files": PolicyAction.ALLOW,
    "git_status": PolicyAction.ALLOW,
    "write_file": PolicyAction.ASK,
    "write_to_file": PolicyAction.ASK,
    "replace_file_content": PolicyAction.ASK,
    "run_command": PolicyAction.ASK,
}


class CommandRulesConfig(BaseModel):
    deny: list[str] = Field(default_factory=list)
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)


class PathRulesConfig(BaseModel):
    deny: list[str] = Field(default_factory=list)
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)


class ExecutionRulesConfig(BaseModel):
    default_policy: PolicyAction = PolicyAction.ASK
    tools: dict[str, PolicyAction] = Field(default_factory=dict)
    commands: CommandRulesConfig = Field(default_factory=CommandRulesConfig)
    paths: PathRulesConfig = Field(default_factory=PathRulesConfig)

    def merge(self, other: Optional["ExecutionRulesConfig"]) -> "ExecutionRulesConfig":
        """Merge another ExecutionRulesConfig (e.g. profile overrides) over this one."""
        if not other:
            return self

        # Deny rules accumulate for maximum security
        deny_cmds = list(dict.fromkeys(self.commands.deny + other.commands.deny))
        allow_cmds = list(dict.fromkeys(self.commands.allow + other.commands.allow))
        ask_cmds = list(dict.fromkeys(self.commands.ask + other.commands.ask))

        deny_paths = list(dict.fromkeys(self.paths.deny + other.paths.deny))
        allow_paths = list(dict.fromkeys(self.paths.allow + other.paths.allow))
        ask_paths = list(dict.fromkeys(self.paths.ask + other.paths.ask))

        merged_tools = dict(self.tools)
        merged_tools.update(other.tools)

        return ExecutionRulesConfig(
            default_policy=other.default_policy if other.default_policy != PolicyAction.ASK else self.default_policy,
            tools=merged_tools,
            commands=CommandRulesConfig(deny=deny_cmds, allow=allow_cmds, ask=ask_cmds),
            paths=PathRulesConfig(deny=deny_paths, allow=allow_paths, ask=ask_paths),
        )


def _matches_command(pattern: str, command: str) -> bool:
    pattern = pattern.strip()
    cmd = command.strip()
    if not pattern or not cmd:
        return False

    cmd_lower = cmd.lower()
    pat_lower = pattern.lower()

    # Root wipeout pattern: match rm -rf /* or rm -rf / * but not subdirectories
    if pat_lower in ("rm -rf /*", "rm -rf / *"):
        return cmd_lower in ("rm -rf /*", "rm -rf / *") or cmd_lower.startswith("rm -rf /* ") or cmd_lower.startswith("rm -rf / * ")

    if any(c in pattern for c in ("*", "?", "[", "]")):
        if fnmatch.fnmatchcase(cmd_lower, pat_lower):
            return True
        return False

    return cmd_lower == pat_lower or cmd_lower.startswith(pat_lower + " ")


def _matches_path(pattern: str, target: Path) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False

    target_str = str(target)
    name = target.name

    try:
        if target.match(pattern):
            return True
    except Exception:
        pass

    if fnmatch.fnmatchcase(name.lower(), pattern.lower()):
        return True
    if fnmatch.fnmatchcase(target_str.lower(), pattern.lower()):
        return True

    # Check parent directory patterns e.g. .ssh in path parts
    clean_pat = pattern.strip("*").strip("/")
    if clean_pat and any(clean_pat.lower() == part.lower() for part in target.parts):
        return True

    return False


class RuleEngine:
    """Evaluates agent tool calls, shell commands, and file paths against security rules."""

    def __init__(self, config: Optional[ExecutionRulesConfig] = None):
        self.config = config or ExecutionRulesConfig()

        # Combine defaults with user configuration
        self.deny_commands = list(dict.fromkeys(DEFAULT_DENY_COMMANDS + self.config.commands.deny))
        self.allow_commands = list(dict.fromkeys(DEFAULT_ALLOW_COMMANDS + self.config.commands.allow))
        self.ask_commands = list(dict.fromkeys(self.config.commands.ask))

        self.deny_paths = list(dict.fromkeys(DEFAULT_DENY_PATHS + self.config.paths.deny))
        self.allow_paths = list(dict.fromkeys(DEFAULT_ALLOW_PATHS + self.config.paths.allow))
        self.ask_paths = list(dict.fromkeys(self.config.paths.ask))

        self.tool_policies = dict(DEFAULT_TOOL_POLICIES)
        self.tool_policies.update(self.config.tools)
        self.default_policy = self.config.default_policy

    def evaluate(
        self,
        tool_name: str,
        args: dict[str, Any],
        cwd: Optional[Path] = None,
        yolo: bool = False,
    ) -> tuple[PolicyAction, str]:
        """Evaluate action policy. Returns (PolicyAction, reason)."""
        work_dir = cwd or Path.cwd()

        # 1. Inspect target path if present (for path-based operations)
        path_arg = (
            args.get("path")
            or args.get("TargetFile")
            or args.get("AbsolutePath")
            or args.get("file_path")
            or args.get("DirectoryPath")
        )
        target_path: Optional[Path] = None
        if path_arg and isinstance(path_arg, (str, Path)):
            raw_path = Path(str(path_arg).strip())
            target_path = raw_path if raw_path.is_absolute() else (work_dir / raw_path).resolve()

        if target_path is not None:
            # Check path DENY
            for pat in self.deny_paths:
                if _matches_path(pat, target_path):
                    return (
                        PolicyAction.DENY,
                        f"Access to path '{target_path}' is denied by rule '{pat}'",
                    )

        # 2. Inspect command if run_command
        if tool_name == "run_command":
            cmd = (args.get("command") or args.get("CommandLine") or "").strip()
            if not cmd:
                return (PolicyAction.DENY, "Empty shell command is not allowed")

            # A. Check command DENY
            for pat in self.deny_commands:
                if _matches_command(pat, cmd):
                    return (
                        PolicyAction.DENY,
                        f"Command '{cmd}' is blocked by security rule '{pat}'",
                    )

            # B. Check command ALLOW
            for pat in self.allow_commands:
                if _matches_command(pat, cmd):
                    return (
                        PolicyAction.ALLOW,
                        f"Command '{cmd}' is auto-approved by rule '{pat}'",
                    )

            # C. Check command ASK
            for pat in self.ask_commands:
                if _matches_command(pat, cmd):
                    if yolo:
                        return (
                            PolicyAction.ALLOW,
                            f"Command '{cmd}' auto-approved via YOLO mode (matched ask rule '{pat}')",
                        )
                    return (PolicyAction.ASK, f"Command '{cmd}' requires approval (matched rule '{pat}')")

        # 3. Path ALLOW / ASK checks if tool had a path
        if target_path is not None:
            for pat in self.allow_paths:
                if _matches_path(pat, target_path):
                    return (
                        PolicyAction.ALLOW,
                        f"Path '{target_path}' is auto-approved by rule '{pat}'",
                    )
            for pat in self.ask_paths:
                if _matches_path(pat, target_path):
                    if yolo:
                        return (
                            PolicyAction.ALLOW,
                            f"Path '{target_path}' auto-approved via YOLO mode",
                        )
                    return (PolicyAction.ASK, f"Path '{target_path}' requires approval")

        # 4. Check tool-level policies
        tool_policy = self.tool_policies.get(tool_name)
        if tool_policy == PolicyAction.DENY:
            return (PolicyAction.DENY, f"Tool '{tool_name}' is disabled by policy")
        if tool_policy == PolicyAction.ALLOW:
            return (PolicyAction.ALLOW, f"Tool '{tool_name}' is auto-approved by policy")
        if tool_policy == PolicyAction.ASK:
            if yolo:
                return (
                    PolicyAction.ALLOW,
                    f"Tool '{tool_name}' auto-approved via YOLO mode",
                )
            return (PolicyAction.ASK, f"Tool '{tool_name}' requires approval")

        # 5. Default fallback policy
        if self.default_policy == PolicyAction.DENY:
            return (PolicyAction.DENY, f"Tool '{tool_name}' blocked by default policy")
        if self.default_policy == PolicyAction.ALLOW or yolo:
            return (PolicyAction.ALLOW, f"Tool '{tool_name}' allowed by default policy{' (YOLO)' if yolo else ''}")

        return (PolicyAction.ASK, f"Tool '{tool_name}' requires approval by default policy")
