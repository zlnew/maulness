import re
from typing import Any, Optional

SAFE_READ_ONLY_COMMANDS = [
    r"^git\s+(status|diff|log|show|branch|rev-parse|describe)",
    r"^(ls|pwd|cat|head|tail|grep|rg|find|echo|which|stat)",
    r"^(cargo\s+check|pytest\s+--collect-only)",
]

SAFE_TOOLS = {
    "view_file",
    "list_dir",
    "grep_search",
    "find_by_name",
    "read_url_content",
    "search_web",
}


class ApprovalClassifier:
    """Evaluates whether an agent action is safe or requires human sign-off."""

    def __init__(self, yolo_mode: bool = False):
        self.yolo_mode = yolo_mode

    def should_auto_approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Return True if action is auto-approved (read-only or YOLO mode enabled)."""
        if self.yolo_mode:
            return True

        if tool_name in SAFE_TOOLS:
            return True

        if tool_name == "run_command":
            cmd = (args.get("command") or args.get("CommandLine") or "").strip()
            for pattern in SAFE_READ_ONLY_COMMANDS:
                if re.match(pattern, cmd, re.IGNORECASE):
                    return True

        return False
