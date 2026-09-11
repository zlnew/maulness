from pathlib import Path
from typing import Any, Optional

from maulness.core.rules import PolicyAction, RuleEngine


class ApprovalClassifier:
    """Evaluates whether an agent action is safe or requires human sign-off."""

    def __init__(self, yolo_mode: bool = False, rule_engine: Optional[RuleEngine] = None):
        self.yolo_mode = yolo_mode
        self.rule_engine = rule_engine or RuleEngine()

    def evaluate(self, tool_name: str, args: dict[str, Any], cwd: Optional[Path] = None) -> tuple[PolicyAction, str]:
        """Evaluate policy action via RuleEngine."""
        return self.rule_engine.evaluate(tool_name, args, cwd=cwd, yolo=self.yolo_mode)

    def should_auto_approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Return True if action is auto-approved (read-only or YOLO mode enabled)."""
        action, _ = self.evaluate(tool_name, args)
        return action == PolicyAction.ALLOW
