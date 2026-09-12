import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from maulness.core.rules import (
    CommandRulesConfig,
    ExecutionRulesConfig,
    PathRulesConfig,
    PolicyAction,
    RuleEngine,
)
from maulness.core.approvals import ApprovalClassifier
from maulness.core.tools import execute_tool_call


def test_default_tool_policies():
    engine = RuleEngine()
    # Read-only tools default to allow
    assert engine.evaluate("read_file", {"path": "test.txt"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("list_dir", {"path": "."})[0] == PolicyAction.ALLOW
    assert engine.evaluate("git_status", {})[0] == PolicyAction.ALLOW

    # Mutating tools default to ask
    assert engine.evaluate("write_file", {"path": "foo.txt", "content": "hi"})[0] == PolicyAction.ASK


def test_default_safe_commands_allow():
    engine = RuleEngine()
    assert engine.evaluate("run_command", {"command": "pwd"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "whoami"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "uname -srm"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "git status -s"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "git diff HEAD~1"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "git log -n 5"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": "pytest tests/ -v"})[0] == PolicyAction.ALLOW
    assert engine.evaluate("run_command", {"command": ".venv/bin/pytest"})[0] == PolicyAction.ALLOW


def test_default_dangerous_commands_deny():
    engine = RuleEngine()
    assert engine.evaluate("run_command", {"command": "rm -rf /"})[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "rm -rf /*"})[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "sudo systemctl restart maulness"})[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "dd if=/dev/zero of=/dev/sda"})[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "mkfs.ext4 /dev/sdb"})[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "git push origin main --force"})[0] == PolicyAction.DENY


def test_mutating_commands_ask_by_default():
    engine = RuleEngine()
    assert engine.evaluate("run_command", {"command": "git push origin main"})[0] == PolicyAction.ASK
    assert engine.evaluate("run_command", {"command": "rm file.txt"})[0] == PolicyAction.ASK
    assert engine.evaluate("run_command", {"command": "mv foo.txt bar.txt"})[0] == PolicyAction.ASK
    assert engine.evaluate("run_command", {"command": "npm install express"})[0] == PolicyAction.ASK


def test_sensitive_paths_deny():
    engine = RuleEngine()
    # .env files
    assert engine.evaluate("read_file", {"path": ".env"})[0] == PolicyAction.DENY
    assert engine.evaluate("read_file", {"path": "/home/user/repo/.env.local"})[0] == PolicyAction.DENY
    assert engine.evaluate("write_file", {"path": ".env", "content": "secret"})[0] == PolicyAction.DENY

    # SSH keys
    assert engine.evaluate("read_file", {"path": "/home/user/.ssh/id_rsa"})[0] == PolicyAction.DENY
    assert engine.evaluate("read_file", {"path": "id_rsa"})[0] == PolicyAction.DENY

    # Root/System paths
    assert engine.evaluate("read_file", {"path": "/etc/shadow"})[0] == PolicyAction.DENY


def test_yolo_mode_bypasses_ask_but_retains_deny():
    engine = RuleEngine()
    # Mutating command in normal mode is ASK, in YOLO mode is ALLOW
    assert engine.evaluate("run_command", {"command": "git push origin main"}, yolo=False)[0] == PolicyAction.ASK
    assert engine.evaluate("run_command", {"command": "git push origin main"}, yolo=True)[0] == PolicyAction.ALLOW

    # Writing file in normal mode is ASK, in YOLO mode is ALLOW
    assert engine.evaluate("write_file", {"path": "foo.txt"}, yolo=False)[0] == PolicyAction.ASK
    assert engine.evaluate("write_file", {"path": "foo.txt"}, yolo=True)[0] == PolicyAction.ALLOW

    # BUT dangerous actions remain DENIED even in YOLO mode!
    assert engine.evaluate("run_command", {"command": "sudo rm -rf /"}, yolo=True)[0] == PolicyAction.DENY
    assert engine.evaluate("run_command", {"command": "rm -rf /*"}, yolo=True)[0] == PolicyAction.DENY
    assert engine.evaluate("read_file", {"path": ".env"}, yolo=True)[0] == PolicyAction.DENY


def test_custom_execution_rules_and_merging():
    global_cfg = ExecutionRulesConfig(
        commands=CommandRulesConfig(
            deny=["dangerous_global_*"],
            allow=["global_allowed_*"],
        ),
        tools={"write_file": PolicyAction.DENY},
    )

    profile_cfg = ExecutionRulesConfig(
        commands=CommandRulesConfig(
            deny=["dangerous_profile_*"],
            allow=["profile_allowed_*"],
        ),
        tools={"write_file": PolicyAction.ALLOW},
    )

    merged = global_cfg.merge(profile_cfg)
    engine = RuleEngine(merged)

    # Deny rules accumulated
    assert "dangerous_global_*" in engine.deny_commands
    assert "dangerous_profile_*" in engine.deny_commands

    # Tool override applied from profile
    assert engine.tool_policies["write_file"] == PolicyAction.ALLOW

    # Profile allowed command matches
    assert engine.evaluate("run_command", {"command": "profile_allowed_cmd"})[0] == PolicyAction.ALLOW
    # Global allowed command matches
    assert engine.evaluate("run_command", {"command": "global_allowed_cmd"})[0] == PolicyAction.ALLOW


def test_approval_classifier_backward_compatibility():
    classifier = ApprovalClassifier(yolo_mode=False)
    assert classifier.should_auto_approve("view_file", {"AbsolutePath": "/test/file"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "git status"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "git push origin main"}) is False
    assert classifier.should_auto_approve("run_command", {"command": "sudo rm -rf /"}) is False


@pytest.mark.asyncio
async def test_execute_tool_call_enforces_deny(tmp_path: Path):
    engine = RuleEngine()
    # Denied command returns security error without executing
    result = await execute_tool_call(
        name="run_command",
        args={"command": "sudo apt install curl"},
        workspace_path=tmp_path,
        rule_engine=engine,
    )
    assert "Execution blocked by security policy" in result


@pytest.mark.asyncio
async def test_execute_tool_call_auto_executes_allow(tmp_path: Path):
    mock_approval = AsyncMock(return_value=False)
    engine = RuleEngine()
    # 'pwd' is in allow list; should run without triggering mock_approval
    result = await execute_tool_call(
        name="run_command",
        args={"command": "pwd"},
        workspace_path=tmp_path,
        on_approval=mock_approval,
        rule_engine=engine,
    )
    mock_approval.assert_not_called()
    assert str(tmp_path.resolve()) in result


@pytest.mark.asyncio
async def test_execute_tool_call_gates_ask(tmp_path: Path):
    mock_approval = AsyncMock(return_value=True)
    engine = RuleEngine()
    # 'git push' is in ask; should invoke on_approval
    result = await execute_tool_call(
        name="run_command",
        args={"command": "git push origin main"},
        workspace_path=tmp_path,
        on_approval=mock_approval,
        rule_engine=engine,
    )
    mock_approval.assert_called_once()


def test_worktree_paths_auto_approved():
    engine = RuleEngine()
    worktree_file = Path("/home/zlnew/www/personal/repo/maulness/.worktrees/pipe-task123/src/cache.py")
    policy, reason = engine.evaluate("write_file", {"path": str(worktree_file), "content": "x = 1"})
    assert policy == PolicyAction.ALLOW
    assert "auto-approved" in reason


def test_rules_merge_none():
    cfg = ExecutionRulesConfig()
    assert cfg.merge(None) is cfg


def test_matches_command_and_path_edge_cases():
    from maulness.core.rules import _matches_command, _matches_path
    assert not _matches_command("", "ls")
    assert not _matches_command("ls", "")
    assert not _matches_path("", Path("test.txt"))

    # Malformed glob pattern in match() (lines 169-170)
    assert not _matches_path("[*[", Path("test.txt"))

    # Filename match (line 173)
    assert _matches_path("test.txt", Path("/some/dir/test.txt"))

    # Parent directory match in parts (line 180)
    assert _matches_path(".ssh", Path("/home/user/.ssh/id_rsa"))


def test_rules_engine_ask_and_deny_branches():
    # 1. Ask commands with and without yolo (lines 260-266)
    cfg = ExecutionRulesConfig(
        commands=CommandRulesConfig(ask=["npm install*"]),
        paths=PathRulesConfig(ask=["*.secret"]),
        tools={"write_file": PolicyAction.DENY},
        default_policy=PolicyAction.DENY,
    )
    engine = RuleEngine(cfg)

    policy, reason = engine.evaluate("run_command", {"command": "npm install lodash"}, yolo=False)
    assert policy == PolicyAction.ASK
    assert "requires approval" in reason

    policy, reason = engine.evaluate("run_command", {"command": "npm install lodash"}, yolo=True)
    assert policy == PolicyAction.ALLOW
    assert "YOLO mode" in reason

    # 2. Ask paths with and without yolo (lines 277-283)
    policy, reason = engine.evaluate("read_file", {"path": "creds.secret"}, yolo=False)
    assert policy == PolicyAction.ASK
    assert "requires approval" in reason

    policy, reason = engine.evaluate("read_file", {"path": "creds.secret"}, yolo=True)
    assert policy == PolicyAction.ALLOW
    assert "YOLO mode" in reason

    # 3. Tool policy DENY (line 288)
    policy, reason = engine.evaluate("write_file", {"path": "any.txt"})
    assert policy == PolicyAction.DENY
    assert "disabled by policy" in reason

    # 4. Default policy DENY (line 301)
    policy, reason = engine.evaluate("unknown_custom_tool", {})
    assert policy == PolicyAction.DENY
    assert "blocked by default policy" in reason

def test_rules_match_path_case_insensitivity_and_exception():
    from maulness.core.rules import _matches_path
    # Case insensitivity (line 173)
    assert _matches_path("*.txt", Path("MyFile.TXT"))

    # Exception in target.match (lines 169-170)
    mock_path = MagicMock(spec=Path)
    mock_path.match.side_effect = ValueError("bad glob")
    mock_path.name = "file.txt"
    mock_path.parts = ("file.txt",)
    assert _matches_path("*.txt", mock_path)


def test_rules_empty_command_and_default_policies():
    engine_ask = RuleEngine(ExecutionRulesConfig(default_policy=PolicyAction.ASK))

    # Empty command (line 240)
    pol, reason = engine_ask.evaluate("run_command", {"command": ""})
    assert pol == PolicyAction.DENY
    assert "Empty shell command" in reason

    # Default policy with YOLO (line 302-303)
    pol, reason = engine_ask.evaluate("custom_action", {}, yolo=True)
    assert pol == PolicyAction.ALLOW
    assert "YOLO" in reason

    # Default policy ASK without YOLO (line 305)
    pol, reason = engine_ask.evaluate("custom_action", {}, yolo=False)
    assert pol == PolicyAction.ASK
    assert "requires approval by default policy" in reason
