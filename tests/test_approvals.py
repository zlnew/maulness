import pytest
from maulness.core.approvals import ApprovalClassifier


def test_safe_tools_auto_approved():
    classifier = ApprovalClassifier(yolo_mode=False)
    assert classifier.should_auto_approve("view_file", {"AbsolutePath": "/test/file"}) is True
    assert classifier.should_auto_approve("list_dir", {"DirectoryPath": "/test"}) is True
    assert classifier.should_auto_approve("grep_search", {"Query": "abc"}) is True


def test_safe_commands_auto_approved():
    classifier = ApprovalClassifier(yolo_mode=False)
    assert classifier.should_auto_approve("run_command", {"command": "git status"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "git diff HEAD~1"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "ls -la /tmp"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "cat package.json"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "pytest --collect-only"}) is True


def test_mutating_actions_require_approval_by_default():
    classifier = ApprovalClassifier(yolo_mode=False)
    assert classifier.should_auto_approve("write_to_file", {"TargetFile": "foo.txt"}) is False
    assert classifier.should_auto_approve("run_command", {"command": "rm -rf /tmp/test"}) is False
    assert classifier.should_auto_approve("run_command", {"command": "git push origin main"}) is False
    assert classifier.should_auto_approve("run_command", {"command": "php artisan migrate"}) is False


def test_yolo_mode_bypasses_all():
    classifier = ApprovalClassifier(yolo_mode=True)
    assert classifier.should_auto_approve("write_to_file", {"TargetFile": "foo.txt"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "rm -rf /tmp/test"}) is True
    assert classifier.should_auto_approve("run_command", {"command": "git push origin main"}) is True
    assert classifier.should_auto_approve("unknown_tool", {}) is True
