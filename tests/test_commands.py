from pathlib import Path
from unittest.mock import MagicMock, patch

from maulness.cli.commands import expand_context_tags, handle_slash_command
from maulness.core.stack import WorkspaceStack


def test_expand_context_tags_no_tag():
    res = expand_context_tags("Hello world without tags", Path("/tmp"))
    assert res == "Hello world without tags"


def test_expand_context_tags_file(tmp_path: Path):
    src_file = tmp_path / "hello.py"
    src_file.write_text("print('Hello from file!')\n", encoding="utf-8")

    prompt = "Review this: @file:hello.py and tell me if it is good."
    expanded = expand_context_tags(prompt, tmp_path)

    assert "[Context from @file:hello.py]:" in expanded
    assert "print('Hello from file!')" in expanded
    assert "Review this:" in expanded


def test_expand_context_tags_missing_file(tmp_path: Path):
    prompt = "Review this: @file:missing.py"
    expanded = expand_context_tags(prompt, tmp_path)
    assert "[File not found: missing.py]" in expanded


def test_expand_context_tags_dir(tmp_path: Path):
    sub = tmp_path / "mydir"
    sub.mkdir()
    (sub / "a.txt").write_text("a")
    (sub / "b.txt").write_text("b")

    prompt = "Look at directory: @dir:mydir"
    expanded = expand_context_tags(prompt, tmp_path)

    assert "[Context from @dir:mydir]:" in expanded
    assert "📄 a.txt" in expanded
    assert "📄 b.txt" in expanded


def test_expand_context_tags_bare_path(tmp_path: Path):
    sub_file = tmp_path / "config.json"
    sub_file.write_text('{"key": "value"}', encoding="utf-8")

    prompt = "Check @config.json please and contact user@domain.com"
    expanded = expand_context_tags(prompt, tmp_path)

    assert "[Context from @config.json]:" in expanded
    assert '{"key": "value"}' in expanded
    # user@domain.com should NOT be replaced because domain.com is not a local workspace file
    assert "user@domain.com" in expanded


def test_expand_context_tags_diff(tmp_path: Path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout="+ Added line in diff\n",
            returncode=0,
        )
        prompt = "Explain these changes: @diff"
        expanded = expand_context_tags(prompt, tmp_path)
        assert "[Context from @diff]:" in expanded
        assert "+ Added line in diff" in expanded


def test_handle_slash_command_non_slash(tmp_path: Path):
    handled, out = handle_slash_command("plain prompt", tmp_path)
    assert handled is False
    assert out == ""


def test_handle_slash_command_help(tmp_path: Path):
    handled, out = handle_slash_command("/help", tmp_path)
    assert handled is True
    assert "Maulness Slash Command Reference" in out
    assert "/undo" in out
    assert "/diff" in out
    assert "/test" in out


def test_handle_slash_command_diff(tmp_path: Path):
    with (
        patch("maulness.core.worktree.WorktreeManager.is_git_repo", return_value=True),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(
            stdout="diff --git a/foo b/foo\n",
            returncode=0,
        )
        handled, out = handle_slash_command("/diff", tmp_path)
        assert handled is True
        assert "```diff" in out
        assert "diff --git a/foo b/foo" in out


def test_handle_slash_command_undo_dirty(tmp_path: Path):
    with (
        patch("maulness.core.worktree.WorktreeManager.is_git_repo", return_value=True),
        patch("subprocess.run") as mock_run,
    ):
        # First call is git status --porcelain
        mock_run.side_effect = [
            MagicMock(stdout=" M file.py\n", returncode=0),
            MagicMock(returncode=0),  # git restore
            MagicMock(returncode=0),  # git clean
        ]
        handled, out = handle_slash_command("/undo", tmp_path)
        assert handled is True
        assert "Undid uncommitted changes" in out


def test_handle_slash_command_undo_clean(tmp_path: Path):
    with (
        patch("maulness.core.worktree.WorktreeManager.is_git_repo", return_value=True),
        patch(
            "maulness.core.worktree.WorktreeManager.rollback_to_previous_milestone",
            return_value=True,
        ),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        handled, out = handle_slash_command("/undo", tmp_path)
        assert handled is True
        assert "Rolled back working tree to previous milestone" in out


def test_handle_slash_command_test(tmp_path: Path):
    fake_stack = WorkspaceStack(
        name="python-uv", language="python", test_command="uv run pytest"
    )
    with (
        patch("maulness.cli.commands.detect_stack", return_value=fake_stack),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="5 passed in 0.12s\n", returncode=0)
        handled, out = handle_slash_command("/test -k utils", tmp_path)
        assert handled is True
        assert "Test Suite Execution: PASSED" in out
        assert "5 passed in 0.12s" in out
        assert "uv run pytest -k utils" in out


def test_handle_slash_command_commit(tmp_path: Path):
    with (
        patch("maulness.core.worktree.WorktreeManager.is_git_repo", return_value=True),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git commit
            MagicMock(stdout="abc1234\n", returncode=0),  # git rev-parse
        ]
        handled, out = handle_slash_command("/commit feat: add feature", tmp_path)
        assert handled is True
        assert "Committed `abc1234`: feat: add feature" in out


def test_handle_slash_command_skills(tmp_path: Path):
    handled, out = handle_slash_command("/skills", tmp_path)
    assert handled is True
    assert "Discovered Skills" in out or "No skills discovered" in out


def test_handle_slash_command_outline(tmp_path: Path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
    )

    handled, out = handle_slash_command("/outline sample.py", tmp_path)
    assert handled is True
    assert "Outline for `sample.py`" in out
    assert "Calculator" in out


def test_handle_slash_command_repo_map(tmp_path: Path):
    py_file = tmp_path / "sample.py"
    py_file.write_text("def my_func(): pass\n")

    handled, out = handle_slash_command("/repo-map", tmp_path)
    assert handled is True
    assert "Workspace Repo Map" in out
