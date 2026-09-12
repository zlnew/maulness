import subprocess
from pathlib import Path
import pytest

from maulness.core.worktree import WorktreeManager


def setup_git_repo(path: Path) -> Path:
    """Helper to initialize a real git repository in a temp directory."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=str(path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@maulness.local"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.name", "Maulness Tester"], cwd=str(path), check=True)
    
    # Create initial commit
    (path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(path), check=True)
    return path


def test_worktree_lifecycle(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "main_repo")
    wm = WorktreeManager()

    assert wm.is_git_repo(repo_path) is True
    assert wm.is_git_repo(tmp_path / "non_existent") is False

    # Create worktree
    wt_path = wm.create_worktree(repo_path, branch_name="feature-test-1")
    assert wt_path.exists()
    assert (wt_path / "README.md").exists()

    # Modify in worktree
    (wt_path / "new_file.txt").write_text("Hello from worktree")
    assert (wt_path / "new_file.txt").exists()
    assert not (repo_path / "new_file.txt").exists()

    # Remove worktree
    success = wm.remove_worktree(repo_path, wt_path, force=True, delete_branch=True)
    assert success is True
    assert not wt_path.exists()


def test_isolated_worktree_context_manager(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "cm_repo")
    wm = WorktreeManager()

    captured_wt_path = None
    with wm.isolated_worktree(repo_path, branch_prefix="iso-test") as wt_path:
        captured_wt_path = wt_path
        assert wt_path.exists()
        assert (wt_path / "README.md").exists()
    # Upon exit, worktree directory should be cleanly removed
    assert not captured_wt_path.exists()


def test_checkpoint_and_rollback(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "cp_repo")
    wm = WorktreeManager()

    # Initial state
    readme = repo_path / "README.md"
    assert readme.read_text() == "# Test Repo\n"

    # Create checkpoint at clean state
    cp1 = wm.create_checkpoint(repo_path, "stage1_start")
    assert cp1 is not None

    # Modify file and create new file
    readme.write_text("# Test Repo Modified\n")
    new_file = repo_path / "dirty.txt"
    new_file.write_text("should be wiped\n")
    assert readme.read_text() == "# Test Repo Modified\n"
    assert new_file.exists()

    # Checkpoint on dirty tree
    cp_dirty = wm.create_checkpoint(repo_path, "dirty_checkpoint")
    assert cp_dirty is not None

    # Rollback to checkpoint 1
    rolled_back = wm.rollback_to_checkpoint(repo_path, cp1)
    assert rolled_back is True
    assert readme.read_text() == "# Test Repo\n"
    assert not new_file.exists()


def test_isolated_worktree_auto_commits_and_preserves_branch(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "preserve_repo")
    wm = WorktreeManager()

    captured_wt_path = None
    created_branch = None
    with wm.isolated_worktree(repo_path, branch_prefix="preserve-test", delete_branch=False) as wt_path:
        captured_wt_path = wt_path
        # Get branch name
        b_res = subprocess.run(["git", "branch", "--show-current"], cwd=str(wt_path), capture_output=True, text=True)
        created_branch = b_res.stdout.strip()
        assert created_branch.startswith("preserve-test-")

        # Write untracked and modified files
        (wt_path / "docs").mkdir()
        (wt_path / "docs" / "research.md").write_text("# Preserved Research Notes\n")

    # 1. Directory must be deleted
    assert not captured_wt_path.exists()

    # 2. Branch must still exist in git repository
    branches_res = subprocess.run(["git", "branch"], cwd=str(repo_path), capture_output=True, text=True)
    assert created_branch in branches_res.stdout

    # 3. Checkout branch and verify committed file content
    subprocess.run(["git", "checkout", created_branch], cwd=str(repo_path), check=True, capture_output=True)
    assert (repo_path / "docs" / "research.md").exists()
    assert (repo_path / "docs" / "research.md").read_text() == "# Preserved Research Notes\n"


def test_isolated_worktree_non_git_repo(tmp_path: Path):
    non_git = tmp_path / "plain_dir"
    non_git.mkdir()
    wm = WorktreeManager()

    with wm.isolated_worktree(non_git) as yielded_path:
        assert yielded_path == non_git

    assert wm.create_checkpoint(non_git, "label") is None
    assert wm.rollback_to_checkpoint(non_git, "dummy_hash") is False


def test_create_worktree_existing_branch(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "existing_branch_repo")
    wm = WorktreeManager()

    # Create branch first
    subprocess.run(["git", "branch", "feature-x"], cwd=str(repo_path), check=True)

    wt = wm.create_worktree(repo_path, branch_name="feature-x")
    assert wt.exists()
    assert (wt / "README.md").exists()
    wm.remove_worktree(repo_path, wt, force=True, delete_branch=True)


def test_create_worktree_failure_raises(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "fail_repo")
    wm = WorktreeManager()

    # Pass nonexistent commit
    with pytest.raises(RuntimeError, match="Failed to create worktree"):
        wm.create_worktree(repo_path, branch_name="bad_branch", base_commit="nonexistent_ref_12345")


def test_checkpoint_and_rollback_exceptions(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "err_repo")
    wm = WorktreeManager()

    orig_run = subprocess.run
    def fail_inside_run(cmd, *args, **kwargs):
        if "--is-inside-work-tree" in cmd:
            return orig_run(cmd, *args, **kwargs)
        raise RuntimeError("Git operation crashed")

    from unittest.mock import patch
    with patch("subprocess.run", side_effect=fail_inside_run):
        assert wm.create_checkpoint(repo_path, "fail") is None
        assert wm.rollback_to_checkpoint(repo_path, "some_hash") is False



def test_worktree_remove_branch_show_current_exception(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "exc_repo")
    wm = WorktreeManager()
    wt = wm.create_worktree(repo_path, branch_name="feat-exc")

    orig_run = subprocess.run
    def custom_run(cmd, *args, **kwargs):
        if "--show-current" in cmd:
            raise RuntimeError("show current error")
        return orig_run(cmd, *args, **kwargs)

    from unittest.mock import patch
    with patch("subprocess.run", side_effect=custom_run):
        # Should catch exception cleanly on line 83-84
        wm.remove_worktree(repo_path, wt, force=True, delete_branch=True)


def test_worktree_fallback_shutil_rmtree(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "shutil_repo")
    wm = WorktreeManager()
    wt = wm.create_worktree(repo_path, branch_name="feat-shutil")

    orig_run = subprocess.run
    def mock_remove_keeps_dir(cmd, *args, **kwargs):
        if "remove" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, *args, **kwargs)

    from unittest.mock import patch
    with patch("subprocess.run", side_effect=mock_remove_keeps_dir):
        # Directory still exists, triggers line 98
        wm.remove_worktree(repo_path, wt, force=True)
        assert not wt.exists()


def test_worktree_auto_commit_on_exit_exception(tmp_path: Path):
    repo_path = setup_git_repo(tmp_path / "autocommit_repo")
    wm = WorktreeManager()

    orig_run = subprocess.run
    def fail_commit_run(cmd, *args, **kwargs):
        if "commit" in cmd:
            raise RuntimeError("Commit failed")
        return orig_run(cmd, *args, **kwargs)

    from unittest.mock import patch
    with patch("subprocess.run", side_effect=fail_commit_run):
        # Auto-commit fails and is caught on lines 153-154
        with wm.isolated_worktree(repo_path, auto_commit=True) as wt:
            (wt / "change.txt").write_text("content")
