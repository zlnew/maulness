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

    # Rollback to checkpoint 1
    rolled_back = wm.rollback_to_checkpoint(repo_path, cp1)
    assert rolled_back is True
    assert readme.read_text() == "# Test Repo\n"
    assert not new_file.exists()

