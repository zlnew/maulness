import contextlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Generator, Optional


class WorktreeManager:
    """Manages Git worktree lifecycle for isolated, concurrent, or multi-worker tasks."""

    def __init__(self, default_base_dir: Optional[Path] = None):
        self.default_base_dir = default_base_dir

    def is_git_repo(self, path: Path) -> bool:
        """Check if path is inside a Git repository."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
            return res.returncode == 0 and res.stdout.strip() == "true"
        except Exception:
            return False

    def create_worktree(
        self,
        repo_path: Path,
        branch_name: str,
        worktree_path: Optional[Path] = None,
        base_commit: str = "HEAD",
    ) -> Path:
        """Create a new Git worktree on a dedicated branch."""
        target_path = (
            worktree_path
            or (self.default_base_dir or (repo_path / ".worktrees")) / branch_name
        ).resolve()

        target_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["git", "worktree", "add", "-b", branch_name, str(target_path), base_commit]
        res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True)

        if res.returncode != 0:
            # If branch already exists, try attaching without -b
            cmd_existing = ["git", "worktree", "add", str(target_path), branch_name]
            res_existing = subprocess.run(
                cmd_existing, cwd=str(repo_path), capture_output=True, text=True
            )
            if res_existing.returncode != 0:
                raise RuntimeError(
                    f"Failed to create worktree: {res.stderr.strip()} / {res_existing.stderr.strip()}"
                )

        return target_path

    def remove_worktree(
        self,
        repo_path: Path,
        worktree_path: Path,
        force: bool = True,
        delete_branch: bool = True,
    ) -> bool:
        """Remove a Git worktree and optionally delete its branch."""
        # Find branch name associated with worktree if deleting branch
        branch_to_delete: Optional[str] = None
        if delete_branch:
            try:
                res = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(worktree_path),
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    branch_to_delete = res.stdout.strip()
            except Exception:
                pass

        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(str(worktree_path))

        res = subprocess.run(cmd, cwd=str(repo_path), capture_output=True, text=True)

        # Always prune stale metadata
        subprocess.run(["git", "worktree", "prune"], cwd=str(repo_path), capture_output=True)

        # Fallback directory cleanup if git worktree remove left artifacts
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

        if delete_branch and branch_to_delete:
            subprocess.run(
                ["git", "branch", "-D", branch_to_delete],
                cwd=str(repo_path),
                capture_output=True,
            )

        return res.returncode == 0

    @contextlib.contextmanager
    def isolated_worktree(
        self,
        repo_path: Path,
        branch_prefix: str = "maulness-task",
        base_commit: str = "HEAD",
        delete_branch: bool = True,
    ) -> Generator[Path, None, None]:
        """Context manager providing an isolated git worktree that automatically cleans up."""
        if not self.is_git_repo(repo_path):
            yield repo_path
            return

        unique_id = uuid.uuid4().hex[:8]
        branch_name = f"{branch_prefix}-{unique_id}"
        worktree_dir = self.create_worktree(
            repo_path=repo_path,
            branch_name=branch_name,
            base_commit=base_commit,
        )

        try:
            yield worktree_dir
        finally:
            self.remove_worktree(
                repo_path=repo_path,
                worktree_path=worktree_dir,
                force=True,
                delete_branch=delete_branch,
            )
