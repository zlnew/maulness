import contextlib
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger("maulness.worktree")


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
        delete_branch: bool = False,
        auto_commit: bool = True,
    ) -> Generator[Path, None, None]:
        """Context manager providing an isolated git worktree that snapshots changes and preserves the branch."""
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
            # Snapshot any uncommitted or untracked changes before teardown
            if auto_commit and worktree_dir.exists():
                try:
                    status_res = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=str(worktree_dir),
                        capture_output=True,
                        text=True,
                    )
                    if status_res.returncode == 0 and status_res.stdout.strip():
                        subprocess.run(
                            ["git", "add", "-A"],
                            cwd=str(worktree_dir),
                            capture_output=True,
                        )
                        subprocess.run(
                            ["git", "commit", "-m", f"chore(pipeline): snapshot changes for {branch_name}"],
                            cwd=str(worktree_dir),
                            capture_output=True,
                        )
                except Exception as ex:
                    logger.debug("Worktree auto-commit failed: %s", ex)
        finally:
            self.remove_worktree(
                repo_path=repo_path,
                worktree_path=worktree_dir,
                force=True,
                delete_branch=delete_branch,
            )

    def create_checkpoint(self, repo_path: Path, label: str) -> Optional[str]:
        """Create a lightweight Git checkpoint commit or capture current HEAD hash."""
        if not self.is_git_repo(repo_path):
            return None
        try:
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            has_changes = bool(status_res.stdout.strip())
            if has_changes:
                subprocess.run(["git", "add", "-A"], cwd=str(repo_path), capture_output=True, timeout=10)
                subprocess.run(
                    ["git", "commit", "-m", f"checkpoint: {label}", "--allow-empty"],
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            rev_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if rev_res.returncode == 0:
                commit_hash = rev_res.stdout.strip()
                logger.info("Created git checkpoint '%s' at %s", label, commit_hash[:8])
                return commit_hash
        except Exception as e:
            logger.warning("Failed to create git checkpoint '%s': %s", label, e)
        return None

    def rollback_to_checkpoint(self, repo_path: Path, commit_hash: str) -> bool:
        """Reset repository working tree cleanly to the specified commit hash."""
        if not self.is_git_repo(repo_path) or not commit_hash:
            return False
        try:
            logger.info("Rolling back working tree in '%s' to %s", repo_path, commit_hash[:8])
            reset_res = subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
            clean_res = subprocess.run(
                ["git", "clean", "-fd"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=15,
            )
            return reset_res.returncode == 0 and clean_res.returncode == 0
        except Exception as e:
            logger.error("Failed to rollback to checkpoint %s: %s", commit_hash, e)
            return False
