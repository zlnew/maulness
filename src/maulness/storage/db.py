import os
from pathlib import Path
from typing import Optional

import aiosqlite

from maulness.core.models import ApprovalStatus, TaskMode, TaskRecord, TaskStatus

DEFAULT_DB_PATH = Path.home() / ".config" / "maulness" / "maulness.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class StorageManager:
    """Async SQLite storage manager for Maulness tasks, sessions, and approvals."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH

    async def initialize(self):
        """Ensure parent directories exist and execute SQLite DDL schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(schema_sql)
            await db.commit()

    async def create_task(
        self,
        task_id: str,
        title: str,
        repo_name: str,
        workspace_path: str,
        mode: TaskMode,
        discord_thread_id: Optional[int] = None,
    ) -> TaskRecord:
        """Create a new task record in the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tasks (id, discord_thread_id, title, repo_name, workspace_path, mode, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    discord_thread_id,
                    title,
                    repo_name,
                    workspace_path,
                    mode.value,
                    TaskStatus.PLANNING.value,
                ),
            )
            await db.commit()

        return TaskRecord(
            id=task_id,
            title=title,
            repo_name=repo_name,
            workspace_path=workspace_path,
            mode=mode,
            status=TaskStatus.PLANNING,
            discord_thread_id=discord_thread_id,
        )

    async def update_task_status(self, task_id: str, status: TaskStatus):
        """Update a task's state machine status."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status.value, task_id),
            )
            await db.commit()

    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch a task record by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = await cursor.fetchone()
            if not row:
                return None

            return TaskRecord(
                id=row["id"],
                title=row["title"],
                repo_name=row["repo_name"],
                workspace_path=row["workspace_path"],
                mode=TaskMode(row["mode"]),
                status=TaskStatus(row["status"]),
                discord_thread_id=row["discord_thread_id"],
            )

    async def list_tasks(self, limit: int = 20) -> list[TaskRecord]:
        """List the most recent tasks."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [
                TaskRecord(
                    id=row["id"],
                    title=row["title"],
                    repo_name=row["repo_name"],
                    workspace_path=row["workspace_path"],
                    mode=TaskMode(row["mode"]),
                    status=TaskStatus(row["status"]),
                    discord_thread_id=row["discord_thread_id"],
                )
                for row in rows
            ]
