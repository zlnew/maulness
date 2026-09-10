import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from maulness.core.models import (
    ApprovalRecord,
    ApprovalStatus,
    SessionRecord,
    TaskMode,
    TaskRecord,
    TaskStatus,
)

DEFAULT_DB_PATH = Path.home() / ".config" / "maulness" / "maulness.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class StorageManager:
    """Async SQLite storage manager for Maulness tasks, sessions, and approvals."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH

    @asynccontextmanager
    async def _connect(self):
        """Yield an aiosqlite connection with foreign keys, WAL mode, and busy timeout enabled."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute("PRAGMA busy_timeout = 5000;")
            yield db

    async def initialize(self):
        """Ensure parent directories exist, configure WAL mode, and execute SQLite DDL schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

        async with self._connect() as db:
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")
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
        async with self._connect() as db:
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
        async with self._connect() as db:
            await db.execute(
                "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status.value, task_id),
            )
            await db.commit()

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task. Cascades to associated sessions, approvals, and events."""
        async with self._connect() as db:
            cursor = await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch a task record by ID."""
        async with self._connect() as db:
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
        async with self._connect() as db:
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

    async def create_session(
        self,
        session_id: str,
        profile: str,
        engine: str,
        task_id: Optional[str] = None,
        acp_session_id: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> SessionRecord:
        """Create a new agent session record."""
        from maulness.core.models import SessionRecord

        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO agent_sessions (id, task_id, profile, engine, acp_session_id, pid, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
                """,
                (session_id, task_id, profile, engine, acp_session_id, pid),
            )
            await db.commit()

        return SessionRecord(
            id=session_id,
            task_id=task_id,
            profile=profile,
            engine=engine,
            acp_session_id=acp_session_id,
            pid=pid,
            status="active",
        )

    async def update_session_status(self, session_id: str, status: str = "closed"):
        """Update session status (active, closed, failed)."""
        async with self._connect() as db:
            await db.execute(
                "UPDATE agent_sessions SET status = ? WHERE id = ?",
                (status, session_id),
            )
            await db.commit()

    async def update_session_acp_id(self, session_id: str, acp_session_id: str):
        """Update backend ACP/Antigravity conversation ID for a session."""
        async with self._connect() as db:
            await db.execute(
                "UPDATE agent_sessions SET acp_session_id = ? WHERE id = ?",
                (acp_session_id, session_id),
            )
            await db.commit()

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Fetch session by ID."""
        from maulness.core.models import SessionRecord

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM agent_sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            if not row:
                return None

            return SessionRecord(
                id=row["id"],
                task_id=row["task_id"],
                profile=row["profile"],
                engine=row["engine"],
                acp_session_id=row["acp_session_id"],
                pid=row["pid"],
                status=row["status"],
            )

    async def list_sessions(self, limit: int = 20) -> list[SessionRecord]:
        """List active and recent sessions."""
        from maulness.core.models import SessionRecord

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM agent_sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
            return [
                SessionRecord(
                    id=row["id"],
                    task_id=row["task_id"],
                    profile=row["profile"],
                    engine=row["engine"],
                    acp_session_id=row["acp_session_id"],
                    pid=row["pid"],
                    status=row["status"],
                )
                for row in rows
            ]

    # ==========================================
    # HITL Approvals CRUD
    # ==========================================
    async def create_approval(
        self,
        approval_id: str,
        task_id: str,
        rpc_request_id: int,
        tool_name: str,
        tool_args: Any,
        discord_message_id: Optional[int] = None,
    ) -> ApprovalRecord:
        """Create a new pending approval record."""
        args_str = tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO approvals (id, task_id, rpc_request_id, tool_name, tool_args, status, discord_message_id)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (approval_id, task_id, rpc_request_id, tool_name, args_str, discord_message_id),
            )
            await db.commit()

        return ApprovalRecord(
            id=approval_id,
            task_id=task_id,
            rpc_request_id=rpc_request_id,
            tool_name=tool_name,
            tool_args=args_str,
            status=ApprovalStatus.PENDING,
            discord_message_id=discord_message_id,
        )

    async def update_approval_status(
        self, approval_id: str, status: ApprovalStatus
    ) -> bool:
        """Update approval status (approved, rejected, timeout)."""
        async with self._connect() as db:
            cursor = await db.execute(
                """
                UPDATE approvals
                SET status = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status.value, approval_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Fetch approval record by ID."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,))
            row = await cursor.fetchone()
            if not row:
                return None

            return ApprovalRecord(
                id=row["id"],
                task_id=row["task_id"],
                rpc_request_id=row["rpc_request_id"],
                tool_name=row["tool_name"],
                tool_args=row["tool_args"],
                status=ApprovalStatus(row["status"]),
                discord_message_id=row["discord_message_id"],
            )

    async def list_approvals(
        self, task_id: Optional[str] = None, limit: int = 20
    ) -> list[ApprovalRecord]:
        """List approval requests."""
        query = "SELECT * FROM approvals"
        params: list[Any] = []
        if task_id:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, tuple(params))
            rows = await cursor.fetchall()
            return [
                ApprovalRecord(
                    id=row["id"],
                    task_id=row["task_id"],
                    rpc_request_id=row["rpc_request_id"],
                    tool_name=row["tool_name"],
                    tool_args=row["tool_args"],
                    status=ApprovalStatus(row["status"]),
                    discord_message_id=row["discord_message_id"],
                )
                for row in rows
            ]

    async def save_session_memory(
        self,
        session_id: str,
        summary: str,
        token_count: int = 0,
    ) -> int:
        """Save a compacted session memory summary to SQLite."""
        async with self._connect() as db:
            cursor = await db.execute(
                """
                INSERT INTO session_memories (session_id, summary, token_count)
                VALUES (?, ?, ?)
                """,
                (session_id, summary, token_count),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_latest_session_memory(self, session_id: str) -> Optional[str]:
        """Retrieve the most recent compacted memory summary for a session."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT summary FROM session_memories
                WHERE session_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,),
            )
            row = await cursor.fetchone()
            return row["summary"] if row else None

    async def reset_stale_sessions(self) -> dict[str, int]:
        """Reset orphan active sessions and mark in-flight tasks as failed/closed."""
        async with self._connect() as db:
            c1 = await db.execute(
                "UPDATE agent_sessions SET status = 'closed' WHERE status = 'active'"
            )
            c2 = await db.execute(
                """
                UPDATE tasks
                SET status = 'failed', updated_at = CURRENT_TIMESTAMP
                WHERE status IN ('planning', 'building', 'review')
                """
            )
            await db.commit()
            return {
                "sessions_closed": c1.rowcount,
                "tasks_failed": c2.rowcount,
            }

    async def get_channel_conversation(self, channel_id: int) -> Optional[str]:
        """Fetch active conversation UUID for a channel."""
        async with self._connect() as db:
            async with db.execute(
                "SELECT conversation_id FROM channel_conversations WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

    async def set_channel_conversation(
        self, channel_id: int, conversation_id: str, profile_name: Optional[str] = None
    ):
        """Save or update active conversation UUID for a channel."""
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO channel_conversations (channel_id, conversation_id, profile_name, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(channel_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    profile_name = excluded.profile_name,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (channel_id, conversation_id, profile_name),
            )
            await db.commit()

    async def clear_channel_conversation(self, channel_id: int):
        """Clear active conversation UUID for a channel."""
        async with self._connect() as db:
            await db.execute(
                "DELETE FROM channel_conversations WHERE channel_id = ?",
                (channel_id,),
            )
            await db.commit()

    async def list_channel_conversations(self) -> dict[int, str]:
        """List all active channel conversation UUID mappings."""
        async with self._connect() as db:
            async with db.execute("SELECT channel_id, conversation_id FROM channel_conversations") as cursor:
                rows = await cursor.fetchall()
                return {row[0]: row[1] for row in rows}

