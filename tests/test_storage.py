import pytest
from pathlib import Path

from maulness.core.models import TaskMode, TaskStatus
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_storage_crud(tmp_path: Path):
    db_file = tmp_path / "test_maulness.db"
    storage = StorageManager(db_path=db_file)

    await storage.initialize()
    assert db_file.exists()

    # Create task
    task = await storage.create_task(
        task_id="task_001",
        title="Test Task",
        repo_name="expense-tracker",
        workspace_path="/tmp/workspace",
        mode=TaskMode.DIRECT,
        discord_thread_id=123456789,
    )
    assert task.id == "task_001"
    assert task.status == TaskStatus.PLANNING

    # Update status
    await storage.update_task_status("task_001", TaskStatus.BUILDING)

    # Fetch task
    fetched = await storage.get_task("task_001")
    assert fetched is not None
    assert fetched.status == TaskStatus.BUILDING
    assert fetched.discord_thread_id == 123456789

    # List tasks
    task_list = await storage.list_tasks()
    assert len(task_list) == 1
    assert task_list[0].id == "task_001"


@pytest.mark.asyncio
async def test_storage_sessions(tmp_path: Path):
    db_file = tmp_path / "test_maulness.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    # Create task first to satisfy foreign key constraint
    await storage.create_task(
        task_id="task_001",
        title="Test Task",
        repo_name="expense-tracker",
        workspace_path="/tmp/workspace",
        mode=TaskMode.DIRECT,
    )

    # Create session
    session = await storage.create_session(
        session_id="sess_001",
        profile="builder",
        engine="acp",
        task_id="task_001",
        pid=12345,
    )
    assert session.id == "sess_001"
    assert session.status == "active"

    # Fetch session
    fetched = await storage.get_session("sess_001")
    assert fetched is not None
    assert fetched.profile == "builder"
    assert fetched.pid == 12345

    # Update session
    await storage.update_session_status("sess_001", "closed")
    updated = await storage.get_session("sess_001")
    assert updated.status == "closed"

    # List sessions
    sessions = await storage.list_sessions()
    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_storage_approvals_and_cascade(tmp_path: Path):
    from maulness.core.models import ApprovalStatus

    db_file = tmp_path / "test_maulness.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    # 1. Create task
    task = await storage.create_task(
        task_id="task_cascade",
        title="Test Task for Cascade",
        repo_name="expense-tracker",
        workspace_path="/tmp/workspace",
        mode=TaskMode.DIRECT,
    )

    # 2. Create session linked to task
    await storage.create_session(
        session_id="sess_cascade",
        task_id="task_cascade",
        profile="builder",
        engine="acp",
    )

    # 3. Create approval linked to task
    appr = await storage.create_approval(
        approval_id="appr_001",
        task_id="task_cascade",
        rpc_request_id=42,
        tool_name="run_command",
        tool_args={"command": "pytest"},
    )
    assert appr.id == "appr_001"
    assert appr.status == ApprovalStatus.PENDING

    # Fetch approval
    fetched_appr = await storage.get_approval("appr_001")
    assert fetched_appr is not None
    assert fetched_appr.tool_name == "run_command"

    # Update approval
    updated = await storage.update_approval_status("appr_001", ApprovalStatus.APPROVED)
    assert updated is True
    assert (await storage.get_approval("appr_001")).status == ApprovalStatus.APPROVED

    # List approvals
    apprs = await storage.list_approvals(task_id="task_cascade")
    assert len(apprs) == 1

    # 4. Test Cascade Deletion: Deleting task must delete associated session and approval
    deleted = await storage.delete_task("task_cascade")
    assert deleted is True

    assert await storage.get_task("task_cascade") is None
    assert await storage.get_session("sess_cascade") is None
    assert await storage.get_approval("appr_001") is None


@pytest.mark.asyncio
async def test_reset_stale_sessions(tmp_path: Path):
    db_file = tmp_path / "test_maulness.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    await storage.create_task(
        task_id="task_inflight",
        title="Inflight Task",
        repo_name="horizonx",
        workspace_path="/tmp/workspace",
        mode=TaskMode.MULTI,
    )
    await storage.create_session(
        session_id="sess_orphan",
        task_id="task_inflight",
        profile="builder",
        engine="acp",
    )

    # Run reset
    res = await storage.reset_stale_sessions()
    assert res["sessions_closed"] == 1
    assert res["tasks_failed"] == 1

    # Verify updated statuses
    sess = await storage.get_session("sess_orphan")
    assert sess.status == "closed"
    task = await storage.get_task("task_inflight")
    assert task.status == TaskStatus.FAILED

