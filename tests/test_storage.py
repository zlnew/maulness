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
