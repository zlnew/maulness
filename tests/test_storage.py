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
async def test_storage_multiple_tasks_same_thread(tmp_path: Path):
    """Verify that multiple tasks can share the same thread/channel without unique constraint violations."""
    db_file = tmp_path / "test_multi_thread.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    t1 = await storage.create_task(
        task_id="task_101",
        title="First Task",
        repo_name="horizonx",
        workspace_path="/tmp/ws",
        mode=TaskMode.MULTI,
        origin_platform="discord",
        origin_channel_id="987654321",
        origin_thread_id="987654321",
    )
    t2 = await storage.create_task(
        task_id="task_102",
        title="Second Task",
        repo_name="horizonx",
        workspace_path="/tmp/ws",
        mode=TaskMode.MULTI,
        origin_platform="discord",
        origin_channel_id="987654321",
        origin_thread_id="987654321",
    )
    assert t1.origin_channel_id == "987654321"
    assert t2.origin_channel_id == "987654321"
    assert t1.discord_thread_id == 987654321


@pytest.mark.asyncio
async def test_storage_multi_platform_and_optional_repo(tmp_path: Path):
    """Verify tasks without repository and tasks originating from Telegram/Slack/WhatsApp."""
    db_file = tmp_path / "test_multi_platform.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    # Task without repo (general Q&A)
    t_general = await storage.create_task(
        task_id="task_general",
        title="What is the meaning of life?",
        repo_name=None,
        workspace_path=None,
        mode=TaskMode.DIRECT,
        origin_platform="telegram",
        origin_channel_id="-1001234567890",
        origin_thread_id="42",
    )
    assert t_general.repo_name is None
    assert t_general.workspace_path is None
    assert t_general.origin_platform == "telegram"
    assert t_general.origin_channel_id == "-1001234567890"
    assert t_general.origin_thread_id == "42"

    fetched = await storage.get_task("task_general")
    assert fetched is not None
    assert fetched.repo_name is None
    assert fetched.origin_platform == "telegram"

    # Multi-platform channel conversations
    await storage.set_channel_conversation(
        channel_id="C01234ABCD",
        conversation_id="uuid-slack-001",
        platform="slack",
    )
    await storage.set_channel_conversation(
        channel_id="-1001234567890",
        conversation_id="uuid-telegram-001",
        platform="telegram",
    )
    assert await storage.get_channel_conversation("C01234ABCD", platform="slack") == "uuid-slack-001"
    assert await storage.get_channel_conversation("-1001234567890", platform="telegram") == "uuid-telegram-001"
    assert await storage.get_channel_conversation("C01234ABCD", platform="discord") is None


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


@pytest.mark.asyncio
async def test_conversation_messages_storage(tmp_path: Path):
    db_file = tmp_path / "test_conv.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    conv_id = "test-conv-uuid-123"

    # Initially empty
    msgs = await storage.get_conversation_messages(conv_id)
    assert msgs == []

    # Add messages
    await storage.add_conversation_message(conv_id, "user", "Hello world")
    await storage.add_conversation_message(conv_id, "assistant", "Hi there!")
    await storage.add_conversation_message(conv_id, "user", "How are you?")

    # Fetch chronological messages
    msgs = await storage.get_conversation_messages(conv_id, limit=10)
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "Hello world"}
    assert msgs[1] == {"role": "assistant", "content": "Hi there!"}
    assert msgs[2] == {"role": "user", "content": "How are you?"}

    # Clear messages
    cleared = await storage.clear_conversation_messages(conv_id)
    assert cleared == 3
    assert await storage.get_conversation_messages(conv_id) == []


@pytest.mark.asyncio
async def test_compacted_memory_storage(tmp_path: Path):
    db_file = tmp_path / "test_memory.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    channel_id = 99887766

    # None initially
    assert await storage.get_latest_compacted_memory(channel_id) is None

    # Save session memory
    await storage.save_session_memory(
        session_id=f"discord_{channel_id}_123",
        summary="User prefers pnpm and service-name addressing in docker",
        token_count=100,
    )

    summary = await storage.get_latest_compacted_memory(channel_id)
    assert summary is not None
    assert "User prefers pnpm" in summary


def test_model_discord_properties():
    from maulness.core.models import TaskRecord, ApprovalRecord, TaskMode, TaskStatus

    # TaskRecord with discord origin
    t1 = TaskRecord(
        id="t1",
        title="T",
        repo_name="R",
        workspace_path="/",
        mode=TaskMode.DIRECT,
        status=TaskStatus.DONE,
        origin_platform="discord",
        origin_thread_id="12345678",
    )
    assert t1.discord_thread_id == 12345678

    # TaskRecord with non-numeric thread id
    t2 = TaskRecord(
        id="t2",
        title="T",
        repo_name="R",
        workspace_path="/",
        mode=TaskMode.DIRECT,
        status=TaskStatus.DONE,
        origin_platform="discord",
        origin_thread_id="not_an_int",
    )
    assert t2.discord_thread_id is None

    # TaskRecord with non-discord origin
    t3 = TaskRecord(
        id="t3",
        title="T",
        repo_name="R",
        workspace_path="/",
        mode=TaskMode.DIRECT,
        status=TaskStatus.DONE,
        origin_platform="cli",
        origin_thread_id="12345678",
    )
    assert t3.discord_thread_id is None

    # ApprovalRecord with discord platform
    a1 = ApprovalRecord(
        id="a1",
        task_id="t1",
        rpc_request_id=1,
        tool_name="run_command",
        tool_args="{}",
        platform="discord",
        platform_message_id="987654321",
    )
    assert a1.discord_message_id == 987654321

    # ApprovalRecord with non-numeric id
    a2 = ApprovalRecord(
        id="a2",
        task_id="t1",
        rpc_request_id=1,
        tool_name="run_command",
        tool_args="{}",
        platform="discord",
        platform_message_id="invalid",
    )
    assert a2.discord_message_id is None

    # ApprovalRecord with non-discord platform
    a3 = ApprovalRecord(
        id="a3",
        task_id="t1",
        rpc_request_id=1,
        tool_name="run_command",
        tool_args="{}",
        platform="slack",
        platform_message_id="987654321",
    )
    assert a3.discord_message_id is None


@pytest.mark.asyncio
async def test_list_channel_conversations(tmp_path: Path):
    db_file = tmp_path / "test_channels.db"
    storage = StorageManager(db_path=db_file)
    await storage.initialize()

    # Set channel conversations
    await storage.set_channel_conversation(111, "conv-111", platform="discord")
    await storage.set_channel_conversation("web_222", "conv-222", platform="web")

    # List all
    all_ch = await storage.list_channel_conversations()
    assert all_ch[111] == "conv-111"
    assert all_ch["web_222"] == "conv-222"

    # Filter by platform
    discord_ch = await storage.list_channel_conversations(platform="discord")
    assert 111 in discord_ch
    assert "web_222" not in discord_ch


@pytest.mark.asyncio
async def test_storage_legacy_migration_and_corrupt_payload(tmp_path: Path):
    import aiosqlite
    db_file = tmp_path / "legacy.db"

    # 1. Create a legacy table schema without 'origin_platform'
    async with aiosqlite.connect(str(db_file)) as db:
        await db.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT);")
        await db.commit()

    storage = StorageManager(db_path=db_file)
    # initialize() should detect missing origin_platform and recreate tables
    await storage.initialize()

    # Verify new schema is active
    task = await storage.create_task(
        task_id="migrated_task",
        title="Migrated",
        repo_name="test",
        workspace_path="/tmp",
        mode=TaskMode.DIRECT,
    )
    assert task.origin_platform == "cli"

    # 2. Empty idempotency key lookup
    assert await storage.get_agent_event_by_idempotency_key("") is None

    # 3. Payload with non-json serializable type (set)
    class NonSerializable:
        def __str__(self):
            return "unserializable_obj"

    ev_id = await storage.record_agent_event(
        task_id="migrated_task",
        stage="test",
        step_index=0,
        event_type="custom",
        payload={"obj": NonSerializable()},
        idempotency_key="idem_custom_1",
    )
    assert ev_id is not None
    ev = await storage.get_agent_event_by_idempotency_key("idem_custom_1")
    assert "unserializable_obj" in str(ev["payload"])

    # 4. Circular reference triggers fallback in record_agent_event
    circular = []
    circular.append(circular)
    ev_circ = await storage.record_agent_event(
        task_id="migrated_task",
        stage="test",
        step_index=1,
        event_type="circular",
        payload={"circ": circular},
        idempotency_key="idem_circ_1",
    )
    assert ev_circ is not None

    # 5. Insert raw non-JSON text into agent_events and fetch
    async with aiosqlite.connect(str(db_file)) as db:
        await db.execute(
            "INSERT INTO agent_events (task_id, stage, step_index, event_type, event_payload, idempotency_key) VALUES (?, ?, ?, ?, ?, ?)",
            ("migrated_task", "test", 1, "raw", "{invalid json content", "idem_raw_1"),
        )
        await db.commit()

    ev_raw = await storage.get_agent_event_by_idempotency_key("idem_raw_1")
    assert ev_raw["payload"] == "{invalid json content"

    all_events = await storage.get_agent_events("migrated_task")
    assert any(e["payload"] == "{invalid json content" for e in all_events)




