import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from maulness.daemon.scheduler import DaemonScheduler
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_daemon_scheduler_start_stop(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    scheduler = DaemonScheduler(storage=storage, bots=[])

    scheduler.start()
    assert scheduler._task is not None
    assert not scheduler._task.done()

    scheduler.stop()
    await asyncio.sleep(0.01)
    assert scheduler._task.done()


@pytest.mark.asyncio
async def test_daemon_scheduler_run_standup(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")

    mock_channel = AsyncMock()
    mock_bot = MagicMock()
    mock_bot.is_ready.return_value = True
    mock_bot.home_channel_id = 12345
    mock_bot.get_channel.return_value = mock_channel

    scheduler = DaemonScheduler(storage=storage, bots=[mock_bot])

    await scheduler.run_morning_standup()

    mock_channel.send.assert_called_once()
    call_kwargs = mock_channel.send.call_args.kwargs
    assert "embed" in call_kwargs
    embed = call_kwargs["embed"]
    assert "Morning Workspace Standup" in embed.title


@pytest.mark.asyncio
async def test_daemon_scheduler_run_loop_triggers_standup(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    scheduler = DaemonScheduler(storage=storage, bots=[])
    scheduler.run_morning_standup = AsyncMock()

    loop_count = 0

    async def fake_sleep(duration):
        nonlocal loop_count
        loop_count += 1
        if loop_count == 1:
            return
        raise asyncio.CancelledError()

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("os.getenv", return_value="08:00"),
        patch("datetime.datetime") as mock_dt,
    ):
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: "2026-09-12" if "%Y" in fmt else "08:00"
        mock_dt.now.return_value = mock_now

        await scheduler._run_loop()

    scheduler.run_morning_standup.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_scheduler_run_loop_handles_exception(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    scheduler = DaemonScheduler(storage=storage, bots=[])

    step = 0

    async def fake_sleep(duration):
        nonlocal step
        step += 1
        if step >= 2:
            raise asyncio.CancelledError()

    with (
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("datetime.datetime") as mock_dt,
    ):
        mock_dt.now.side_effect = [RuntimeError("Test error"), MagicMock()]
        await scheduler._run_loop()


@pytest.mark.asyncio
async def test_daemon_scheduler_standup_with_backups_and_git_error(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")

    fake_repo_dir = tmp_path / "repos"
    fake_repo_dir.mkdir()
    repo_a = fake_repo_dir / "repo_a"
    repo_a.mkdir()
    (repo_a / ".git").mkdir()

    fake_backup_dir = tmp_path / "workspace" / "_backups" / "postgres"
    fake_backup_dir.mkdir(parents=True)
    sql_file = fake_backup_dir / "dump_2026.sql.gz"
    sql_file.write_bytes(b"SQL_DATA" * 1000)

    mock_channel = AsyncMock()
    mock_channel.send.side_effect = Exception("Discord API error")
    mock_bot = MagicMock()
    mock_bot.is_ready.return_value = True
    mock_bot.home_channel_id = 9999
    mock_bot.get_channel.return_value = mock_channel

    scheduler = DaemonScheduler(storage=storage, bots=[mock_bot])

    with (
        patch("maulness.daemon.scheduler.config") as mock_cfg,
        patch("subprocess.run", side_effect=RuntimeError("git failure")),
    ):
        mock_cfg.repo_dir = fake_repo_dir
        mock_cfg.workspace_root = tmp_path / "workspace"

        await scheduler.run_morning_standup()

    mock_channel.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_scheduler_standup_no_active_bot(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    scheduler = DaemonScheduler(storage=storage, bots=[])

    with patch("maulness.daemon.scheduler.config") as mock_cfg:
        mock_cfg.repo_dir = tmp_path / "non_existent_repos"
        mock_cfg.workspace_root = tmp_path / "workspace"
        await scheduler.run_morning_standup()

