import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
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
