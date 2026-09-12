import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maulness.daemon import service


async def fake_start(token):
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_daemon_service_profile_filter_with_token(tmp_path: Path):
    mock_profile = MagicMock()
    mock_profile.name = "custom_profile"
    mock_profile.env_vars = {"DISCORD_BOT_TOKEN": "token_123"}

    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.discord.bot.MaulnessBot") as mock_bot_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [mock_profile]
        mock_pm.get_profile.return_value = mock_profile
        mock_pm_cls.return_value = mock_pm

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock(side_effect=fake_start)
        mock_bot.close = AsyncMock()
        mock_bot.profile_name = "custom_profile"
        mock_bot_cls.return_value = mock_bot

        mock_sched = MagicMock()
        mock_sched_cls.return_value = mock_sched

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main(profile_filter="custom_profile")

        mock_sm.initialize.assert_awaited_once()
        mock_pm.get_profile.assert_called_with("custom_profile")
        mock_bot.start.assert_called_with("token_123")
        mock_sched.start.assert_called_once()
        mock_sched.stop.assert_called_once()
        mock_bot.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_service_profile_filter_without_token(tmp_path: Path):
    mock_profile = MagicMock()
    mock_profile.name = "no_token_profile"
    mock_profile.env_vars = {}

    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.discord_bot_token = None
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [mock_profile]
        mock_pm.get_profile.return_value = mock_profile
        mock_pm_cls.return_value = mock_pm

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main(profile_filter="no_token_profile")

        mock_sm.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_daemon_service_multiplex_mode(tmp_path: Path):
    p1 = MagicMock()
    p1.name = "p1"
    p1.env_vars = {"DISCORD_BOT_TOKEN": "token_a"}

    p2 = MagicMock()
    p2.name = "p2"
    p2.env_vars = {"DISCORD_BOT_TOKEN": "token_a"}

    p3 = MagicMock()
    p3.name = "p3"
    p3.env_vars = {"DISCORD_BOT_TOKEN": "token_b"}

    p4 = MagicMock()
    p4.name = "p4"
    p4.env_vars = {}

    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.discord.bot.MaulnessBot") as mock_bot_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = True
        mock_cfg.gateway_multiplex_profile_allowlist = ["p1", "p2", "p3", "p4"]

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [p1, p2, p3, p4]
        mock_pm_cls.return_value = mock_pm

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock(side_effect=fake_start)
        mock_bot.close = AsyncMock()
        mock_bot.profiles = [p1, p2]
        mock_bot_cls.return_value = mock_bot

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main()

        assert mock_bot.start.call_count == 2
        mock_bot.close.assert_awaited()


@pytest.mark.asyncio
async def test_daemon_service_multiplex_default_token_fallback(tmp_path: Path):
    p_default = MagicMock()
    p_default.name = "default"
    p_default.env_vars = {}

    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.discord.bot.MaulnessBot") as mock_bot_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = True
        mock_cfg.gateway_multiplex_profile_allowlist = []
        mock_cfg.discord_bot_token = "fallback_token"

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [p_default]
        mock_pm_cls.return_value = mock_pm

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock(side_effect=fake_start)
        mock_bot.close = AsyncMock()
        mock_bot.profile_name = "default"
        del mock_bot.profiles
        mock_bot_cls.return_value = mock_bot

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main()

        mock_bot.start.assert_called_with("fallback_token")
        mock_bot.close.assert_awaited()


@pytest.mark.asyncio
async def test_daemon_service_multiplex_no_tokens(tmp_path: Path):
    p = MagicMock()
    p.name = "no_token"
    p.env_vars = {}

    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = True
        mock_cfg.gateway_multiplex_profile_allowlist = []
        mock_cfg.discord_bot_token = None

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [p]
        mock_pm_cls.return_value = mock_pm

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main()


@pytest.mark.asyncio
async def test_daemon_service_default_mode(tmp_path: Path):
    real_event = asyncio.Event()

    with (
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.discord.bot.MaulnessBot") as mock_bot_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = False
        mock_cfg.has_discord = True
        mock_cfg.discord_bot_token = "default_token"

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = []
        mock_pm_cls.return_value = mock_pm

        mock_bot = MagicMock()
        mock_bot.start = AsyncMock(side_effect=fake_start)
        mock_bot.close = AsyncMock()
        mock_bot.profile_name = "default"
        del mock_bot.profiles
        mock_bot_cls.return_value = mock_bot

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main()

        mock_bot.start.assert_called_with("default_token")
        mock_bot.close.assert_awaited()


@pytest.mark.asyncio
async def test_daemon_service_signal_handling(tmp_path: Path):
    real_event = asyncio.Event()
    signal_callbacks = []

    mock_loop = MagicMock()

    def fake_add_signal_handler(sig, callback):
        signal_callbacks.append(callback)

    mock_loop.add_signal_handler = fake_add_signal_handler

    with (
        patch("maulness.daemon.service.asyncio.get_running_loop", return_value=mock_loop),
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = False
        mock_cfg.has_discord = False

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = []
        mock_pm_cls.return_value = mock_pm

        async def trigger_signal():
            await asyncio.sleep(0.02)
            for cb in signal_callbacks:
                cb()

        asyncio.create_task(trigger_signal())
        await service.main()

        assert real_event.is_set()


@pytest.mark.asyncio
async def test_daemon_service_signal_not_implemented(tmp_path: Path):
    real_event = asyncio.Event()
    mock_loop = MagicMock()
    mock_loop.add_signal_handler.side_effect = NotImplementedError

    with (
        patch("maulness.daemon.service.asyncio.get_running_loop", return_value=mock_loop),
        patch("maulness.daemon.service.StorageManager") as mock_sm_cls,
        patch("maulness.core.profiles.ProfileManager") as mock_pm_cls,
        patch("maulness.daemon.scheduler.DaemonScheduler") as mock_sched_cls,
        patch("maulness.daemon.service.config") as mock_cfg,
        patch("maulness.daemon.service.asyncio.Event", return_value=real_event),
    ):
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.gateway_multiplex_profiles = False
        mock_cfg.has_discord = False

        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = []
        mock_pm_cls.return_value = mock_pm

        async def stop_soon():
            await asyncio.sleep(0.02)
            real_event.set()

        asyncio.create_task(stop_soon())
        await service.main()


def test_daemon_service_main_entrypoint():
    import runpy
    import sys

    def fake_asyncio_run(coro):
        coro.close()

    with (
        patch.object(sys, "argv", ["service.py", "--profile", "test_p"]),
        patch("asyncio.run", side_effect=fake_asyncio_run) as mock_asyncio_run,
    ):
        runpy.run_path(str(Path(service.__file__)), run_name="__main__")
        mock_asyncio_run.assert_called_once()
