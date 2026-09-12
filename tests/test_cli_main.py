from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from maulness.cli import main
from maulness.core.models import (
    SessionRecord,
    TaskMode,
    TaskRecord,
    TaskStatus,
)

runner = CliRunner()


def test_cli_main_entrypoint():
    with patch("maulness.cli.main.app") as mock_app:
        main.main()
        mock_app.assert_called_once()


def test_cli_status(tmp_path: Path):
    with (
        patch("subprocess.run") as mock_run,
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.ProfileManager") as mock_pm_cls,
    ):
        mock_run.side_effect = [
            MagicMock(stdout="active", returncode=0),  # systemctl
            MagicMock(returncode=0),  # which agy
        ]
        mock_cfg.gateway_multiplex_profiles = True
        mock_cfg.gateway_multiplex_profile_allowlist = ["default", "builder"]
        mock_cfg.agy_cmd = ["agy"]
        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)
        mock_cfg.db_path = tmp_path / "test.db"
        mock_cfg.db_path.touch()

        mock_prof = MagicMock()
        mock_prof.name = "builder"
        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [mock_prof]
        mock_pm_cls.return_value = mock_pm

        result = runner.invoke(main.app, ["status"])
        assert result.exit_code == 0
        assert "Maulness System Status" in result.output


def test_cli_status_standby(tmp_path: Path):
    with (
        patch("subprocess.run") as mock_run,
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.ProfileManager") as mock_pm_cls,
    ):
        mock_run.side_effect = [
            Exception("systemd unavailable"),
            MagicMock(returncode=1),  # which agy missing
        ]
        mock_cfg.gateway_multiplex_profiles = False
        mock_cfg.has_discord = False
        mock_cfg.agy_cmd = ["agy"]
        mock_cfg.get_current_workspace.return_value = ("root", tmp_path)
        mock_cfg.db_path = tmp_path / "uninit.db"

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = []
        mock_pm_cls.return_value = mock_pm

        result = runner.invoke(main.app, ["status"])
        assert result.exit_code == 0
        assert "Standby" in result.output


def test_cli_run(tmp_path: Path):
    with (
        patch("maulness.cli.main.TaskRunner") as mock_runner_cls,
        patch("maulness.cli.main.config") as mock_cfg,
    ):
        mock_runner = MagicMock()
        mock_runner.run_direct = AsyncMock()
        mock_runner_cls.return_value = mock_runner

        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)
        mock_cfg.resolve_repo_path.return_value = tmp_path

        result = runner.invoke(
            main.app,
            [
                "run",
                "Fix bug in handler",
                "-r",
                "test-repo",
                "-p",
                "builder",
                "-w",
                "-y",
            ],
        )
        assert result.exit_code == 0
        mock_runner.run_direct.assert_awaited_once()


def test_cli_run_afk_alias(tmp_path: Path):
    with (
        patch("maulness.cli.main.TaskRunner") as mock_runner_cls,
        patch("maulness.cli.main.config") as mock_cfg,
    ):
        mock_runner = MagicMock()
        mock_runner.run_direct = AsyncMock()
        mock_runner_cls.return_value = mock_runner

        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)
        mock_cfg.resolve_repo_path.return_value = tmp_path

        result = runner.invoke(
            main.app,
            ["run", "Refactor module", "-r", "test-repo", "--afk"],
        )
        assert result.exit_code == 0
        mock_runner.run_direct.assert_awaited_once()
        call_kwargs = mock_runner.run_direct.call_args[1]
        assert call_kwargs["yolo"] is True


def test_cli_chat_plain_mode(tmp_path: Path):
    inputs = [
        "",  # Empty prompt exercises line 203
        "/yolo",
        "/worktree",
        "/diff",
        "/profile",
        "/profile reviewer",
        "!git status",
        "/pipeline standard fix auth",
        "/task generic build task",  # Exercises line 277
        "Explain architecture",
        "/clear",
        "/exit",
    ]

    with (
        patch("maulness.cli.main.TaskRunner") as mock_runner_cls,
        patch("maulness.cli.main.PipelineOrchestrator") as mock_orch_cls,
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.console.input", side_effect=inputs),
        patch("subprocess.run") as mock_run,
    ):
        mock_runner = MagicMock()

        async def fake_run_direct(**kwargs):
            if "on_init" in kwargs and kwargs["on_init"]:
                await kwargs["on_init"]("conv_999")

        mock_runner.run_direct = AsyncMock(side_effect=fake_run_direct)
        mock_runner_cls.return_value = mock_runner

        mock_orch = MagicMock()
        mock_orch.run_pipeline = AsyncMock()
        mock_orch_cls.return_value = mock_orch

        mock_run.return_value = MagicMock(returncode=0, stdout="diff output", stderr="")
        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)

        result = runner.invoke(main.app, ["chat", "--plain"])
        assert result.exit_code == 0
        assert mock_orch.run_pipeline.call_count == 2
        assert mock_runner.run_direct.call_count == 1


def test_cli_tasks(tmp_path: Path):
    with patch("maulness.cli.main.StorageManager") as mock_sm_cls:
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()

        task1 = TaskRecord(
            id="t1",
            title="Task One",
            repo_name="horizonx",
            workspace_path="/tmp",
            mode=TaskMode.DIRECT,
            status=TaskStatus.DONE,
            origin_platform="discord",
            origin_channel_id="111",
            origin_thread_id="222",
        )
        task2 = TaskRecord(
            id="t2",
            title="Task Two",
            repo_name="expense-tracker",
            workspace_path="/tmp",
            mode=TaskMode.MULTI,
            status=TaskStatus.FAILED,
        )

        mock_sm.list_tasks = AsyncMock(return_value=[task1, task2])
        mock_sm.get_task = AsyncMock(return_value=task1)
        mock_sm.update_task_status = AsyncMock()
        mock_sm.list_sessions = AsyncMock(return_value=[])
        mock_sm_cls.return_value = mock_sm

        # 1. list
        res = runner.invoke(main.app, ["task", "list"])
        assert res.exit_code == 0
        assert "t1" in res.output

        # 2. show
        res = runner.invoke(main.app, ["task", "show", "t1"])
        assert res.exit_code == 0
        assert "Task Details" in res.output

        # 3. abort
        res = runner.invoke(main.app, ["task", "abort", "t1"])
        assert res.exit_code == 0
        assert "Aborted task 't1'" in res.output


def test_cli_tasks_empty_and_not_found():
    with patch("maulness.cli.main.StorageManager") as mock_sm_cls:
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm.list_tasks = AsyncMock(return_value=[])
        mock_sm.get_task = AsyncMock(return_value=None)
        mock_sm_cls.return_value = mock_sm

        res = runner.invoke(main.app, ["task", "list"])
        assert "No tasks recorded yet" in res.output

        res = runner.invoke(main.app, ["task", "show", "nonexistent"])
        assert "not found" in res.output

        res = runner.invoke(main.app, ["task", "abort", "nonexistent"])
        assert "not found" in res.output


def test_cli_sessions(tmp_path: Path):
    with patch("maulness.cli.main.StorageManager") as mock_sm_cls:
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()

        sess = SessionRecord(
            id="sess_123",
            task_id="t1",
            profile="builder",
            engine="gemini",
            status="active",
            acp_session_id="acp_999",
            pid=1234,
        )
        mock_sm.list_sessions = AsyncMock(return_value=[sess])
        mock_sm.get_session = AsyncMock(return_value=sess)
        mock_sm_cls.return_value = mock_sm

        # list
        res = runner.invoke(main.app, ["session", "list"])
        assert res.exit_code == 0
        assert "sess_123" in res.output

        # show
        res = runner.invoke(main.app, ["session", "show", "sess_123"])
        assert res.exit_code == 0
        assert "sess_123" in res.output


def test_cli_sessions_empty_and_not_found():
    with patch("maulness.cli.main.StorageManager") as mock_sm_cls:
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm.list_sessions = AsyncMock(return_value=[])
        mock_sm.get_session = AsyncMock(return_value=None)
        mock_sm_cls.return_value = mock_sm

        res = runner.invoke(main.app, ["session", "list"])
        assert "No sessions recorded yet" in res.output

        res = runner.invoke(main.app, ["session", "show", "missing"])
        assert "not found" in res.output


def test_cli_profiles():
    with patch("maulness.cli.main.ProfileManager") as mock_pm_cls:
        mock_pm = MagicMock()
        mock_prof = MagicMock()
        mock_prof.name = "builder"
        mock_prof.provider = "gemini"
        mock_prof.model = "gemini-2.5-pro"
        mock_prof.command = None
        mock_prof.description = "Autonomous code builder"
        mock_prof.soul_content = "Build solid code"
        mock_prof.skills_dir = None
        mock_prof.model_dump.return_value = {"identity": {"name": "builder"}}

        mock_pm.list_profiles.return_value = [mock_prof]
        mock_pm.get_profile.return_value = mock_prof
        mock_pm_cls.return_value = mock_pm

        res = runner.invoke(main.app, ["profile", "list"])
        assert res.exit_code == 0
        assert "builder" in res.output

        res = runner.invoke(main.app, ["profile", "show", "builder"])
        assert res.exit_code == 0
        assert "builder" in res.output


def test_cli_daemon_commands():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        res = runner.invoke(main.app, ["daemon", "status"])
        assert res.exit_code == 0

        res = runner.invoke(main.app, ["daemon", "start"])
        assert "Started maulness.service successfully" in res.output

        res = runner.invoke(main.app, ["daemon", "stop"])
        assert "Stopped maulness.service" in res.output

        res = runner.invoke(main.app, ["daemon", "restart"])
        assert "Restarted maulness.service" in res.output

        res = runner.invoke(main.app, ["daemon", "logs", "--no-follow"])
        assert res.exit_code == 0

        # Test failure paths
        mock_run.return_value = MagicMock(returncode=1)
        res = runner.invoke(main.app, ["daemon", "start"])
        assert "Failed to start" in res.output

        res = runner.invoke(main.app, ["daemon", "stop"])
        assert "Failed to stop" in res.output

        res = runner.invoke(main.app, ["daemon", "restart"])
        assert "Failed to restart" in res.output


def test_cli_discord_run():
    with patch(
        "maulness.daemon.service.main", new_callable=AsyncMock
    ) as mock_service_main:
        res = runner.invoke(main.app, ["discord", "run", "-p", "builder"])
        assert res.exit_code == 0
        mock_service_main.assert_awaited_once_with(profile_filter="builder")


def test_cli_soul_memory_user(tmp_path: Path):
    with (
        patch("maulness.cli.main.config") as mock_cfg,
        patch("subprocess.run") as mock_run,
        patch("maulness.cli.main.ProfileManager") as mock_pm_cls,
    ):
        mock_cfg.config_dir = tmp_path
        mock_run.return_value = MagicMock(returncode=0)

        mock_pm = MagicMock()
        mock_prof = MagicMock()
        mock_prof.name = "test_prof"
        mock_prof.soul_content = "Prof Soul"
        mock_prof.memory_content = "Prof Memory"
        mock_prof.user_content = "Prof User"
        mock_pm.get_profile.return_value = mock_prof
        mock_pm_cls.return_value = mock_pm

        # Soul show & edit
        res = runner.invoke(main.app, ["soul", "show", "-p", "test_prof"])
        assert "Prof Soul" in res.output
        res = runner.invoke(main.app, ["soul", "show"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["soul", "edit", "-p", "test_prof"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["soul", "edit"])
        assert res.exit_code == 0

        # Memory show & edit
        res = runner.invoke(main.app, ["memory", "show", "-p", "test_prof"])
        assert "Prof Memory" in res.output
        res = runner.invoke(main.app, ["memory", "show"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["memory", "edit", "-p", "test_prof"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["memory", "edit"])
        assert res.exit_code == 0

        # User show & edit
        res = runner.invoke(main.app, ["user", "show", "-p", "test_prof"])
        assert "Prof User" in res.output
        res = runner.invoke(main.app, ["user", "show"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["user", "edit", "-p", "test_prof"])
        assert res.exit_code == 0
        res = runner.invoke(main.app, ["user", "edit"])
        assert res.exit_code == 0


def test_cli_db_commands():
    with patch("maulness.cli.main.StorageManager") as mock_sm_cls:
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        mock_sm.reset_stale_sessions = AsyncMock(
            return_value={"sessions_closed": 2, "tasks_failed": 1}
        )
        mock_sm_cls.return_value = mock_sm

        res = runner.invoke(main.app, ["db", "migrate"])
        assert res.exit_code == 0
        assert "Applied database DDL schemas" in res.output

        res = runner.invoke(main.app, ["db", "reset"])
        assert res.exit_code == 0
        assert "closed 2 sessions" in res.output


def test_cli_status_single_discord(tmp_path: Path):
    with (
        patch("subprocess.run") as mock_run,
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.ProfileManager") as mock_pm_cls,
    ):
        mock_run.return_value = MagicMock(stdout="active", returncode=0)
        mock_cfg.gateway_multiplex_profiles = False
        mock_cfg.has_discord = True
        mock_cfg.agy_cmd = ["agy"]
        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)
        mock_cfg.db_path = tmp_path / "test.db"

        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = []
        mock_pm_cls.return_value = mock_pm

        res = runner.invoke(main.app, ["status"])
        assert res.exit_code == 0
        assert "Configured" in res.output


def test_cli_chat_invocation():
    with patch("maulness.cli.main.run_plain_chat") as mock_plain:
        main.chat(profile="builder")
        mock_plain.assert_called_once_with(initial_profile="builder")


def test_cli_chat_keyboard_interrupt(tmp_path: Path):
    with (
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.console.input", side_effect=KeyboardInterrupt),
    ):
        mock_cfg.get_current_workspace.return_value = ("test-repo", tmp_path)
        res = runner.invoke(main.app, ["chat", "--plain"])
        assert res.exit_code == 0
        assert "Session terminated" in res.output


def test_cli_task_abort_killing_sessions(tmp_path: Path):
    with (
        patch("maulness.cli.main.StorageManager") as mock_sm_cls,
        patch("os.kill") as mock_kill,
    ):
        mock_sm = MagicMock()
        mock_sm.initialize = AsyncMock()
        task = TaskRecord(
            id="t_kill",
            title="Task to kill",
            repo_name="horizonx",
            workspace_path="/tmp",
            mode=TaskMode.DIRECT,
            status=TaskStatus.BUILDING,
        )
        sess = SessionRecord(
            id="s_kill",
            task_id="t_kill",
            profile="builder",
            engine="gemini",
            status="active",
            pid=99999,
        )
        mock_sm.get_task = AsyncMock(return_value=task)
        mock_sm.update_task_status = AsyncMock()
        mock_sm.list_sessions = AsyncMock(return_value=[sess])
        mock_sm.update_session_status = AsyncMock()
        mock_sm_cls.return_value = mock_sm

        mock_kill.side_effect = ProcessLookupError()

        res = runner.invoke(main.app, ["task", "abort", "t_kill"])
        assert res.exit_code == 0
        assert "closed 1 sessions" in res.output
        mock_kill.assert_called_once_with(99999, 9)


def test_cli_daemon_logs_with_follow():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        res = runner.invoke(main.app, ["daemon", "logs", "--follow", "--lines", "25"])
        assert res.exit_code == 0
        called_cmd = mock_run.call_args[0][0]
        assert "-f" in called_cmd
        assert "-n25" in called_cmd


def test_cli_soul_memory_user_missing_files(tmp_path: Path):
    empty_dir = tmp_path / "empty_cfg"
    empty_dir.mkdir()

    with (
        patch("maulness.cli.main.config") as mock_cfg,
        patch("maulness.cli.main.ProfileManager") as mock_pm_cls,
        patch.object(Path, "exists", return_value=False),
    ):
        mock_cfg.config_dir = empty_dir

        mock_pm = MagicMock()
        mock_prof = MagicMock()
        mock_prof.name = "empty_prof"
        mock_prof.soul_content = None
        mock_prof.memory_content = None
        mock_prof.user_content = None
        mock_pm.get_profile.return_value = mock_prof
        mock_pm_cls.return_value = mock_pm

        res = runner.invoke(main.app, ["soul", "show", "-p", "empty_prof"])
        assert "No profile-specific SOUL.md found" in res.output

        res = runner.invoke(main.app, ["soul", "show"])
        assert "No SOUL.md found" in res.output

        res = runner.invoke(main.app, ["memory", "show", "-p", "empty_prof"])
        assert "No profile-specific MEMORY.md found" in res.output

        res = runner.invoke(main.app, ["memory", "show"])
        assert "No MEMORY.md found" in res.output

        res = runner.invoke(main.app, ["user", "show", "-p", "empty_prof"])
        assert "No profile-specific USER.md found" in res.output

        res = runner.invoke(main.app, ["user", "show"])
        assert "No USER.md found" in res.output


def test_cli_dunder_main():

    with (
        patch("maulness.cli.main.app") as mock_app,
        patch.object(main, "__name__", "__main__"),
    ):
        main.main()
        mock_app.assert_called_once()


def test_cli_soul_memory_user_root_exists_and_fallback_edit(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "SOUL.md").write_text("Root SOUL content")
    (cfg_dir / "MEMORY.md").write_text("Root MEMORY content")
    (cfg_dir / "USER.md").write_text("Root USER content")

    with (
        patch("maulness.cli.main.config") as mock_cfg,
        patch("subprocess.run") as mock_run,
    ):
        mock_cfg.config_dir = cfg_dir

        res = runner.invoke(main.app, ["soul", "show"])
        assert "Root SOUL content" in res.output

        res = runner.invoke(main.app, ["memory", "show"])
        assert "Root MEMORY content" in res.output

        res = runner.invoke(main.app, ["user", "show"])
        assert "Root USER content" in res.output

    # Test edit fallback when target does not exist and templates do not exist
    empty_cfg = tmp_path / "empty_cfg"
    empty_cfg.mkdir()
    with (
        patch("maulness.cli.main.config") as mock_cfg,
        patch("subprocess.run") as mock_run,
        patch.object(Path, "exists", return_value=False),
    ):
        mock_cfg.config_dir = empty_cfg

        res = runner.invoke(main.app, ["soul", "edit"])
        assert res.exit_code == 0

        res = runner.invoke(main.app, ["memory", "edit"])
        assert res.exit_code == 0

        res = runner.invoke(main.app, ["user", "edit"])
        assert res.exit_code == 0


def test_cli_dunder_main_module():
    import runpy

    with patch("sys.argv", ["maulness", "--help"]):
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("maulness.cli.main", run_name="__main__")
        assert exc.value.code == 0
