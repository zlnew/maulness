import os
from pathlib import Path
from unittest.mock import patch
import yaml

from maulness import config as config_module
from maulness.config import Config


def test_config_defaults(tmp_path: Path):
    empty_cfg_dir = tmp_path / "empty_cfg"
    empty_cfg_dir.mkdir()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("maulness.config.CONFIG_DIR", empty_cfg_dir),
    ):
        cfg = Config()
        assert cfg.max_tool_turns == 50
        assert cfg.max_relays == 5
        assert cfg.stream_idle_timeout_seconds == 180.0
        assert cfg.sandbox_mode == "auto"
        assert not cfg.gateway_multiplex_profiles
        assert cfg.gateway_multiplex_profile_allowlist == []
        assert cfg.gateway_profile_routes == []
        assert cfg.agy_cmd == ["agy"]
        assert not cfg.has_discord


def test_config_has_discord(tmp_path: Path):
    cfg = Config()
    cfg.discord_bot_token = "token123"
    cfg.owner_discord_id = 456789
    assert cfg.has_discord is True

    cfg.discord_bot_token = None
    assert cfg.has_discord is False


def test_config_agy_cmd_custom():
    with patch.dict(os.environ, {"AGY_CMD": "custom-agy --flag"}):
        cfg = Config()
        assert cfg.agy_cmd == ["custom-agy", "--flag"]


def test_config_load_yaml_full(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_yaml = cfg_dir / "config.yaml"

    content = {
        "workspace": {"root": str(tmp_path / "ws")},
        "storage": {"db_path": str(tmp_path / "custom.db")},
        "execution": {
            "stream_idle_timeout_seconds": 45.0,
            "max_tool_turns": 10,
            "max_relays": 2,
            "sandbox": "none",
            "default_policy": "allow",
            "rules": {"commands": {"deny": ["sudo"]}},
        },
        "gateway": {
            "multiplex_profiles": True,
            "multiplex_profile_allowlist": ["builder", "planner"],
            "profile_routes": [{"prefix": "plan", "profile": "planner"}],
        },
    }
    cfg_yaml.write_text(yaml.safe_dump(content), encoding="utf-8")

    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.workspace_root == (tmp_path / "ws").resolve()
        assert cfg.repo_dir == (tmp_path / "ws" / "repo").resolve()
        assert cfg.db_path == (tmp_path / "custom.db").resolve()
        assert cfg.stream_idle_timeout_seconds == 45.0
        assert cfg.max_tool_turns == 10
        assert cfg.max_relays == 2
        assert cfg.sandbox_mode == "none"
        assert cfg.gateway_multiplex_profiles is True
        assert cfg.gateway_multiplex_profile_allowlist == ["builder", "planner"]
        assert len(cfg.gateway_profile_routes) == 1
        assert cfg.global_rules.commands.deny == ["sudo"]


def test_config_load_yaml_sandbox_variants(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_yaml = cfg_dir / "config.yaml"

    # Test bwrap
    cfg_yaml.write_text("execution:\n  sandbox: bwrap\n")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.sandbox_mode == "bwrap"

    # Test auto
    cfg_yaml.write_text("execution:\n  sandbox: '1'\n")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.sandbox_mode == "auto"

    # Test sandbox_mode explicit key
    cfg_yaml.write_text("execution:\n  sandbox_mode: custom_mode\n")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.sandbox_mode == "custom_mode"


def test_config_load_yaml_corrupt_or_non_dict(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_yaml = cfg_dir / "config.yaml"

    # Non-dict yaml
    cfg_yaml.write_text("just a string")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.sandbox_mode == "auto"

    # Malformed yaml
    cfg_yaml.write_text("invalid: [broken yaml")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.sandbox_mode == "auto"


def test_config_load_yaml_invalid_rules_payload(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_yaml = cfg_dir / "config.yaml"

    # Invalid rules (commands is not a dict)
    content = {
        "execution": {
            "rules": {"commands": "invalid_string_not_dict"},
        }
    }
    cfg_yaml.write_text(yaml.safe_dump(content), encoding="utf-8")
    with patch("maulness.config.CONFIG_DIR", cfg_dir):
        cfg = Config()
        assert cfg.global_rules.commands.deny == []


def test_config_env_multiplex_override(tmp_path: Path):
    with patch.dict(os.environ, {"GATEWAY_MULTIPLEX_PROFILES": "yes"}):
        cfg = Config()
        assert cfg.gateway_multiplex_profiles is True

    with patch.dict(os.environ, {"GATEWAY_MULTIPLEX_PROFILES": "0"}):
        cfg = Config()
        assert cfg.gateway_multiplex_profiles is False


def test_config_get_current_workspace(tmp_path: Path):
    ws_root = tmp_path / "personal"
    repo_dir = ws_root / "repo"
    my_repo = repo_dir / "my-service" / "src"
    my_repo.mkdir(parents=True)

    cfg = Config()
    cfg.workspace_root = ws_root
    cfg.repo_dir = repo_dir

    # Inside repo_dir
    name, path = cfg.get_current_workspace(custom_path=my_repo)
    assert name == "my-service"
    assert path == repo_dir / "my-service"

    # Outside repo_dir
    external = tmp_path / "other"
    external.mkdir()
    name, path = cfg.get_current_workspace(custom_path=external)
    assert name == "other"
    assert path == external


def test_config_resolve_repo_path(tmp_path: Path):
    ws_root = tmp_path / "workspace"
    repo_dir = ws_root / "repo"
    repo_dir.mkdir(parents=True)

    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()

    repo_service = repo_dir / "my-api"
    repo_service.mkdir()

    ws_tool = ws_root / "scripts"
    ws_tool.mkdir()

    cfg = Config()
    cfg.workspace_root = ws_root
    cfg.repo_dir = repo_dir

    # 1. Existing absolute or direct directory
    assert cfg.resolve_repo_path(direct_dir) == direct_dir.resolve()

    # 2. In repo_dir
    assert cfg.resolve_repo_path("my-api") == repo_service.resolve()

    # 3. In workspace_root
    assert cfg.resolve_repo_path("scripts") == ws_tool.resolve()

    # 4. Fallback when not found returns cwd
    missing = cfg.resolve_repo_path("nonexistent")
    assert missing == Path.cwd().resolve()


def test_config_env_file_loading_precedence():
    def fake_exists_dotenv(self):
        return self == config_module.DOTENV_FILE

    def fake_exists_env(self):
        return self == config_module.ENV_FILE

    def fake_exists_local(self):
        return self == config_module.LOCAL_ENV_FILE

    # 1. DOTENV_FILE exists
    with (
        patch.object(Path, "exists", new=fake_exists_dotenv),
        patch("maulness.config.load_dotenv") as mock_load,
    ):
        config_module.load_env_files()
        mock_load.assert_called_with(config_module.DOTENV_FILE)

    # 2. ENV_FILE exists
    with (
        patch.object(Path, "exists", new=fake_exists_env),
        patch("maulness.config.load_dotenv") as mock_load,
    ):
        config_module.load_env_files()
        mock_load.assert_called_with(config_module.ENV_FILE)

    # 3. LOCAL_ENV_FILE exists
    with (
        patch.object(Path, "exists", new=fake_exists_local),
        patch("maulness.config.load_dotenv") as mock_load,
    ):
        config_module.load_env_files()
        mock_load.assert_called_with(config_module.LOCAL_ENV_FILE)

    # 4. None exists
    with (
        patch.object(Path, "exists", return_value=False),
        patch("maulness.config.load_dotenv") as mock_load,
    ):
        config_module.load_env_files()
        mock_load.assert_not_called()
