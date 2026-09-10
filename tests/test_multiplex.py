import pytest
from maulness.config import Config
from maulness.core.profiles import Profile
from maulness.discord.bot import MaulnessBot
from maulness.storage.db import StorageManager


def test_config_gateway_multiplex_parsing(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
gateway:
  multiplex_profiles: true
  multiplex_profile_allowlist:
    - personal-orchestrator
    - office-orchestrator
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("maulness.config.CONFIG_DIR", tmp_path)
    cfg = Config()
    assert cfg.gateway_multiplex_profiles is True
    assert cfg.gateway_multiplex_profile_allowlist == [
        "personal-orchestrator",
        "office-orchestrator",
    ]


def test_bot_profile_binding(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    prof = Profile(
        name="office-bot",
        provider="acp",
        env_vars={
            "DISCORD_BOT_TOKEN": "test_office_token_xyz",
            "OWNER_DISCORD_ID": "123456789",
            "DISCORD_GUILD_ID": "987654321",
        },
    )
    bot = MaulnessBot(storage=storage, profile=prof)
    assert bot.profile_name == "office-bot"
    assert bot.bound_profile.name == "office-bot"
    assert bot.owner_id == 123456789
    assert bot.guild_id == 987654321
