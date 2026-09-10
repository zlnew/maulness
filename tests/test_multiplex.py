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


def test_bot_multi_profile_channel_routing(tmp_path):
    storage = StorageManager(db_path=tmp_path / "test.db")
    prof_default = Profile(
        name="default",
        provider="acp",
        env_vars={"DISCORD_HOME_CHANNEL": "1001", "DISCORD_BOT_TOKEN": "tok1"},
    )
    prof_life = Profile(
        name="life",
        provider="acp",
        env_vars={"DISCORD_HOME_CHANNEL": "2002", "DISCORD_BOT_TOKEN": "tok1"},
    )
    prof_research = Profile(
        name="research",
        provider="acp",
        env_vars={"DISCORD_HOME_CHANNEL": "3003", "DISCORD_BOT_TOKEN": "tok1"},
    )
    bot = MaulnessBot(storage=storage, profiles=[prof_default, prof_life, prof_research])
    assert bot.profile_name == "default"
    assert len(bot.profiles) == 3

    assert bot.resolve_profile_for_channel(1001).name == "default"
    assert bot.resolve_profile_for_channel(2002).name == "life"
    assert bot.resolve_profile_for_channel(3003).name == "research"
    assert bot.resolve_profile_for_channel(9999) is None

