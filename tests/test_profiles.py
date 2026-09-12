import os
from pathlib import Path
from unittest.mock import patch
import pytest

from maulness.core.profiles import ProfileManager, Profile


def test_profile_manager_list():
    pm = ProfileManager()
    profiles = pm.list_profiles()
    names = {p.name for p in profiles}

    assert "default" in names
    assert "planner" in names
    assert "builder" in names
    assert "reviewer" in names


def test_get_specific_profile():
    pm = ProfileManager()
    planner = pm.get_profile("planner")
    assert planner.name == "planner"
    assert planner.provider == "gemini"
    assert "shipwright" in planner.effective_system_prompt().lower()
    # Check that SOUL.md is injected
    assert "Maul" in planner.effective_system_prompt()


def test_get_builder_profile():
    pm = ProfileManager()
    builder = pm.get_profile("builder")
    assert builder.name == "builder"
    assert builder.provider == "acp"
    assert "agy" in (builder.command or "")


def test_fallback_profile():
    pm = ProfileManager()
    unknown = pm.get_profile("nonexistent_role")
    assert unknown.name == "nonexistent_role"
    assert unknown.provider in ("gemini", "acp")


def test_directory_profile_loading(tmp_path):
    prof_dir = tmp_path / "custom"
    prof_dir.mkdir()
    (prof_dir / "config.yaml").write_text("identity:\n  name: custom\nagent:\n  provider: gemini\n  model: gemini-2.5-flash", encoding="utf-8")
    (prof_dir / "SOUL.md").write_text("# Custom Doctrine\nBe extremely fast.", encoding="utf-8")
    (prof_dir / ".env").write_text("GEMINI_API_KEY=test_custom_key_123\n", encoding="utf-8")

    skills_dir = prof_dir / "skills" / "my-custom-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: my-custom-skill\ndescription: Custom skill\n---\nRun custom.", encoding="utf-8")

    pm = ProfileManager(profiles_dir=tmp_path)
    profile = pm.get_profile("custom")

    assert profile.name == "custom"
    assert profile.provider == "gemini"
    assert profile.soul_content is not None
    assert "Be extremely fast." in profile.soul_content
    assert profile.get_api_key() == "test_custom_key_123"
    assert "Custom Doctrine" in profile.effective_system_prompt()
    assert "my-custom-skill" in profile.effective_system_prompt()


def test_soul_params_interpolation():
    profile = Profile(
        identity={
            "name": "orchestrator",
            "soul_params": {"role": "Vice-Captain", "captain": "Maul", "fleet": "Grand Fleet"},
        },
        agent={"provider": "acp", "command": "agy"},
        execution={"workspace": "personal"},
        soul_content="Ye be Silvers Rayleigh, {{role}} of {{captain}}'s {fleet}. Acting in {workspace}.",
    )
    prompt = profile.effective_system_prompt()
    assert "Vice-Captain of Maul's Grand Fleet" in prompt
    assert "Acting in personal" in prompt


def test_grouped_yaml_profile_loading(tmp_path):
    prof_dir = tmp_path / "grouped_bot"
    prof_dir.mkdir()
    grouped_yaml = """
identity:
  name: grouped_bot
  description: "Grouped bot test"
  system_prompt: "You are grouped."
agent:
  provider: opencode_zen
  model: gpt-4o
  base_url: https://api.opencode.ai/v1
  reasoning_effort: high
  vertex:
    enabled: false
parameters:
  temperature: 0.3
  max_tokens: 2048
execution:
  workspace: inherit
  yolo: true
resilience:
  fallbacks:
    - provider: gemini
      model: gemini-2.5-flash
      reasoning_effort: medium
"""
    (prof_dir / "config.yaml").write_text(grouped_yaml, encoding="utf-8")

    pm = ProfileManager(profiles_dir=tmp_path)
    p = pm.get_profile("grouped_bot")

    assert p.name == "grouped_bot"
    assert p.provider == "opencode_zen"
    assert p.model == "gpt-4o"
    assert p.base_url == "https://api.opencode.ai/v1"
    assert p.reasoning_effort == "high"
    assert p.temperature == 0.3
    assert p.max_tokens == 2048
    assert p.yolo is True
    assert len(p.fallbacks) == 1
    assert p.fallbacks[0]["provider"] == "gemini"
    assert p.fallbacks[0]["model"] == "gemini-2.5-flash"
    assert p.fallbacks[0]["reasoning_effort"] == "medium"
    assert p.resilience.fallbacks[0].provider == "gemini"
    assert p.resilience.fallbacks[0].model == "gemini-2.5-flash"
    assert p.resilience.fallbacks[0].reasoning_effort == "medium"


def test_standard_api_key_resolution_without_api_key_env():
    profile = Profile(
        identity={"name": "claude-dev"},
        agent={"provider": "anthropic"},
        env_vars={"ANTHROPIC_API_KEY": "sk-ant-test-123"},
    )
    assert profile.get_api_key() == "sk-ant-test-123"


def test_dual_target_memory_loading_and_injection(tmp_path):
    prof_dir = tmp_path / "memory_bot"
    prof_dir.mkdir()
    (prof_dir / "config.yaml").write_text("identity:\n  name: memory_bot\nagent:\n  provider: gemini\n  model: gemini-2.5-flash", encoding="utf-8")
    (prof_dir / "SOUL.md").write_text("# Persona\nI am helpful.", encoding="utf-8")
    (prof_dir / "USER.md").write_text("# User Style\nMaul prefers concise bullet points.", encoding="utf-8")
    (prof_dir / "MEMORY.md").write_text("# Workspace Memory\nDocker port 8000 is used by expense-tracker.", encoding="utf-8")

    pm = ProfileManager(profiles_dir=tmp_path)
    profile = pm.get_profile("memory_bot")

    assert profile.user_content == "# User Style\nMaul prefers concise bullet points."
    assert profile.memory_content == "# Workspace Memory\nDocker port 8000 is used by expense-tracker."

    effective = profile.effective_system_prompt()
    assert "User Profile & Working Style (USER.md)" in effective
    assert "Maul prefers concise bullet points" in effective
    assert "Workspace Knowledge & Lessons Learned (MEMORY.md)" in effective
    assert "Docker port 8000 is used by expense-tracker" in effective


def test_reasoning_effort_config():
    p = Profile(
        identity={"name": "thinker"},
        agent={"provider": "gemini", "model": "gemini-2.5-pro", "reasoning_effort": "high"},
    )
    assert p.reasoning_effort == "high"
    assert p.agent_cfg.reasoning_effort == "high"


def test_reasoning_effort_defaults_to_none():
    """Profile without reasoning_effort defaults to None."""
    p = Profile(identity={"name": "gemini-bot"}, agent={"provider": "gemini"})
    assert p.reasoning_effort is None


def test_fallback_reasoning_effort_propagated():
    """FallbackItem.reasoning_effort is propagated through factory into provider."""
    from maulness.core.providers.factory import get_provider_for_profile
    from maulness.core.providers.fallback import FallbackProviderChain

    profile = Profile(
        identity={"name": "smart-gemini"},
        agent={"provider": "gemini", "reasoning_effort": "high"},
        resilience={
            "fallbacks": [
                {
                    "provider": "anthropic",
                    "model": "claude-3-7-sonnet-20250219",
                    "reasoning_effort": "medium",
                }
            ]
        },
    )
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, FallbackProviderChain)
    fb_profile = provider.fallbacks[0].profile
    assert fb_profile.reasoning_effort == "medium"
    assert fb_profile.provider == "anthropic"


def test_profile_inherits_root_env(tmp_path, monkeypatch):
    """Profiles without a local .env should inherit base environment keys."""
    from maulness.config import config
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("GOOGLE_API_KEY=shared-google-key-123\n", encoding="utf-8")
    monkeypatch.setattr(config, "config_dir", config_dir)

    prof_dir = tmp_path / "profiles" / "planner_test"
    prof_dir.mkdir(parents=True)
    (prof_dir / "config.yaml").write_text("identity:\n  name: planner_test\nagent:\n  provider: gemini", encoding="utf-8")

    pm = ProfileManager(profiles_dir=tmp_path / "profiles")
    profile = pm.get_profile("planner_test")

    assert profile.get_api_key() == "shared-google-key-123"

def test_profile_properties_and_coercion():
    from maulness.core.profiles import AgentConfig, ParameterConfig, ExecutionConfig

    # Line 48: _coerce_string
    mc = AgentConfig.model_validate("gemini-2.5-pro")
    assert mc.model == "gemini-2.5-pro"

    # Lines 181, 185, 197, 214: properties
    p = Profile(
        identity={"name": "prop_agent"},
        agent={"provider": "gemini"},
        parameters={"top_p": 0.95, "top_k": 40},
        execution={"worktree": True, "sandbox": "bwrap"},
    )
    assert p.top_p == 0.95
    assert p.top_k == 40
    assert p.worktree is True
    assert p.sandbox_mode == "bwrap"


def test_profile_api_key_fallbacks_and_unknown_provider():
    # Line 254: opencode / openai_compatible
    p_opencode = Profile(
        identity={"name": "coder"},
        agent={"provider": "opencode"},
        env_vars={"OPENCODE_API_KEY": "open-key-123"},
    )
    assert p_opencode.get_api_key() == "open-key-123"

    # Line 264: unknown provider without key returns None
    p_unknown = Profile(
        identity={"name": "unknown"},
        agent={"provider": "custom_unsupported"},
    )
    assert p_unknown.get_api_key() is None


def test_profile_manager_edge_cases_and_default_env(tmp_path, monkeypatch):
    from pathlib import Path
    from maulness.config import config

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    default_dir = cfg_dir / "profiles" / "default"
    default_dir.mkdir(parents=True)
    (default_dir / ".env").write_text("DEFAULT_VAR=loaded_from_default\n", encoding="utf-8")
    monkeypatch.setattr(config, "config_dir", cfg_dir)

    pm = ProfileManager(profiles_dir=cfg_dir / "profiles")

    # Lines 370-372: fallback profile default env
    fb = pm.get_profile("new_fallback_agent")
    assert fb.env_vars.get("DEFAULT_VAR") == "loaded_from_default"

    # Line 390: resolve_workspace_for_profile fallback to resolve_repo_path
    p_ws = Profile(
        identity={"name": "ws_agent"},
        agent={"provider": "gemini"},
        execution={"workspace": "expense-tracker"},
    )
    monkeypatch.setattr(config, "resolve_repo_path", lambda target: Path("/resolved") / target)
    resolved = pm.resolve_workspace_for_profile(p_ws, tmp_path)
    assert resolved == Path("/resolved/expense-tracker")

    # Line 397: _discover_in_dir on nonexistent
    assert pm._discover_in_dir(tmp_path / "nonexistent") == []

    # Line 413: _load_directory_profile non-dict config
    bad_dir = tmp_path / "profiles" / "bad_profile"
    bad_dir.mkdir(parents=True)
    (bad_dir / "config.yaml").write_text("just a string\n", encoding="utf-8")
    assert pm._load_directory_profile(bad_dir, bad_dir / "config.yaml") is None

    # Lines 446-448: _load_directory_profile default_env
    good_dir = tmp_path / "profiles" / "good_profile"
    good_dir.mkdir(parents=True)
    (good_dir / "config.yaml").write_text("agent:\n  provider: gemini\n", encoding="utf-8")
    p_good = pm._load_directory_profile(good_dir, good_dir / "config.yaml")
    assert p_good.env_vars.get("DEFAULT_VAR") == "loaded_from_default"

    # Lines 467-469: _load_directory_profile exception
    err_dir = tmp_path / "profiles" / "err_profile"
    err_dir.mkdir(parents=True)
    (err_dir / "config.yaml").write_text(": invalid yaml [\n", encoding="utf-8")
    assert pm._load_directory_profile(err_dir, err_dir / "config.yaml") is None

def test_profile_all_properties_and_vertex_coverage(tmp_path: Path):
    from maulness.core.profiles import VertexConfig

    # 1. Invalid ACP whitespace command (line 108)
    with pytest.raises(ValueError, match="has no 'command'"):
        Profile(identity={"name": "bad_acp"}, agent={"provider": "acp", "command": "   "})

    # 2. Vertex properties (lines 162, 166, 170)
    p_vert = Profile(
        identity={"name": "vert_agent", "description": "A vertex agent"},
        agent={
            "provider": "gemini",
            "vertex": {"enabled": True, "project": "my-gcp-project", "location": "europe-west1"},
        },
        execution={"max_tool_turns": 15, "max_relays": 3},
    )
    assert p_vert.description == "A vertex agent"
    assert p_vert.vertex is True
    assert p_vert.project == "my-gcp-project"
    assert p_vert.location == "europe-west1"
    assert p_vert.max_tool_turns == 15
    assert p_vert.max_relays == 3

    # 3. Default sandbox_mode, max_tool_turns, max_relays when None (lines 204, 210, 216)
    from maulness.config import config
    p_def = Profile(identity={"name": "def_agent"}, agent={"provider": "gemini"})
    assert p_def.sandbox_mode == "auto"
    assert p_def.max_tool_turns == config.max_tool_turns
    assert p_def.max_relays == config.max_relays

    # 4. get_api_key from custom api_key_env and system env (lines 239, 251)
    p_custom_env = Profile(
        identity={"name": "custom_env_agent"},
        agent={"provider": "anthropic", "api_key_env": "MY_SPECIAL_ANTHROPIC_KEY"},
    )
    with patch.dict(os.environ, {"MY_SPECIAL_ANTHROPIC_KEY": "special-secret-key-123"}):
        assert p_custom_env.get_api_key() == "special-secret-key-123"

    # 5. Generic fallback for openai_compatible (line 255)
    p_comp = Profile(identity={"name": "comp"}, agent={"provider": "openai_compatible"})
    with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-generic-key"}):
        assert p_comp.get_api_key() == "openai-generic-key"

    # 6. Generic fallback for gemini (line 257) via config.gemini_api_key
    p_gem = Profile(identity={"name": "gem"}, agent={"provider": "gemini"})
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "gemini_api_key", "cfg-gemini-key"),
    ):
        assert p_gem.get_api_key() == "cfg-gemini-key"

    # 7. get_rule_engine (lines 269-271)
    engine = p_vert.get_rule_engine()
    assert engine is not None

    # 8. resolve_workspace_for_profile relative folder exists (lines 389, 391)
    pm = ProfileManager(profiles_dir=tmp_path)
    sub_ws = tmp_path / "subfolder"
    sub_ws.mkdir()
    p_sub = Profile(identity={"name": "sub"}, agent={"provider": "gemini"}, execution={"workspace": "subfolder"})
    assert pm.resolve_workspace_for_profile(p_sub, tmp_path) == sub_ws

    # workspace is inherit or None
    p_inherit = Profile(identity={"name": "inh"}, agent={"provider": "gemini"}, execution={"workspace": "inherit"})
    assert pm.resolve_workspace_for_profile(p_inherit, tmp_path) == tmp_path
