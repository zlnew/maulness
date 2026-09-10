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
