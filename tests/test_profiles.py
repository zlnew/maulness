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
    (prof_dir / "config.yaml").write_text("name: custom\nprovider: gemini\nmodel: gemini-2.5-flash", encoding="utf-8")
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
        name="orchestrator",
        provider="acp",
        workspace="personal",
        soul_params={"role": "Vice-Captain", "captain": "Maul", "fleet": "Grand Fleet"},
        soul_content="Ye be Silvers Rayleigh, {{role}} of {{captain}}'s {fleet}. Acting in {workspace}.",
    )
    prompt = profile.effective_system_prompt()
    assert "Vice-Captain of Maul's Grand Fleet" in prompt
    assert "Acting in personal" in prompt


def test_standard_api_key_resolution_without_api_key_env():
    profile = Profile(
        name="claude-dev",
        provider="anthropic",
        env_vars={"ANTHROPIC_API_KEY": "sk-ant-test-123"},
    )
    assert profile.get_api_key() == "sk-ant-test-123"


def test_dual_target_memory_loading_and_injection(tmp_path):
    prof_dir = tmp_path / "memory_bot"
    prof_dir.mkdir()
    (prof_dir / "config.yaml").write_text("name: memory_bot\nprovider: gemini\nmodel: gemini-2.5-flash", encoding="utf-8")
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

