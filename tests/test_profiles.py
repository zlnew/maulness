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
    assert "Shipwright" in planner.effective_system_prompt()
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
