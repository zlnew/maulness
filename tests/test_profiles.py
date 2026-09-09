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
    assert "agy --acp" in (builder.command or "")


def test_fallback_profile():
    pm = ProfileManager()
    unknown = pm.get_profile("nonexistent_role")
    assert unknown.name == "nonexistent_role"
    assert unknown.provider in ("gemini", "acp")
