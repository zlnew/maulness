import pytest
from pathlib import Path
from maulness.core.skills import SkillManager, Skill


def test_discover_skills_in_dir(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: custom-tool\ndescription: Custom testing tool\n---\n# Instructions\nDo something useful.",
        encoding="utf-8",
    )

    manager = SkillManager(root_skills_dir=tmp_path)
    skills = manager.discover_skills_in_dir(tmp_path)

    assert "custom-tool" in skills
    assert skills["custom-tool"].description == "Custom testing tool"
    assert "Do something useful." in skills["custom-tool"].instruction


def test_profile_skill_overrides_root_skill(tmp_path):
    root_dir = tmp_path / "root_skills"
    root_dir.mkdir()
    s1 = root_dir / "audit"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("---\nname: audit\ndescription: Root audit\n---\nRoot instructions", encoding="utf-8")

    profile_dir = tmp_path / "profile_skills"
    profile_dir.mkdir()
    s2 = profile_dir / "audit"
    s2.mkdir()
    (s2 / "SKILL.md").write_text("---\nname: audit\ndescription: Profile audit\n---\nProfile instructions", encoding="utf-8")

    manager = SkillManager(root_skills_dir=root_dir, template_skills_dir=tmp_path / "empty_templates")
    effective = manager.list_skills(profile_skills_dir=profile_dir)

    assert len(effective) == 1
    assert effective["audit"].description == "Profile audit"
    assert "Profile instructions" in effective["audit"].instruction


def test_skills_summary_formatting():
    manager = SkillManager()
    skills = {
        "alpha": Skill(name="alpha", description="Alpha skill", path=Path("/tmp/alpha"), instruction=""),
        "beta": Skill(name="beta", description="Beta skill", path=Path("/tmp/beta"), instruction=""),
    }
    summary = manager.format_skills_summary(skills)
    assert "## Available Skills" in summary
    assert "- **alpha**: Alpha skill" in summary
    assert "- **beta**: Beta skill" in summary
