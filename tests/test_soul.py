from pathlib import Path
from maulness.core.soul import get_soul_content


def test_get_soul_content():
    soul = get_soul_content()
    assert "Maul" in soul
    assert "zlnew" in soul
    assert "Core Directives" in soul


def test_custom_soul_path(tmp_path: Path):
    custom = tmp_path / "CUSTOM_SOUL.md"
    custom.write_text("Custom Soul Content for Testing", encoding="utf-8")
    loaded = get_soul_content(custom_path=custom)
    assert loaded == "Custom Soul Content for Testing"
