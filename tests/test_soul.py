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


def test_fallback_soul_content(tmp_path: Path, monkeypatch):
    from maulness.core import soul

    nonexistent = tmp_path / "nonexistent.md"
    dummy_fallback = tmp_path / "fallback.md"
    dummy_fallback.write_text("Fallback doctrine", encoding="utf-8")

    monkeypatch.setattr(soul, "FALLBACK_SOUL_PATH", dummy_fallback)
    loaded = get_soul_content(custom_path=nonexistent)
    assert loaded == "Fallback doctrine"

    # When fallback also does not exist
    monkeypatch.setattr(soul, "FALLBACK_SOUL_PATH", tmp_path / "missing_fallback.md")
    loaded_default = get_soul_content(custom_path=nonexistent)
    assert "You are Maul's personal AI agent assistant" in loaded_default

