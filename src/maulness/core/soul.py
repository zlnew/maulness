from pathlib import Path
from typing import Optional

from maulness.config import config

DEFAULT_SOUL_PATH = config.config_dir / "SOUL.md"
FALLBACK_SOUL_PATH = Path(__file__).parent.parent.parent / "templates" / "SOUL.md"


def get_soul_content(custom_path: Optional[Path] = None) -> str:
    """Load personal doctrine from ~/.config/maulness/SOUL.md or fallback template."""
    target_path = custom_path or DEFAULT_SOUL_PATH
    if target_path.exists():
        return target_path.read_text(encoding="utf-8").strip()

    if FALLBACK_SOUL_PATH.exists():
        return FALLBACK_SOUL_PATH.read_text(encoding="utf-8").strip()

    return (
        "You are Maul's personal AI agent assistant. Keep replies concise and direct."
    )
