import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field

from maulness.config import config
from maulness.core.soul import get_soul_content

USER_PROFILES_DIR = config.config_dir / "profiles"
TEMPLATE_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "profiles"


class Profile(BaseModel):
    name: str
    description: str = ""
    provider: str = "acp"  # acp, gemini, anthropic, openai, openrouter
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    command: Optional[str] = None  # ACP command (e.g. "agy --acp")
    inject_soul: bool = True
    system_prompt: str = ""
    workspace: Optional[str] = None  # Per-profile workspace directory or repo name

    def get_api_key(self) -> Optional[str]:
        """Fetch API key from the environment variable specified in the profile."""
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)

    def effective_system_prompt(self) -> str:
        """Combine role-specific system prompt with personal SOUL.md doctrine."""
        parts = []
        if self.system_prompt.strip():
            parts.append(self.system_prompt.strip())

        if self.inject_soul:
            soul = get_soul_content()
            if soul:
                parts.append("\n---\n## Personal Operating Doctrine (SOUL.md)\n" + soul)

        return "\n\n".join(parts)


class ProfileManager:
    """Manages discovery and instantiation of Maulness profiles."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or USER_PROFILES_DIR

    def list_profiles(self) -> list[Profile]:
        """List all discovered profiles (user profiles take precedence over templates)."""
        profiles: dict[str, Profile] = {}

        # 1. Load built-in templates first
        if TEMPLATE_PROFILES_DIR.exists():
            for file in sorted(TEMPLATE_PROFILES_DIR.glob("*.yaml")):
                p = self._load_file(file)
                if p:
                    profiles[p.name] = p

        # 2. Overlay user profiles in ~/.config/maulness/profiles
        if self.profiles_dir.exists():
            for file in sorted(self.profiles_dir.glob("*.yaml")):
                p = self._load_file(file)
                if p:
                    profiles[p.name] = p

        return list(profiles.values())

    def get_profile(self, name: str) -> Profile:
        """Get a profile by name. Falls back to default if not found."""
        profiles = {p.name: p for p in self.list_profiles()}
        if name in profiles:
            return profiles[name]

        # If not found, return fallback default profile
        return Profile(
            name=name,
            description="Ephemeral default fallback profile",
            provider="acp" if name in ("default", "builder") else "gemini",
            model="gemini-2.5-flash",
            api_key_env="GEMINI_API_KEY",
            command="agy" if name in ("default", "builder") else None,
        )

    def resolve_workspace_for_profile(self, profile: Profile, fallback_workspace: Path) -> Path:
        """Resolve effective workspace path for a profile."""
        if profile.workspace and profile.workspace.strip().lower() != "inherit":
            return config.resolve_repo_path(profile.workspace.strip())
        return fallback_workspace

    def _load_file(self, path: Path) -> Optional[Profile]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return Profile(**data)
        except Exception:
            pass
        return None
