import logging
import os
from pathlib import Path
from typing import Any, Optional
from dotenv import dotenv_values
import yaml
from pydantic import BaseModel, Field

from maulness.config import config
from maulness.core.skills import SkillManager
from maulness.core.soul import get_soul_content

logger = logging.getLogger("maulness.profiles")

USER_PROFILES_DIR = config.config_dir / "profiles"
TEMPLATE_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "profiles"


class Profile(BaseModel):
    name: str
    description: str = ""
    provider: str = "acp"  # acp, antigravity_sdk, gemini, anthropic, openai, openrouter
    model: Optional[str] = None
    api_key_env: Optional[str] = None  # Deprecated legacy field, optional for backwards compatibility
    temperature: float = 0.7
    max_tokens: int = 4096
    command: Optional[str] = None  # ACP command (e.g. "agy")
    inject_soul: bool = True
    soul_params: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = ""
    workspace: Optional[str] = None  # Per-profile workspace directory or repo name
    vertex: bool = False
    project: Optional[str] = None  # Google Cloud Project ID (Vertex AI)
    location: Optional[str] = "us-central1"

    # Hermes Directory-based attributes
    profile_dir: Optional[Path] = None
    soul_content: Optional[str] = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    skills_dir: Optional[Path] = None

    def get_api_key(self) -> Optional[str]:
        """Fetch API key prioritizing profile-specific .env, then system environment."""
        # Provider-to-env-var standard mapping
        std_key_names = {
            "gemini": "GEMINI_API_KEY",
            "antigravity_sdk": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        key_name = self.api_key_env or std_key_names.get(self.provider)

        # 1. Check profile-specific .env
        if key_name and key_name in self.env_vars:
            return self.env_vars[key_name]
        for candidate in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            if candidate in self.env_vars and self.provider in candidate.lower():
                return self.env_vars[candidate]

        # 2. Check system environment
        if key_name:
            val = os.getenv(key_name)
            if val:
                return val

        return None

    def interpolate_soul_text(self, text: str) -> str:
        """Substitute {param} and {{param}} placeholders with soul_params and runtime context."""
        ctx: dict[str, Any] = {
            "name": self.name,
            "profile": self.name,
            "workspace": self.workspace or "",
            "provider": self.provider,
            "model": self.model or "",
            "user": os.getenv("USER", "Maul"),
            **self.soul_params,
        }
        result = text
        for k, v in ctx.items():
            result = result.replace(f"{{{{{k}}}}}", str(v))
            result = result.replace(f"{{{k}}}", str(v))
        return result

    def effective_system_prompt(self) -> str:
        """Combine role-specific system prompt, profile SOUL.md, root SOUL.md, and skills summary."""
        parts = []
        if self.system_prompt.strip():
            parts.append(self.interpolate_soul_text(self.system_prompt.strip()))

        # Profile-specific SOUL doctrine (profiles/<name>/SOUL.md)
        if self.soul_content and self.soul_content.strip():
            interpolated_soul = self.interpolate_soul_text(self.soul_content.strip())
            parts.append("\n---\n## Profile Operating Doctrine (SOUL.md)\n" + interpolated_soul)

        # Global personal doctrine (~/.config/maulness/SOUL.md)
        if self.inject_soul:
            soul = get_soul_content()
            if soul:
                interpolated_global_soul = self.interpolate_soul_text(soul)
                parts.append("\n---\n## Personal Operating Doctrine (SOUL.md)\n" + interpolated_global_soul)

        # Append effective skills index
        skill_mgr = SkillManager()
        skills = skill_mgr.list_skills(profile_skills_dir=self.skills_dir)
        skills_summary = skill_mgr.format_skills_summary(skills)
        if skills_summary:
            parts.append(skills_summary)

        return "\n\n".join(parts)


class ProfileManager:
    """Manages discovery and instantiation of Hermes-style directory profiles."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or USER_PROFILES_DIR

    def list_profiles(self) -> list[Profile]:
        """List all discovered profiles (user directories take precedence over templates)."""
        profiles: dict[str, Profile] = {}

        # 1. Load built-in templates first
        if TEMPLATE_PROFILES_DIR.exists():
            for p in self._discover_in_dir(TEMPLATE_PROFILES_DIR):
                profiles[p.name] = p

        # 2. Overlay user profiles in ~/.config/maulness/profiles
        if self.profiles_dir.exists():
            for p in self._discover_in_dir(self.profiles_dir):
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
            command="agy" if name in ("default", "builder") else None,
        )

    def resolve_workspace_for_profile(self, profile: Profile, fallback_workspace: Path) -> Path:
        """Resolve effective workspace path for a profile."""
        if profile.workspace and profile.workspace.strip().lower() != "inherit":
            return config.resolve_repo_path(profile.workspace.strip())
        return fallback_workspace

    def _discover_in_dir(self, directory: Path) -> list[Profile]:
        """Discover directory-based profiles and fallback standalone YAML files."""
        discovered: list[Profile] = []
        if not directory.exists() or not directory.is_dir():
            return discovered

        for entry in sorted(directory.iterdir()):
            # Hermes Directory Mode: <name>/config.yaml
            if entry.is_dir():
                config_file = entry / "config.yaml"
                if config_file.exists():
                    p = self._load_directory_profile(entry, config_file)
                    if p:
                        discovered.append(p)

            # Legacy single-file mode: <name>.yaml
            elif entry.is_file() and entry.suffix in (".yaml", ".yml"):
                p = self._load_file_profile(entry)
                if p:
                    discovered.append(p)

        return discovered

    def _load_directory_profile(self, profile_dir: Path, config_file: Path) -> Optional[Profile]:
        try:
            data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None

            if "name" not in data:
                data["name"] = profile_dir.name

            soul_content = None
            soul_file = profile_dir / "SOUL.md"
            if soul_file.exists():
                soul_content = soul_file.read_text(encoding="utf-8").strip()

            env_vars: dict[str, str] = {}
            env_file = profile_dir / ".env"
            if env_file.exists():
                env_vars = {k: v for k, v in dotenv_values(env_file).items() if v is not None}

            skills_dir = profile_dir / "skills"
            if not skills_dir.exists():
                skills_dir = None

            return Profile(
                **data,
                profile_dir=profile_dir,
                soul_content=soul_content,
                env_vars=env_vars,
                skills_dir=skills_dir,
            )
        except Exception as e:
            logger.warning("Failed to load profile from %s: %s", profile_dir, e)
            return None

    def _load_file_profile(self, path: Path) -> Optional[Profile]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return Profile(**data)
        except Exception as e:
            logger.warning("Failed to load profile from %s: %s", path, e)
        return None
