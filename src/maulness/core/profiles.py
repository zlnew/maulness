import logging
import os
from pathlib import Path
from typing import Any, Optional
from dotenv import dotenv_values
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from maulness.config import config
from maulness.core.skills import SkillManager
from maulness.core.soul import get_soul_content

logger = logging.getLogger("maulness.profiles")

USER_PROFILES_DIR = config.config_dir / "profiles"
TEMPLATE_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "profiles"


class IdentityConfig(BaseModel):
    name: str = "default"
    description: str = ""
    system_prompt: str = ""
    soul_inject: bool = True
    soul_params: dict[str, Any] = Field(default_factory=dict)


class VertexConfig(BaseModel):
    enabled: bool = True
    project: Optional[str] = None
    location: Optional[str] = "us-central1"


class ModelConfig(BaseModel):
    provider: str = "acp"
    name: Optional[str] = None  # target model identifier
    base_url: Optional[str] = None  # custom endpoint (e.g. OpenCode, Ollama, DeepSeek)
    api_key_env: Optional[str] = None
    command: Optional[str] = None  # CLI command for ACP (e.g. "agy", "opencode run")
    vertex: Optional[VertexConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _remap_model_alias(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if isinstance(data, dict):
            if "model" in data and "name" not in data:
                data["name"] = data.pop("model")
        return data


class ParameterConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class ExecutionConfig(BaseModel):
    workspace: Optional[str] = None
    yolo: bool = False
    rate_limit_per_minute: int = 20
    system_prompt_mode: str = "prepend"


class FallbackItem(BaseModel):
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    command: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ResilienceConfig(BaseModel):
    fallbacks: list[FallbackItem] = Field(default_factory=list)


class Profile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    identity: IdentityConfig
    model_cfg: ModelConfig = Field(default_factory=ModelConfig, alias="model")
    parameters: ParameterConfig = Field(default_factory=ParameterConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)

    # Hermes Directory-based attributes (loaded dynamically)
    profile_dir: Optional[Path] = None
    soul_content: Optional[str] = None
    memory_content: Optional[str] = None
    user_content: Optional[str] = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    skills_dir: Optional[Path] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_config_structure(cls, data: Any) -> Any:
        """Transparently accepts both new grouped YAML and legacy flat YAML."""
        if not isinstance(data, dict):
            return data

        # Check if already using the new grouped structure
        if "identity" in data and isinstance(data["identity"], (dict, IdentityConfig)):
            # Normalize vertex if passed as flat inside model or top-level
            if "cloud" in data and isinstance(data["cloud"], dict):
                cloud_data = data["cloud"]
                if "model" in data and isinstance(data["model"], dict):
                    data["model"]["vertex"] = {
                        "enabled": cloud_data.get("vertex", True),
                        "project": cloud_data.get("project"),
                        "location": cloud_data.get("location", "us-central1"),
                    }
            return data

        # Otherwise, dynamically reshape flat legacy keys into structured groups
        vertex_cfg = None
        if data.get("vertex") or data.get("project"):
            vertex_cfg = {
                "enabled": bool(data.get("vertex", True)),
                "project": data.get("project"),
                "location": data.get("location", "us-central1"),
            }

        fallbacks_raw = data.get("fallbacks", [])
        fallbacks = []
        for fb in fallbacks_raw:
            if isinstance(fb, dict):
                fallbacks.append(fb)
            elif isinstance(fb, str):
                fallbacks.append({"provider": fb})

        return {
            "identity": {
                "name": data.get("name", "default"),
                "description": data.get("description", ""),
                "system_prompt": data.get("system_prompt", ""),
                "soul_inject": data.get("soul_inject", data.get("inject_soul", True)),
                "soul_params": data.get("soul_params", {}),
            },
            "model": {
                "provider": data.get("provider", "acp"),
                "name": data.get("model") or data.get("model_name"),
                "base_url": data.get("base_url"),
                "api_key_env": data.get("api_key_env"),
                "command": data.get("command"),
                "vertex": vertex_cfg,
            },
            "parameters": {
                "temperature": data.get("temperature", 0.7),
                "max_tokens": data.get("max_tokens", 4096),
                "top_p": data.get("top_p"),
                "top_k": data.get("top_k"),
            },
            "execution": {
                "workspace": data.get("workspace"),
                "yolo": data.get("yolo", False),
                "worktree": data.get("worktree", False),
            },
            "resilience": {
                "fallbacks": fallbacks,
            },
            # Preserve runtime fields
            "profile_dir": data.get("profile_dir"),
            "soul_content": data.get("soul_content"),
            "memory_content": data.get("memory_content"),
            "user_content": data.get("user_content"),
            "env_vars": data.get("env_vars", {}),
            "skills_dir": data.get("skills_dir"),
        }

    @model_validator(mode="after")
    def _validate_acp_command(self) -> "Profile":
        """Enforce explicit command requirement for ACP provider."""
        if self.model_cfg.provider.lower().strip() == "acp" and not (self.model_cfg.command and self.model_cfg.command.strip()):
            raise ValueError(
                f"Profile '{self.name}' specifies provider 'acp' but has no 'command' configured in config.yaml"
            )
        return self

    # --------------------------------------------------------------------------
    # Backward-Compatible Properties (Existing code continues to work seamlessly)
    # --------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def description(self) -> str:
        return self.identity.description

    @property
    def system_prompt(self) -> str:
        return self.identity.system_prompt

    @property
    def inject_soul(self) -> bool:
        return self.identity.soul_inject

    @property
    def soul_params(self) -> dict[str, Any]:
        return self.identity.soul_params

    @property
    def provider(self) -> str:
        return self.model_cfg.provider

    @property
    def model(self) -> Optional[str]:
        return self.model_cfg.name

    @property
    def model_name(self) -> Optional[str]:
        return self.model_cfg.name

    @property
    def base_url(self) -> Optional[str]:
        return self.model_cfg.base_url

    @property
    def api_key_env(self) -> Optional[str]:
        return self.model_cfg.api_key_env

    @property
    def command(self) -> Optional[str]:
        return self.model_cfg.command

    @property
    def vertex(self) -> bool:
        return bool(self.model_cfg.vertex and self.model_cfg.vertex.enabled)

    @property
    def project(self) -> Optional[str]:
        return self.model_cfg.vertex.project if self.model_cfg.vertex else None

    @property
    def location(self) -> Optional[str]:
        return self.model_cfg.vertex.location if self.model_cfg.vertex else "us-central1"

    @property
    def temperature(self) -> float:
        return self.parameters.temperature

    @property
    def max_tokens(self) -> int:
        return self.parameters.max_tokens

    @property
    def top_p(self) -> Optional[float]:
        return self.parameters.top_p

    @property
    def top_k(self) -> Optional[int]:
        return self.parameters.top_k

    @property
    def workspace(self) -> Optional[str]:
        return self.execution.workspace

    @property
    def yolo(self) -> bool:
        return self.execution.yolo

    @property
    def worktree(self) -> bool:
        return self.execution.worktree

    @property
    def fallbacks(self) -> list[dict[str, Any]]:
        return [f.model_dump(exclude_none=True) for f in self.resilience.fallbacks]

    def get_api_key(self) -> Optional[str]:
        """Fetch API key prioritizing profile-specific .env, then system environment."""
        std_key_names = {
            "gemini": "GEMINI_API_KEY",
            "antigravity_sdk": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "opencode": "OPENCODE_API_KEY",
            "opencode_go": "OPENCODE_API_KEY",
            "opencode_zen": "OPENCODE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        prov = self.provider.lower().strip()
        key_name = self.api_key_env or std_key_names.get(prov)

        # 1. Check profile-specific .env
        if key_name and key_name in self.env_vars:
            return self.env_vars[key_name]
        for candidate in (
            "OPENCODE_API_KEY",
            "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            if candidate in self.env_vars and prov in candidate.lower():
                return self.env_vars[candidate]

        # 2. Check system environment
        if key_name:
            val = os.getenv(key_name)
            if val:
                return val

        # 3. Generic fallback for OpenAI-compatible providers
        if prov in ("openai_compatible", "opencode", "opencode_go", "opencode_zen"):
            return os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")

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

        # User-specific operating preferences (USER.md)
        if self.user_content and self.user_content.strip():
            interpolated_user = self.interpolate_soul_text(self.user_content.strip())
            parts.append("\n---\n## User Profile & Working Style (USER.md)\n" + interpolated_user)

        # Durable workspace memory & lessons learned (MEMORY.md)
        if self.memory_content and self.memory_content.strip():
            interpolated_mem = self.interpolate_soul_text(self.memory_content.strip())
            parts.append("\n---\n## Workspace Knowledge & Lessons Learned (MEMORY.md)\n" + interpolated_mem)

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

            # Directory name is the canonical profile identifier
            data["name"] = profile_dir.name

            soul_content = None
            soul_file = profile_dir / "SOUL.md"
            if soul_file.exists():
                soul_content = soul_file.read_text(encoding="utf-8").strip()

            memory_content = None
            memory_file = profile_dir / "MEMORY.md"
            if memory_file.exists():
                memory_content = memory_file.read_text(encoding="utf-8").strip()
            else:
                root_mem = config.config_dir / "MEMORY.md"
                if root_mem.exists():
                    memory_content = root_mem.read_text(encoding="utf-8").strip()

            user_content = None
            user_file = profile_dir / "USER.md"
            if user_file.exists():
                user_content = user_file.read_text(encoding="utf-8").strip()
            else:
                root_user = config.config_dir / "USER.md"
                if root_user.exists():
                    user_content = root_user.read_text(encoding="utf-8").strip()

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
                memory_content=memory_content,
                user_content=user_content,
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
