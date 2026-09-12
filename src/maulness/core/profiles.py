import logging
import os
from pathlib import Path
from typing import Any, Optional
from dotenv import dotenv_values
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from maulness.config import config
from maulness.core.rules import ExecutionRulesConfig, PolicyAction, RuleEngine
from maulness.core.skills import SkillManager
from maulness.core.soul import get_soul_content
from maulness.core.stack import get_stack_doctrine

logger = logging.getLogger("maulness.profiles")

USER_PROFILES_DIR = config.config_dir / "profiles"
TEMPLATE_PROFILES_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "templates" / "profiles"
)


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


class AgentConfig(BaseModel):
    """Routing + execution identity block (mirrors Hermes 'agent:' schema)."""

    provider: str = "acp"
    model: Optional[str] = None  # target model identifier (e.g. "gemini-2.5-pro")
    base_url: Optional[str] = None  # custom endpoint (e.g. OpenCode, Ollama, DeepSeek)
    api_key_env: Optional[str] = None
    command: Optional[str] = None  # CLI command for ACP (e.g. "agy", "opencode run")
    reasoning_effort: Optional[str] = None  # "low" | "medium" | "high"
    vertex: Optional[VertexConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"model": data}
        return data


class ParameterConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class ExecutionConfig(BaseModel):
    workspace: Optional[str] = None
    yolo: bool = False
    worktree: bool = False
    rate_limit_per_minute: int = 20
    system_prompt_mode: str = "prepend"
    default_policy: PolicyAction = PolicyAction.ASK
    rules: ExecutionRulesConfig = Field(default_factory=ExecutionRulesConfig)
    max_tool_turns: Optional[int] = None
    max_relays: Optional[int] = None
    sandbox: Optional[str] = None


class FallbackItem(BaseModel):
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    command: Optional[str] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ResilienceConfig(BaseModel):
    fallbacks: list[FallbackItem] = Field(default_factory=list)


class Profile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    agent_cfg: AgentConfig = Field(default_factory=AgentConfig, alias="agent")
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

    @model_validator(mode="after")
    def _validate_acp_command(self) -> "Profile":
        """Enforce explicit command requirement for ACP provider."""
        if self.agent_cfg.provider.lower().strip() == "acp" and not (
            self.agent_cfg.command and self.agent_cfg.command.strip()
        ):
            raise ValueError(
                f"Profile '{self.name}' specifies provider 'acp' but has no 'command' configured in config.yaml"
            )
        return self

    # --------------------------------------------------------------------------
    # Convenience Properties
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
        return self.agent_cfg.provider

    @property
    def model(self) -> Optional[str]:
        return self.agent_cfg.model

    @property
    def base_url(self) -> Optional[str]:
        return self.agent_cfg.base_url

    @property
    def api_key_env(self) -> Optional[str]:
        return self.agent_cfg.api_key_env

    @property
    def command(self) -> Optional[str]:
        return self.agent_cfg.command

    @property
    def reasoning_effort(self) -> Optional[str]:
        return self.agent_cfg.reasoning_effort

    @property
    def vertex(self) -> bool:
        return bool(self.agent_cfg.vertex and self.agent_cfg.vertex.enabled)

    @property
    def project(self) -> Optional[str]:
        return self.agent_cfg.vertex.project if self.agent_cfg.vertex else None

    @property
    def location(self) -> Optional[str]:
        return (
            self.agent_cfg.vertex.location if self.agent_cfg.vertex else "us-central1"
        )

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
    def max_tool_turns(self) -> int:
        if self.execution and self.execution.max_tool_turns is not None:
            return self.execution.max_tool_turns
        return config.max_tool_turns

    @property
    def max_relays(self) -> int:
        if self.execution and self.execution.max_relays is not None:
            return self.execution.max_relays
        return config.max_relays

    @property
    def sandbox_mode(self) -> str:
        if self.execution and self.execution.sandbox is not None:
            return str(self.execution.sandbox)
        return config.sandbox_mode

    @property
    def fallbacks(self) -> list[dict[str, Any]]:
        return [f.model_dump(exclude_none=True) for f in self.resilience.fallbacks]

    def get_api_key(self) -> Optional[str]:
        """Fetch API key prioritizing profile-specific .env, then system environment."""
        prov = self.provider.lower().strip().replace("-", "_")
        std_key_map: dict[str, list[str]] = {
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "antigravity_sdk": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "openai": ["OPENAI_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "opencode": [
                "OPENCODE_API_KEY",
                "OPENCODE_ZEN_API_KEY",
                "OPENCODE_GO_API_KEY",
            ],
            "opencode_go": ["OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"],
            "opencode_zen": ["OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "ollama": ["OLLAMA_API_KEY"],
        }
        candidate_keys = []
        if self.api_key_env:
            candidate_keys.append(self.api_key_env)
        candidate_keys.extend(std_key_map.get(prov, [f"{prov.upper()}_API_KEY"]))

        # 1. Check profile-specific .env
        for key in candidate_keys:
            if key in self.env_vars and self.env_vars[key]:
                return self.env_vars[key]

        # 2. Check system environment
        for key in candidate_keys:
            val = os.getenv(key)
            if val:
                return val

        # 3. Generic fallback
        if prov in ("openai_compatible", "opencode", "opencode_go", "opencode_zen"):
            return (
                self.env_vars.get("OPENCODE_API_KEY")
                or os.getenv("OPENCODE_API_KEY")
                or self.env_vars.get("OPENAI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
        if prov in ("gemini", "antigravity_sdk"):
            return (
                self.env_vars.get("GOOGLE_API_KEY")
                or self.env_vars.get("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or config.gemini_api_key
            )

        return None

    def get_rule_engine(self) -> RuleEngine:
        """Construct a RuleEngine merging global rules with profile-specific execution rules."""
        global_rules = config.global_rules
        merged = global_rules.merge(self.execution.rules)
        return RuleEngine(merged)

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

    def effective_system_prompt(self, workspace_path: Optional[Path] = None) -> str:
        """Combine role-specific system prompt, profile SOUL.md, root SOUL.md, workspace doctrine, stack rules, and skills summary."""
        parts = []
        if self.system_prompt.strip():
            parts.append(self.interpolate_soul_text(self.system_prompt.strip()))

        # Profile-specific SOUL doctrine (profiles/<name>/SOUL.md)
        if self.soul_content and self.soul_content.strip():
            interpolated_soul = self.interpolate_soul_text(self.soul_content.strip())
            parts.append(
                "\n---\n## Profile Operating Doctrine (SOUL.md)\n" + interpolated_soul
            )

        # Global personal doctrine (~/.config/maulness/SOUL.md)
        if self.inject_soul:
            soul = get_soul_content()
            if soul:
                interpolated_global_soul = self.interpolate_soul_text(soul)
                parts.append(
                    "\n---\n## Personal Operating Doctrine (SOUL.md)\n"
                    + interpolated_global_soul
                )

        # User-specific operating preferences (USER.md)
        if self.user_content and self.user_content.strip():
            interpolated_user = self.interpolate_soul_text(self.user_content.strip())
            parts.append(
                "\n---\n## User Profile & Working Style (USER.md)\n" + interpolated_user
            )

        # Durable workspace memory & lessons learned (MEMORY.md)
        if self.memory_content and self.memory_content.strip():
            interpolated_mem = self.interpolate_soul_text(self.memory_content.strip())
            parts.append(
                "\n---\n## Workspace Knowledge & Lessons Learned (MEMORY.md)\n"
                + interpolated_mem
            )

        # Workspace Doctrine (repo/AGENTS.md)
        if workspace_path:
            ws = Path(workspace_path).resolve()
            agents_md = ws / "AGENTS.md"
            if not agents_md.exists():
                agents_md = ws / ".agents" / "AGENTS.md"
            if agents_md.exists():
                try:
                    agents_content = agents_md.read_text(encoding="utf-8").strip()
                    if agents_content:
                        parts.append(
                            "\n---\n## Workspace Doctrine (AGENTS.md)\n"
                            + self.interpolate_soul_text(agents_content)
                        )
                except Exception:
                    pass

            # Stack-Specific Rules (max 3-5 rules capped)
            stack_doc = get_stack_doctrine(ws)
            if stack_doc:
                parts.append("\n---\n" + stack_doc)

        # Append effective skills index
        skill_mgr = SkillManager()
        skills = skill_mgr.list_skills(profile_skills_dir=self.skills_dir)
        skills_summary = skill_mgr.format_skills_summary(skills)
        if skills_summary:
            parts.append(skills_summary)

        # Tool Calling Doctrine (strictly enforces real function calling over simulated markdown)
        parts.append(
            "---\n## Tool Calling Doctrine\n"
            "- You have real workspace tools available via function calling: "
            "`run_command`, `read_file`, `write_file`, `replace_file_content`, `patch_file`, "
            "`list_dir`, `search_files`, `git_status`, `get_outline`, `find_symbol`, `web_search`, `fetch_doc_markdown`.\n"
            "- Structural inspection: Use `get_outline` for inspecting file class/function signatures without reading entire files, and `find_symbol` to locate symbol definitions across workspace.\n"
            "- External knowledge: Use `web_search` and `fetch_doc_markdown` for online documentation, API signatures, and library references.\n"
            "- NEVER simulate, fabricate, or hallucinate tool execution syntax (such as `> **tool_name**` or markdown breadcrumbs) in plain text.\n"
            "- When you need to inspect files, execute shell commands, or check git status, you MUST call the appropriate function tool. Never guess, assume, or invent filesystem contents."
        )

        return "\n\n".join(parts)


class ProfileManager:
    """Manages discovery and instantiation of Hermes-style directory profiles."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or USER_PROFILES_DIR

    def list_profiles(self) -> list[Profile]:
        """List all discovered profiles strictly from the profiles directory (~/.config/maulness/profiles)."""
        profiles: dict[str, Profile] = {}

        # Scan profiles from user config dir (~/.config/maulness/profiles)
        if self.profiles_dir.exists():
            for p in self._discover_in_dir(self.profiles_dir):
                profiles[p.name] = p

        # If user profiles dir is empty or doesn't exist, seed/fallback from templates
        if not profiles and TEMPLATE_PROFILES_DIR.exists():
            for p in self._discover_in_dir(TEMPLATE_PROFILES_DIR):
                profiles[p.name] = p

        return list(profiles.values())

    def get_profile(self, name: str) -> Profile:
        """Get a profile by name. Falls back to default if not found."""
        profiles = {p.name: p for p in self.list_profiles()}
        if name in profiles:
            return profiles[name]

        # If not found, return fallback default profile with inherited base environment
        base_env: dict[str, str] = {}
        root_env = config.config_dir / ".env"
        if root_env.exists():
            base_env.update(
                {k: v for k, v in dotenv_values(root_env).items() if v is not None}
            )
        else:
            default_env = config.config_dir / "profiles" / "default" / ".env"
            if default_env.exists():
                base_env.update(
                    {
                        k: v
                        for k, v in dotenv_values(default_env).items()
                        if v is not None
                    }
                )

        return Profile(
            identity=IdentityConfig(
                name=name, description="Ephemeral default fallback profile"
            ),
            agent=AgentConfig(
                provider="acp" if name in ("default", "builder") else "gemini",
                model="gemini-3.6-flash",
                command="agy" if name in ("default", "builder") else None,
            ),
            env_vars=base_env,
        )

    def resolve_workspace_for_profile(
        self, profile: Profile, fallback_workspace: Path
    ) -> Path:
        """Resolve effective workspace path for a profile."""
        if profile.workspace and profile.workspace.strip().lower() != "inherit":
            rel_candidate = (fallback_workspace / profile.workspace.strip()).resolve()
            if rel_candidate.exists() and rel_candidate.is_dir():
                return rel_candidate
            return config.resolve_repo_path(profile.workspace.strip())
        return fallback_workspace

    def _discover_in_dir(self, directory: Path) -> list[Profile]:
        """Discover Hermes-style directory-based profiles (<name>/config.yaml)."""
        discovered: list[Profile] = []
        if not directory.exists() or not directory.is_dir():
            return discovered

        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                config_file = entry / "config.yaml"
                if config_file.exists():
                    p = self._load_directory_profile(entry, config_file)
                    if p:
                        discovered.append(p)

        return discovered

    def _load_directory_profile(
        self, profile_dir: Path, config_file: Path
    ) -> Optional[Profile]:
        try:
            data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None

            # Directory name is the canonical profile identifier
            data.setdefault("identity", {})["name"] = profile_dir.name

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
            root_env = config.config_dir / ".env"
            if root_env.exists():
                env_vars.update(
                    {k: v for k, v in dotenv_values(root_env).items() if v is not None}
                )
            else:
                default_env = config.config_dir / "profiles" / "default" / ".env"
                if default_env.exists():
                    env_vars.update(
                        {
                            k: v
                            for k, v in dotenv_values(default_env).items()
                            if v is not None
                        }
                    )

            env_file = profile_dir / ".env"
            if env_file.exists():
                env_vars.update(
                    {k: v for k, v in dotenv_values(env_file).items() if v is not None}
                )

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
