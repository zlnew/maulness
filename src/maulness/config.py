import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "maulness"
ENV_FILE = CONFIG_DIR / "env"
DOTENV_FILE = CONFIG_DIR / ".env"
LOCAL_ENV_FILE = Path.cwd() / ".env"

# Load ~/.config/maulness/.env or ~/.config/maulness/env, then local .env
if DOTENV_FILE.exists():
    load_dotenv(DOTENV_FILE)
elif ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif LOCAL_ENV_FILE.exists():
    load_dotenv(LOCAL_ENV_FILE)


class Config:
    def __init__(self):
        self.config_dir: Path = CONFIG_DIR
        self.db_path: Path = Path(os.getenv("MAULNESS_DB_PATH", str(CONFIG_DIR / "maulness.db")))
        self.skills_dir: Path = CONFIG_DIR / "skills"
        self.workspace_root: Path = Path(os.getenv("WORKSPACE_ROOT", "/home/zlnew/www/personal"))
        self.repo_dir: Path = self.workspace_root / "repo"

        # Discord (Optional at startup)
        self.discord_bot_token: Optional[str] = os.getenv("DISCORD_BOT_TOKEN")
        self.discord_guild_id: Optional[int] = (
            int(os.getenv("DISCORD_GUILD_ID")) if os.getenv("DISCORD_GUILD_ID") else None
        )
        self.discord_forum_channel_id: Optional[int] = (
            int(os.getenv("DISCORD_FORUM_CHANNEL_ID"))
            if os.getenv("DISCORD_FORUM_CHANNEL_ID")
            else None
        )
        self.owner_discord_id: Optional[int] = (
            int(os.getenv("OWNER_DISCORD_ID")) if os.getenv("OWNER_DISCORD_ID") else None
        )

        # Gemini API Key (Optional)
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

        # Streaming & Execution Watchdog
        self.stream_idle_timeout_seconds: float = float(
            os.getenv("STREAM_IDLE_TIMEOUT_SECONDS", "180.0")
        )

        # Gateway Multiplexing (Hermes-style)
        self.gateway_multiplex_profiles: bool = False
        self.gateway_multiplex_profile_allowlist: list[str] = []
        self.gateway_profile_routes: list[dict] = []
        self._load_yaml_config()

    @property
    def agy_cmd(self) -> list[str]:
        """Backward-compatible default CLI binary token list."""
        return os.getenv("AGY_CMD", "agy").split()

    def _load_yaml_config(self) -> None:
        """Load root configuration from ~/.config/maulness/config.yaml."""
        cfg_yaml = self.config_dir / "config.yaml"
        if cfg_yaml.exists():
            try:
                import yaml
                data = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Workspace group
                    ws = data.get("workspace", {})
                    if isinstance(ws, dict) and "root" in ws:
                        self.workspace_root = Path(ws["root"]).expanduser().resolve()
                        self.repo_dir = self.workspace_root / "repo"

                    # Storage group
                    storage = data.get("storage", {})
                    if isinstance(storage, dict) and "db_path" in storage:
                        self.db_path = Path(storage["db_path"]).expanduser().resolve()

                    # Execution group
                    exec_cfg = data.get("execution", {})
                    if isinstance(exec_cfg, dict):
                        if "stream_idle_timeout_seconds" in exec_cfg:
                            self.stream_idle_timeout_seconds = float(exec_cfg["stream_idle_timeout_seconds"])

                    # Gateway group
                    gw = data.get("gateway", {})
                    if isinstance(gw, dict):
                        self.gateway_multiplex_profiles = bool(gw.get("multiplex_profiles", False))
                        allowlist = gw.get("multiplex_profile_allowlist", [])
                        if isinstance(allowlist, list):
                            self.gateway_multiplex_profile_allowlist = [str(x) for x in allowlist]
                        routes = gw.get("profile_routes", [])
                        if isinstance(routes, list):
                            self.gateway_profile_routes = routes
            except Exception:
                pass

        # Environment variable override takes precedence
        env_multiplex = os.getenv("GATEWAY_MULTIPLEX_PROFILES")
        if env_multiplex is not None:
            self.gateway_multiplex_profiles = env_multiplex.strip().lower() in ("1", "true", "yes", "on")

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_bot_token and self.owner_discord_id)

    def get_current_workspace(self, custom_path: Optional[Path] = None) -> tuple[str, Path]:
        """Detect repository name and directory from current working directory (cwd)."""
        target = (custom_path or Path.cwd()).resolve()

        # If inside workspace_root/repo/<name>, detect the repo folder
        try:
            rel = target.relative_to(self.repo_dir)
            repo_name = rel.parts[0]
            return repo_name, self.repo_dir / repo_name
        except ValueError:
            # Fallback to current folder name and target
            return target.name, target

    def resolve_repo_path(self, target: str | Path) -> Path:
        """Resolve a repository name or path to an absolute workspace directory path."""
        # 1. If it's already an existing directory (absolute or relative)
        candidate = Path(str(target)).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate

        # 2. Check inside repo_dir (e.g. repo/expense-tracker, repo/maulness)
        repo_candidate = (self.repo_dir / str(target)).resolve()
        if repo_candidate.exists() and repo_candidate.is_dir():
            return repo_candidate

        # 3. Check inside workspace_root (e.g. personal)
        ws_candidate = (self.workspace_root / str(target)).resolve()
        if ws_candidate.exists() and ws_candidate.is_dir():
            return ws_candidate

        # 4. Safe fallback to current working directory
        return Path.cwd().resolve()


config = Config()
