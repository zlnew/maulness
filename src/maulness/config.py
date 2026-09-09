import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

CONFIG_DIR = Path.home() / ".config" / "maulness"
ENV_FILE = CONFIG_DIR / "env"
LOCAL_ENV_FILE = Path.cwd() / ".env"

# Load ~/.config/maulness/env first, then local .env if present
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif LOCAL_ENV_FILE.exists():
    load_dotenv(LOCAL_ENV_FILE)


class Config:
    def __init__(self):
        self.config_dir: Path = CONFIG_DIR
        self.db_path: Path = CONFIG_DIR / "maulness.db"
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

        # Antigravity binary command
        self.agy_cmd: list[str] = os.getenv("AGY_CMD", "agy --acp").split()

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_bot_token and self.owner_discord_id)

    def resolve_repo_path(self, repo_name: str) -> Path:
        """Resolve a repository name to an absolute workspace directory path."""
        target = self.repo_dir / repo_name
        if not target.exists():
            # Check if it's already an absolute or relative path
            candidate = Path(repo_name).resolve()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Repository '{repo_name}' not found in {self.repo_dir}")
        return target


config = Config()
