import asyncio
import datetime
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import discord

from maulness.config import config
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.daemon.scheduler")


class DaemonScheduler:
    """Background scheduler for automated standups, repo hygiene, and backup watchdogs."""

    def __init__(self, storage: StorageManager, bots: list[Any]):
        self.storage = storage
        self.bots = bots
        self._task: Optional[asyncio.Task] = None
        self._last_standup_date: Optional[str] = None

    def start(self):
        """Launch background cron scheduler loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())
            logger.info("DaemonScheduler background task started")

    def stop(self):
        """Cancel background cron scheduler loop."""
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run_loop(self):
        """Evaluate scheduled triggers every 30 seconds."""
        while True:
            try:
                now = datetime.datetime.now()
                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M")

                # 1. Morning Standup (08:00 Local Time)
                target_time = os.getenv("STANDUP_TIME", "08:00")
                if current_time == target_time and self._last_standup_date != current_date:
                    self._last_standup_date = current_date
                    await self.run_morning_standup()

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in DaemonScheduler loop: %s", e)
                await asyncio.sleep(30)

    async def run_morning_standup(self):
        """Compile workspace status, repo diffs, and backup health, dispatching to Discord."""
        logger.info("Running automated morning standup sweep...")
        repo_dir = config.repo_dir
        repo_summaries = []

        if repo_dir.exists():
            for repo_path in sorted(repo_dir.iterdir()):
                if (repo_path / ".git").exists():
                    try:
                        branch_proc = subprocess.run(
                            ["git", "branch", "--show-current"],
                            cwd=str(repo_path),
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        branch = branch_proc.stdout.strip() or "detached"

                        status_proc = subprocess.run(
                            ["git", "status", "--porcelain"],
                            cwd=str(repo_path),
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        dirty_lines = [l for l in status_proc.stdout.splitlines() if l.strip()]
                        dirty_str = f"⚠️ {len(dirty_lines)} uncommitted" if dirty_lines else "✅ clean"
                        repo_summaries.append(f"• **`{repo_path.name}`** (`{branch}`): {dirty_str}")
                    except Exception as e:
                        repo_summaries.append(f"• **`{repo_path.name}`**: Error inspecting git: {e}")

        # Check DB backups
        backup_dir = config.workspace_root / "_backups" / "postgres"
        backup_status = "None found"
        if backup_dir.exists():
            backup_files = sorted(backup_dir.glob("*.sql.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
            if backup_files:
                latest = backup_files[0]
                mtime = datetime.datetime.fromtimestamp(latest.stat().st_mtime)
                size_mb = latest.stat().st_size / (1024 * 1024)
                backup_status = f"`{latest.name}` ({size_mb:.2f} MB) at {mtime.strftime('%Y-%m-%d %H:%M')}"

        embed = discord.Embed(
            title="🌅 Morning Workspace Standup",
            description="Automated morning health sweep across personal portfolio repos and databases.",
            color=0x10B981,
            timestamp=datetime.datetime.now(),
        )
        embed.add_field(
            name="📦 Repositories Status",
            value="\n".join(repo_summaries) if repo_summaries else "No repositories found.",
            inline=False,
        )
        embed.add_field(
            name="💾 Latest Postgres Backup",
            value=backup_status,
            inline=False,
        )

        # Dispatch embed via first active bot that has a home channel
        sent = False
        for bot in self.bots:
            if getattr(bot, "is_ready", lambda: False)() and getattr(bot, "home_channel_id", None):
                ch = bot.get_channel(bot.home_channel_id)
                if ch:
                    try:
                        await ch.send(embed=embed)
                        sent = True
                        logger.info("Dispatched morning standup embed to channel %s", bot.home_channel_id)
                        break
                    except Exception as e:
                        logger.warning("Failed to send standup embed to channel %s: %s", bot.home_channel_id, e)

        if not sent:
            logger.info("Morning standup completed locally (no active Discord home channel found to post embed)")
