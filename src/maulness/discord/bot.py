import asyncio
import logging
import discord
from discord import app_commands
from discord.ext import commands

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.debouncer import MessageStreamDebouncer
from maulness.core.runner import TaskRunner
from maulness.discord.views import ApprovalView
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.discord")


class MaulnessBot(commands.Bot):
    """Discord Bot gateway for Maulness with strict single-user authorization."""

    def __init__(self, storage: StorageManager):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!mn ", intents=intents)
        self.storage = storage
        self.runner = TaskRunner(storage=storage)

    async def setup_hook(self):
        """Sync slash commands on startup."""
        await self._register_slash_commands()
        if config.discord_guild_id:
            guild = discord.Object(id=config.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Synced slash commands to guild %s", config.discord_guild_id)
        else:
            await self.tree.sync()
            logger.info("Synced global slash commands")

    async def on_ready(self):
        logger.info("Maulness Discord bot online as %s (ID: %s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message):
        # Strict single-user authorization gate
        if config.owner_discord_id and message.author.id != config.owner_discord_id:
            return  # Drop silently

        await self.process_commands(message)

    async def _register_slash_commands(self):
        @self.tree.command(name="status", description="Show Maulness system status and active sessions")
        async def status_cmd(interaction: discord.Interaction):
            if config.owner_discord_id and interaction.user.id != config.owner_discord_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            tasks = await self.storage.list_tasks(limit=5)
            embed = discord.Embed(title="Maulness System Status", color=0xDC2626)
            embed.add_field(name="Gateway", value="Online (Connected)", inline=True)
            embed.add_field(name="Recent Tasks", value=str(len(tasks)), inline=True)

            task_summary = "\n".join(
                f"`{t.id}` | **{t.repo_name}** | `{t.status.value}`" for t in tasks
            ) or "No active tasks."
            embed.add_field(name="Recent Activity", value=task_summary, inline=False)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="run", description="Execute a direct task on a local repository")
        @app_commands.describe(
            repo="Target repository name (e.g. expense-tracker, horizonx, peek)",
            prompt="The instruction or task for Antigravity to perform",
        )
        async def run_cmd(interaction: discord.Interaction, repo: str, prompt: str):
            if config.owner_discord_id and interaction.user.id != config.owner_discord_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()
            status_msg = await interaction.followup.send(
                f"⏳ **Initializing task on `{repo}`...**\n> {prompt[:100]}"
            )

            # Streaming debouncer setup
            async def flush_chunk(text: str, is_final: bool):
                try:
                    await status_msg.edit(content=f"**[{repo}] Running...**\n\n{text[:1900]}")
                except Exception as e:
                    logger.error("Failed to edit Discord message: %s", e)

            debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

            async def on_thought(event: AgentThoughtEvent):
                pass  # Optional: accumulate internal thought

            async def on_message(event: AgentMessageEvent):
                await debouncer.write(event.delta)

            async def on_approval(event: ApprovalRequestEvent) -> bool:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                view = ApprovalView(future=fut)
                approval_msg = await interaction.channel.send(
                    f"⚠️ **Approval Required**\n"
                    f"**Tool:** `{event.tool_name}`\n"
                    f"**Args:** ```json\n{event.args}\n```",
                    view=view,
                )
                try:
                    return await asyncio.wait_for(fut, timeout=600.0)
                except asyncio.TimeoutError:
                    return False

            task_record = await self.runner.run_direct(
                repo_name=repo,
                prompt=prompt,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
            )
            await debouncer.close()
            await interaction.channel.send(
                f"✅ **Task `{task_record.id}` Finished ({task_record.status.value})**"
            )
