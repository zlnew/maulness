import asyncio
import io
import logging
from typing import Any, Optional
import discord
from discord import app_commands
from discord.ext import commands

from maulness.config import config
from maulness.core.debouncer import MessageStreamDebouncer
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
    TaskStatus,
)
from maulness.core.pipeline import PipelineOrchestrator
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.runner import TaskRunner
from maulness.discord.views import ApprovalView
from maulness.storage.db import StorageManager

logger = logging.getLogger("maulness.discord")


class MaulnessBot(commands.Bot):
    """Discord Bot gateway for Maulness with strict single-user authorization and profile binding."""

    def __init__(
        self,
        storage: StorageManager,
        profile_name: Optional[str] = None,
        profile: Optional[Any] = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!mn ", intents=intents)
        self.storage = storage
        self.profile_manager = ProfileManager()

        if profile:
            self.bound_profile = profile
            self.profile_name = profile.name
        elif profile_name:
            self.bound_profile = self.profile_manager.get_profile(profile_name)
            self.profile_name = profile_name
        else:
            self.bound_profile = self.profile_manager.get_profile("default")
            self.profile_name = "default"

        self.runner = TaskRunner(storage=storage, profile_manager=self.profile_manager)
        self.pipeline = PipelineOrchestrator(storage=storage, profile_manager=self.profile_manager)

        # Scoped credentials from profile .env or config fallback
        self.owner_id = self._resolve_owner_id()
        self.guild_id = self._resolve_guild_id()

    def _resolve_owner_id(self) -> Optional[int]:
        if "OWNER_DISCORD_ID" in self.bound_profile.env_vars:
            try:
                return int(self.bound_profile.env_vars["OWNER_DISCORD_ID"])
            except ValueError:
                pass
        return config.owner_discord_id

    def _resolve_guild_id(self) -> Optional[int]:
        if "DISCORD_GUILD_ID" in self.bound_profile.env_vars:
            try:
                return int(self.bound_profile.env_vars["DISCORD_GUILD_ID"])
            except ValueError:
                pass
        return config.discord_guild_id

    async def setup_hook(self):
        """Sync slash commands on startup."""
        await self._register_slash_commands()
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("[%s] Synced slash commands to guild %s", self.profile_name, self.guild_id)
        else:
            await self.tree.sync()
            logger.info("[%s] Synced global slash commands", self.profile_name)

    async def on_ready(self):
        logger.info(
            "Maulness Discord bot [%s] online as %s (ID: %s)",
            self.profile_name,
            self.user,
            self.user.id,
        )

    async def on_message(self, message: discord.Message):
        # Strict single-user authorization gate
        if self.owner_id and message.author.id != self.owner_id:
            return  # Drop silently

        await self.process_commands(message)

    async def _update_forum_tags(self, thread: discord.Thread, status_tag_name: str):
        """Update Discord Forum post tags dynamically based on task status."""
        try:
            if not isinstance(thread.parent, discord.ForumChannel):
                return
            available = {tag.name.lower(): tag for tag in thread.parent.available_tags}
            new_tags = []
            for tag in thread.applied_tags:
                # Keep repo or mode tags, replace status tag
                if tag.name.lower() not in ("planning", "building", "review", "done", "failed"):
                    new_tags.append(tag)

            target_tag = available.get(status_tag_name.lower())
            if target_tag and target_tag not in new_tags:
                new_tags.append(target_tag)

            await thread.edit(applied_tags=new_tags)
        except Exception as e:
            logger.debug("Could not update forum tags: %s", e)

    async def _register_slash_commands(self):
        @self.tree.command(name="status", description="Show Maulness system status and active sessions")
        async def status_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            tasks = await self.storage.list_tasks(limit=5)
            embed = discord.Embed(title="Maulness System Status", color=0xDC2626)
            embed.add_field(name="Gateway", value="Online (Connected)", inline=True)
            embed.add_field(name="Bound Profile", value=f"`{self.profile_name}`", inline=True)
            embed.add_field(name="Recent Tasks", value=str(len(tasks)), inline=True)

            task_summary = "\n".join(
                f"`{t.id}` | **{t.repo_name}** | `{t.status.value}`" for t in tasks
            ) or "No active tasks."
            embed.add_field(name="Recent Activity", value=task_summary, inline=False)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="profiles", description="List all available execution profiles")
        async def profiles_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            profiles = self.profile_manager.list_profiles()
            embed = discord.Embed(title="Available Agent Profiles", color=0x9333EA)
            for p in profiles:
                target = p.model or p.command or "-"
                is_bound = " [BOUND]" if p.name == self.profile_name else ""
                embed.add_field(
                    name=f"`{p.name}` ({p.provider}){is_bound}",
                    value=f"**Target:** `{target}`\n{p.description}",
                    inline=False,
                )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="ask", description="Fast conversational Q&A without spawning local subprocesses")
        @app_commands.describe(query="Your question, math, or query")
        async def ask_cmd(interaction: discord.Interaction, query: str):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()
            status_msg = await interaction.followup.send(f"💭 **Thinking [{self.profile_name}]...**\n> {query[:100]}")

            async def flush_chunk(text: str, is_final: bool):
                try:
                    await status_msg.edit(content=text[:1900])
                except Exception as e:
                    logger.debug("Failed to edit Discord message: %s", e)

            debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

            async def on_message(event: AgentMessageEvent):
                await debouncer.write(event.delta)

            profile = self.bound_profile
            provider = get_provider_for_profile(profile)

            try:
                await provider.run(
                    session_id=f"ask_{interaction.id}",
                    prompt=query,
                    on_message=on_message,
                )
            except Exception as e:
                await interaction.channel.send(f"❌ **Error:** {e}")
            finally:
                await debouncer.close()

        @self.tree.command(name="run", description="Execute a direct task on a repository")
        @app_commands.describe(
            repo="Target repository name (e.g. expense-tracker, horizonx, peek)",
            prompt="The instruction or task to perform",
            profile="Profile to use (builder, planner, default, reviewer)",
        )
        async def run_cmd(
            interaction: discord.Interaction,
            repo: str,
            prompt: str,
            profile: Optional[str] = None,
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            effective_profile = profile or self.profile_name
            await interaction.response.defer()
            status_msg = await interaction.followup.send(
                f"⏳ **Initializing task on `{repo}` using profile `{effective_profile}`...**\n> {prompt[:100]}"
            )

            # Update tag if in a forum thread
            if isinstance(interaction.channel, discord.Thread):
                await self._update_forum_tags(interaction.channel, "Building")

            all_text = []

            async def flush_chunk(text: str, is_final: bool):
                try:
                    await status_msg.edit(content=f"**[{repo}] ({effective_profile}) Running...**\n\n{text[:1900]}")
                except Exception as e:
                    logger.debug("Failed to edit Discord message: %s", e)

            debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

            async def on_thought(event: AgentThoughtEvent):
                pass

            async def on_message(event: AgentMessageEvent):
                all_text.append(event.delta)
                await debouncer.write(event.delta)

            async def on_approval(event: ApprovalRequestEvent) -> bool:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                view = ApprovalView(future=fut)
                await interaction.channel.send(
                    f"⚠️ **Approval Required (HITL)**\n"
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
                profile_name=effective_profile,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
            )
            await debouncer.close()

            # Handle file attachment spillover if output was very large
            full_output = "".join(all_text)
            files = []
            if len(full_output) > 2000:
                fp = io.BytesIO(full_output.encode("utf-8"))
                files.append(discord.File(fp=fp, filename=f"{task_record.id}_output.txt"))

            if isinstance(interaction.channel, discord.Thread):
                status_tag = "Done" if task_record.status == TaskStatus.DONE else "Failed"
                await self._update_forum_tags(interaction.channel, status_tag)

            await interaction.channel.send(
                f"✅ **Task `{task_record.id}` Finished ({task_record.status.value})**",
                files=files,
            )

        @self.tree.command(name="task", description="Create and run a Multi-Route Kanban task (Planner ➔ Builder ➔ Reviewer)")
        @app_commands.describe(
            repo="Target repository name",
            title="Task headline",
            prompt="Detailed task requirements",
        )
        async def task_cmd(interaction: discord.Interaction, repo: str, title: str, prompt: str):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()
            await interaction.followup.send(f"🚀 **Launching Pipeline for `{title}` on `{repo}`...**")

            thread_id = interaction.channel_id if isinstance(interaction.channel, discord.Thread) else None
            if isinstance(interaction.channel, discord.Thread):
                await self._update_forum_tags(interaction.channel, "Planning")

            async def on_thought(event: AgentThoughtEvent):
                pass

            async def on_message(event: AgentMessageEvent):
                pass

            async def on_approval(event: ApprovalRequestEvent) -> bool:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                view = ApprovalView(future=fut)
                await interaction.channel.send(
                    f"⚠️ **Approval Required**\n**Tool:** `{event.tool_name}`\n```json\n{event.args}\n```",
                    view=view,
                )
                try:
                    return await asyncio.wait_for(fut, timeout=600.0)
                except asyncio.TimeoutError:
                    return False

            task_record = await self.pipeline.run_pipeline(
                repo_name=repo,
                title=title,
                prompt=prompt,
                discord_thread_id=thread_id,
                on_thought=on_thought,
                on_message=on_message,
                on_approval=on_approval,
                auto_proceed=True,  # In Discord, auto-proceed through stages while holding tool approvals
            )

            if isinstance(interaction.channel, discord.Thread):
                status_tag = "Done" if task_record.status == TaskStatus.DONE else "Failed"
                await self._update_forum_tags(interaction.channel, status_tag)

            await interaction.channel.send(
                f"🎉 **Pipeline Completed for `{title}`!**\nStatus: `{task_record.status.value}`"
            )
