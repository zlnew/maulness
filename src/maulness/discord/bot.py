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
        self.forum_channel_id = self._resolve_forum_channel_id()
        self.home_channel_id = self._resolve_home_channel_id()

        # Active tasks tracking for /abort
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.channel_tasks: dict[int, str] = {}

    def _resolve_home_channel_id(self) -> Optional[int]:
        if "DISCORD_HOME_CHANNEL" in self.bound_profile.env_vars:
            try:
                return int(self.bound_profile.env_vars["DISCORD_HOME_CHANNEL"])
            except ValueError:
                pass
        return self._resolve_forum_channel_id()

    def _resolve_forum_channel_id(self) -> Optional[int]:
        if "DISCORD_FORUM_CHANNEL_ID" in self.bound_profile.env_vars:
            try:
                return int(self.bound_profile.env_vars["DISCORD_FORUM_CHANNEL_ID"])
            except ValueError:
                pass
        return config.discord_forum_channel_id

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

    def _should_handle_message(self, message: discord.Message) -> bool:
        """Determine whether this specific profile bot instance should claim and handle this message."""
        is_mentioned = False
        if self.user:
            is_mentioned = (
                self.user in message.mentions
                or f"<@{self.user.id}>" in message.content
                or f"<@!{self.user.id}>" in message.content
            )

        channel_id = message.channel.id
        parent_id = getattr(message.channel, "parent_id", None)

        target_channels = set()
        if self.home_channel_id:
            target_channels.add(self.home_channel_id)
        if self.forum_channel_id:
            target_channels.add(self.forum_channel_id)

        is_home_channel = (channel_id in target_channels or (parent_id is not None and parent_id in target_channels))
        is_dm = isinstance(message.channel, discord.DMChannel)

        if is_home_channel:
            return True

        if is_mentioned:
            # When life and research share the same user ID, tie-break by designated channel
            if self.profile_name == "research" and not is_home_channel:
                return False
            return True

        if is_dm:
            return True

        return False

    async def _execute_chat_prompt(
        self,
        channel: discord.abc.Messageable,
        prompt: str,
        author_mention: str,
        session_key: str,
    ):
        """Execute a conversational chat prompt in Discord and stream the response."""
        status_msg = await channel.send(f"💭 **{self.profile_name}** is thinking...\n> {prompt[:100]}")

        async def flush_chunk(text: str, is_final: bool):
            try:
                display_text = text if len(text) <= 1950 else text[:1950] + "…"
                await status_msg.edit(content=display_text)
            except Exception as e:
                logger.debug("Failed to edit Discord message: %s", e)

        debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

        async def on_thought(event: AgentThoughtEvent):
            pass

        async def on_message_chunk(event: AgentMessageEvent):
            await debouncer.write(event.delta)

        async def on_approval(event: ApprovalRequestEvent) -> bool:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            view = ApprovalView(future=fut)
            await channel.send(
                f"⚠️ **Approval Required (HITL)**\n"
                f"**Tool:** `{event.tool_name}`\n"
                f"**Args:** ```json\n{event.args}\n```",
                view=view,
            )
            try:
                return await asyncio.wait_for(fut, timeout=600.0)
            except asyncio.TimeoutError:
                return False

        profile = self.bound_profile
        provider = get_provider_for_profile(profile)
        workspace = self.profile_manager.resolve_workspace_for_profile(profile, config.workspace_root)

        try:
            async with channel.typing():
                res = await provider.run(
                    session_id=session_key,
                    prompt=prompt,
                    workspace_path=workspace,
                    conversation_id=session_key,
                    on_thought=on_thought,
                    on_message=on_message_chunk,
                    on_approval=on_approval,
                )
                if res and not debouncer.buffer:
                    await debouncer.write(res)
        except Exception as e:
            logger.exception("[%s] Error executing chat prompt: %s", self.profile_name, e)
            try:
                await channel.send(f"❌ **Error [{self.profile_name}]:** {e}")
            except Exception:
                pass
        finally:
            await debouncer.close()

    async def on_message(self, message: discord.Message):
        # Ignore own messages and other bots
        if message.author.bot or (self.user and message.author.id == self.user.id):
            return

        # Strict single-user authorization gate
        if self.owner_id and message.author.id != self.owner_id:
            return

        # Check prefix commands if any
        if message.content.startswith(self.command_prefix):
            await self.process_commands(message)
            return

        # Check if message is directed to this bot
        if not self._should_handle_message(message):
            return

        raw_prompt = message.content
        if self.user:
            raw_prompt = raw_prompt.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "").strip()

        if not raw_prompt:
            return

        logger.info(
            "[%s] Handling Discord message in channel %s: %s",
            self.profile_name,
            message.channel.id,
            raw_prompt[:80],
        )

        session_key = f"chat_{message.channel.id}"
        await self._execute_chat_prompt(
            channel=message.channel,
            prompt=raw_prompt,
            author_mention=message.author.mention,
            session_key=session_key,
        )

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

        @self.tree.command(name="thread", description="Create a new thread and start a session in it")
        @app_commands.describe(
            name="Thread name / topic",
            message="Optional first message or instruction to run in the thread",
        )
        async def thread_cmd(
            interaction: discord.Interaction,
            name: str,
            message: Optional[str] = None,
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()

            channel = interaction.channel
            created_thread = None

            if isinstance(channel, discord.ForumChannel):
                thread_with_msg = await channel.create_thread(
                    name=name[:100],
                    content=message or f"🧵 Thread started by {interaction.user.mention}",
                )
                created_thread = thread_with_msg.thread
            elif isinstance(channel, discord.TextChannel):
                created_thread = await channel.create_thread(
                    name=name[:100],
                    type=discord.ChannelType.public_thread,
                )
                if message:
                    await created_thread.send(f"{interaction.user.mention}: {message}")
            elif isinstance(channel, discord.Thread):
                created_thread = channel
            else:
                await interaction.followup.send("⚠️ Cannot create a thread in this channel type.", ephemeral=True)
                return

            await interaction.followup.send(f"🧵 Created thread: {created_thread.mention}")

            if message and created_thread:
                await self._execute_chat_prompt(
                    channel=created_thread,
                    prompt=message,
                    author_mention=interaction.user.mention,
                    session_key=f"thread_{created_thread.id}",
                )

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

        REPO_CHOICES = [
            app_commands.Choice(name="expense-tracker", value="expense-tracker"),
            app_commands.Choice(name="horizonx", value="horizonx"),
            app_commands.Choice(name="peek", value="peek"),
            app_commands.Choice(name="aprizqyhub.my.id", value="aprizqyhub.my.id"),
            app_commands.Choice(name="neo-portfolio", value="neo-portfolio"),
            app_commands.Choice(name="maulness", value="maulness"),
        ]

        @self.tree.command(name="abort", description="Terminate running ACP session or task immediately")
        @app_commands.describe(task_id="Optional specific task ID to abort")
        async def abort_cmd(interaction: discord.Interaction, task_id: Optional[str] = None):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_id = task_id or self.channel_tasks.get(interaction.channel_id)
            if not target_id or target_id not in self.active_tasks:
                await interaction.response.send_message(
                    "⚠️ No active task found running in this channel.", ephemeral=True
                )
                return

            async_task = self.active_tasks.get(target_id)
            if async_task and not async_task.done():
                async_task.cancel()
                await self.storage.update_task_status(target_id, TaskStatus.FAILED)
                if isinstance(interaction.channel, discord.Thread):
                    await self._update_forum_tags(interaction.channel, "Failed")
                await interaction.response.send_message(
                    f"🛑 **Task `{target_id}` aborted immediately by user.**"
                )
            else:
                await interaction.response.send_message(
                    f"ℹ️ Task `{target_id}` is already finished.", ephemeral=True
                )

        @self.tree.command(name="run", description="Execute a direct task on a repository")
        @app_commands.describe(
            repo="Target repository name (e.g. expense-tracker, horizonx, peek)",
            prompt="The instruction or task to perform",
            profile="Profile to use (builder, planner, default, reviewer)",
            worktree="Execute in an isolated git worktree (non-destructive)",
            yolo="YOLO mode: bypass HITL approval on mutating actions",
        )
        @app_commands.choices(repo=REPO_CHOICES)
        async def run_cmd(
            interaction: discord.Interaction,
            repo: str,
            prompt: str,
            profile: Optional[str] = None,
            worktree: bool = False,
            yolo: bool = False,
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            effective_profile = profile or self.profile_name
            await interaction.response.defer()

            # Check if we should dispatch to forum channel
            exec_channel = interaction.channel
            if (
                self.forum_channel_id
                and not isinstance(interaction.channel, discord.Thread)
                and interaction.channel_id != self.forum_channel_id
            ):
                forum = self.get_channel(self.forum_channel_id)
                if isinstance(forum, discord.ForumChannel):
                    applied_tags = []
                    avail = {t.name.lower(): t for t in forum.available_tags}
                    for tag_key in ("direct", "building", repo.lower()):
                        if tag_key in avail:
                            applied_tags.append(avail[tag_key])
                    thread_with_msg = await forum.create_thread(
                        name=f"[{repo}] {prompt[:70]}",
                        content=f"🚀 **Direct Task Launched**\n**Repo:** `{repo}` | **Profile:** `{effective_profile}`\n> {prompt[:200]}",
                        applied_tags=applied_tags,
                    )
                    exec_channel = thread_with_msg.thread
                    await interaction.followup.send(
                        f"📋 Created task thread in workbench: {exec_channel.mention}"
                    )

            status_msg = await exec_channel.send(
                f"⏳ **Initializing task on `{repo}` using profile `{effective_profile}`...**\n> {prompt[:100]}"
            )

            if isinstance(exec_channel, discord.Thread):
                await self._update_forum_tags(exec_channel, "Building")

            all_text = []

            async def flush_chunk(text: str, is_final: bool):
                try:
                    await status_msg.edit(
                        content=f"**[{repo}] ({effective_profile}) Running...**\n\n{text[:1900]}"
                    )
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
                await exec_channel.send(
                    f"⚠️ **Approval Required (HITL)**\n"
                    f"**Tool:** `{event.tool_name}`\n"
                    f"**Args:** ```json\n{event.args}\n```",
                    view=view,
                )
                try:
                    return await asyncio.wait_for(fut, timeout=600.0)
                except asyncio.TimeoutError:
                    return False

            async def _run_task_coro():
                task_id = None
                try:
                    task_record = await self.runner.run_direct(
                        repo_name=repo,
                        prompt=prompt,
                        profile_name=effective_profile,
                        use_worktree=worktree,
                        yolo=yolo,
                        on_thought=on_thought,
                        on_message=on_message,
                        on_approval=on_approval,
                    )
                    task_id = task_record.id
                    await debouncer.close()

                    full_output = "".join(all_text)
                    files = []
                    if len(full_output) > 2000:
                        fp = io.BytesIO(full_output.encode("utf-8"))
                        files.append(discord.File(fp=fp, filename=f"{task_record.id}_output.txt"))

                    if isinstance(exec_channel, discord.Thread):
                        status_tag = "Done" if task_record.status == TaskStatus.DONE else "Failed"
                        await self._update_forum_tags(exec_channel, status_tag)

                    await exec_channel.send(
                        f"✅ **Task `{task_record.id}` Finished ({task_record.status.value})**",
                        files=files,
                    )
                except asyncio.CancelledError:
                    logger.info("Task %s cancelled via /abort", task_id or repo)
                    await exec_channel.send(f"🛑 **Task was aborted by user.**")
                finally:
                    if task_id and task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    if exec_channel.id in self.channel_tasks:
                        del self.channel_tasks[exec_channel.id]

            bg_task = asyncio.create_task(_run_task_coro())
            # Track task by channel
            self.channel_tasks[exec_channel.id] = f"run_{exec_channel.id}"
            self.active_tasks[f"run_{exec_channel.id}"] = bg_task

        @self.tree.command(name="task", description="Create and run a Multi-Route Kanban task (Planner ➔ Builder ➔ Reviewer)")
        @app_commands.describe(
            repo="Target repository name",
            title="Task headline",
            prompt="Detailed task requirements",
            worktree="Run builder in an isolated git worktree",
            yolo="YOLO mode: bypass HITL approval on mutating actions",
        )
        @app_commands.choices(repo=REPO_CHOICES)
        async def task_cmd(
            interaction: discord.Interaction,
            repo: str,
            title: str,
            prompt: str,
            worktree: bool = False,
            yolo: bool = False,
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()

            exec_channel = interaction.channel
            if (
                self.forum_channel_id
                and not isinstance(interaction.channel, discord.Thread)
                and interaction.channel_id != self.forum_channel_id
            ):
                forum = self.get_channel(self.forum_channel_id)
                if isinstance(forum, discord.ForumChannel):
                    applied_tags = []
                    avail = {t.name.lower(): t for t in forum.available_tags}
                    for tag_key in ("pipeline", "planning", repo.lower()):
                        if tag_key in avail:
                            applied_tags.append(avail[tag_key])
                    thread_with_msg = await forum.create_thread(
                        name=f"[{repo}] {title[:70]}",
                        content=f"🚀 **Pipeline Kanban Task**\n**Goal:** {title}\n> {prompt[:200]}",
                        applied_tags=applied_tags,
                    )
                    exec_channel = thread_with_msg.thread
                    await interaction.followup.send(
                        f"📋 Created pipeline thread in workbench: {exec_channel.mention}"
                    )

            status_msg = await exec_channel.send(f"🚀 **Launching Pipeline for `{title}` on `{repo}`...**")

            thread_id = exec_channel.id if isinstance(exec_channel, discord.Thread) else None
            if isinstance(exec_channel, discord.Thread):
                await self._update_forum_tags(exec_channel, "Planning")

            async def on_thought(event: AgentThoughtEvent):
                pass

            async def on_message(event: AgentMessageEvent):
                pass

            async def on_approval(event: ApprovalRequestEvent) -> bool:
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                view = ApprovalView(future=fut)
                await exec_channel.send(
                    f"⚠️ **Approval Required**\n**Tool:** `{event.tool_name}`\n```json\n{event.args}\n```",
                    view=view,
                )
                try:
                    return await asyncio.wait_for(fut, timeout=600.0)
                except asyncio.TimeoutError:
                    return False

            async def _run_pipeline_coro():
                task_id = None
                try:
                    task_record = await self.pipeline.run_pipeline(
                        repo_name=repo,
                        title=title,
                        prompt=prompt,
                        discord_thread_id=thread_id,
                        use_worktree=worktree,
                        yolo=yolo,
                        on_thought=on_thought,
                        on_message=on_message,
                        on_approval=on_approval,
                        auto_proceed=True,
                    )
                    task_id = task_record.id

                    if isinstance(exec_channel, discord.Thread):
                        status_tag = "Done" if task_record.status == TaskStatus.DONE else "Failed"
                        await self._update_forum_tags(exec_channel, status_tag)

                    await exec_channel.send(
                        f"🎉 **Pipeline Completed for `{title}`!**\nStatus: `{task_record.status.value}`"
                    )
                except asyncio.CancelledError:
                    logger.info("Pipeline %s cancelled via /abort", task_id or title)
                    await exec_channel.send(f"🛑 **Pipeline was aborted by user.**")
                finally:
                    if task_id and task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    if exec_channel.id in self.channel_tasks:
                        del self.channel_tasks[exec_channel.id]

            bg_task = asyncio.create_task(_run_pipeline_coro())
            self.channel_tasks[exec_channel.id] = f"task_{exec_channel.id}"
            self.active_tasks[f"task_{exec_channel.id}"] = bg_task

        @self.tree.command(name="memory", description="Display active workspace memory (MEMORY.md) and user profile (USER.md)")
        @app_commands.describe(profile="Optional profile name to inspect")
        async def memory_cmd(interaction: discord.Interaction, profile: Optional[str] = None):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_prof = self.profile_manager.get_profile(profile or self.profile_name)
            embed = discord.Embed(
                title=f"Durable Memory & User Profile — `{target_prof.name}`",
                color=0x06B6D4,
            )

            mem_text = (target_prof.memory_content or "No MEMORY.md found.")[:1000]
            user_text = (target_prof.user_content or "No USER.md found.")[:1000]

            embed.add_field(name="🧠 Workspace Memory (MEMORY.md)", value=mem_text, inline=False)
            embed.add_field(name="👤 User Profile (USER.md)", value=user_text, inline=False)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="yolo", description="Show YOLO mode status and usage guidance")
        async def yolo_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            embed = discord.Embed(
                title="⚡ YOLO Mode Guidance",
                description="YOLO mode auto-approves mutating actions (shell commands, file edits, git operations) without waiting for button confirmations in Discord.",
                color=0xEF4444,
            )
            embed.add_field(
                name="Usage in `/run`",
                value="`/run repo:<name> prompt:<text> yolo:True`",
                inline=False,
            )
            embed.add_field(
                name="Usage in `/task`",
                value="`/task repo:<name> title:<text> prompt:<text> yolo:True`",
                inline=False,
            )
            embed.add_field(
                name="CLI Equivalent",
                value="`maulness run <prompt> --yolo` or `/yolo` in `maulness chat`",
                inline=False,
            )
            await interaction.response.send_message(embed=embed)
