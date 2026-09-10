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
_synced_tokens: set[str] = set()


class MaulnessBot(commands.Bot):
    """Discord Bot gateway for Maulness with strict single-user authorization and profile binding."""

    def __init__(
        self,
        storage: StorageManager,
        profile_name: Optional[str] = None,
        profile: Optional[Any] = None,
        profiles: Optional[list[Any]] = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!mn ", intents=intents)
        self.storage = storage
        self.profile_manager = ProfileManager()

        if profiles:
            self.profiles = list(profiles)
            default_prof = next((p for p in profiles if p.name == "default"), profiles[0])
            self.bound_profile = default_prof
            self.profile_name = default_prof.name
        elif profile:
            self.profiles = [profile]
            self.bound_profile = profile
            self.profile_name = profile.name
        elif profile_name:
            prof = self.profile_manager.get_profile(profile_name)
            self.profiles = [prof]
            self.bound_profile = prof
            self.profile_name = profile_name
        else:
            prof = self.profile_manager.get_profile("default")
            self.profiles = [prof]
            self.bound_profile = prof
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

        # Active conversation UUID tracking per channel/thread for multi-turn session persistence
        self.channel_conversations: dict[int, str] = {}

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
        """Sync slash commands on startup and load persisted channel conversations."""
        try:
            self.channel_conversations = await self.storage.list_channel_conversations()
            logger.info("[%s] Loaded %d active channel conversation mapping(s)", self.profile_name, len(self.channel_conversations))
        except Exception as e:
            logger.debug("[%s] Could not load persisted channel conversations: %s", self.profile_name, e)

        await self._register_slash_commands()

        # Deduplicate sync per application/token to prevent Discord HTTP 429 rate limit
        token_key = self.bound_profile.env_vars.get("DISCORD_BOT_TOKEN", "")[:24]
        if token_key:
            if token_key in _synced_tokens:
                logger.info("[%s] Slash commands already synced for shared token", self.profile_name)
                return
            _synced_tokens.add(token_key)

        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("[%s] Synced slash commands to guild %s", self.profile_name, self.guild_id)
        else:
            await self.tree.sync()
            logger.info("[%s] Synced global slash commands", self.profile_name)

    async def on_ready(self):
        profile_names = [p.name for p in self.profiles]
        logger.info(
            "Maulness Discord bot %s online as %s (ID: %s)",
            profile_names,
            self.user,
            self.user.id,
        )

    def resolve_profile_for_channel(self, channel_id: int, parent_id: Optional[int] = None) -> Optional[Any]:
        """Find the matching profile from self.profiles for a given channel or thread."""
        channel_ids = {channel_id}
        if parent_id is not None:
            channel_ids.add(parent_id)

        # 1. Check configured gateway profile_routes
        for route in getattr(config, "gateway_profile_routes", []):
            chat_id = route.get("chat_id")
            if chat_id:
                try:
                    if int(chat_id) in channel_ids:
                        prof_name = route.get("profile")
                        matched = next((p for p in self.profiles if p.name == prof_name), None)
                        if matched:
                            return matched
                except (ValueError, TypeError):
                    pass

        # 2. Check each profile's DISCORD_HOME_CHANNEL or DISCORD_FORUM_CHANNEL_ID
        for p in self.profiles:
            home = p.env_vars.get("DISCORD_HOME_CHANNEL")
            forum = p.env_vars.get("DISCORD_FORUM_CHANNEL_ID")
            for ch in (home, forum):
                if ch:
                    try:
                        if int(ch) in channel_ids:
                            return p
                    except (ValueError, TypeError):
                        pass

        # 3. Check instance fallback channel IDs
        target_channels = set()
        if self.home_channel_id:
            target_channels.add(self.home_channel_id)
        if self.forum_channel_id:
            target_channels.add(self.forum_channel_id)
        if channel_ids.intersection(target_channels):
            return self.bound_profile

        return None

    def resolve_profile_for_message(self, message: discord.Message) -> Optional[Any]:
        """Determine which profile should handle this message, or None if it should be ignored."""
        channel_id = message.channel.id
        parent_id = getattr(message.channel, "parent_id", None)
        is_dm = isinstance(message.channel, discord.DMChannel)

        matched = self.resolve_profile_for_channel(channel_id, parent_id)
        if matched is not None:
            return matched

        is_mentioned = False
        if self.user:
            is_mentioned = (
                self.user in message.mentions
                or f"<@{self.user.id}>" in message.content
                or f"<@!{self.user.id}>" in message.content
            )

        if is_mentioned or is_dm:
            # Standalone satellite profiles ignore mentions in unmapped channels
            if len(self.profiles) == 1 and self.profile_name in ("life", "research"):
                return None
            return self.bound_profile

        return None

    def _should_handle_message(self, message: discord.Message) -> bool:
        """Determine whether this specific bot instance should claim and handle this message."""
        return self.resolve_profile_for_message(message) is not None

    async def _execute_chat_prompt(
        self,
        channel: discord.abc.Messageable,
        prompt: str,
        author_mention: str,
        session_key: str,
        profile: Optional[Any] = None,
    ):
        """Execute a conversational chat prompt in Discord and stream the response."""
        target_profile = profile or self.bound_profile
        status_msg = await channel.send(f"💭 **{target_profile.name}** is thinking...\n> {prompt[:100]}")

        async def flush_chunk(text: str, is_final: bool):
            if not text.strip():
                return
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

        provider = get_provider_for_profile(target_profile)
        workspace = self.profile_manager.resolve_workspace_for_profile(target_profile, config.workspace_root)

        channel_id = getattr(channel, "id", None)
        active_conv_id = self.channel_conversations.get(channel_id) if channel_id else None

        current_task = asyncio.current_task()
        if channel_id and current_task:
            self.channel_tasks[channel_id] = session_key
            self.active_tasks[session_key] = current_task

        async def on_init(conv_id: str):
            if channel_id and conv_id:
                self.channel_conversations[channel_id] = conv_id
                try:
                    await self.storage.set_channel_conversation(channel_id, conv_id, target_profile.name)
                except Exception as e:
                    logger.debug("Failed to persist channel conversation: %s", e)
                logger.info("[%s] Bound channel %s to conversation UUID %s", target_profile.name, channel_id, conv_id)

        try:
            async with channel.typing():
                res = await provider.run(
                    session_id=session_key,
                    prompt=prompt,
                    workspace_path=workspace,
                    conversation_id=active_conv_id,
                    on_init=on_init,
                    on_thought=on_thought,
                    on_message=on_message_chunk,
                    on_approval=on_approval,
                )
                if res and not debouncer.full_text:
                    await debouncer.write(res)
                last_cid = getattr(provider, "last_conversation_id", None)
                if channel_id and last_cid:
                    self.channel_conversations[channel_id] = last_cid
                    try:
                        await self.storage.set_channel_conversation(channel_id, last_cid, target_profile.name)
                    except Exception as e:
                        logger.debug("Failed to persist channel conversation: %s", e)
        except asyncio.CancelledError:
            logger.info("[%s] Chat prompt cancelled by user in channel %s", target_profile.name, channel_id)
            try:
                await channel.send("🛑 **Session stopped by user.**")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.exception("[%s] Error executing chat prompt: %s", target_profile.name, e)
            try:
                await channel.send(f"❌ **Error [{target_profile.name}]:** {e}")
            except Exception:
                pass
        finally:
            if channel_id and self.channel_tasks.get(channel_id) == session_key:
                self.channel_tasks.pop(channel_id, None)
            self.active_tasks.pop(session_key, None)
            await debouncer.close()

    def _create_help_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title="🤖 Maulness Discord Assistant — Help & Commands",
            description=(
                "Personal multi-agent harness powered by Antigravity CLI (`agy`), "
                "orchestrating planning, building, and review workflows."
            ),
            color=0x3B82F6,
        )
        embed.add_field(
            name="💬 Conversation & Session",
            value=(
                "• `/new` or `/reset` (or `/clear`): Start a fresh session with clean doctrine\n"
                "• `/context`: View active conversation UUID, model, workspace, and memory\n"
                "• `/model [name]`: View or switch active model for this profile\n"
                "• `/reasoning [effort]`: Inspect or set model reasoning effort\n"
                "• `/ask <query>`: Quick direct question without local repository tools"
            ),
            inline=False,
        )
        embed.add_field(
            name="🛑 Task & Process Control",
            value=(
                "• `/stop` or `/abort [task_id]`: Immediately cancel running task/prompt\n"
                "• `/yolo`: Guidance on auto-approving mutating actions"
            ),
            inline=False,
        )
        embed.add_field(
            name="🛠️ Workspace & Engineering",
            value=(
                "• `/run <repo> <prompt>`: Execute a direct task on a personal repo\n"
                "• `/task <repo> <title> <prompt>`: Multi-Route Kanban ticket (Planner ➔ Builder ➔ Reviewer)\n"
                "• `/thread <name> [message]`: Create an agent thread and start a session"
            ),
            inline=False,
        )
        embed.add_field(
            name="📊 System & Memory",
            value=(
                "• `/status`: View system state, gateway connections, and recent tasks\n"
                "• `/profiles`: List all configured agent execution profiles\n"
                "• `/memory [profile]`: Inspect durable memory (`MEMORY.md`) and user profile (`USER.md`)"
            ),
            inline=False,
        )
        embed.set_footer(text="Tip: You can also chat directly in designated channels, or type !new / !stop.")
        return embed

    async def on_message(self, message: discord.Message):
        # Ignore own messages and other bots
        if message.author.bot or (self.user and message.author.id == self.user.id):
            return

        # Strict single-user authorization gate
        if self.owner_id and message.author.id != self.owner_id:
            return

        # Check prefix/slash text commands
        raw_text = message.content.strip()
        if raw_text.startswith(("!", "/")):
            parts = raw_text[1:].strip().split(maxsplit=1)
            cmd_name = parts[0].lower() if parts else ""
            cmd_args = parts[1].strip() if len(parts) > 1 else ""

            if cmd_name in ("new", "reset", "clear"):
                old_id = self.channel_conversations.pop(message.channel.id, None)
                await self.storage.clear_channel_conversation(message.channel.id)
                target_id = self.channel_tasks.pop(message.channel.id, None)
                if target_id and target_id in self.active_tasks:
                    task = self.active_tasks.pop(target_id, None)
                    if task and not task.done():
                        task.cancel()

                target_profile = self.resolve_profile_for_channel(message.channel.id) or self.bound_profile
                embed = discord.Embed(
                    title="✨ Session Reset",
                    description=(
                        f"Cleared active conversation context in this channel.\n"
                        f"Next message will begin fresh with **`{target_profile.name}`**'s operating doctrine."
                    ),
                    color=0x10B981,
                )
                if old_id:
                    embed.set_footer(text=f"Previous session: {old_id}")
                await message.channel.send(embed=embed)
                return

            elif cmd_name in ("stop", "abort"):
                target_id = self.channel_tasks.get(message.channel.id)
                if not target_id or target_id not in self.active_tasks:
                    await message.channel.send("ℹ️ No active task or agent execution running in this channel. (Use `/new` to reset context)")
                    return
                async_task = self.active_tasks.get(target_id)
                if async_task and not async_task.done():
                    async_task.cancel()
                    await message.channel.send(f"🛑 **Execution `{target_id}` stopped immediately by user.**")
                else:
                    await message.channel.send(f"ℹ️ Execution `{target_id}` is already finished.")
                return

            elif cmd_name == "status":
                tasks = await self.storage.list_tasks(limit=5)
                embed = discord.Embed(title="Maulness System Status", color=0xDC2626)
                embed.add_field(name="Gateway", value="Online (Connected)", inline=True)
                embed.add_field(name="Profiles", value=", ".join(f"`{p.name}`" for p in self.profiles), inline=True)
                embed.add_field(name="Recent Tasks", value=str(len(tasks)), inline=True)
                task_summary = "\n".join(
                    f"`{t.id}` | **{t.repo_name}** | `{t.status.value}`" for t in tasks
                ) or "No active tasks."
                embed.add_field(name="Recent Activity", value=task_summary, inline=False)
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "context":
                target_profile = self.resolve_profile_for_channel(message.channel.id) or self.bound_profile
                conv_id = self.channel_conversations.get(message.channel.id)
                workspace = self.profile_manager.resolve_workspace_for_profile(target_profile, config.workspace_root)
                embed = discord.Embed(title="🧭 Active Session Context", color=0x3B82F6)
                embed.add_field(name="Profile", value=f"`{target_profile.name}` ({target_profile.provider})", inline=True)
                target = target_profile.model or target_profile.command or "default"
                embed.add_field(name="Model / Command", value=f"`{target}`", inline=True)
                embed.add_field(name="Workspace", value=f"`{workspace}`", inline=False)
                embed.add_field(
                    name="Conversation UUID",
                    value=f"`{conv_id}`" if conv_id else "*Fresh (starts on next message)*",
                    inline=False,
                )
                has_soul = bool(target_profile.soul_content and target_profile.soul_content.strip())
                has_user = bool(target_profile.user_content and target_profile.user_content.strip())
                has_mem = bool(target_profile.memory_content and target_profile.memory_content.strip())
                memory_status = (
                    f"• SOUL.md: {'✅ Loaded' if has_soul else '❌ None'}\n"
                    f"• USER.md: {'✅ Loaded' if has_user else '❌ None'}\n"
                    f"• MEMORY.md: {'✅ Loaded' if has_mem else '❌ None'}"
                )
                embed.add_field(name="Memory & Identity", value=memory_status, inline=False)
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "help":
                embed = self._create_help_embed()
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "model":
                target_profile = self.resolve_profile_for_channel(message.channel.id) or self.bound_profile
                if cmd_args:
                    target_profile.model = cmd_args
                    await message.channel.send(f"✅ Model for profile **`{target_profile.name}`** set to `{cmd_args}`.")
                    return
                current_model = target_profile.model or target_profile.command or "default"
                fallbacks = [
                    fb.get("model") if isinstance(fb, dict) else str(fb)
                    for fb in (target_profile.fallbacks or [])
                ]
                embed = discord.Embed(title=f"🤖 Model Info — `{target_profile.name}`", color=0x6366F1)
                embed.add_field(name="Primary Model / Command", value=f"`{current_model}`", inline=False)
                embed.add_field(name="Provider", value=f"`{target_profile.provider}`", inline=True)
                embed.add_field(
                    name="Fallbacks",
                    value=", ".join(f"`{fb}`" for fb in fallbacks) if fallbacks else "*None*",
                    inline=True,
                )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "memory":
                target_prof = self.profile_manager.get_profile(cmd_args or self.profile_name)
                embed = discord.Embed(
                    title=f"Durable Memory & User Profile — `{target_prof.name}`",
                    color=0x06B6D4,
                )
                mem_text = (target_prof.memory_content or "No MEMORY.md found.")[:1000]
                user_text = (target_prof.user_content or "No USER.md found.")[:1000]
                embed.add_field(name="🧠 Workspace Memory (MEMORY.md)", value=mem_text, inline=False)
                embed.add_field(name="👤 User Profile (USER.md)", value=user_text, inline=False)
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "profiles":
                profiles = self.profile_manager.list_profiles()
                embed = discord.Embed(title="Available Agent Profiles", color=0x9333EA)
                active_names = {p.name for p in self.profiles}
                for p in profiles:
                    target = p.model or p.command or "-"
                    is_bound = " [ACTIVE]" if p.name in active_names else ""
                    embed.add_field(
                        name=f"`{p.name}` ({p.provider}){is_bound}",
                        value=f"**Target:** `{target}`\n{p.description}",
                        inline=False,
                    )
                await message.channel.send(embed=embed)
                return

        # Check if message is directed to this bot and resolve target profile
        target_profile = self.resolve_profile_for_message(message)
        if not target_profile:
            return

        raw_prompt = message.content
        if self.user:
            raw_prompt = raw_prompt.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "").strip()

        if not raw_prompt:
            return

        logger.info(
            "[%s] Handling Discord message in channel %s: %s",
            target_profile.name,
            message.channel.id,
            raw_prompt[:80],
        )

        session_key = f"chat_{message.channel.id}"
        await self._execute_chat_prompt(
            channel=message.channel,
            prompt=raw_prompt,
            author_mention=message.author.mention,
            session_key=session_key,
            profile=target_profile,
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
            embed.add_field(name="Profiles", value=", ".join(f"`{p.name}`" for p in self.profiles), inline=True)
            embed.add_field(name="Recent Tasks", value=str(len(tasks)), inline=True)

            task_summary = "\n".join(
                f"`{t.id}` | **{t.repo_name}** | `{t.status.value}`" for t in tasks
            ) or "No active tasks."
            embed.add_field(name="Recent Activity", value=task_summary, inline=False)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="new", description="Start a new conversation session in this channel/thread")
        async def new_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            old_id = self.channel_conversations.pop(interaction.channel_id, None)
            await self.storage.clear_channel_conversation(interaction.channel_id)

            # Cancel running task if any in this channel
            target_id = self.channel_tasks.pop(interaction.channel_id, None)
            if target_id and target_id in self.active_tasks:
                task = self.active_tasks.pop(target_id, None)
                if task and not task.done():
                    task.cancel()

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            embed = discord.Embed(
                title="✨ Session Reset",
                description=(
                    f"Cleared active conversation context in this channel.\n"
                    f"Next message will begin fresh with **`{target_profile.name}`**'s operating doctrine."
                ),
                color=0x10B981,
            )
            if old_id:
                embed.set_footer(text=f"Previous session: {old_id}")
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="reset", description="Reset your agent session in this channel")
        async def reset_cmd(interaction: discord.Interaction):
            await new_cmd(interaction)

        @self.tree.command(name="clear", description="Clear conversation history and reset session in this channel")
        async def clear_cmd(interaction: discord.Interaction):
            await new_cmd(interaction)

        @self.tree.command(name="context", description="Show active session context, profile, model, and memory")
        async def context_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            conv_id = self.channel_conversations.get(interaction.channel_id)
            workspace = self.profile_manager.resolve_workspace_for_profile(target_profile, config.workspace_root)

            embed = discord.Embed(title="🧭 Active Session Context", color=0x3B82F6)
            embed.add_field(name="Profile", value=f"`{target_profile.name}` ({target_profile.provider})", inline=True)
            target = target_profile.model or target_profile.command or "default"
            embed.add_field(name="Model / Command", value=f"`{target}`", inline=True)
            embed.add_field(name="Workspace", value=f"`{workspace}`", inline=False)
            embed.add_field(
                name="Conversation UUID",
                value=f"`{conv_id}`" if conv_id else "*Fresh (starts on next message)*",
                inline=False,
            )

            has_soul = bool(target_profile.soul_content and target_profile.soul_content.strip())
            has_user = bool(target_profile.user_content and target_profile.user_content.strip())
            has_mem = bool(target_profile.memory_content and target_profile.memory_content.strip())

            memory_status = (
                f"• SOUL.md: {'✅ Loaded' if has_soul else '❌ None'}\n"
                f"• USER.md: {'✅ Loaded' if has_user else '❌ None'}\n"
                f"• MEMORY.md: {'✅ Loaded' if has_mem else '❌ None'}"
            )
            embed.add_field(name="Memory & Identity", value=memory_status, inline=False)
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="model", description="Show or change the active model for this profile")
        @app_commands.describe(name="Optional model identifier to set or switch to")
        async def model_cmd(interaction: discord.Interaction, name: Optional[str] = None):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            if name:
                target_profile.model = name
                await interaction.response.send_message(
                    f"✅ Model for profile **`{target_profile.name}`** set to `{name}`.", ephemeral=True
                )
                return

            current_model = target_profile.model or target_profile.command or "default"
            fallbacks = [
                fb.get("model") if isinstance(fb, dict) else str(fb)
                for fb in (target_profile.fallbacks or [])
            ]
            embed = discord.Embed(title=f"🤖 Model Info — `{target_profile.name}`", color=0x6366F1)
            embed.add_field(name="Primary Model / Command", value=f"`{current_model}`", inline=False)
            embed.add_field(name="Provider", value=f"`{target_profile.provider}`", inline=True)
            embed.add_field(
                name="Fallbacks",
                value=", ".join(f"`{fb}`" for fb in fallbacks) if fallbacks else "*None*",
                inline=True,
            )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="reasoning", description="Show or configure model reasoning effort")
        @app_commands.describe(effort="Reasoning level (none, low, medium, high, max)")
        @app_commands.choices(effort=[
            app_commands.Choice(name="none — minimal/disabled", value="none"),
            app_commands.Choice(name="low", value="low"),
            app_commands.Choice(name="medium", value="medium"),
            app_commands.Choice(name="high", value="high"),
            app_commands.Choice(name="max — maximum reasoning", value="max"),
        ])
        async def reasoning_cmd(interaction: discord.Interaction, effort: Optional[str] = None):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            if effort:
                target_profile.env_vars["REASONING_EFFORT"] = effort
                await interaction.response.send_message(
                    f"⚙️ Reasoning effort for profile **`{target_profile.name}`** set to `{effort}`.",
                    ephemeral=True,
                )
                return

            curr = target_profile.env_vars.get("REASONING_EFFORT", "model-default (high)")
            embed = discord.Embed(title=f"🧠 Reasoning Effort — `{target_profile.name}`", color=0x8B5CF6)
            embed.add_field(name="Current Effort", value=f"`{curr}`", inline=True)
            embed.add_field(name="Profile Provider", value=f"`{target_profile.provider}`", inline=True)
            embed.set_footer(text="Select an effort option from `/reasoning` to change.")
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="help", description="Show available commands and usage guide")
        async def help_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            embed = self._create_help_embed()
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
                channel_id = created_thread.id
                parent_id = getattr(created_thread, "parent_id", None)
                target_profile = self.resolve_profile_for_channel(channel_id, parent_id) or self.bound_profile
                await self._execute_chat_prompt(
                    channel=created_thread,
                    prompt=message,
                    author_mention=interaction.user.mention,
                    session_key=f"thread_{created_thread.id}",
                    profile=target_profile,
                )

        @self.tree.command(name="profiles", description="List all available execution profiles")
        async def profiles_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            profiles = self.profile_manager.list_profiles()
            embed = discord.Embed(title="Available Agent Profiles", color=0x9333EA)
            active_names = {p.name for p in self.profiles}
            for p in profiles:
                target = p.model or p.command or "-"
                is_bound = " [ACTIVE]" if p.name in active_names else ""
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

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            await interaction.response.defer()
            status_msg = await interaction.followup.send(f"💭 **Thinking [{target_profile.name}]...**\n> {query[:100]}")

            async def flush_chunk(text: str, is_final: bool):
                try:
                    await status_msg.edit(content=text[:1900])
                except Exception as e:
                    logger.debug("Failed to edit Discord message: %s", e)

            debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

            async def on_message(event: AgentMessageEvent):
                await debouncer.write(event.delta)

            provider = get_provider_for_profile(target_profile)

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
            app_commands.Choice(name="horizonx-dashboard", value="horizonx-dashboard"),
            app_commands.Choice(name="peek", value="peek"),
            app_commands.Choice(name="aprizqyhub.my.id", value="aprizqyhub.my.id"),
            app_commands.Choice(name="neo-portfolio", value="neo-portfolio"),
            app_commands.Choice(name="maulness", value="maulness"),
            app_commands.Choice(name="gacha", value="gacha"),
        ]

        async def _stop_or_abort_handler(interaction: discord.Interaction, task_id: Optional[str] = None):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("⛔ Unauthorized", ephemeral=True)
                return

            target_id = task_id or self.channel_tasks.get(interaction.channel_id)
            if not target_id or target_id not in self.active_tasks:
                await interaction.response.send_message(
                    "ℹ️ No active task or agent execution running in this channel. (Use `/new` to reset context)",
                    ephemeral=True,
                )
                return

            async_task = self.active_tasks.get(target_id)
            if async_task and not async_task.done():
                async_task.cancel()
                if not target_id.startswith("chat_"):
                    try:
                        await self.storage.update_task_status(target_id, TaskStatus.FAILED)
                    except Exception:
                        pass
                if isinstance(interaction.channel, discord.Thread):
                    await self._update_forum_tags(interaction.channel, "Failed")
                await interaction.response.send_message(
                    f"🛑 **Execution `{target_id}` stopped immediately by user.**"
                )
            else:
                await interaction.response.send_message(
                    f"ℹ️ Execution `{target_id}` is already finished.", ephemeral=True
                )

        @self.tree.command(name="stop", description="Stop the running agent task or streaming response in this channel")
        @app_commands.describe(task_id="Optional specific task ID to stop")
        async def stop_cmd(interaction: discord.Interaction, task_id: Optional[str] = None):
            await _stop_or_abort_handler(interaction, task_id)

        @self.tree.command(name="abort", description="Terminate running ACP session or task immediately")
        @app_commands.describe(task_id="Optional specific task ID to abort")
        async def abort_cmd(interaction: discord.Interaction, task_id: Optional[str] = None):
            await _stop_or_abort_handler(interaction, task_id)

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

            target_profile = self.resolve_profile_for_channel(interaction.channel_id) or self.bound_profile
            effective_profile = profile or target_profile.name
            await interaction.response.defer()

            # Check if we should dispatch to forum channel
            exec_channel = interaction.channel
            created_in_forum = False
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
                    created_in_forum = True
                    await interaction.followup.send(
                        f"📋 Created task thread in workbench: {exec_channel.mention}"
                    )

            if not created_in_forum:
                status_msg = await interaction.followup.send(
                    f"⏳ **Initializing task on `{repo}` using profile `{effective_profile}`...**\n> {prompt[:100]}"
                )
            else:
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
            created_in_forum = False
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
                    created_in_forum = True
                    await interaction.followup.send(
                        f"📋 Created pipeline thread in workbench: {exec_channel.mention}"
                    )

            if not created_in_forum:
                status_msg = await interaction.followup.send(f"🚀 **Launching Pipeline for `{title}` on `{repo}`...**")
            else:
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
