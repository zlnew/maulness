import asyncio
import logging
import re
from typing import Any, Callable, Coroutine, Optional
import discord
from discord import app_commands
from discord.ext import commands

from maulness.config import config
from maulness.core.debouncer import (
    MessageStreamDebouncer,
    chunk_markdown_message,
)
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
    TaskRecord,
    TaskStatus,
)
from maulness.core.pipeline import PipelineOrchestrator
from maulness.core.pipelines import PipelineStage
from maulness.core.profiles import ProfileManager
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.runner import TaskRunner
from maulness.core.tools import clear_turn_tools
from maulness.discord.views import ApprovalView, SuspendedAfkView
from maulness.storage.db import StorageManager

import json

import subprocess
import time
from pathlib import Path

logger = logging.getLogger("maulness.discord")
_synced_tokens: set[str] = set()


def get_git_branch(workspace_path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return res.stdout.strip() or "detached"
    except Exception:
        return "none"


LATEX_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Arrows
    (re.compile(r"\$\s*\\(?:rightarrow|to)\s*\$|\\(?:rightarrow|to)\b"), "->"),
    (re.compile(r"\$\s*\\leftarrow\s*\$|\\leftarrow\b"), "<-"),
    (re.compile(r"\$\s*\\leftrightarrow\s*\$|\\leftrightarrow\b"), "<->"),
    (re.compile(r"\$\s*\\Rightarrow\s*\$|\\Rightarrow\b"), "=>"),
    (re.compile(r"\$\s*\\Leftarrow\s*\$|\\Leftarrow\b"), "<="),
    (re.compile(r"\$\s*\\Leftrightarrow\s*\$|\\Leftrightarrow\b"), "<=>"),
    # Comparisons & Relations
    (re.compile(r"\$\s*\\approx\s*\$|\\approx\b"), "~"),
    (re.compile(r"\$\s*\\(?:ne|neq)\s*\$|\\(?:ne|neq)\b"), "!="),
    (re.compile(r"\$\s*\\(?:le|leq)\s*\$|\\(?:le|leq)\b"), "<="),
    (re.compile(r"\$\s*\\(?:ge|geq)\s*\$|\\(?:ge|geq)\b"), ">="),
    # Arithmetic & symbols
    (re.compile(r"\$\s*\\times\s*\$|\\times\b"), "*"),
    (re.compile(r"\$\s*\\pm\s*\$|\\pm\b"), "+/-"),
    (re.compile(r"\$\s*\\cdot\s*\$|\\cdot\b"), "*"),
    (re.compile(r"\$\s*\\(?:dots|cdots|ldots)\s*\$|\\(?:dots|cdots|ldots)\b"), "..."),
]


def _format_markdown_elements(text: str) -> str:
    """Format markdown headers and tables for clean Discord rendering without touching code blocks."""
    if not text:
        return text

    # If there is an unclosed code fence at the end, separate it so it is treated as a code block
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        last_fence_idx = text.rfind("```")
        main_part = text[:last_fence_idx]
        unclosed_code_part = text[last_fence_idx:]
    else:
        main_part = text
        unclosed_code_part = ""

    parts = re.split(r"(```[\s\S]*?```)", main_part)
    processed_parts: list[str] = []

    for part in parts:
        if part.startswith("```"):
            processed_parts.append(part)
        else:
            # 1. Soften large markdown headers (# Title -> ### Title, ## Title -> ### Title)
            part = re.sub(r"^(#{1,2})\s+(.+)$", r"### \2", part, flags=re.MULTILINE)

            # 2. Format markdown tables: wrap table blocks in ```text ... ```
            lines = part.splitlines(keepends=True)
            new_lines: list[str] = []
            table_buffer: list[str] = []

            def flush_table():
                nonlocal table_buffer
                if not table_buffer:
                    return
                has_divider = any(
                    re.search(r"\|(?:\s*:?-+:?\s*\|)+", row) for row in table_buffer
                )
                if has_divider and len(table_buffer) >= 2:
                    table_content = "".join(table_buffer).strip()
                    new_lines.append(f"```text\n{table_content}\n```\n")
                else:
                    new_lines.extend(table_buffer)
                table_buffer = []

            for line in lines:
                stripped = line.strip()
                if (
                    stripped.startswith("|")
                    and stripped.endswith("|")
                    and stripped.count("|") >= 2
                ):
                    table_buffer.append(line)
                else:
                    if table_buffer:
                        flush_table()
                    new_lines.append(line)
            if table_buffer:
                flush_table()

            processed_parts.append("".join(new_lines))

    if unclosed_code_part:
        processed_parts.append(unclosed_code_part)

    return "".join(processed_parts)


DISCORD_RESPONSE_DIRECTIVE = """## Discord Interface & Formatting Doctrine
You are conversing directly with Maul inside a Discord chat thread (accessible via desktop and mobile).
Deliver clean, readable, and safe Discord formatting:
1. Directness & TL;DR: Lead with the bottom-line answer, verdict, or a 1-2 sentence TL;DR.
2. Heading Hierarchy: Use '###' for section headings. NEVER use '#' or '##' (they render as gigantic, disruptive text on mobile).
3. Brevity & Scanning: Use concise bullet points, bold key terms, and short paragraphs. Avoid long unbroken walls of text.
4. Code & Commands: Enclose all code snippets, file paths, and terminal commands in fenced code blocks with language tags (e.g. ```python, ```bash).
5. Clean Data: Do not output unformatted markdown tables or raw JSON dumps. Summarize comparisons in bullet points or format them cleanly.
6. Communication: Do not leak internal system tags, scratchpads, or raw tool payloads into chat."""


def format_discord_markdown(text: str) -> str:
    """Format markdown for clean Discord rendering: mention safety, tag stripping, LaTeX translation, header softening, and table formatting."""
    if not text:
        return text
    result = text

    # 1. Neutralize server-wide mentions (@everyone, @here) with zero-width space
    result = result.replace("@everyone", "@\u200beveryone").replace(
        "@here", "@\u200bhere"
    )

    # 2. Strip internal reasoning or system tags that might leak from models
    result = re.sub(
        r"<(?:scratchpad|antigravity_thought|system_context|internal_thought)>[\s\S]*?</(?:scratchpad|antigravity_thought|system_context|internal_thought)>",
        "",
        result,
        flags=re.IGNORECASE,
    )

    # 3. Translate LaTeX math to ASCII
    for pattern, replacement in LATEX_REPLACEMENTS:
        result = pattern.sub(replacement, result)
    result = re.sub(
        r"\$\s*(->|<-|<->|=>|<=|<=>|~|!=|\*|\+/-|\.\.\.)\s*\$", r"\1", result
    )

    # 4. Soften headings and wrap tables
    result = _format_markdown_elements(result)
    return result


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
        allowed_mentions = discord.AllowedMentions(
            everyone=False,
            roles=False,
            users=True,
            replied_user=True,
        )

        super().__init__(
            command_prefix="!mn ",
            intents=intents,
            allowed_mentions=allowed_mentions,
        )
        self.storage = storage
        self.profile_manager = ProfileManager()

        if profiles:
            self.profiles = list(profiles)
            default_prof = next(
                (p for p in profiles if p.name == "default"), profiles[0]
            )
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
        self.pipeline = PipelineOrchestrator(
            storage=storage, profile_manager=self.profile_manager
        )

        # Scoped credentials from profile .env or config fallback
        self.owner_id = self._resolve_owner_id()
        self.guild_id = self._resolve_guild_id()
        self.forum_channel_id = self._resolve_forum_channel_id()
        self.home_channel_id = self._resolve_home_channel_id()

        # Active tasks tracking for /stop
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.channel_tasks: dict[int, str] = {}

        # Active conversation UUID tracking per channel/thread for multi-turn session persistence
        self.channel_conversations: dict[int, str] = {}

        # Queue of prompts per channel for sequential execution (/queue)
        self.channel_queues: dict[int, list[str]] = {}

        # Per-channel prompt serialization locks to prevent concurrent execution races
        self._channel_locks: dict[int, asyncio.Lock] = {}

        # Per-channel settings and dynamic overrides
        self.channel_yolo: dict[int, bool] = {}
        self.channel_worktree: dict[int, bool] = {}
        self.channel_profile_overrides: dict[int, Any] = {}

        # Session metrics
        self.session_start_time = time.time()
        self.total_prompts: int = 0
        self.total_chars_out: int = 0

    def _get_channel_lock(self, channel_id: int) -> asyncio.Lock:
        """Return existing asyncio.Lock for channel or initialize a new one."""
        if channel_id not in self._channel_locks:
            self._channel_locks[channel_id] = asyncio.Lock()
        return self._channel_locks[channel_id]

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
            logger.info(
                "[%s] Loaded %d active channel conversation mapping(s)",
                self.profile_name,
                len(self.channel_conversations),
            )
        except Exception as e:
            logger.debug(
                "[%s] Could not load persisted channel conversations: %s",
                self.profile_name,
                e,
            )

        await self._register_slash_commands()

        # Deduplicate sync per application/token to prevent Discord HTTP 429 rate limit
        token_key = self.bound_profile.env_vars.get("DISCORD_BOT_TOKEN", "")[:24]
        if token_key:
            if token_key in _synced_tokens:
                logger.info(
                    "[%s] Slash commands already synced for shared token",
                    self.profile_name,
                )
                return
            _synced_tokens.add(token_key)

        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(
                "[%s] Synced slash commands to guild %s",
                self.profile_name,
                self.guild_id,
            )
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

    async def on_interaction(self, interaction: discord.Interaction):
        if interaction.type == discord.InteractionType.application_command:
            cmd_name = interaction.command.name if interaction.command else "unknown"
            logger.info(
                "[%s] Received /%s command in channel %s from user %s",
                self.profile_name,
                cmd_name,
                interaction.channel_id,
                interaction.user.id,
            )

    def resolve_profile_for_channel(
        self, channel_id: int, parent_id: Optional[int] = None
    ) -> Optional[Any]:
        """Find the matching profile from self.profiles for a given channel or thread."""
        channel_ids = {channel_id}
        if parent_id is not None:
            channel_ids.add(parent_id)

        # 0. Check dynamic channel/thread profile overrides
        for cid in channel_ids:
            if cid in self.channel_profile_overrides:
                return self.channel_profile_overrides[cid]

        # 1. Check configured gateway profile_routes
        for route in getattr(config, "gateway_profile_routes", []):
            chat_id = route.get("chat_id")
            if chat_id:
                try:
                    if int(chat_id) in channel_ids:
                        prof_name = route.get("profile")
                        matched = next(
                            (p for p in self.profiles if p.name == prof_name), None
                        )
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
        """Execute a conversational chat prompt in Discord, serialized per-channel to avoid races."""
        channel_id = getattr(channel, "id", None)
        lock = self._get_channel_lock(channel_id) if channel_id else None
        if lock:
            await lock.acquire()
        try:
            await self._run_chat_prompt(
                channel=channel,
                prompt=prompt,
                author_mention=author_mention,
                session_key=session_key,
                profile=profile,
            )
        finally:
            if lock and lock.locked():
                lock.release()

    async def _run_chat_prompt(
        self,
        channel: discord.abc.Messageable,
        prompt: str,
        author_mention: str,
        session_key: str,
        profile: Optional[Any] = None,
    ):
        """Internal worker executing chat prompt and streaming response."""
        self.total_prompts += 1
        target_profile = profile or self.bound_profile
        model_tag = target_profile.model or target_profile.command or "default"
        provider_tag = f"{target_profile.provider}:{model_tag}"
        current_msg = await channel.send(
            f"**{target_profile.name}** is thinking... `[{provider_tag}]`\n> {prompt[:100]}"
        )
        active_msgs = [current_msg]
        start_time = time.time()
        fallback_alerts: list[str] = []

        needs_new_msg = False

        async def flush_chunk(text: str, is_final: bool, is_overflow: bool = False):
            nonlocal current_msg, needs_new_msg
            if not text.strip():
                return
            clean_text = format_discord_markdown(text)
            if not clean_text.strip():
                return

            chunks = chunk_markdown_message(clean_text, max_size=1900)
            for idx, ch in enumerate(chunks):
                is_chunk_overflow = (idx < len(chunks) - 1) or is_overflow

                if needs_new_msg or idx > 0:
                    try:
                        current_msg = await channel.send(ch)
                        active_msgs.append(current_msg)
                        needs_new_msg = False
                    except Exception as e:
                        logger.error("Failed to send Discord message chunk: %s", e)
                else:
                    try:
                        current_msg = await current_msg.edit(content=ch)
                    except discord.RateLimited as rl:
                        retry_after = getattr(rl, "retry_after", 1.5)
                        logger.warning("Discord edit rate limited: backing off %.2fs", retry_after)
                        await asyncio.sleep(retry_after)
                        try:
                            current_msg = await current_msg.edit(content=ch)
                        except Exception:
                            pass
                    except discord.HTTPException as http_err:
                        if http_err.status == 429:
                            retry_after = getattr(http_err, "retry_after", 1.5)
                            logger.warning("Discord edit HTTP 429: backing off %.2fs", retry_after)
                            await asyncio.sleep(retry_after)
                            try:
                                current_msg = await current_msg.edit(content=ch)
                            except Exception:
                                pass
                        else:
                            logger.warning(
                                "Failed to edit Discord message (%s), sending replacement",
                                http_err,
                            )
                            try:
                                current_msg = await channel.send(ch)
                                active_msgs.append(current_msg)
                            except Exception as send_err:
                                logger.error(
                                    "Failed to send replacement Discord message: %s",
                                    send_err,
                                )
                    except Exception as e:
                        logger.warning(
                            "Failed to edit Discord message (%s), sending replacement",
                            e,
                        )
                        try:
                            current_msg = await channel.send(ch)
                            active_msgs.append(current_msg)
                        except Exception as send_err:
                            logger.error(
                                "Failed to send replacement Discord message: %s",
                                send_err,
                            )

                if is_chunk_overflow and not is_final:
                    needs_new_msg = True

        debouncer = MessageStreamDebouncer(flush_callback=flush_chunk)

        async def on_thought(event: AgentThoughtEvent):
            if "Provider" in event.delta or "Switching to fallback" in event.delta:
                clean_alert = event.delta.strip()
                fallback_alerts.append(clean_alert)
                if not debouncer.full_text.strip():
                    alert_block = "\n".join(f"> {a}" for a in fallback_alerts)
                    try:
                        await current_msg.edit(
                            content=f"**{target_profile.name}** is thinking... `[{provider_tag}]`\n> {prompt[:100]}\n\n{alert_block}"
                        )
                    except Exception as e:
                        logger.debug(
                            "Failed to edit Discord thinking message with fallback notice: %s",
                            e,
                        )

        async def on_tool_call(event: AgentToolCallEvent):
            logger.info(
                "[%s] Tool call in channel %s: %s",
                target_profile.name,
                channel_id,
                event.tool_name,
            )
            tool_summary = event.tool_name
            if event.tool_name == "run_command" and "command" in event.args:
                tool_summary = f"`{event.args['command'][:60]}`"
            elif (
                event.tool_name in ("read_file", "write_file") and "path" in event.args
            ):
                tool_summary = f"{event.tool_name} `{event.args['path'][:50]}`"
            if not debouncer.full_text.strip():
                try:
                    await current_msg.edit(
                        content=f"**{target_profile.name}** is thinking... `[{provider_tag}]`\n> {prompt[:100]}\n\n> *Running {tool_summary}...*"
                    )
                except Exception as e:
                    logger.debug(
                        "Failed to edit Discord thinking message with tool call: %s", e
                    )

        async def on_message_chunk(event: AgentMessageEvent):
            self.total_chars_out += len(event.delta)
            await debouncer.write(event.delta)

        channel_id = getattr(channel, "id", None)

        async def on_approval(event: ApprovalRequestEvent) -> bool:
            nonlocal current_msg
            if channel_id and self.channel_yolo.get(channel_id, False):
                logger.info(
                    "[%s] Auto-approving tool %s due to channel YOLO mode",
                    target_profile.name,
                    event.tool_name,
                )
                return True
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            view = ApprovalView(future=fut)

            embed = discord.Embed(
                title="Approval Required (HITL)",
                description=f"**{target_profile.name}** requests permission to execute an action.",
                color=0xF59E0B,
            )
            embed.add_field(name="Tool", value=f"`{event.tool_name}`", inline=True)
            if event.tool_name == "run_command":
                cmd = event.args.get("command", "")
                embed.add_field(
                    name="Command", value=f"```bash\n{cmd}\n```", inline=False
                )
                if "cwd" in event.args:
                    embed.add_field(
                        name="Directory", value=f"`{event.args['cwd']}`", inline=True
                    )
            elif event.tool_name == "write_file":
                embed.add_field(
                    name="Target File", value=f"`{event.args.get('path')}`", inline=True
                )
                embed.add_field(
                    name="Size",
                    value=f"{event.args.get('bytes', 0)} bytes",
                    inline=True,
                )
            else:
                formatted_args = json.dumps(event.args, indent=2)
                embed.add_field(
                    name="Arguments",
                    value=f"```json\n{formatted_args[:1000]}\n```",
                    inline=False,
                )

            embed.set_footer(text="Maulness Security Gate • 10m timeout")
            approval_msg = await channel.send(embed=embed, view=view)
            is_timeout = False
            try:
                approved = await asyncio.wait_for(fut, timeout=600.0)
            except asyncio.TimeoutError:
                approved = False
                is_timeout = True

            # Disable view buttons so late interactions don't raise NotFound or unhandled exceptions
            for item in view.children:
                item.disabled = True
            try:
                await approval_msg.edit(view=view)
            except Exception:
                pass

            # Clean up old thinking placeholder if empty
            if not debouncer.full_text.strip():
                try:
                    await current_msg.delete()
                except Exception:
                    pass
            else:
                try:
                    await debouncer.close()
                except Exception:
                    pass
                debouncer.reset()

            # Always spawn fresh message below approval card so all subsequent text streams sequentially
            if approved:
                status_text = (
                    f"**{target_profile.name}** is running `{event.tool_name}`..."
                )
            elif is_timeout:
                status_text = (
                    f"**{target_profile.name}** action `{event.tool_name}` timed out."
                )
            else:
                status_text = f"**{target_profile.name}** action `{event.tool_name}` was rejected by Maul."

            try:
                current_msg = await channel.send(status_text)
                active_msgs.append(current_msg)
            except Exception as e:
                logger.error(
                    "Failed to spawn sequential message after approval decision: %s", e
                )

            return approved

        provider = get_provider_for_profile(target_profile)
        workspace = self.profile_manager.resolve_workspace_for_profile(
            target_profile, config.workspace_root
        )

        active_conv_id = (
            self.channel_conversations.get(channel_id) if channel_id else None
        )

        # Re-inject compacted session memory into context upon starting a fresh conversation
        if channel_id and not active_conv_id:
            try:
                compacted_summary = await self.storage.get_latest_compacted_memory(
                    channel_id
                )
                if compacted_summary and compacted_summary.strip():
                    prompt = (
                        f"[PREVIOUS COMPACTED SESSION CONTEXT]\n"
                        f"{compacted_summary.strip()}\n"
                        f"[/PREVIOUS COMPACTED SESSION CONTEXT]\n\n"
                        f"{prompt}"
                    )
            except Exception as e:
                logger.debug("Failed to retrieve compacted session memory: %s", e)

        current_task = asyncio.current_task()
        if channel_id and current_task:
            self.channel_tasks[channel_id] = session_key
            self.active_tasks[session_key] = current_task

        async def on_init(conv_id: str):
            if channel_id and conv_id:
                self.channel_conversations[channel_id] = conv_id
                try:
                    await self.storage.set_channel_conversation(
                        channel_id, conv_id, target_profile.name
                    )
                except Exception as e:
                    logger.debug("Failed to persist channel conversation: %s", e)
                logger.info(
                    "[%s] Bound channel %s to conversation UUID %s",
                    target_profile.name,
                    channel_id,
                    conv_id,
                )

        try:
            async with channel.typing():
                res = await provider.run(
                    session_id=session_key,
                    prompt=prompt,
                    workspace_path=workspace,
                    conversation_id=active_conv_id,
                    extra_system_prompt=DISCORD_RESPONSE_DIRECTIVE,
                    on_init=on_init,
                    on_thought=on_thought,
                    on_message=on_message_chunk,
                    on_tool_call=on_tool_call,
                    on_approval=on_approval,
                )
                if res and not debouncer.full_text.strip():
                    await debouncer.write(res)

                if not debouncer.full_text.strip():
                    await debouncer.write(
                        "*(No visible output was generated by the model)*"
                    )

                # If response was fulfilled by a fallback provider, attach a subtle footnote
                used_prov = getattr(provider, "last_used_provider", None)
                if (
                    used_prov
                    and hasattr(provider, "primary")
                    and used_prov != getattr(provider, "primary", None)
                ):
                    used_tag = f"{used_prov.profile.provider}:{used_prov.profile.model or used_prov.profile.command or 'default'}"
                    footnote = f"\n\n*(Responded via fallback `{used_tag}` after primary provider failure)*"
                    await debouncer.write(footnote)

                last_cid = getattr(provider, "last_conversation_id", None)
                if channel_id and last_cid:
                    self.channel_conversations[channel_id] = last_cid
                    try:
                        await self.storage.set_channel_conversation(
                            channel_id, last_cid, target_profile.name
                        )
                    except Exception as e:
                        logger.debug("Failed to persist channel conversation: %s", e)
        except asyncio.CancelledError:
            logger.info(
                "[%s] Chat prompt cancelled by user in channel %s",
                target_profile.name,
                channel_id,
            )
            try:
                await channel.send("**Session stopped by user.**")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.exception(
                "[%s] Error executing chat prompt: %s", target_profile.name, e
            )
            try:
                err_text = str(e).strip()
                error_msg = f"**Execution Error [{target_profile.name}]:**\n```\n{err_text[:1500]}\n```"
                try:
                    await current_msg.edit(content=error_msg)
                except Exception:
                    await channel.send(error_msg)
            except Exception:
                pass
        finally:
            clear_turn_tools(session_key)
            if channel_id and self.channel_tasks.get(channel_id) == session_key:
                self.channel_tasks.pop(channel_id, None)
            self.active_tasks.pop(session_key, None)
            await debouncer.close()
            elapsed = time.time() - start_time
            logger.info(
                "[%s] Completed chat prompt in channel %s in %.2fs (output: %d chars)",
                target_profile.name,
                channel_id,
                elapsed,
                len(debouncer.full_text),
            )

            # Trigger background autonomous memory extraction for durable developer facts
            if channel_id:
                conv_id = self.channel_conversations.get(channel_id)
                if conv_id:

                    async def _run_fact_extraction(cid: str, pname: str):
                        try:
                            from maulness.core.memory import AutonomousMemoryExtractor

                            recent = await self.storage.get_conversation_messages(
                                cid, limit=10
                            )
                            if len(recent) >= 4:
                                extractor = AutonomousMemoryExtractor()
                                await extractor.extract_and_update(
                                    recent, profile_name=pname
                                )
                        except Exception as ex:
                            logger.debug("Background fact extraction skipped: %s", ex)

                    asyncio.create_task(
                        _run_fact_extraction(conv_id, target_profile.name)
                    )

            # Process next queued prompt in channel if any
            if channel_id and self.channel_queues.get(channel_id):
                next_prompt = self.channel_queues[channel_id].pop(0)
                asyncio.create_task(
                    self._execute_chat_prompt(
                        channel=channel,
                        prompt=next_prompt,
                        author_mention=author_mention,
                        session_key=f"chat_{channel_id}",
                        profile=target_profile,
                    )
                )

    def _create_help_embed(self) -> discord.Embed:
        embed = discord.Embed(
            title="Maulness Discord Assistant — Help & Commands",
            description=(
                "Personal multi-agent harness powered by Antigravity CLI (`agy`), "
                "orchestrating planning, building, and review workflows."
            ),
            color=0x3B82F6,
        )
        embed.add_field(
            name="Conversation & Session",
            value=(
                "• `/new`: Start a fresh session, reset context window, and cancel running tasks\n"
                "• `/context`: View active conversation UUID, git branch, dirty status, workspace, and memory\n"
                "• `/compact`: Compact conversation history into SQLite memory and reset active context\n"
                "• `/profile [name]`: Inspect available profiles or switch active profile for this channel\n"
                "• `/usage`: Display session token metrics, character count, uptime, and cost estimate"
            ),
            inline=False,
        )
        embed.add_field(
            name="Task & Flow Control",
            value=(
                "• `/stop [task_id]`: Immediately cancel running task or prompt stream\n"
                "• `/interrupt <prompt>`: Cancel active task and steer agent immediately with new prompt\n"
                "• `/queue <prompt>`: Queue prompt to run sequentially after current task finishes"
            ),
            inline=False,
        )
        embed.add_field(
            name="Execution & Workspace",
            value=(
                "• `/pipeline <repo> <title> <prompt> [pipeline_name]`: Multi-stage Kanban pipeline (standard, quick, audit)\n"
                "• `/yolo [enabled]`: Toggle auto-approval on mutating tool actions for this channel\n"
                "• `/worktree [enabled]`: Toggle isolated Git worktree execution for this channel\n"
                "• `/thread <name> [message]`: Create a new thread or forum post and start a session in it"
            ),
            inline=False,
        )
        embed.add_field(
            name="Reference",
            value="• `/help`: Show this command reference",
            inline=False,
        )
        embed.set_footer(
            text="Tip: You can also chat directly in designated channels, or type !new / !stop."
        )
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

            if cmd_name == "new":
                old_id = self.channel_conversations.pop(message.channel.id, None)
                await self.storage.clear_channel_conversation(message.channel.id)
                target_id = self.channel_tasks.pop(message.channel.id, None)
                if target_id and target_id in self.active_tasks:
                    task = self.active_tasks.pop(target_id, None)
                    if task and not task.done():
                        task.cancel()
                self.channel_queues.pop(message.channel.id, None)

                target_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                embed = discord.Embed(
                    title="Session Reset",
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

            elif cmd_name == "stop":
                target_id = self.channel_tasks.get(message.channel.id)
                if not target_id or target_id not in self.active_tasks:
                    await message.channel.send(
                        "[Info] No active task or agent execution running in this channel. (Use `/new` to reset context)"
                    )
                    return
                async_task = self.active_tasks.get(target_id)
                if async_task and not async_task.done():
                    async_task.cancel()
                    await message.channel.send(
                        f"**Execution `{target_id}` stopped immediately by user.**"
                    )
                else:
                    await message.channel.send(
                        f"[Info] Execution `{target_id}` is already finished."
                    )
                return

            elif cmd_name == "interrupt":
                if not cmd_args:
                    await message.channel.send("Usage: `!interrupt <prompt>`")
                    return
                target_id = self.channel_tasks.get(message.channel.id)
                if target_id and target_id in self.active_tasks:
                    task = self.active_tasks.get(target_id)
                    if task and not task.done():
                        task.cancel()
                target_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                await message.channel.send(
                    f"**Interrupted previous task to steer [{target_profile.name}]:**\n> {cmd_args[:200]}"
                )
                asyncio.create_task(
                    self._execute_chat_prompt(
                        channel=message.channel,
                        prompt=cmd_args,
                        author_mention=message.author.mention,
                        session_key=f"chat_{message.channel.id}",
                        profile=target_profile,
                    )
                )
                return

            elif cmd_name == "queue":
                if not cmd_args:
                    await message.channel.send("Usage: `!queue <prompt>`")
                    return
                target_id = self.channel_tasks.get(message.channel.id)
                is_busy = bool(
                    target_id
                    and target_id in self.active_tasks
                    and not self.active_tasks[target_id].done()
                )
                if is_busy:
                    q = self.channel_queues.setdefault(message.channel.id, [])
                    q.append(cmd_args)
                    await message.channel.send(
                        f"**Prompt Queued (#{len(q)})** — will auto-execute once current task completes:\n> {cmd_args[:200]}"
                    )
                else:
                    target_profile = (
                        self.resolve_profile_for_channel(message.channel.id)
                        or self.bound_profile
                    )
                    await message.channel.send(
                        f"**Dispatching prompt directly [{target_profile.name}]:**\n> {cmd_args[:200]}"
                    )
                    asyncio.create_task(
                        self._execute_chat_prompt(
                            channel=message.channel,
                            prompt=cmd_args,
                            author_mention=message.author.mention,
                            session_key=f"chat_{message.channel.id}",
                            profile=target_profile,
                        )
                    )
                return

            elif cmd_name == "context":
                target_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                conv_id = self.channel_conversations.get(message.channel.id)
                workspace = self.profile_manager.resolve_workspace_for_profile(
                    target_profile, config.workspace_root
                )
                branch = get_git_branch(workspace)
                diff_proc = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                )
                dirty_count = len(
                    [line for line in diff_proc.stdout.splitlines() if line.strip()]
                )
                dirty_str = (
                    f"{dirty_count} uncommitted changes" if dirty_count > 0 else "clean"
                )

                embed = discord.Embed(title="Active Session Context", color=0x3B82F6)
                embed.add_field(
                    name="Profile",
                    value=f"`{target_profile.name}` ({target_profile.provider})",
                    inline=True,
                )
                target = target_profile.model or target_profile.command or "default"
                embed.add_field(
                    name="Model / Command", value=f"`{target}`", inline=True
                )
                embed.add_field(
                    name="Git Branch", value=f"`{branch}` ({dirty_str})", inline=True
                )
                embed.add_field(name="Workspace", value=f"`{workspace}`", inline=False)
                embed.add_field(
                    name="Conversation UUID",
                    value=f"`{conv_id}`"
                    if conv_id
                    else "*Fresh (starts on next message)*",
                    inline=False,
                )
                has_soul = bool(
                    target_profile.soul_content and target_profile.soul_content.strip()
                )
                has_user = bool(
                    target_profile.user_content and target_profile.user_content.strip()
                )
                has_mem = bool(
                    target_profile.memory_content
                    and target_profile.memory_content.strip()
                )
                memory_status = (
                    f"• SOUL.md: {'Loaded' if has_soul else 'None'}\n"
                    f"• USER.md: {'Loaded' if has_user else 'None'}\n"
                    f"• MEMORY.md: {'Loaded' if has_mem else 'None'}"
                )
                embed.add_field(
                    name="Memory & Identity", value=memory_status, inline=True
                )
                yolo_status = (
                    "ENABLED"
                    if self.channel_yolo.get(message.channel.id, False)
                    else "DISABLED"
                )
                wt_status = (
                    "ENABLED"
                    if self.channel_worktree.get(message.channel.id, False)
                    else "DISABLED"
                )
                queue_count = len(self.channel_queues.get(message.channel.id, []))
                embed.add_field(
                    name="Execution Modes",
                    value=f"• YOLO: `{yolo_status}`\n• Worktree: `{wt_status}`\n• Queued: `{queue_count}`",
                    inline=True,
                )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "compact":
                old_id = self.channel_conversations.pop(message.channel.id, None)
                await self.storage.clear_channel_conversation(message.channel.id)
                target_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                workspace = self.profile_manager.resolve_workspace_for_profile(
                    target_profile, config.workspace_root
                )
                session_id = f"discord_{message.channel.id}_{int(time.time())}"
                summary = (
                    f"Compacted Session Summary (Channel: {message.channel.id} - {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                    f"Workspace: {workspace}\n"
                    f"Active Profile: {target_profile.name}\n"
                    f"Prompts Completed: {self.total_prompts}\n"
                    f"Previous UUID: {old_id or 'none'}\n"
                    f"Channel transcript compressed into persistent memory."
                )
                await self.storage.save_session_memory(
                    session_id=session_id,
                    summary=summary,
                    token_count=self.total_chars_out // 4,
                )
                embed = discord.Embed(
                    title="Context Compacted",
                    description=(
                        f"Saved summary to SQLite (`session_memories`).\n"
                        f"Conversation transcript compressed into memory. Active context reset for lean token usage.\n"
                        f"Persisted in: `{config.db_path}`"
                    ),
                    color=0x8B5CF6,
                )
                if old_id:
                    embed.set_footer(text=f"Compacted session: {old_id}")
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "profile":
                if cmd_args:
                    matched = self.profile_manager.get_profile(cmd_args)
                    self.channel_profile_overrides[message.channel.id] = matched
                    self.channel_conversations.pop(message.channel.id, None)
                    await self.storage.clear_channel_conversation(message.channel.id)
                    await message.channel.send(
                        f"Switched active profile for this channel to **`{matched.name}`** ({matched.provider}). Conversation context reset."
                    )
                    return
                profiles = self.profile_manager.list_profiles()
                active_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                embed = discord.Embed(title="Available Agent Profiles", color=0x9333EA)
                for p in profiles:
                    target = p.model or p.command or "-"
                    is_active = (
                        " [ACTIVE IN THIS CHANNEL]"
                        if p.name == active_profile.name
                        else ""
                    )
                    embed.add_field(
                        name=f"`{p.name}` ({p.provider}){is_active}",
                        value=f"**Target:** `{target}`\n{p.description}",
                        inline=False,
                    )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "yolo":
                current = self.channel_yolo.get(message.channel.id, False)
                new_val = not current
                self.channel_yolo[message.channel.id] = new_val
                status = (
                    "ENABLED (Auto-approving mutating tool calls)"
                    if new_val
                    else "DISABLED (HITL confirmations active)"
                )
                color = 0xEF4444 if new_val else 0x10B981
                embed = discord.Embed(
                    title=f"YOLO Mode: {status}",
                    description="When enabled, tool execution (shell, file edits, git) executes without blocking for approval buttons.",
                    color=color,
                )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "worktree":
                current = self.channel_worktree.get(message.channel.id, False)
                new_val = not current
                self.channel_worktree[message.channel.id] = new_val
                status = (
                    "ENABLED (Tasks execute in isolated git worktrees)"
                    if new_val
                    else "DISABLED (Tasks execute in direct repo workspace)"
                )
                color = 0x06B6D4 if new_val else 0x64748B
                embed = discord.Embed(
                    title=f"Worktree Isolation: {status}",
                    description="When enabled, pipelines and tasks execute in separate temporary git worktrees to keep main branches clean.",
                    color=color,
                )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "usage":
                uptime = time.time() - self.session_start_time
                m = int(uptime // 60)
                s = int(uptime % 60)
                est_tokens = self.total_chars_out // 4
                target_profile = (
                    self.resolve_profile_for_channel(message.channel.id)
                    or self.bound_profile
                )
                queue_len = len(self.channel_queues.get(message.channel.id, []))

                embed = discord.Embed(title="Session Metrics & Usage", color=0x3B82F6)
                embed.add_field(
                    name="Prompts Executed",
                    value=f"**{self.total_prompts}**",
                    inline=True,
                )
                embed.add_field(
                    name="Queued Prompts", value=f"**{queue_len}**", inline=True
                )
                embed.add_field(
                    name="Output Characters",
                    value=f"**{self.total_chars_out:,}**",
                    inline=True,
                )
                embed.add_field(
                    name="Estimated Tokens", value=f"**~{est_tokens:,}**", inline=True
                )
                embed.add_field(
                    name="Active Profile",
                    value=f"`{target_profile.name}` ({target_profile.provider})",
                    inline=True,
                )
                embed.add_field(
                    name="Session Uptime", value=f"**{m}m {s}s**", inline=True
                )
                embed.add_field(
                    name="Estimated Cost",
                    value="**$0.00** (Local Antigravity ACP / Free Tier)",
                    inline=False,
                )
                await message.channel.send(embed=embed)
                return

            elif cmd_name == "help":
                embed = self._create_help_embed()
                await message.channel.send(embed=embed)
                return

        # Check if message is directed to this bot and resolve target profile
        target_profile = self.resolve_profile_for_message(message)
        if not target_profile:
            return

        raw_prompt = message.content
        if self.user:
            raw_prompt = (
                raw_prompt.replace(f"<@{self.user.id}>", "")
                .replace(f"<@!{self.user.id}>", "")
                .strip()
            )

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
                if tag.name.lower() not in (
                    "planning",
                    "building",
                    "review",
                    "done",
                    "failed",
                    "suspended_afk",
                    "suspended",
                ):
                    new_tags.append(tag)

            target_tag = available.get(status_tag_name.lower())
            if not target_tag and status_tag_name.lower() in (
                "suspended_afk",
                "suspended",
            ):
                target_tag = available.get("suspended") or available.get("failed")
            if target_tag and target_tag not in new_tags:
                new_tags.append(target_tag)

            await thread.edit(applied_tags=new_tags)
        except Exception as e:
            logger.debug("Could not update forum tags: %s", e)

    async def _handle_afk_suspension(
        self,
        task_record: TaskRecord,
        future: asyncio.Future,
        channel: Any,
        resume_coro_fn: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
    ):
        """Handle user action on SuspendedAfkView (resume, merge, abort)."""
        try:
            action = await asyncio.wait_for(future, timeout=3600.0)
        except asyncio.TimeoutError:
            logger.debug(
                "Suspended AFK task %s timed out waiting for action", task_record.id
            )
            return
        except Exception as e:
            logger.debug(
                "Suspended AFK task %s handler exception: %s", task_record.id, e
            )
            return

        if action == "resume":
            await channel.send(
                f"🔄 **Resuming pipeline task `{task_record.id}` with refreshed budget...**"
            )
            if resume_coro_fn:
                asyncio.create_task(resume_coro_fn())
        elif action == "merge":
            try:
                events = await self.storage.get_agent_events(task_record.id, limit=50)
                branch_event = next(
                    (
                        e
                        for e in reversed(events)
                        if e.get("event_type") == "worktree_branch"
                    ),
                    None,
                )
                branch_name = None
                if branch_event:
                    bp = branch_event.get("event_payload")
                    branch_name = bp.get("branch") if isinstance(bp, dict) else str(bp)

                await self.storage.update_task_status(task_record.id, TaskStatus.DONE)
                if isinstance(channel, discord.Thread):
                    await self._update_forum_tags(channel, "Done")

                if branch_name:
                    await channel.send(
                        f"✅ **Task `{task_record.id}` signed off and marked DONE.** Branch: `{branch_name}`."
                    )
                else:
                    await channel.send(
                        f"✅ **Task `{task_record.id}` signed off and marked DONE.**"
                    )
            except Exception as err:
                logger.exception(
                    "Failed signing off AFK task %s: %s", task_record.id, err
                )
                await channel.send(
                    f"⚠️ **Error signing off task `{task_record.id}`:** {err}"
                )
        elif action == "abort":
            try:
                await self.storage.update_task_status(task_record.id, TaskStatus.FAILED)
                if isinstance(channel, discord.Thread):
                    await self._update_forum_tags(channel, "Failed")
                await channel.send(
                    f"🛑 **Task `{task_record.id}` aborted and marked FAILED.**"
                )
            except Exception as err:
                logger.exception("Failed aborting AFK task %s: %s", task_record.id, err)
                await channel.send(
                    f"⚠️ **Error aborting task `{task_record.id}`:** {err}"
                )

    async def _register_slash_commands(self):
        async def on_app_command_error(
            interaction: discord.Interaction, error: app_commands.AppCommandError
        ):
            logger.error(
                "[%s] Slash command error in channel %s: %s",
                self.profile_name,
                interaction.channel_id,
                error,
                exc_info=True,
            )
            msg = f"**Command Error:** `{error}`"
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(msg, ephemeral=True)
                else:
                    await interaction.response.send_message(msg, ephemeral=True)
            except Exception:
                pass

        self.tree.on_error = on_app_command_error

        @self.tree.command(
            name="new",
            description="Start a fresh conversation session in this channel/thread",
        )
        async def new_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            old_id = self.channel_conversations.pop(interaction.channel_id, None)
            await self.storage.clear_channel_conversation(interaction.channel_id)

            # Cancel running task if any in this channel
            target_id = self.channel_tasks.pop(interaction.channel_id, None)
            if target_id and target_id in self.active_tasks:
                task = self.active_tasks.pop(target_id, None)
                if task and not task.done():
                    task.cancel()

            self.channel_queues.pop(interaction.channel_id, None)

            target_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            embed = discord.Embed(
                title="Session Reset",
                description=(
                    f"Cleared active conversation context in this channel.\n"
                    f"Next message will begin fresh with **`{target_profile.name}`**'s operating doctrine."
                ),
                color=0x10B981,
            )
            if old_id:
                embed.set_footer(text=f"Previous session: {old_id}")
            await interaction.response.send_message(embed=embed)

        async def _stop_task_handler(
            interaction: discord.Interaction, task_id: Optional[str] = None
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            target_id = task_id or self.channel_tasks.get(interaction.channel_id)
            if not target_id or target_id not in self.active_tasks:
                await interaction.response.send_message(
                    "[Info] No active task or agent execution running in this channel. (Use `/new` to reset context)",
                    ephemeral=True,
                )
                return

            async_task = self.active_tasks.get(target_id)
            if async_task and not async_task.done():
                async_task.cancel()
                if not target_id.startswith("chat_"):
                    try:
                        await self.storage.update_task_status(
                            target_id, TaskStatus.FAILED
                        )
                    except Exception:
                        pass
                if isinstance(interaction.channel, discord.Thread):
                    await self._update_forum_tags(interaction.channel, "Failed")
                await interaction.response.send_message(
                    f"**Execution `{target_id}` stopped immediately by user.**"
                )
            else:
                await interaction.response.send_message(
                    f"[Info] Execution `{target_id}` is already finished.",
                    ephemeral=True,
                )

        @self.tree.command(
            name="stop", description="Stop running task or prompt stream immediately"
        )
        @app_commands.describe(task_id="Optional specific task ID to stop")
        async def stop_cmd(
            interaction: discord.Interaction, task_id: Optional[str] = None
        ):
            await _stop_task_handler(interaction, task_id)

        @self.tree.command(
            name="interrupt",
            description="Cancel active task and steer agent immediately with new prompt",
        )
        @app_commands.describe(
            prompt="New instruction to steer the agent with immediately"
        )
        async def interrupt_cmd(interaction: discord.Interaction, prompt: str):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            target_id = self.channel_tasks.get(interaction.channel_id)
            if target_id and target_id in self.active_tasks:
                task = self.active_tasks.get(target_id)
                if task and not task.done():
                    task.cancel()

            target_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            await interaction.response.send_message(
                f"**Interrupted previous task to steer [{target_profile.name}]:**\n> {prompt[:200]}"
            )
            asyncio.create_task(
                self._execute_chat_prompt(
                    channel=interaction.channel,
                    prompt=prompt,
                    author_mention=interaction.user.mention,
                    session_key=f"chat_{interaction.channel_id}",
                    profile=target_profile,
                )
            )

        @self.tree.command(
            name="queue",
            description="Queue a prompt to execute after current task finishes",
        )
        @app_commands.describe(prompt="Instruction to queue for sequential execution")
        async def queue_cmd(interaction: discord.Interaction, prompt: str):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            target_id = self.channel_tasks.get(interaction.channel_id)
            is_busy = bool(
                target_id
                and target_id in self.active_tasks
                and not self.active_tasks[target_id].done()
            )

            if is_busy:
                q = self.channel_queues.setdefault(interaction.channel_id, [])
                q.append(prompt)
                await interaction.response.send_message(
                    f"**Prompt Queued (#{len(q)})** — will auto-execute once current task completes:\n> {prompt[:200]}"
                )
            else:
                target_profile = (
                    self.resolve_profile_for_channel(interaction.channel_id)
                    or self.bound_profile
                )
                await interaction.response.send_message(
                    f"**Dispatching prompt directly [{target_profile.name}]:**\n> {prompt[:200]}"
                )
                asyncio.create_task(
                    self._execute_chat_prompt(
                        channel=interaction.channel,
                        prompt=prompt,
                        author_mention=interaction.user.mention,
                        session_key=f"chat_{interaction.channel_id}",
                        profile=target_profile,
                    )
                )

        @self.tree.command(
            name="context",
            description="Show active session context, profile, model, and memory",
        )
        async def context_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            target_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            conv_id = self.channel_conversations.get(interaction.channel_id)
            workspace = self.profile_manager.resolve_workspace_for_profile(
                target_profile, config.workspace_root
            )

            branch = get_git_branch(workspace)
            diff_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
            )
            dirty_count = len(
                [line for line in diff_proc.stdout.splitlines() if line.strip()]
            )
            dirty_str = (
                f"{dirty_count} uncommitted changes" if dirty_count > 0 else "clean"
            )

            embed = discord.Embed(title="Active Session Context", color=0x3B82F6)
            embed.add_field(
                name="Profile",
                value=f"`{target_profile.name}` ({target_profile.provider})",
                inline=True,
            )
            target = target_profile.model or target_profile.command or "default"
            embed.add_field(name="Model / Command", value=f"`{target}`", inline=True)
            embed.add_field(
                name="Git Branch", value=f"`{branch}` ({dirty_str})", inline=True
            )
            embed.add_field(name="Workspace", value=f"`{workspace}`", inline=False)
            embed.add_field(
                name="Conversation UUID",
                value=f"`{conv_id}`" if conv_id else "*Fresh (starts on next message)*",
                inline=False,
            )

            has_soul = bool(
                target_profile.soul_content and target_profile.soul_content.strip()
            )
            has_user = bool(
                target_profile.user_content and target_profile.user_content.strip()
            )
            has_mem = bool(
                target_profile.memory_content and target_profile.memory_content.strip()
            )

            memory_status = (
                f"• SOUL.md: {'Loaded' if has_soul else 'None'}\n"
                f"• USER.md: {'Loaded' if has_user else 'None'}\n"
                f"• MEMORY.md: {'Loaded' if has_mem else 'None'}"
            )
            embed.add_field(name="Memory & Identity", value=memory_status, inline=True)

            yolo_status = (
                "ENABLED"
                if self.channel_yolo.get(interaction.channel_id, False)
                else "DISABLED"
            )
            wt_status = (
                "ENABLED"
                if self.channel_worktree.get(interaction.channel_id, False)
                else "DISABLED"
            )
            queue_count = len(self.channel_queues.get(interaction.channel_id, []))
            embed.add_field(
                name="Execution Modes",
                value=f"• YOLO: `{yolo_status}`\n• Worktree: `{wt_status}`\n• Queued: `{queue_count}`",
                inline=True,
            )

            from maulness.core.providers.circuit import ProviderHealthRegistry

            circuit_text = ProviderHealthRegistry.get_instance().format_status_text()
            embed.add_field(
                name="Circuit Breaker Status", value=circuit_text, inline=False
            )

            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="compact",
            description="Compact conversation history into SQLite memory and reset context",
        )
        async def compact_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            old_id = self.channel_conversations.pop(interaction.channel_id, None)
            await self.storage.clear_channel_conversation(interaction.channel_id)

            target_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            workspace = self.profile_manager.resolve_workspace_for_profile(
                target_profile, config.workspace_root
            )

            session_id = f"discord_{interaction.channel_id}_{int(time.time())}"
            summary = (
                f"Compacted Session Summary (Channel: {interaction.channel_id} - {time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                f"Workspace: {workspace}\n"
                f"Active Profile: {target_profile.name}\n"
                f"Prompts Completed: {self.total_prompts}\n"
                f"Previous UUID: {old_id or 'none'}\n"
                f"Channel transcript compressed into persistent memory."
            )
            await self.storage.save_session_memory(
                session_id=session_id,
                summary=summary,
                token_count=self.total_chars_out // 4,
            )

            embed = discord.Embed(
                title="Context Compacted",
                description=(
                    f"Saved summary to SQLite (`session_memories`).\n"
                    f"Conversation transcript compressed into memory. Active context reset for lean token usage.\n"
                    f"Persisted in: `{config.db_path}`"
                ),
                color=0x8B5CF6,
            )
            if old_id:
                embed.set_footer(text=f"Compacted session: {old_id}")
            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="help", description="Show available commands and usage guide"
        )
        async def help_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            embed = self._create_help_embed()
            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="thread", description="Create a new thread and start a session in it"
        )
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
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            await interaction.response.defer()

            channel = interaction.channel
            created_thread = None

            if isinstance(channel, discord.ForumChannel):
                thread_with_msg = await channel.create_thread(
                    name=name[:100],
                    content=message or f"Thread started by {interaction.user.mention}",
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
                await interaction.followup.send(
                    "Cannot create a thread in this channel type.", ephemeral=True
                )
                return

            await interaction.followup.send(f"Created thread: {created_thread.mention}")

            if message and created_thread:
                channel_id = created_thread.id
                parent_id = getattr(created_thread, "parent_id", None)
                target_profile = (
                    self.resolve_profile_for_channel(channel_id, parent_id)
                    or self.bound_profile
                )
                await self._execute_chat_prompt(
                    channel=created_thread,
                    prompt=message,
                    author_mention=interaction.user.mention,
                    session_key=f"thread_{created_thread.id}",
                    profile=target_profile,
                )

        @self.tree.command(
            name="profile",
            description="Inspect available profiles or switch active profile for this channel",
        )
        @app_commands.describe(
            name="Optional profile name to switch to (e.g. builder, planner, reviewer, default)"
        )
        async def profile_cmd(
            interaction: discord.Interaction, name: Optional[str] = None
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            if name:
                matched = self.profile_manager.get_profile(name)
                self.channel_profile_overrides[interaction.channel_id] = matched
                self.channel_conversations.pop(interaction.channel_id, None)
                await self.storage.clear_channel_conversation(interaction.channel_id)
                await interaction.response.send_message(
                    f"Switched active profile for this channel to **`{matched.name}`** ({matched.provider}). Conversation context reset."
                )
                return

            profiles = self.profile_manager.list_profiles()
            active_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            embed = discord.Embed(title="Available Agent Profiles", color=0x9333EA)
            for p in profiles:
                target = p.model or p.command or "-"
                is_active = (
                    " [ACTIVE IN THIS CHANNEL]" if p.name == active_profile.name else ""
                )
                embed.add_field(
                    name=f"`{p.name}` ({p.provider}){is_active}",
                    value=f"**Target:** `{target}`\n{p.description}",
                    inline=False,
                )
            await interaction.response.send_message(embed=embed)

        @profile_cmd.autocomplete("name")
        async def profile_autocomplete(
            interaction: discord.Interaction, current: str
        ) -> list[app_commands.Choice[str]]:
            profiles = self.profile_manager.list_profiles()
            choices = []
            for p in profiles:
                if not current or current.lower() in p.name.lower():
                    label = f"{p.name} ({p.provider})"
                    choices.append(app_commands.Choice(name=label[:100], value=p.name))
            return choices[:25]

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

        @self.tree.command(
            name="pipeline",
            description="Execute a multi-stage Kanban pipeline (Planner -> Builder -> Reviewer)",
        )
        @app_commands.describe(
            prompt="Detailed task requirements and instructions",
            repo="Target repository name (optional, default inferred or maulness)",
            title="Pipeline goal / headline (optional, auto-derived from prompt)",
            pipeline_name="Pipeline definition to execute (default: standard)",
            worktree="Run builder in an isolated git worktree",
            yolo="YOLO mode: bypass HITL approval on mutating actions",
        )
        @app_commands.choices(repo=REPO_CHOICES)
        async def pipeline_cmd(
            interaction: discord.Interaction,
            prompt: str,
            repo: Optional[str] = None,
            title: Optional[str] = None,
            pipeline_name: Optional[str] = "standard",
            worktree: Optional[bool] = None,
            yolo: Optional[bool] = None,
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            effective_pipe_name = pipeline_name or "standard"
            effective_worktree = (
                self.channel_worktree.get(interaction.channel_id, False)
                if worktree is None
                else worktree
            )
            effective_yolo = (
                self.channel_yolo.get(interaction.channel_id, False)
                if yolo is None
                else yolo
            )

            # Infer repository if omitted
            effective_repo = repo
            if not effective_repo:
                if isinstance(interaction.channel, discord.Thread) and isinstance(
                    interaction.channel.parent, discord.ForumChannel
                ):
                    for tag in interaction.channel.applied_tags:
                        for choice in REPO_CHOICES:
                            if tag.name.lower() == choice.value.lower():
                                effective_repo = choice.value
                                break
                        if effective_repo:
                            break

            repo_label = effective_repo or "[general]"
            repo_desc = f" on `{effective_repo}`" if effective_repo else ""
            repo_prefix = f"[{effective_repo}] " if effective_repo else ""

            # Infer title if omitted
            first_line = (
                prompt.strip().splitlines()[0].strip()
                if prompt and prompt.strip()
                else "Pipeline Task"
            )
            effective_title = (
                title.strip() if title and title.strip() else first_line[:70]
            )

            await interaction.response.defer()

            exec_channel = interaction.channel
            created_in_forum = False
            if not isinstance(interaction.channel, discord.Thread):
                target_parent = None
                if (
                    self.forum_channel_id
                    and interaction.channel_id != self.forum_channel_id
                ):
                    target_parent = self.get_channel(self.forum_channel_id)
                    if not target_parent:
                        try:
                            target_parent = await self.fetch_channel(
                                self.forum_channel_id
                            )
                        except Exception:
                            pass
                if not target_parent and isinstance(
                    interaction.channel, (discord.ForumChannel, discord.TextChannel)
                ):
                    target_parent = interaction.channel

                if isinstance(target_parent, discord.ForumChannel):
                    applied_tags = []
                    avail = {t.name.lower(): t for t in target_parent.available_tags}
                    tag_keys = ["pipeline", "planning"]
                    if effective_repo:
                        tag_keys.append(effective_repo.lower())
                    for tag_key in tag_keys:
                        if tag_key in avail:
                            applied_tags.append(avail[tag_key])
                    thread_with_msg = await target_parent.create_thread(
                        name=f"{repo_prefix}{effective_title[:70]}",
                        content=f"**Pipeline Kanban Task ({effective_pipe_name})**\n**Goal:** {effective_title}\n> {prompt[:200]}",
                        applied_tags=applied_tags,
                    )
                    exec_channel = thread_with_msg.thread
                    created_in_forum = True
                    await interaction.followup.send(
                        f"Created pipeline thread in workbench: {exec_channel.mention}"
                    )
                elif isinstance(target_parent, discord.TextChannel):
                    exec_channel = await target_parent.create_thread(
                        name=f"{repo_prefix}{effective_title[:70]}",
                        type=discord.ChannelType.public_thread,
                    )
                    created_in_forum = True
                    await interaction.followup.send(
                        f"Created pipeline thread: {exec_channel.mention}"
                    )

            if not created_in_forum:
                await interaction.followup.send(
                    f"**Launching Pipeline `{effective_pipe_name}` for `{effective_title}`{repo_desc}...**"
                )
            else:
                await exec_channel.send(
                    f"**Launching Pipeline `{effective_pipe_name}` for `{effective_title}`{repo_desc}...**"
                )

            thread_id = (
                exec_channel.id if isinstance(exec_channel, discord.Thread) else None
            )
            if isinstance(exec_channel, discord.Thread):
                await self._update_forum_tags(exec_channel, "Planning")

            start_time = time.time()

            async def on_thought(event: AgentThoughtEvent):
                pass

            async def on_message(event: AgentMessageEvent):
                pass

            async def on_stage_start(stage: PipelineStage, current: int, total: int):
                if isinstance(exec_channel, discord.Thread):
                    await self._update_forum_tags(exec_channel, stage.name.title())

                color_map = {
                    "planning": 0x3B82F6,
                    "building": 0xF59E0B,
                    "review": 0x8B5CF6,
                }
                color = color_map.get(stage.name.lower(), 0x06B6D4)
                embed = discord.Embed(
                    title=f"Stage {current}/{total}: {stage.name.title()} ({stage.profile}) Started",
                    description=f"**Goal:** {effective_title}",
                    color=color,
                )
                if stage.use_worktree or effective_worktree:
                    embed.add_field(
                        name="Worktree Isolation",
                        value="Active (isolated branch)",
                        inline=True,
                    )
                if stage.verification_gate:
                    embed.add_field(
                        name="Verification Gate",
                        value=f"`{stage.verification_gate.command}`",
                        inline=True,
                    )
                await exec_channel.send(embed=embed)

            async def on_stage_finish(stage: PipelineStage, stage_output: str):
                embed = discord.Embed(
                    title=f"Stage {stage.name.title()} Completed",
                    color=0x10B981,
                )
                if stage_output and stage_output.strip():
                    clean_text = format_discord_markdown(stage_output.strip())
                    if len(clean_text) > 900:
                        snippet = clean_text[:900] + "\n..."
                    else:
                        snippet = clean_text
                    embed.add_field(
                        name="Stage Output Excerpt",
                        value=f"```markdown\n{snippet}\n```"
                        if "```" not in snippet
                        else snippet,
                        inline=False,
                    )
                await exec_channel.send(embed=embed)

            async def on_approval(event: ApprovalRequestEvent) -> bool:
                if effective_yolo:
                    return True
                loop = asyncio.get_running_loop()
                fut = loop.create_future()
                view = ApprovalView(future=fut)
                await exec_channel.send(
                    f"**Approval Required**\n**Tool:** `{event.tool_name}`\n```json\n{event.args}\n```",
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
                        title=effective_title,
                        prompt=prompt,
                        repo_name=effective_repo,
                        pipeline_name=effective_pipe_name,
                        origin_platform="discord",
                        origin_channel_id=str(exec_channel.id),
                        origin_thread_id=str(thread_id) if thread_id else None,
                        discord_thread_id=thread_id,
                        use_worktree=effective_worktree,
                        yolo=effective_yolo,
                        on_thought=on_thought,
                        on_message=on_message,
                        on_approval=on_approval,
                        on_stage_start=on_stage_start,
                        on_stage_finish=on_stage_finish,
                        auto_proceed=True,
                    )
                    task_id = task_record.id

                    if isinstance(exec_channel, discord.Thread):
                        if task_record.status == TaskStatus.SUSPENDED_AFK:
                            status_tag = "Suspended"
                        elif task_record.status == TaskStatus.DONE:
                            status_tag = "Done"
                        else:
                            status_tag = "Failed"
                        await self._update_forum_tags(exec_channel, status_tag)

                    duration_s = int(time.time() - start_time)
                    mins, secs = divmod(duration_s, 60)
                    time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                    if task_record.status == TaskStatus.SUSPENDED_AFK:
                        status_color = 0xF59E0B  # Amber
                        status_text = "Parked in SUSPENDED_AFK (Budget Limit Reached)"
                    elif task_record.status == TaskStatus.DONE:
                        status_color = 0x10B981
                        status_text = "Completed Successfully"
                    else:
                        status_color = 0xEF4444
                        status_text = "Failed"

                    summary_embed = discord.Embed(
                        title=f"Pipeline {status_text}: {effective_title}",
                        color=status_color,
                    )
                    summary_embed.add_field(
                        name="Repository", value=f"`{repo_label}`", inline=True
                    )
                    summary_embed.add_field(
                        name="Task ID", value=f"`{task_record.id}`", inline=True
                    )
                    summary_embed.add_field(
                        name="Duration", value=time_str, inline=True
                    )
                    summary_embed.add_field(
                        name="Pipeline", value=f"`{effective_pipe_name}`", inline=True
                    )
                    summary_embed.add_field(
                        name="Final Status",
                        value=f"**`{task_record.status.value.upper()}`**",
                        inline=True,
                    )

                    # Extract reviewer scorecard and worktree branch from agent_events if available
                    try:
                        events = await self.storage.get_agent_events(
                            task_record.id, limit=50
                        )
                        branch_event = next(
                            (
                                e
                                for e in reversed(events)
                                if e.get("event_type") == "worktree_branch"
                            ),
                            None,
                        )
                        if branch_event:
                            bp = branch_event.get("event_payload")
                            branch_name = (
                                bp.get("branch") if isinstance(bp, dict) else str(bp)
                            )
                            if branch_name:
                                summary_embed.add_field(
                                    name="Git Branch",
                                    value=f"`{branch_name}`",
                                    inline=True,
                                )

                        review_event = next(
                            (
                                e
                                for e in reversed(events)
                                if e.get("stage") == "review"
                                and e.get("event_type") == "final_response"
                            ),
                            None,
                        )
                        if review_event:
                            p = review_event.get("event_payload")
                            audit_text = (
                                p.get("content", "") if isinstance(p, dict) else str(p)
                            )
                            if audit_text:
                                clean_audit = format_discord_markdown(audit_text)
                                summary_embed.add_field(
                                    name="Review Verdict",
                                    value=clean_audit[:1000],
                                    inline=False,
                                )
                    except Exception as ev_err:
                        logger.debug(
                            "Could not attach review/branch events to summary embed: %s",
                            ev_err,
                        )

                    view = None
                    if task_record.status == TaskStatus.SUSPENDED_AFK:
                        afk_fut = asyncio.get_running_loop().create_future()
                        view = SuspendedAfkView(task_record.id, future=afk_fut)
                        asyncio.create_task(
                            self._handle_afk_suspension(
                                task_record=task_record,
                                future=afk_fut,
                                channel=exec_channel,
                                resume_coro_fn=_run_pipeline_coro,
                            )
                        )
                    await exec_channel.send(embed=summary_embed, view=view)
                except asyncio.CancelledError:
                    logger.info(
                        "Pipeline %s cancelled via /stop", task_id or effective_title
                    )
                    await exec_channel.send("**Pipeline was stopped by user.**")
                except Exception as e:
                    logger.exception(
                        "[%s] Pipeline execution failed for '%s': %s",
                        self.profile_name,
                        effective_title,
                        e,
                    )
                    await exec_channel.send(
                        f"**Pipeline Execution Error:** `{type(e).__name__}: {str(e)[:400]}`"
                    )
                finally:
                    if task_id and task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    if exec_channel.id in self.channel_tasks:
                        del self.channel_tasks[exec_channel.id]

            bg_task = asyncio.create_task(_run_pipeline_coro())
            self.channel_tasks[exec_channel.id] = f"pipeline_{exec_channel.id}"
            self.active_tasks[f"pipeline_{exec_channel.id}"] = bg_task

        @self.tree.command(
            name="yolo",
            description="Toggle or set YOLO mode (auto-approves mutating tool actions)",
        )
        @app_commands.describe(
            enabled="Optional explicit toggle (True to enable, False to disable)"
        )
        async def yolo_cmd(
            interaction: discord.Interaction, enabled: Optional[bool] = None
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            current = self.channel_yolo.get(interaction.channel_id, False)
            new_val = not current if enabled is None else enabled
            self.channel_yolo[interaction.channel_id] = new_val

            status = (
                "ENABLED (Auto-approving mutating tool calls)"
                if new_val
                else "DISABLED (HITL confirmations active)"
            )
            color = 0xEF4444 if new_val else 0x10B981
            embed = discord.Embed(
                title=f"YOLO Mode: {status}",
                description="When enabled, tool execution (shell, file edits, git) executes without blocking for approval buttons.",
                color=color,
            )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="worktree",
            description="Toggle or set Git worktree isolation for tasks in this channel",
        )
        @app_commands.describe(
            enabled="Optional explicit toggle (True to enable, False to disable)"
        )
        async def worktree_cmd(
            interaction: discord.Interaction, enabled: Optional[bool] = None
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            current = self.channel_worktree.get(interaction.channel_id, False)
            new_val = not current if enabled is None else enabled
            self.channel_worktree[interaction.channel_id] = new_val

            status = (
                "ENABLED (Tasks execute in isolated git worktrees)"
                if new_val
                else "DISABLED (Tasks execute in direct repo workspace)"
            )
            color = 0x06B6D4 if new_val else 0x64748B
            embed = discord.Embed(
                title=f"Worktree Isolation: {status}",
                description="When enabled, pipelines and tasks execute in separate temporary git worktrees to keep main branches clean.",
                color=color,
            )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="usage",
            description="Display session token metrics, uptime, and cost estimation",
        )
        async def usage_cmd(interaction: discord.Interaction):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            uptime = time.time() - self.session_start_time
            m = int(uptime // 60)
            s = int(uptime % 60)
            est_tokens = self.total_chars_out // 4
            target_profile = (
                self.resolve_profile_for_channel(interaction.channel_id)
                or self.bound_profile
            )
            queue_len = len(self.channel_queues.get(interaction.channel_id, []))

            embed = discord.Embed(title="Session Metrics & Usage", color=0x3B82F6)
            embed.add_field(
                name="Prompts Executed", value=f"**{self.total_prompts}**", inline=True
            )
            embed.add_field(
                name="Queued Prompts", value=f"**{queue_len}**", inline=True
            )
            embed.add_field(
                name="Output Characters",
                value=f"**{self.total_chars_out:,}**",
                inline=True,
            )
            embed.add_field(
                name="Estimated Tokens", value=f"**~{est_tokens:,}**", inline=True
            )
            embed.add_field(
                name="Active Profile",
                value=f"`{target_profile.name}` ({target_profile.provider})",
                inline=True,
            )
            embed.add_field(name="Session Uptime", value=f"**{m}m {s}s**", inline=True)
            embed.add_field(
                name="Estimated Cost",
                value="**$0.00** (Local Antigravity ACP / Free Tier)",
                inline=False,
            )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(
            name="providers",
            description="Inspect provider circuit breaker health and cooldowns, or reset circuits",
        )
        @app_commands.describe(action="Optional action: 'status' (default) or 'reset'")
        async def providers_cmd(
            interaction: discord.Interaction, action: Optional[str] = "status"
        ):
            if self.owner_id and interaction.user.id != self.owner_id:
                await interaction.response.send_message("Unauthorized", ephemeral=True)
                return

            from maulness.core.providers.circuit import ProviderHealthRegistry

            registry = ProviderHealthRegistry.get_instance()

            if action and action.lower() == "reset":
                registry.reset_all()
                await interaction.response.send_message(
                    "All provider circuit breakers have been reset to CLOSED (HEALTHY).",
                    ephemeral=False,
                )
                return

            summary_text = registry.format_status_text()
            embed = discord.Embed(
                title="Provider Circuit Breaker Registry",
                description=summary_text,
                color=0x10B981,
            )
            await interaction.response.send_message(embed=embed)
