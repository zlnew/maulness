import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider

logger = logging.getLogger("maulness.providers.sdk")


class AntigravitySdkProvider(BaseProvider):
    """In-process Antigravity provider using the official google-antigravity Python SDK."""

    def __init__(self, profile: Profile):
        super().__init__(profile)
        self.last_conversation_id: Optional[str] = None

    async def run(
        self,
        session_id: str,
        prompt: str,
        workspace_path: Optional[Path] = None,
        conversation_id: Optional[str] = None,
        on_init: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        on_thought: Optional[
            Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_message: Optional[
            Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_tool_call: Optional[
            Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]
        ] = None,
        on_approval: Optional[
            Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]
        ] = None,
    ) -> str:
        """Run an agent session using the in-process google.antigravity Agent."""
        try:
            from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
        except ImportError as e:
            raise ImportError(
                "google-antigravity SDK is not installed. Install it via `uv pip install google-antigravity`"
            ) from e

        cwd = Path(workspace_path) if workspace_path else config.workspace_root
        api_key = self.profile.get_api_key() or config.gemini_api_key

        from maulness.core.skills import SkillManager

        skill_mgr = SkillManager()
        skills_paths = skill_mgr.get_skills_paths(
            profile_skills_dir=self.profile.skills_dir
        )

        sdk_config = LocalAgentConfig(
            system_instructions=self.profile.effective_system_prompt(),
            workspaces=[str(cwd)],
            capabilities=CapabilitiesConfig(),
            model=self.profile.model or "gemini-2.5-flash",
            api_key=api_key,
            vertex=self.profile.vertex or None,
            project=self.profile.project or None,
            location=self.profile.location or "us-central1",
            conversation_id=conversation_id,
            skills_paths=skills_paths or None,
            env=self.profile.env_vars or None,
        )

        accumulated: list[str] = []

        async with Agent(sdk_config) as agent:
            conv_id = getattr(agent, "conversation_id", None) or conversation_id
            if conv_id:
                self.last_conversation_id = conv_id
                if on_init:
                    await on_init(conv_id)

            response = await agent.chat(prompt)

            async def _stream_thoughts():
                try:
                    async for thought in response.thoughts:
                        if on_thought:
                            await on_thought(
                                AgentThoughtEvent(
                                    delta=str(thought), session_id=session_id
                                )
                            )
                except Exception as e:
                    logger.debug("Thought streaming ended or error: %s", e)

            async def _stream_tools():
                try:
                    async for call in response.tool_calls:
                        if on_tool_call:
                            await on_tool_call(
                                AgentToolCallEvent(
                                    call_id=str(getattr(call, "id", "")),
                                    tool_name=getattr(call, "name", "tool"),
                                    args=getattr(call, "args", {}),
                                    session_id=session_id,
                                )
                            )
                except Exception as e:
                    logger.debug("Tool call streaming ended or error: %s", e)

            thought_task = asyncio.create_task(_stream_thoughts())
            tool_task = asyncio.create_task(_stream_tools())

            try:
                async for chunk in response.chunks:
                    accumulated.append(chunk)
                    if on_message:
                        await on_message(
                            AgentMessageEvent(delta=chunk, session_id=session_id)
                        )
            finally:
                await asyncio.gather(thought_task, tool_task, return_exceptions=True)

        return "".join(accumulated)
