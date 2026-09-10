from pathlib import Path
from typing import Any, Callable, Coroutine, Optional
from google import genai
from google.genai import types

from maulness.config import config
from maulness.core.models import (
    AgentMessageEvent,
    AgentThoughtEvent,
    AgentToolCallEvent,
    ApprovalRequestEvent,
)
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Direct Google Gemini API provider using the official google-genai SDK."""

    async def run(
        self,
        session_id: str,
        prompt: str,
        workspace_path: Optional[Path] = None,
        conversation_id: Optional[str] = None,
        on_init: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
        on_thought: Optional[Callable[[AgentThoughtEvent], Coroutine[Any, Any, None]]] = None,
        on_message: Optional[Callable[[AgentMessageEvent], Coroutine[Any, Any, None]]] = None,
        on_tool_call: Optional[Callable[[AgentToolCallEvent], Coroutine[Any, Any, None]]] = None,
        on_approval: Optional[Callable[[ApprovalRequestEvent], Coroutine[Any, Any, bool]]] = None,
    ) -> str:
        api_key = self.profile.get_api_key() or config.gemini_api_key
        if not api_key:
            import shutil
            if shutil.which(config.agy_cmd[0]):
                from maulness.core.providers.acp_provider import AcpProvider
                fallback_provider = AcpProvider(self.profile)
                return await fallback_provider.run(
                    session_id=session_id,
                    prompt=prompt,
                    workspace_path=workspace_path,
                    conversation_id=conversation_id,
                    on_init=on_init,
                    on_thought=on_thought,
                    on_message=on_message,
                    on_tool_call=on_tool_call,
                    on_approval=on_approval,
                )
            raise ValueError(
                f"Missing API key for profile '{self.profile.name}'. "
                f"Set {self.profile.api_key_env or 'GEMINI_API_KEY'} in ~/.config/maulness/env"
            )

        client = genai.Client(api_key=api_key)
        model_name = self.profile.model or "gemini-2.5-flash"

        gen_config = types.GenerateContentConfig(
            system_instruction=self.profile.effective_system_prompt(),
            temperature=self.profile.temperature,
            max_output_tokens=self.profile.max_tokens,
        )

        accumulated = []
        response_stream = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=gen_config,
        )

        async for chunk in response_stream:
            # Check for thinking parts if present
            if hasattr(chunk, "candidates") and chunk.candidates:
                for cand in chunk.candidates:
                    if hasattr(cand, "content") and cand.content:
                        for part in cand.content.parts:
                            if getattr(part, "thought", None) and on_thought:
                                await on_thought(
                                    AgentThoughtEvent(delta=part.text, session_id=session_id)
                                )

            if chunk.text:
                accumulated.append(chunk.text)
                if on_message:
                    await on_message(
                        AgentMessageEvent(delta=chunk.text, session_id=session_id)
                    )

        return "".join(accumulated)
