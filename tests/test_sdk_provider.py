import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from maulness.core.models import AgentMessageEvent, AgentThoughtEvent
from maulness.core.profiles import Profile
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.providers.sdk_provider import AntigravitySdkProvider


def test_sdk_provider_factory():
    profile = Profile(identity={"name": "sdk_agent"}, agent={"provider": "antigravity_sdk", "model": "gemini-2.5-flash"})
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, AntigravitySdkProvider)


@pytest.mark.asyncio
async def test_sdk_provider_run(monkeypatch):
    profile = Profile(
        identity={"name": "test_sdk"},
        agent={"provider": "antigravity_sdk", "model": "gemini-2.5-flash", "api_key_env": "TEST_KEY"},
    )
    provider = AntigravitySdkProvider(profile)

    class MockResponse:
        def __init__(self):
            self.chunks = self._async_chunks(["Hello ", "from ", "SDK!"])
            self.thoughts = self._async_thoughts(["Thinking step 1", "Thinking step 2"])
            self.tool_calls = self._async_tools([])

        async def _async_chunks(self, items):
            for item in items:
                yield item

        async def _async_thoughts(self, items):
            for item in items:
                yield item

        async def _async_tools(self, items):
            for item in items:
                yield item

    class MockAgent:
        def __init__(self, config):
            self.config = config
            self.conversation_id = "test-sdk-conv-uuid"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def chat(self, prompt):
            return MockResponse()

    # Mock the google.antigravity imports
    mock_ga = MagicMock()
    mock_ga.Agent = MockAgent
    mock_ga.LocalAgentConfig = MagicMock(return_value=MagicMock())
    mock_ga.CapabilitiesConfig = MagicMock(return_value=MagicMock())

    import sys
    monkeypatch.setitem(sys.modules, "google.antigravity", mock_ga)

    received_chunks = []
    received_thoughts = []
    captured_init = []

    async def on_message(event: AgentMessageEvent):
        received_chunks.append(event.delta)

    async def on_thought(event: AgentThoughtEvent):
        received_thoughts.append(event.delta)

    async def on_init(conv_id: str):
        captured_init.append(conv_id)

    result = await provider.run(
        session_id="sess_123",
        prompt="Hi SDK",
        on_message=on_message,
        on_thought=on_thought,
        on_init=on_init,
    )

    assert result == "Hello from SDK!"
    assert received_chunks == ["Hello ", "from ", "SDK!"]
    assert received_thoughts == ["Thinking step 1", "Thinking step 2"]
    assert captured_init == ["test-sdk-conv-uuid"]
    assert provider.last_conversation_id == "test-sdk-conv-uuid"
