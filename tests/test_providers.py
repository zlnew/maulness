import pytest
from maulness.core.profiles import Profile
from maulness.core.providers.acp_provider import AcpProvider
from maulness.core.providers.api_provider import UnifiedApiProvider
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.providers.gemini_provider import GeminiProvider


def test_provider_factory_acp():
    profile = Profile(identity={"name": "builder"}, agent={"provider": "acp", "command": "agy"})
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, AcpProvider)


def test_provider_factory_acp_missing_command_raises():
    with pytest.raises(ValueError, match="specifies provider 'acp' but has no 'command'"):
        Profile(identity={"name": "invalid_builder"}, agent={"provider": "acp"})


def test_provider_factory_opencode_go():
    profile = Profile(
        identity={"name": "cheap_coder"},
        agent={"provider": "opencode_go", "model": "minimax-01"},
        env_vars={"OPENCODE_API_KEY": "test_key"},
    )
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, UnifiedApiProvider)
    assert provider.base_url == "https://api.opencode.ai/v1"
    assert provider.api_key == "test_key"


def test_provider_factory_custom_base_url():
    profile = Profile(
        identity={"name": "local_llm"},
        agent={"provider": "openai_compatible", "model": "llama3", "base_url": "http://localhost:11434/v1"},
        env_vars={"OPENAI_API_KEY": "dummy"},
    )
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, UnifiedApiProvider)
    assert provider.base_url == "http://localhost:11434/v1"


def test_provider_factory_gemini():
    profile = Profile(identity={"name": "planner"}, agent={"provider": "gemini"})
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, GeminiProvider)


def test_provider_factory_unified_api():
    profile = Profile(identity={"name": "coder"}, agent={"provider": "anthropic"})
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, UnifiedApiProvider)


def test_provider_factory_fallback_chain():
    profile = Profile(
        identity={"name": "resilient"},
        agent={"provider": "gemini"},
        resilience={
            "fallbacks": [
                {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "base_url": "https://api.anthropic.com/v1",
                }
            ]
        },
    )
    provider = get_provider_for_profile(profile)
    from maulness.core.providers.fallback import FallbackProviderChain

    assert isinstance(provider, FallbackProviderChain)
    assert len(provider.fallbacks) == 1
    assert isinstance(provider.primary, GeminiProvider)
    assert isinstance(provider.fallbacks[0], UnifiedApiProvider)
    assert provider.fallbacks[0].base_url == "https://api.anthropic.com/v1"


@pytest.mark.asyncio
async def test_fallback_provider_chain_execution_failover():
    from unittest.mock import AsyncMock
    from maulness.core.providers.fallback import FallbackProviderChain
    from maulness.core.providers.base import BaseProvider

    p1 = Profile(identity={"name": "p1"}, agent={"provider": "gemini"})
    p2 = Profile(identity={"name": "p2"}, agent={"provider": "anthropic"})

    mock_primary = AsyncMock(spec=BaseProvider)
    mock_primary.profile = p1
    mock_primary.run.side_effect = RuntimeError("Primary quota exceeded 429")

    mock_fallback = AsyncMock(spec=BaseProvider)
    mock_fallback.profile = p2
    mock_fallback.run.return_value = "Fallback succeeded!"

    chain = FallbackProviderChain(primary=mock_primary, fallbacks=[mock_fallback])
    result = await chain.run(session_id="s1", prompt="test prompt")

    assert result == "Fallback succeeded!"
    mock_primary.run.assert_called_once()
    mock_fallback.run.assert_called_once()

