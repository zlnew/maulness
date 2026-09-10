import pytest
from maulness.core.profiles import Profile
from maulness.core.providers.acp_provider import AcpProvider
from maulness.core.providers.api_provider import UnifiedApiProvider
from maulness.core.providers.factory import get_provider_for_profile
from maulness.core.providers.gemini_provider import GeminiProvider


def test_provider_factory_acp():
    profile = Profile(name="builder", provider="acp")
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, AcpProvider)


def test_provider_factory_gemini():
    profile = Profile(name="planner", provider="gemini")
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, GeminiProvider)


def test_provider_factory_unified_api():
    profile = Profile(name="coder", provider="anthropic")
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, UnifiedApiProvider)


def test_provider_factory_fallback_chain():
    profile = Profile(
        name="resilient",
        provider="gemini",
        fallbacks=[{"provider": "anthropic", "model": "claude-3-5-sonnet"}],
    )
    provider = get_provider_for_profile(profile)
    from maulness.core.providers.fallback import FallbackProviderChain
    assert isinstance(provider, FallbackProviderChain)
    assert len(provider.fallbacks) == 1
    assert isinstance(provider.primary, GeminiProvider)
    assert isinstance(provider.fallbacks[0], UnifiedApiProvider)


@pytest.mark.asyncio
async def test_fallback_provider_chain_execution_failover():
    from unittest.mock import AsyncMock
    from maulness.core.providers.fallback import FallbackProviderChain
    from maulness.core.providers.base import BaseProvider

    p1 = Profile(name="p1", provider="gemini")
    p2 = Profile(name="p2", provider="anthropic")

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

