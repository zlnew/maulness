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
    profile = Profile(name="coder", provider="anthropic", api_key_env="ANTHROPIC_API_KEY")
    provider = get_provider_for_profile(profile)
    assert isinstance(provider, UnifiedApiProvider)
