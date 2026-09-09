from maulness.core.profiles import Profile
from maulness.core.providers.acp_provider import AcpProvider
from maulness.core.providers.api_provider import UnifiedApiProvider
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.gemini_provider import GeminiProvider


def get_provider_for_profile(profile: Profile) -> BaseProvider:
    """Factory function returning the corresponding execution provider for a profile."""
    provider_name = profile.provider.lower().strip()

    if provider_name == "acp":
        return AcpProvider(profile)
    elif provider_name == "gemini":
        return GeminiProvider(profile)
    elif provider_name in ("openai", "anthropic", "openrouter"):
        return UnifiedApiProvider(profile)
    else:
        # Default fallback to ACP if command exists or Gemini
        if profile.command:
            return AcpProvider(profile)
        return GeminiProvider(profile)
