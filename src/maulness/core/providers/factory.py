from maulness.core.profiles import Profile
from maulness.core.providers.acp_provider import AcpProvider
from maulness.core.providers.api_provider import UnifiedApiProvider
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.fallback import FallbackProviderChain
from maulness.core.providers.gemini_provider import GeminiProvider
from maulness.core.providers.sdk_provider import AntigravitySdkProvider


def _instantiate_single_provider(profile: Profile) -> BaseProvider:
    provider_name = profile.provider.lower().strip()

    if provider_name == "acp":
        return AcpProvider(profile)
    elif provider_name in ("antigravity_sdk", "sdk", "antigravity"):
        return AntigravitySdkProvider(profile)
    elif provider_name == "gemini":
        return GeminiProvider(profile)
    elif provider_name in ("openai", "anthropic", "openrouter"):
        return UnifiedApiProvider(profile)
    else:
        if profile.command:
            return AcpProvider(profile)
        return GeminiProvider(profile)


def get_provider_for_profile(profile: Profile) -> BaseProvider:
    """Factory function returning the corresponding execution provider or fallback chain."""
    primary = _instantiate_single_provider(profile)

    if not profile.fallbacks:
        return primary

    fallback_providers: list[BaseProvider] = []
    for fb in profile.fallbacks:
        fb_dict = fb if isinstance(fb, dict) else {"provider": str(fb)}
        fb_profile_data = profile.model_dump(mode="python", exclude={"fallbacks"})
        fb_profile_data.update(fb_dict)
        fb_profile = Profile(**fb_profile_data)
        fallback_providers.append(_instantiate_single_provider(fb_profile))

    return FallbackProviderChain(primary=primary, fallbacks=fallback_providers)
