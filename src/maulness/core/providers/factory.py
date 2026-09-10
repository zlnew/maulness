from maulness.core.profiles import Profile
from maulness.core.providers.acp_provider import AcpProvider
from maulness.core.providers.api_provider import UnifiedApiProvider
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.fallback import FallbackProviderChain
from maulness.core.providers.gemini_provider import GeminiProvider
from maulness.core.providers.sdk_provider import AntigravitySdkProvider


def _instantiate_single_provider(profile: Profile) -> BaseProvider:
    provider_name = profile.provider.lower().strip().replace("-", "_")

    if provider_name == "acp":
        return AcpProvider(profile)
    elif provider_name in ("antigravity_sdk", "sdk", "antigravity"):
        return AntigravitySdkProvider(profile)
    elif provider_name == "gemini":
        return GeminiProvider(profile)
    elif provider_name in (
        "openai",
        "anthropic",
        "openrouter",
        "opencode",
        "opencode_go",
        "opencode_zen",
        "deepseek",
        "ollama",
        "openai_compatible",
    ):
        return UnifiedApiProvider(profile)
    else:
        if profile.command:
            return AcpProvider(profile)
        if profile.base_url:
            return UnifiedApiProvider(profile)
        return GeminiProvider(profile)


def get_provider_for_profile(profile: Profile) -> BaseProvider:
    """Factory function returning the corresponding execution provider or fallback chain."""
    primary = _instantiate_single_provider(profile)

    if not profile.fallbacks:
        return primary

    fallback_providers: list[BaseProvider] = []
    for fb in profile.fallbacks:
        fb_dict = fb.model_dump(exclude_none=True) if hasattr(fb, "model_dump") else (fb if isinstance(fb, dict) else {"provider": str(fb)})
        fb_profile = profile.model_copy(deep=True)
        fb_profile.resilience.fallbacks = []

        if "provider" in fb_dict and fb_dict["provider"]:
            fb_profile.agent_cfg.provider = fb_dict["provider"]
        target_model = fb_dict.get("model")
        if target_model:
            fb_profile.agent_cfg.model = target_model
            fb_profile.identity.name = f"{profile.name}-{target_model}"
        if "base_url" in fb_dict and fb_dict["base_url"]:
            fb_profile.agent_cfg.base_url = fb_dict["base_url"]
        if "api_key_env" in fb_dict and fb_dict["api_key_env"]:
            fb_profile.agent_cfg.api_key_env = fb_dict["api_key_env"]
        if "command" in fb_dict and fb_dict["command"]:
            fb_profile.agent_cfg.command = fb_dict["command"]
        if "reasoning_effort" in fb_dict and fb_dict["reasoning_effort"]:
            fb_profile.agent_cfg.reasoning_effort = fb_dict["reasoning_effort"]

        fallback_providers.append(_instantiate_single_provider(fb_profile))

    return FallbackProviderChain(primary=primary, fallbacks=fallback_providers)
