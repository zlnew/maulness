import pytest
from maulness.core.providers.circuit import ProviderHealthRegistry


@pytest.fixture(autouse=True)
def reset_provider_health_registry():
    """Ensure circuit breaker health registry is clean before each test."""
    ProviderHealthRegistry.reset_instance()
    yield
    ProviderHealthRegistry.reset_instance()
