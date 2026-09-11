import pytest
from maulness.core.providers.circuit import ProviderHealthRegistry


@pytest.fixture(autouse=True)
def reset_provider_health_registry():
    """Ensure circuit breaker health registry is clean before each test."""
    ProviderHealthRegistry.reset_instance()
    yield
    ProviderHealthRegistry.reset_instance()


@pytest.fixture(autouse=True)
async def isolate_test_db(tmp_path, monkeypatch):
    """Ensure every test operates on an isolated SQLite database."""
    from maulness.config import config
    from maulness.storage.db import StorageManager
    test_db = tmp_path / "maulness_test.db"
    monkeypatch.setenv("MAULNESS_DB_PATH", str(test_db))
    monkeypatch.setattr(config, "db_path", test_db)
    storage = StorageManager(db_path=test_db)
    await storage.initialize()
    yield
