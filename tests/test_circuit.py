import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from maulness.core.models import AgentThoughtEvent
from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.circuit import (
    CircuitState,
    ErrorClassification,
    ProviderHealth,
    ProviderHealthRegistry,
    classify_provider_error,
)
from maulness.core.providers.fallback import FallbackProviderChain


def test_classify_provider_error():
    c_429, msg_429 = classify_provider_error(Exception("429 Resource exhausted"))
    assert c_429 == ErrorClassification.RATE_LIMIT
    assert "429" in msg_429

    c_to, msg_to = classify_provider_error(TimeoutError("Request timed out"))
    assert c_to == ErrorClassification.TIMEOUT

    c_503, _ = classify_provider_error(Exception("503 Service Unavailable"))
    assert c_503 == ErrorClassification.SERVER_ERROR

    c_401, _ = classify_provider_error(Exception("401 Unauthorized API key"))
    assert c_401 == ErrorClassification.AUTH_ERROR

    c_400, _ = classify_provider_error(Exception("400 Bad Request: prompt too long"))
    assert c_400 == ErrorClassification.PAYLOAD_ERROR


def test_provider_health_state_transitions():
    health = ProviderHealth(provider_key="gemini:gemini-3.8-flash")
    now = 1000.0

    # 1. Initially CLOSED
    assert health.is_available(now) is True
    assert health.state == CircuitState.CLOSED

    # 2. 400 error does not trip circuit
    tripped, _, _ = health.record_failure(Exception("400 Bad Request"))
    assert tripped is False
    assert health.state == CircuitState.CLOSED

    # 3. 429 Rate Limit trips circuit to OPEN with 60s cooldown
    tripped, msg, cd = health.record_failure(Exception("429 Quota exceeded"), now=now)
    assert tripped is True
    assert cd == 60
    assert health.state == CircuitState.OPEN
    assert health.consecutive_failures == 1

    # 4. During cooldown, not available
    assert health.is_available(now + 30.0) is False

    # 5. After cooldown (now + 61), transitions to HALF_OPEN (probation probe)
    assert health.is_available(now + 65.0) is True
    assert health.state == CircuitState.HALF_OPEN

    # 6. Probation fails -> trips back to OPEN with backoff (120s)
    tripped, _, cd2 = health.record_failure(Exception("429 Quota exceeded"), now=now + 65.0)
    assert tripped is True
    assert cd2 == 120
    assert health.consecutive_failures == 2
    assert health.state == CircuitState.OPEN

    # 7. Expire again and succeed -> resets to CLOSED
    assert health.is_available(health.cooldown_until + 1.0) is True
    health.record_success()
    assert health.state == CircuitState.CLOSED
    assert health.consecutive_failures == 0


@pytest.mark.asyncio
async def test_fallback_chain_fast_bypass():
    ProviderHealthRegistry.reset_instance()
    registry = ProviderHealthRegistry.get_instance()

    primary_profile = Profile(
        identity={"name": "primary"},
        agent={"provider": "gemini", "model": "gemini-3.8-flash"},
    )
    fallback_profile = Profile(
        identity={"name": "fallback"},
        agent={"provider": "ollama", "model": "gemma"},
    )

    primary = MagicMock(spec=BaseProvider)
    primary.profile = primary_profile
    primary.run = AsyncMock(side_effect=Exception("429 Quota exceeded"))

    fallback = MagicMock(spec=BaseProvider)
    fallback.profile = fallback_profile
    fallback.run = AsyncMock(return_value="Response from fallback model")

    chain = FallbackProviderChain(primary=primary, fallbacks=[fallback])

    # Turn 1: Primary fails with 429, fallback succeeds, primary trips OPEN
    thought_deltas = []
    async def capture_thought(ev: AgentThoughtEvent):
        thought_deltas.append(ev.delta)

    res1 = await chain.run(session_id="s1", prompt="Prompt 1", on_thought=capture_thought)
    assert res1 == "Response from fallback model"
    assert primary.run.call_count == 1
    assert fallback.run.call_count == 1

    # Verify circuit for primary is OPEN
    health_primary = registry.get_or_create("gemini:gemini-3.8-flash")
    assert health_primary.state == CircuitState.OPEN

    # Turn 2: Primary is in OPEN cooldown. MUST fast-bypass primary with 0 calls to primary.run()!
    thought_deltas.clear()
    res2 = await chain.run(session_id="s2", prompt="Prompt 2", on_thought=capture_thought)
    assert res2 == "Response from fallback model"
    # Call count on primary MUST STILL BE 1 (fast-bypassed!)
    assert primary.run.call_count == 1
    assert fallback.run.call_count == 2
    # Verify thought notice mentions bypassing
    assert any("Bypassed" in d and "gemini:gemini-3.8-flash" in d for d in thought_deltas)

    # Turn 3: Time advances past cooldown -> primary is probed in HALF_OPEN
    # Configure primary to recover
    primary.run = AsyncMock(return_value="Recovered response from primary")
    health_primary.cooldown_until = time.time() - 1.0  # Force cooldown expiry

    res3 = await chain.run(session_id="s3", prompt="Prompt 3")
    assert res3 == "Recovered response from primary"
    assert primary.run.call_count == 1
    # Circuit MUST now be reset to CLOSED
    assert health_primary.state == CircuitState.CLOSED
    assert health_primary.consecutive_failures == 0


def test_classify_additional_errors():
    c_500, msg_500 = classify_provider_error(Exception("500 INTERNAL_SERVER_ERROR"))
    assert c_500 == ErrorClassification.SERVER_ERROR

    c_502, msg_502 = classify_provider_error(Exception("502 Bad Gateway"))
    assert c_502 == ErrorClassification.SERVER_ERROR

    c_api, msg_api = classify_provider_error(Exception("API_KEY_INVALID"))
    assert c_api == ErrorClassification.AUTH_ERROR

    c_403, msg_403 = classify_provider_error(Exception("403 Forbidden: PERMISSION_DENIED"))
    assert c_403 == ErrorClassification.AUTH_ERROR

    c_unk, msg_unk = classify_provider_error(Exception("Some weird unknown failure"))
    assert c_unk == ErrorClassification.UNKNOWN


def test_circuit_cooldown_types():
    health_to = ProviderHealth(provider_key="timeout_prov")
    tripped, _, cd = health_to.record_failure(TimeoutError("Read timed out"), now=100.0)
    assert tripped is True
    assert cd == 45

    health_srv = ProviderHealth(provider_key="srv_prov")
    tripped, _, cd = health_srv.record_failure(Exception("503 Unavailable"), now=100.0)
    assert tripped is True
    assert cd == 30

    health_auth = ProviderHealth(provider_key="auth_prov")
    tripped, _, cd = health_auth.record_failure(Exception("401 Unauthorized"), now=100.0)
    assert tripped is True
    assert cd == 3600

    health_unk = ProviderHealth(provider_key="unk_prov")
    tripped, _, cd = health_unk.record_failure(Exception("Something weird"), now=100.0)
    assert tripped is True
    assert cd == 30



def test_registry_format_and_reset():
    reg = ProviderHealthRegistry()
    reg.reset_instance()
    reg = ProviderHealthRegistry.get_instance()

    # Empty
    assert "No provider health data" in reg.format_status_text()

    # Add closed provider
    reg.record_success("p1")
    assert reg.is_available("p1") is True

    # Add open provider
    reg.record_failure("p2", Exception("429 Quota"))

    # Add half-open provider
    h3 = reg.get_or_create("p3")
    h3.state = CircuitState.HALF_OPEN
    assert h3.is_available() is True
    h3.state = "unknown"
    assert h3.is_available() is False
    h3.state = CircuitState.HALF_OPEN

    status_text = reg.format_status_text()
    assert "**HEALTHY** (CLOSED)" in status_text
    assert "**COOLDOWN** (OPEN" in status_text
    assert "**PROBATION** (HALF_OPEN" in status_text

    # Reset all
    reg.reset_all()
    status_text2 = reg.format_status_text()
    assert "**COOLDOWN**" not in status_text2
    assert reg.get_or_create("p2").state == CircuitState.CLOSED


