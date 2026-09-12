import asyncio
import json
from unittest.mock import MagicMock, patch
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


def test_provider_factory_unknown_provider_fallbacks():
    # antigravity provider
    from maulness.core.providers.sdk_provider import AntigravitySdkProvider
    p_sdk = Profile(identity={"name": "sdk_prof"}, agent={"provider": "antigravity"})
    assert isinstance(get_provider_for_profile(p_sdk), AntigravitySdkProvider)

    # with command
    p_cmd = Profile(identity={"name": "custom"}, agent={"provider": "custom_agent", "command": "my_agent"})
    assert isinstance(get_provider_for_profile(p_cmd), AcpProvider)

    # with base_url
    p_url = Profile(identity={"name": "custom"}, agent={"provider": "custom_model", "base_url": "http://localhost:8000/v1"})
    assert isinstance(get_provider_for_profile(p_url), UnifiedApiProvider)

    # neither
    p_gem = Profile(identity={"name": "custom"}, agent={"provider": "custom_default"})
    assert isinstance(get_provider_for_profile(p_gem), GeminiProvider)


def test_provider_factory_fallback_chain_command_and_env():
    profile = Profile(
        identity={"name": "resilient"},
        agent={"provider": "gemini"},
        resilience={
            "fallbacks": [
                {
                    "provider": "acp",
                    "command": "custom_subagent",
                },
                {
                    "provider": "openai",
                    "api_key_env": "CUSTOM_KEY",
                    "reasoning_effort": "high",
                },
            ]
        },
    )
    provider = get_provider_for_profile(profile)
    assert len(provider.fallbacks) == 2
    assert isinstance(provider.fallbacks[0], AcpProvider)
    assert isinstance(provider.fallbacks[1], UnifiedApiProvider)


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
    thought_events = []
    async def on_thought(ev):
        thought_events.append(ev.delta)

    result = await chain.run(session_id="s1", prompt="test prompt", on_thought=on_thought)

    assert result == "Fallback succeeded!"
    assert chain.last_used_provider == mock_fallback
    assert len(thought_events) == 1
    assert "Provider [gemini:default] failed (Rate limit / quota exceeded (429))" in thought_events[0]
    assert "Switching to fallback [anthropic:default]" in thought_events[0]
    mock_primary.run.assert_called_once()
    mock_fallback.run.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_provider_chain_all_fail_aggregates_errors():
    from unittest.mock import AsyncMock
    from maulness.core.providers.fallback import FallbackProviderChain
    from maulness.core.providers.base import BaseProvider

    p1 = Profile(identity={"name": "p1"}, agent={"provider": "gemini", "model": "gemini-3.8-flash"})
    p2 = Profile(identity={"name": "p2"}, agent={"provider": "ollama", "model": "gemma4:31b-cloud"})

    mock1 = AsyncMock(spec=BaseProvider)
    mock1.profile = p1
    mock1.run.side_effect = RuntimeError("HTTP 500 Internal Server Error")

    mock2 = AsyncMock(spec=BaseProvider)
    mock2.profile = p2
    mock2.run.side_effect = TimeoutError("Stream timed out")

    chain = FallbackProviderChain(primary=mock1, fallbacks=[mock2])
    with pytest.raises(RuntimeError) as exc_info:
        await chain.run(session_id="s2", prompt="test")

    err_text = str(exc_info.value)
    assert "All configured providers failed" in err_text
    assert "gemini:gemini-3.8-flash -> Internal server error (500)" in err_text
    assert "ollama:gemma4:31b-cloud -> Request timed out" in err_text


def test_format_user_friendly_error():
    from maulness.core.providers.fallback import format_user_friendly_error

    assert "500" in format_user_friendly_error(RuntimeError("500 Internal Server Error"))
    assert "429" in format_user_friendly_error(Exception("RESOURCE_EXHAUSTED: quota reached"))
    assert "401" in format_user_friendly_error(Exception("CreditsError: balance 0"))
    assert "timed out" in format_user_friendly_error(TimeoutError("No response for 180s"))
    assert "API key" in format_user_friendly_error(Exception("API_KEY_INVALID"))


@pytest.mark.asyncio
async def test_fallback_provider_chain_rejects_empty_response():
    from unittest.mock import AsyncMock
    from maulness.core.providers.fallback import FallbackProviderChain
    from maulness.core.providers.base import BaseProvider

    p1 = Profile(identity={"name": "p1"}, agent={"provider": "gemini", "model": "gemini-3.8-flash"})
    p2 = Profile(identity={"name": "p2"}, agent={"provider": "ollama", "model": "gemma4:31b-cloud"})

    mock1 = AsyncMock(spec=BaseProvider)
    mock1.profile = p1
    mock1.run.return_value = "   "  # Empty / whitespace response

    mock2 = AsyncMock(spec=BaseProvider)
    mock2.profile = p2
    mock2.run.return_value = "Valid fallback response"

    chain = FallbackProviderChain(primary=mock1, fallbacks=[mock2])
    result = await chain.run(session_id="empty_test", prompt="hello")

    assert result == "Valid fallback response"
    assert chain.last_used_provider == mock2
    mock1.run.assert_called_once()
    mock2.run.assert_called_once()


@pytest.mark.asyncio
async def test_unified_api_provider_tool_loop(tmp_path):
    import json
    from unittest.mock import patch, MagicMock, AsyncMock
    from maulness.core.providers.api_provider import UnifiedApiProvider
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "test.db")
    await storage.initialize()

    profile = Profile(
        identity={"name": "tool_tester"},
        agent={"provider": "ollama", "model": "llama3", "base_url": "http://localhost:11434/v1"},
    )
    provider = UnifiedApiProvider(profile, storage=storage)

    # Mock tool call in turn 1 and final text in turn 2
    turn_1_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "run_command", "arguments": '{"command": "echo 42"}'}
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    turn_2_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "The answer is 42."
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1_lines, turn_2_lines]

    class MockStreamCtx:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.is_error = False

            async def aiter_lines():
                for l in self.lines:
                    yield l

            resp.aiter_lines = aiter_lines
            return resp

        async def __aexit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamCtx(lines)

    with patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        result = await provider.run(
            session_id="test_sess",
            prompt="What is the answer?",
            yolo=True,
        )

    assert "run_command: echo 42" in result
    assert "The answer is 42." in result


@pytest.mark.asyncio
async def test_unified_api_provider_intercepts_simulated_tool_call(tmp_path):
    import json
    from unittest.mock import patch, MagicMock
    from maulness.core.providers.api_provider import UnifiedApiProvider
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "test.db")
    await storage.initialize()

    profile = Profile(
        identity={"name": "sim_tester"},
        agent={"provider": "ollama", "model": "gemma", "base_url": "http://localhost:11434/v1"},
    )
    provider = UnifiedApiProvider(profile, storage=storage)

    # Turn 1: Model attempts to hallucinate markdown tool call in plain text instead of function calling
    turn_1_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "> **`run_command: echo 'real_intercepted'`**:\n```\nfake markdown output\n```"
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 2: Synthesized answer
    turn_2_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "Done executing intercepted command."
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1_lines, turn_2_lines]

    class MockStreamCtx:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.is_error = False

            async def aiter_lines():
                for l in self.lines:
                    yield l

            resp.aiter_lines = aiter_lines
            return resp

        async def __aexit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamCtx(lines)

    with patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        result = await provider.run(
            session_id="sim_sess",
            prompt="Run echo intercepted",
            workspace_path=tmp_path,
            yolo=True,
        )

    # Verify real tool execution occurred and replaced the fake output
    assert "run_command: echo 'real_intercepted'" in result
    assert "real_intercepted" in result
    assert "fake markdown output" not in result
    assert "Done executing intercepted command." in result


@pytest.mark.asyncio
async def test_unified_api_provider_max_tool_turns_forces_synthesis(tmp_path):
    import json
    from unittest.mock import patch, MagicMock
    from maulness.core.providers.api_provider import UnifiedApiProvider
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "test.db")
    await storage.initialize()

    # Profile capped at 2 tool turns
    profile = Profile(
        identity={"name": "cap_tester"},
        agent={"provider": "ollama", "model": "llama3", "base_url": "http://localhost:11434/v1"},
        execution={"max_tool_turns": 2},
    )
    provider = UnifiedApiProvider(profile, storage=storage)

    turn_1_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "run_command", "arguments": '{"command": "echo step1"}'}
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    turn_2_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "run_command", "arguments": '{"command": "echo step2"}'}
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    turn_3_forced_synthesis_lines = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "Synthesized final answer after 2 tool turns."
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1_lines, turn_2_lines, turn_3_forced_synthesis_lines]
    payloads_captured = []

    class MockStreamCtx:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.is_error = False

            async def aiter_lines():
                for l in self.lines:
                    yield l

            resp.aiter_lines = aiter_lines
            return resp

        async def __aexit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        payloads_captured.append(dict(json) if json else {})
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamCtx(lines)

    with patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        result = await provider.run(
            session_id="capped_sess",
            prompt="Perform multi-step exploration",
            workspace_path=tmp_path,
            yolo=True,
        )

    # 1. Turn 1 and 2 should have had tools in payload
    assert "tools" in payloads_captured[0]
    assert "tools" in payloads_captured[1]

    # 2. Turn 3 (forced synthesis) must NOT have tools in payload
    assert "tools" not in payloads_captured[2]
    # And must have the system prompt requesting final synthesis
    last_msg = payloads_captured[2]["messages"][-1]
    assert "Maximum allowed tool execution limit" in last_msg["content"] or "maximum allowed tool" in last_msg["content"].lower()

    # 3. Output must contain both breadcrumbs AND the synthesized final answer
    assert "run_command: echo step1" in result
    assert "run_command: echo step2" in result
    assert "Synthesized final answer after 2 tool turns." in result


@pytest.mark.asyncio
async def test_openai_compatible_multi_relay_continuation_and_compaction(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    profile = Profile(
        identity={"name": "relay_agent"},
        agent={"provider": "openrouter", "model": "test-model"},
        execution={"max_tool_turns": 1, "max_relays": 2, "yolo": True},
        env_vars={"OPENROUTER_API_KEY": "dummy_key"},
    )
    provider = UnifiedApiProvider(profile)

    # Turn 1: Relay 1, tool call
    turn_1 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo relay_1_action"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 2: Relay 1, forced checkpoint evaluation -> outputs [STATUS: IN_PROGRESS]
    turn_2 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": (
                        "[STATUS: IN_PROGRESS]\n"
                        "Accomplished: Ran initial inspection.\n"
                        "Key Findings: Environment is ready.\n"
                        "Next Step: Execute build step."
                    )
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 3: Relay 2, tool call (tools re-enabled, context compacted)
    turn_3 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo relay_2_action"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 4: Relay 2, final synthesis
    turn_4 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "All tasks completed successfully across both relays. [STATUS: COMPLETE]"
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1, turn_2, turn_3, turn_4]
    payloads_captured = []

    class MockStreamCtx:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.is_error = False

            async def aiter_lines():
                for l in self.lines:
                    yield l

            resp.aiter_lines = aiter_lines
            return resp

        async def __aexit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        payloads_captured.append(dict(json) if json else {})
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamCtx(lines)

    with patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        result = await provider.run(
            session_id="relay_sess",
            prompt="Build and verify long-horizon project",
            workspace_path=tmp_path,
            yolo=True,
        )

    # 1. Verify 4 HTTP turns occurred (2 per relay)
    assert len(payloads_captured) == 4

    # 2. Relay 1 turn 1 had tools; turn 2 had tools disabled for evaluation
    assert "tools" in payloads_captured[0]
    assert "tools" not in payloads_captured[1]

    # 3. Relay 2 turn 1 had tools RE-ENABLED and context COMPACTED
    assert "tools" in payloads_captured[2]
    relay_2_msgs = payloads_captured[2]["messages"]
    # Should have assistant checkpoint summary
    asst_checkpoint = next((m for m in relay_2_msgs if m.get("role") == "assistant" and "[AUTONOMOUS RELAY CHECKPOINTS]" in m.get("content", "")), None)
    assert asst_checkpoint is not None
    assert "Accomplished: Ran initial inspection" in asst_checkpoint["content"]
    assert "Key Findings: Environment is ready" in asst_checkpoint["content"]

    # Bulky raw tool result from Relay 1 should NOT be in Relay 2 messages
    assert not any(m.get("role") == "tool" and "relay_1_action" in m.get("content", "") for m in relay_2_msgs)

    # 4. Result contains breadcrumbs from both relays, checkpoint notice, and final cleaned text
    assert "run_command: echo relay_1_action" in result
    assert "[Relay Checkpoint 1/2]" in result
    assert "Auto-advancing with: *Execute build step.*" in result
    assert "run_command: echo relay_2_action" in result
    assert "All tasks completed successfully across both relays." in result
    # Protocol tags must be cleaned from final text
    assert "[STATUS: COMPLETE]" not in result


@pytest.mark.asyncio
async def test_openai_compatible_max_relay_ceiling(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    # Only 1 relay allowed
    profile = Profile(
        identity={"name": "ceiling_agent"},
        agent={"provider": "openrouter", "model": "test-model"},
        execution={"max_tool_turns": 1, "max_relays": 1, "yolo": True},
        env_vars={"OPENROUTER_API_KEY": "dummy_key"},
    )
    provider = UnifiedApiProvider(profile)

    turn_1 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo step1"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 2: Model attempts to continue with IN_PROGRESS even though ceiling is reached
    turn_2 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": (
                        "[STATUS: IN_PROGRESS]\n"
                        "Accomplished: Finished step 1.\n"
                        "Next Step: Would like to continue step 2."
                    )
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1, turn_2]
    payloads_captured = []

    class MockStreamCtx:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            resp = MagicMock()
            resp.is_error = False

            async def aiter_lines():
                for l in self.lines:
                    yield l

            resp.aiter_lines = aiter_lines
            return resp

        async def __aexit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        payloads_captured.append(dict(json) if json else {})
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamCtx(lines)

    with patch("httpx.AsyncClient.stream", side_effect=mock_stream):
        result = await provider.run(
            session_id="ceiling_sess",
            prompt="Run task with ceiling",
            workspace_path=tmp_path,
            yolo=True,
        )

    # Must stop after turn 2 without looping further
    assert len(payloads_captured) == 2
    assert "Maximum relay budget of 1 relays reached. Task paused at checkpoint." in result


@pytest.mark.asyncio
async def test_gemini_multi_relay_continuation_and_compaction(tmp_path):
    from unittest.mock import MagicMock, patch
    from maulness.core.profiles import Profile
    from maulness.core.providers.gemini_provider import GeminiProvider

    profile = Profile(
        identity={"name": "gemini_relay_agent"},
        agent={"provider": "gemini", "model": "gemini-2.5-flash"},
        execution={"max_tool_turns": 1, "max_relays": 2, "yolo": True},
        env_vars={"GEMINI_API_KEY": "dummy_gemini_key"},
    )
    provider = GeminiProvider(profile)

    from google.genai import types

    # Chunk 1: Tool call run_command echo g_step1
    part_1 = types.Part.from_function_call(name="run_command", args={"command": "echo g_step1"})
    cand_1 = MagicMock()
    cand_1.content.parts = [part_1]
    chunk_1 = MagicMock()
    chunk_1.candidates = [cand_1]
    chunk_1.text = None

    # Chunk 2: Checkpoint response [STATUS: IN_PROGRESS]
    part_2 = types.Part.from_text(text=(
        "[STATUS: IN_PROGRESS]\n"
        "Accomplished: Completed Gemini step 1.\n"
        "Next Step: Execute Gemini step 2."
    ))
    cand_2 = MagicMock()
    cand_2.content.parts = [part_2]
    chunk_2 = MagicMock()
    chunk_2.candidates = [cand_2]
    chunk_2.text = part_2.text

    # Chunk 3: Final answer in Relay 2
    part_3 = types.Part.from_text(text="Gemini completed all relays successfully.")
    cand_3 = MagicMock()
    cand_3.content.parts = [part_3]
    chunk_3 = MagicMock()
    chunk_3.candidates = [cand_3]
    chunk_3.text = part_3.text

    async def stream_gen(chunks):
        for c in chunks:
            yield c

    stream_calls = [
        stream_gen([chunk_1]),
        stream_gen([chunk_2]),
        stream_gen([chunk_3]),
    ]

    recorded_contents = []

    async def mock_generate_stream(*args, **kwargs):
        recorded_contents.append(list(kwargs.get("contents", [])))
        if stream_calls:
            return stream_calls.pop(0)
        return stream_gen([])

    mock_client = MagicMock()
    mock_client.aio.models.generate_content_stream = mock_generate_stream

    with patch("google.genai.Client", return_value=mock_client):
        result = await provider.run(
            session_id="gemini_relay_sess",
            prompt="Run multi-step Gemini task",
            workspace_path=tmp_path,
        )

    # 3 turns executed across 2 relays
    assert len(recorded_contents) == 3

    # Turn 3 (Relay 2 start) must have context compacted with checkpoint ledger
    relay_2_contents = recorded_contents[2]
    checkpoint_content = next(
        (c for c in relay_2_contents if any("[AUTONOMOUS RELAY CHECKPOINTS]" in getattr(p, "text", "") for p in c.parts)),
        None
    )
    assert checkpoint_content is not None
    assert any("Completed Gemini step 1" in getattr(p, "text", "") for p in checkpoint_content.parts)

    # Output verification
    assert "run_command: echo g_step1" in result
    assert "[Relay Checkpoint 1/2]" in result
    assert "Auto-advancing with: *Execute Gemini step 2.*" in result
    assert "Gemini completed all relays successfully." in result


@pytest.mark.asyncio
async def test_unified_api_provider_natural_progress_map_auto_advances(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    profile = Profile(
        identity={"name": "natural_progress_agent"},
        agent={"provider": "openrouter", "model": "test-model"},
        execution={"max_tool_turns": 1, "max_relays": 2, "yolo": True},
        env_vars={"OPENROUTER_API_KEY": "dummy_key"},
    )
    provider = UnifiedApiProvider(profile)

    # Turn 1: Relay 1, tool call
    turn_1 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo domain_inspected"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 2: Natural progress map without literal [STATUS: IN_PROGRESS] tag
    turn_2 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": (
                        "I have inspected the domain layer.\n\n"
                        "Current Status:\n"
                        "- [x] internal/domain -> COMPLETE\n"
                        "- [ ] internal/application -> NEXT\n"
                        "- [ ] internal/adapters -> PENDING\n\n"
                        "Next Step: I am diving into internal/application to inspect services."
                    )
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 3: Relay 2, tool call
    turn_3 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo app_inspected"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]

    # Turn 4: Relay 2, final answer
    turn_4 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "All packages read and analyzed 100%."
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1, turn_2, turn_3, turn_4]

    class MockStreamContext:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.is_error = False

            async def aiter_lines():
                for line in self.lines:
                    yield line

            mock_resp.aiter_lines = aiter_lines
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(method, url, headers=None, json=None):
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamContext(lines)

    mock_client = MagicMock()
    mock_client.stream = mock_stream

    class MockAsyncClientConstructor:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient", MockAsyncClientConstructor):
        result = await provider.run(
            session_id="natural_prog_sess",
            prompt="Read 100% of horizonx",
            workspace_path=tmp_path,
        )

    assert "[Relay Checkpoint 1/2]" in result
    assert "diving into internal/application to inspect services" in result
    assert "echo app_inspected" in result
    assert "All packages read and analyzed 100%." in result


@pytest.mark.asyncio
async def test_unified_api_provider_unlimited_tool_turns(tmp_path):
    import json
    from unittest.mock import MagicMock, patch
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    # max_tool_turns=0 signifies unlimited turns
    profile = Profile(
        identity={"name": "unlimited_agent"},
        agent={"provider": "openrouter", "model": "test-model"},
        execution={"max_tool_turns": 0, "max_relays": 1, "yolo": True},
        env_vars={"OPENROUTER_API_KEY": "dummy_key"},
    )
    provider = UnifiedApiProvider(profile)

    # 3 consecutive tool turns without forcing synthesis
    turn_1 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo step_1"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]
    turn_2 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo step_2"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]
    turn_3 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo step_3"}),
                        }
                    }]
                }
            }]
        }),
        'data: [DONE]'
    ]
    turn_4 = [
        'data: ' + json.dumps({
            "choices": [{
                "delta": {
                    "content": "Completed all steps in a single unbounded session."
                }
            }]
        }),
        'data: [DONE]'
    ]

    calls = [turn_1, turn_2, turn_3, turn_4]

    class MockStreamContext:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            mock_resp = MagicMock()
            mock_resp.is_error = False

            async def aiter_lines():
                for line in self.lines:
                    yield line

            mock_resp.aiter_lines = aiter_lines
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(method, url, headers=None, json=None):
        lines = calls.pop(0) if calls else ['data: [DONE]']
        return MockStreamContext(lines)

    mock_client = MagicMock()
    mock_client.stream = mock_stream

    class MockAsyncClientConstructor:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient", MockAsyncClientConstructor):
        result = await provider.run(
            session_id="unlimited_sess",
            prompt="Run unbounded tool calls",
            workspace_path=tmp_path,
        )

    assert "echo step_1" in result
    assert "echo step_2" in result
    assert "echo step_3" in result
    assert "Completed all steps in a single unbounded session." in result
    assert "[Relay Checkpoint" not in result


def test_compact_in_flight_tool_messages():
    from maulness.core.providers.api_provider import compact_in_flight_tool_messages

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Fix code"},
        # Turn 1: tool call
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_command", "arguments": '{"command": "pytest"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Very long output\n" * 30},
        # Turn 2: tool call
        {"role": "assistant", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "foo.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "File lines\n" * 30},
        # Turn 3: tool call
        {"role": "assistant", "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "replace_file_content", "arguments": '{"path": "foo.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_3", "content": "Replacement result\n" * 30},
        # Turn 4: tool call (recent)
        {"role": "assistant", "tool_calls": [{"id": "call_4", "type": "function", "function": {"name": "run_command", "arguments": '{"command": "pytest"}'}}]},
        {"role": "tool", "tool_call_id": "call_4", "content": "Recent output\n" * 30},
    ]

    compact_in_flight_tool_messages(messages, keep_recent=3)

    assert "[Tool result for 'run_command' (`pytest`) compacted:" in messages[3]["content"]
    assert "Very long output" not in messages[3]["content"]
    assert "File lines" in messages[5]["content"]
    assert "Replacement result" in messages[7]["content"]
    assert "Recent output" in messages[9]["content"]


@pytest.mark.asyncio
async def test_anthropic_turn_streaming(monkeypatch):
    import json
    from unittest.mock import AsyncMock, MagicMock
    import httpx
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    profile = Profile(
        identity={"name": "claude_coder"},
        agent={"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
        env_vars={"ANTHROPIC_API_KEY": "test_anthropic_key"},
        reasoning={"effort": "high"},
    )
    provider = UnifiedApiProvider(profile)

    lines = [
        'event: content_block_delta\n',
        'data: {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Let me think about it"}}\n',
        '\n',
        'event: content_block_delta\n',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Here is the response"}}\n',
        '\n',
    ]

    class MockStreamResponse:
        def __init__(self):
            self.status_code = 200
            self.is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_lines(self):
            for l in lines:
                yield l

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, **kwargs):
            return MockStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    thoughts = []
    messages = []

    async def on_thought(ev):
        thoughts.append(ev.delta)

    async def on_msg(ev):
        messages.append(ev.delta)

    res = await provider._generate_anthropic_turn(
        messages=[{"role": "user", "content": "Help me"}],
        tools=None,
        session_id="test_sess",
        on_thought=on_thought,
        on_message=on_msg,
    )

    assert res.content == "Here is the response"
    assert res.thought == "Let me think about it"
    assert thoughts == ["Let me think about it"]
    assert messages == ["Here is the response"]


@pytest.mark.asyncio
async def test_anthropic_turn_http_error(monkeypatch):
    import httpx
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    profile = Profile(
        identity={"name": "claude_coder"},
        agent={"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
        env_vars={"ANTHROPIC_API_KEY": "test_anthropic_key"},
    )
    provider = UnifiedApiProvider(profile)

    class MockErrorResponse:
        def __init__(self):
            self.status_code = 400
            self.is_error = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aread(self):
            return b'{"error": "invalid_request_error", "message": "max_tokens too high"}'

    class MockErrorClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, method, url, **kwargs):
            return MockErrorResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockErrorClient)

    with pytest.raises(RuntimeError, match="Anthropic.*returned HTTP 400"):
        await provider._generate_anthropic_turn(
            messages=[{"role": "user", "content": "Help"}],
            tools=None,
            session_id="test_err_sess",
        )


def test_gemini_helpers_and_compact():
    from maulness.core.providers.gemini_provider import (
        compact_gemini_in_flight_contents,
        _find_tool_name,
    )
    from unittest.mock import MagicMock

    # 1. _find_tool_name
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_123", "function": {"name": "special_tool"}}]}
    ]
    assert _find_tool_name(messages, 1, "call_123") == "special_tool"
    assert _find_tool_name([], 0, "call_fallback_tool") == "tool"

    # 2. compact_gemini_in_flight_contents
    class MockPart:
        def __init__(self, result_text):
            resp = MagicMock()
            resp.name = "run_command"
            resp.response = {"result": result_text}
            self.function_response = resp

    class MockContent:
        def __init__(self, part):
            self.parts = [part]

    contents = [
        MockContent(MockPart("short")),
        MockContent(MockPart("large content " * 30)),
        MockContent(MockPart("another large " * 30)),
        MockContent(MockPart("recent 1")),
        MockContent(MockPart("recent 2")),
        MockContent(MockPart("recent 3")),
    ]

    compact_gemini_in_flight_contents(contents, keep_recent=3)
    # Older large parts should be replaced by tombstones
    first_resp = contents[1].parts[0].function_response.response["result"]
    assert "[Tool result for 'run_command'" in first_resp


def test_gemini_client_resolution(monkeypatch):
    from maulness.config import config
    from maulness.core.profiles import Profile
    from maulness.core.providers.gemini_provider import GeminiProvider

    monkeypatch.setattr(config, "gemini_api_key", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 1. Missing API key with no command raises ValueError
    p_no_key = Profile(identity={"name": "no_key"}, agent={"provider": "gemini"})
    prov_no_key = GeminiProvider(p_no_key)
    with pytest.raises(ValueError, match="Missing API key"):
        prov_no_key._get_client()

    # 2. Missing API key with available command returns None (triggers ACP fallback)
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/agy")
    p_cmd_fallback = Profile(identity={"name": "with_cmd"}, agent={"provider": "gemini", "command": "agy"})
    prov_cmd_fallback = GeminiProvider(p_cmd_fallback)
    assert prov_cmd_fallback._get_client() is None









@pytest.mark.asyncio
async def test_base_provider_abstract_run():
    from maulness.core.providers.base import BaseProvider
    class ConcreteProvider(BaseProvider):
        async def run(self, *args, **kwargs):
            return await super().run(*args, **kwargs)

    prof = Profile(identity={"name": "test"}, agent={"provider": "gemini", "model": "gemini-2.5-flash"})
    prov = ConcreteProvider(prof)
    res = await prov.run("sess", "prompt")
    assert res is None


@pytest.mark.asyncio
async def test_unified_api_provider_properties_and_missing_key():
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    # 1. Properties: unknown provider
    p_unk = Profile(
        identity={"name": "unk_prov"},
        agent={"provider": "unknown_provider", "model": "test-model"},
    )
    prov_unk = UnifiedApiProvider(p_unk)
    assert prov_unk.last_conversation_id is None
    assert prov_unk.base_url is None
    assert prov_unk.api_key is None

    # 2. Missing key for non-ollama raises ValueError (lines 84-85)
    with pytest.raises(ValueError, match="Missing API key"):
        await prov_unk.run(session_id="s1", prompt="hello")

    # 3. on_init callback (line 93)
    p_ollama = Profile(
        identity={"name": "ollama_agent"},
        agent={"provider": "ollama", "model": "llama3"},
    )
    prov_ollama = UnifiedApiProvider(p_ollama)

    init_calls = []
    async def fake_on_init(conv_id: str):
        init_calls.append(conv_id)

    async def fake_turn_generator(*args, **kwargs):
        from maulness.core.kernel import ModelTurnOutput
        return ModelTurnOutput(content="Done", tool_calls=[])

    with patch.object(prov_ollama, "_generate_openai_turn", side_effect=fake_turn_generator):
        res = await prov_ollama.run(session_id="s2", prompt="hi", on_init=fake_on_init)
        assert res == "Done"
        assert len(init_calls) == 1
        assert prov_ollama.last_conversation_id == init_calls[0]


@pytest.mark.asyncio
async def test_unified_api_provider_openai_turn_edge_cases(monkeypatch):
    import httpx
    from maulness.core.models import AgentThoughtEvent, AgentMessageEvent
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    # Profile with reasoning_effort and opencode provider
    profile = Profile(
        identity={"name": "opencode_agent"},
        agent={
            "provider": "opencode",
            "base_url": "https://api.opencode.ai/v1/chat/completions",
            "reasoning_effort": "high",
        },
        env_vars={"OPENCODE_API_KEY": "test_key"},
    )
    provider = UnifiedApiProvider(profile)

    # 1. client.stream raises Exception (lines 167-168)
    class StreamCrashClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            raise ConnectionError("DNS failure")

    monkeypatch.setattr(httpx, "AsyncClient", StreamCrashClient)
    with pytest.raises(RuntimeError, match="Failed to initiate stream"):
        await provider._generate_openai_turn([], None, session_id="s")

    # 2. response.is_error (lines 172-174)
    class MockErrorResponse:
        def __init__(self):
            self.status_code = 502
            self.is_error = True
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def aread(self): return b"Bad Gateway error"

    class StreamErrorClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockErrorResponse()

    monkeypatch.setattr(httpx, "AsyncClient", StreamErrorClient)
    with pytest.raises(RuntimeError, match="returned HTTP 502: Bad Gateway"):
        await provider._generate_openai_turn([], None, session_id="s")

    # 3. Timeout on first chunk vs subsequent chunk (lines 186-193)
    class MockFirstTimeoutLines:
        def __aiter__(self): return self
        async def __anext__(self):
            raise asyncio.TimeoutError()

    class MockFirstTimeoutResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockFirstTimeoutLines()

    class FirstTimeoutClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockFirstTimeoutResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FirstTimeoutClient)
    with pytest.raises(TimeoutError, match="timed out waiting for first token"):
        await provider._generate_openai_turn([], None, session_id="s")

    # Subsequent chunk timeout
    class MockLaterTimeoutLines:
        def __init__(self):
            self.count = 0
        def __aiter__(self): return self
        async def __anext__(self):
            self.count += 1
            if self.count == 1:
                return "data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}"
            raise asyncio.TimeoutError()

    class MockLaterTimeoutResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockLaterTimeoutLines()

    class LaterTimeoutClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockLaterTimeoutResponse()

    monkeypatch.setattr(httpx, "AsyncClient", LaterTimeoutClient)
    with pytest.raises(TimeoutError, match="Stream idle watchdog triggered"):
        await provider._generate_openai_turn([], None, session_id="s")

    # 4. Successful turn with:
    # - invalid json line ignored (lines 199-200)
    # - reasoning delta & on_thought (lines 209-211)
    # - tool call with id concat, tool call without name skipped, tool call with bad json args (lines 223, 244, 248-249)
    # - content delta & on_message (lines 234-237)
    class MockSuccessLines:
        def __init__(self):
            self.lines = [
                "data: {invalid json not parseable}",
                "data: " + json.dumps({"choices": [{"delta": {"reasoning": "pondering deep questions..."}}]}),
                "data: " + json.dumps({"choices": [{"delta": {"content": "Result: ", "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run_cmd", "arguments": '{"cmd": '}}]}}]}),
                "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "23", "function": {"arguments": '"ls"}'}}, {"index": 1, "function": {"name": ""}}, {"index": 2, "id": "call_bad", "function": {"name": "bad_tool", "arguments": "not valid json"}}]}}]}),
                "data: [DONE]",
            ]
            self.idx = 0
        def __aiter__(self): return self
        async def __anext__(self):
            if self.idx >= len(self.lines):
                raise StopAsyncIteration
            val = self.lines[self.idx]
            self.idx += 1
            return val

    class MockSuccessResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockSuccessLines()

    class SuccessClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockSuccessResponse()

    monkeypatch.setattr(httpx, "AsyncClient", SuccessClient)
    thoughts = []
    messages = []
    async def record_thought(ev: AgentThoughtEvent): thoughts.append(ev.delta)
    async def record_msg(ev: AgentMessageEvent): messages.append(ev.delta)

    out = await provider._generate_openai_turn(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"function": {"name": "run_cmd"}}],
        session_id="s",
        on_thought=record_thought,
        on_message=record_msg,
    )
    assert out.content == "Result: "
    assert out.thought == "pondering deep questions..."
    assert len(out.tool_calls) == 2
    assert out.tool_calls[0]["id"] == "call_123"
    assert out.tool_calls[0]["name"] == "run_cmd"
    assert out.tool_calls[0]["arguments"] == {"cmd": "ls"}
    assert out.tool_calls[1]["name"] == "bad_tool"
    assert out.tool_calls[1]["arguments"] == {"raw": "not valid json"}
    assert thoughts == ["pondering deep questions..."]
    assert messages == ["Result: "]


@pytest.mark.asyncio
async def test_unified_api_provider_anthropic_turn_edge_cases(monkeypatch):
    import httpx
    from maulness.core.models import AgentThoughtEvent, AgentMessageEvent
    from maulness.core.profiles import Profile
    from maulness.core.providers.api_provider import UnifiedApiProvider

    # Anthropic profile with url not ending in /messages, reasoning_effort
    profile = Profile(
        identity={"name": "claude_agent"},
        agent={
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "reasoning_effort": "medium",
        },
        env_vars={"ANTHROPIC_API_KEY": "test_claude_key"},
    )
    provider = UnifiedApiProvider(profile)

    # 1. client.stream raises Exception (lines 315-316)
    class StreamCrashClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs):
            raise ConnectionResetError("Reset")

    monkeypatch.setattr(httpx, "AsyncClient", StreamCrashClient)
    with pytest.raises(RuntimeError, match="Failed to initiate stream with Anthropic"):
        await provider._generate_anthropic_turn([], None, session_id="s")

    # 2. Timeout on first chunk vs subsequent chunk (lines 334-341)
    class MockFirstTimeoutLines:
        def __aiter__(self): return self
        async def __anext__(self): raise asyncio.TimeoutError()

    class MockFirstResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockFirstTimeoutLines()

    class FirstTimeoutClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockFirstResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FirstTimeoutClient)
    with pytest.raises(TimeoutError, match="timed out waiting for first token"):
        await provider._generate_anthropic_turn([], None, session_id="s")

    class MockLaterTimeoutLines:
        def __init__(self): self.count = 0
        def __aiter__(self): return self
        async def __anext__(self):
            self.count += 1
            if self.count == 1:
                return "data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"text_delta\", \"text\": \"hi\"}}"
            raise asyncio.TimeoutError()

    class MockLaterResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockLaterTimeoutLines()

    class LaterTimeoutClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockLaterResponse()

    monkeypatch.setattr(httpx, "AsyncClient", LaterTimeoutClient)
    with pytest.raises(TimeoutError, match="Stream idle watchdog triggered"):
        await provider._generate_anthropic_turn([], None, session_id="s")

    # 3. Successful stream with thinking_delta, invalid json line (lines 347-348, 357-360)
    class MockAnthropicSuccessLines:
        def __init__(self):
            self.lines = [
                "data: {bad json invalid}",
                "data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"thinking_delta\", \"thinking\": \"deep anthropic thought\"}}",
                "data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"text_delta\", \"text\": \"Claude reply\"}}",
            ]
            self.idx = 0
        def __aiter__(self): return self
        async def __anext__(self):
            if self.idx >= len(self.lines): raise StopAsyncIteration
            val = self.lines[self.idx]
            self.idx += 1
            return val

    class MockAnthropicSuccessResponse:
        is_error = False
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def aiter_lines(self): return MockAnthropicSuccessLines()

    class AnthropicSuccessClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def stream(self, *args, **kwargs): return MockAnthropicSuccessResponse()

    monkeypatch.setattr(httpx, "AsyncClient", AnthropicSuccessClient)
    thoughts = []
    messages = []
    async def record_thought(ev: AgentThoughtEvent): thoughts.append(ev.delta)
    async def record_msg(ev: AgentMessageEvent): messages.append(ev.delta)

    out = await provider._generate_anthropic_turn(
        messages=[{"role": "system", "content": "Sys prompt"}, {"role": "user", "content": "Hi"}],
        tools=None,
        session_id="s",
        on_thought=record_thought,
        on_message=record_msg,
    )
    assert out.content == "Claude reply"
    assert out.thought == "deep anthropic thought"
    assert thoughts == ["deep anthropic thought"]
    assert messages == ["Claude reply"]


@pytest.mark.asyncio
async def test_gemini_provider_full_coverage(monkeypatch):
    from maulness.core.models import AgentThoughtEvent, AgentMessageEvent
    from maulness.core.profiles import Profile
    from maulness.core.providers.gemini_provider import (
        GeminiProvider,
        compact_gemini_in_flight_contents,
        _find_tool_name,
    )

    # 1. compact_gemini_in_flight_contents early return (line 39)
    assert compact_gemini_in_flight_contents([], keep_recent=5) is None

    # 2. _find_tool_name call_id prefix format (line 63) and fallback (line 64)
    assert _find_tool_name([], 0, "call_123_custom") == "custom"
    assert _find_tool_name([], 0, "nonprefixed") == "tool"

    # Lines 96-97: messages_to_gemini_contents with invalid json arguments
    from maulness.core.providers.gemini_provider import messages_to_gemini_contents
    msg = {"role": "assistant", "tool_calls": [{"name": "cmd", "arguments": "invalid json not dict"}]}
    contents = messages_to_gemini_contents([msg])
    assert contents[0].parts[0].function_call.args == {}

    # 3. Vertex client instantiation (lines 134-137)
    p_vert = Profile(
        identity={"name": "vert_agent"},
        agent={"provider": "gemini", "vertex": {"enabled": True, "project": "proj-1", "location": "us-central1"}},
    )
    prov_vert = GeminiProvider(p_vert)
    with patch("google.genai.Client") as mock_genai_client:
        cl = prov_vert._get_client()
        mock_genai_client.assert_called_with(vertexai=True, project="proj-1", location="us-central1")
        assert cl == mock_genai_client()

    # 4. run with on_init callback and fallback to AcpProvider when client is None (lines 167, 171-184)
    p_fallback = Profile(identity={"name": "fb"}, agent={"provider": "gemini", "command": "agy"})
    prov_fallback = GeminiProvider(p_fallback)
    prov_fallback._client = None
    with patch.object(prov_fallback, "_get_client", return_value=None):
        init_calls = []
        async def fake_on_init(conv_id): init_calls.append(conv_id)
        with patch("maulness.core.providers.acp_provider.AcpProvider.run", return_value="Fallback ACP output"):
            res = await prov_fallback.run("s", "prompt", on_init=fake_on_init)
            assert res == "Fallback ACP output"
            assert len(init_calls) == 1

    # 5. generate_turn client is None raises RuntimeError (line 212)
    prov_no_client = GeminiProvider(p_fallback)
    with patch.object(prov_no_client, "_get_client", return_value=None):
        with pytest.raises(RuntimeError, match="Gemini client not initialized"):
            await prov_no_client.generate_turn([], None, session_id="s")

    # 6. generate_turn with reasoning_effort (lines 224-226) and timeouts (lines 260-261, 280-286)
    p_gem_effort = Profile(
        identity={"name": "effort_agent"},
        agent={"provider": "gemini", "model": "gemini-2.5-flash", "reasoning_effort": "high"},
        env_vars={"GEMINI_API_KEY": "fake_key"},
    )
    prov_effort = GeminiProvider(p_gem_effort)

    # Handshake timeout
    mock_client = MagicMock()
    prov_effort._client = mock_client

    async def fake_handshake_timeout(*args, **kwargs):
        raise asyncio.TimeoutError()

    mock_client.aio.models.generate_content_stream = fake_handshake_timeout
    with pytest.raises(TimeoutError, match="Gemini API connection handshake timed out"):
        await prov_effort.generate_turn([], None, session_id="s")

    # First token timeout
    class MockStreamFirstTimeout:
        def __aiter__(self): return self
        async def __anext__(self): raise asyncio.TimeoutError()

    async def fake_first_token_stream(*args, **kwargs):
        return MockStreamFirstTimeout()

    mock_client.aio.models.generate_content_stream = fake_first_token_stream
    with pytest.raises(TimeoutError, match="timed out waiting for first token response"):
        await prov_effort.generate_turn([], None, session_id="s")

    # Subsequent token timeout
    class MockStreamLaterTimeout:
        def __init__(self): self.count = 0
        def __aiter__(self): return self
        async def __anext__(self):
            self.count += 1
            if self.count == 1:
                chunk = MagicMock()
                part = MagicMock(text="Hello", thought=False, function_call=None)
                cand = MagicMock()
                cand.content.parts = [part]
                chunk.candidates = [cand]
                chunk.text = None
                return chunk
            raise asyncio.TimeoutError()

    async def fake_later_token_stream(*args, **kwargs):
        return MockStreamLaterTimeout()

    mock_client.aio.models.generate_content_stream = fake_later_token_stream
    with pytest.raises(TimeoutError, match="Gemini stream idle watchdog triggered"):
        await prov_effort.generate_turn([], None, session_id="s")

    # Successful turn testing:
    # - part.thought with on_thought (lines 313-316)
    # - part.text with on_message (line 321)
    # - chunk.text fallback with on_message (lines 324-327)
    class MockSuccessStream:
        def __init__(self):
            # Chunk 1: thought part and text part
            c1 = MagicMock()
            p_th = MagicMock(text="Gemini thinking...", thought=True, function_call=None)
            p_txt = MagicMock(text="Main answer", thought=False, function_call=None)
            cand1 = MagicMock()
            cand1.content.parts = [p_th, p_txt]
            c1.candidates = [cand1]
            c1.text = None

            # Chunk 2: chunk.text fallback (no candidates)
            c2 = MagicMock()
            c2.candidates = []
            c2.text = " Extra fallback text"

            self.chunks = [c1, c2]
            self.idx = 0

        def __aiter__(self): return self
        async def __anext__(self):
            if self.idx >= len(self.chunks): raise StopAsyncIteration
            chunk = self.chunks[self.idx]
            self.idx += 1
            return chunk

    async def fake_success_stream(*args, **kwargs):
        return MockSuccessStream()

    mock_client.aio.models.generate_content_stream = fake_success_stream
    thoughts = []
    messages = []
    async def record_thought(ev: AgentThoughtEvent): thoughts.append(ev.delta)
    async def record_msg(ev: AgentMessageEvent): messages.append(ev.delta)

    out = await prov_effort.generate_turn(
        messages=[{"role": "user", "content": "Question"}],
        tools=None,
        session_id="s",
        on_thought=record_thought,
        on_message=record_msg,
    )
    assert out.content == "Main answer Extra fallback text"
    assert out.thought == "Gemini thinking..."
    assert thoughts == ["Gemini thinking..."]
    assert messages == ["Main answer", " Extra fallback text"]

