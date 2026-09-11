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





