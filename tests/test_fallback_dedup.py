import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from maulness.core.profiles import Profile
from maulness.core.providers.base import BaseProvider
from maulness.core.providers.fallback import FallbackProviderChain
from maulness.core.tools import execute_tool_call, get_turn_executed_tools, clear_turn_tools
from maulness.core.debouncer import MessageStreamDebouncer


@pytest.mark.asyncio
async def test_fallback_chain_passes_tool_context_and_deduplicates(tmp_path: Path):
    p1 = Profile(identity={"name": "p1"}, agent={"provider": "gemini", "model": "gemini-3.7-flash"})
    p2 = Profile(identity={"name": "p2"}, agent={"provider": "ollama", "model": "gemma4:31b-cloud"})

    sess_id = "test_sess_dedup_1"

    # Primary provider executes 'pwd', then crashes on synthesis turn
    async def primary_run(**kwargs):
        session_id = kwargs.get("session_id")
        # Run tool
        res = await execute_tool_call(
            name="run_command",
            args={"command": "pwd"},
            workspace_path=tmp_path,
            session_id=session_id,
        )
        assert str(tmp_path.resolve()) in res
        # Simulate failure during synthesis
        raise RuntimeError("503 Service Unavailable during synthesis")

    mock_primary = AsyncMock(spec=BaseProvider)
    mock_primary.profile = p1
    mock_primary.run.side_effect = primary_run

    received_fallback_prompt = []

    # Fallback provider receives prompt with tool context
    async def fallback_run(**kwargs):
        prompt_received = kwargs.get("prompt")
        received_fallback_prompt.append(prompt_received)
        session_id = kwargs.get("session_id")

        # Fallback provider attempts to run the same tool call again
        res2 = await execute_tool_call(
            name="run_command",
            args={"command": "pwd"},
            workspace_path=tmp_path,
            session_id=session_id,
        )
        # Should get the exact same cached output without re-executing
        assert str(tmp_path.resolve()) in res2
        return "The working directory is: " + res2

    mock_fallback = AsyncMock(spec=BaseProvider)
    mock_fallback.profile = p2
    mock_fallback.run.side_effect = fallback_run

    chain = FallbackProviderChain(primary=mock_primary, fallbacks=[mock_fallback])

    result = await chain.run(session_id=sess_id, prompt="What is current pwd?")

    # 1. Fallback returned synthesized answer
    assert "The working directory is:" in result
    assert str(tmp_path.resolve()) in result

    # 2. Fallback provider prompt contained system note with completed tool results
    assert len(received_fallback_prompt) == 1
    assert "SYSTEM NOTE: The following tool(s) were already executed" in received_fallback_prompt[0]
    assert "run_command" in received_fallback_prompt[0]
    assert str(tmp_path.resolve()) in received_fallback_prompt[0]

    # 3. Turn tools were cleaned up in finally block
    assert len(get_turn_executed_tools(sess_id)) == 0


def test_debouncer_reset():
    flushed = []
    async def flush_cb(text, is_final=False, is_overflow=False):
        flushed.append(text)

    debouncer = MessageStreamDebouncer(flush_callback=flush_cb)
    debouncer._buffer.append("prior text")
    assert debouncer.full_text == "prior text"

    debouncer.reset()
    assert debouncer.full_text == ""
    assert len(debouncer._buffer) == 0
