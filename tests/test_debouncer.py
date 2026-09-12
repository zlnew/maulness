import asyncio
import pytest

from maulness.core.debouncer import MessageStreamDebouncer


@pytest.mark.asyncio
async def test_debouncer_throttles_flushes():
    flushed_chunks = []

    async def flush_cb(text: str, is_final: bool):
        flushed_chunks.append((text, is_final))

    debouncer = MessageStreamDebouncer(flush_callback=flush_cb, interval_seconds=0.05)

    # Rapid writes within the 50ms interval
    await debouncer.write("Hello ")
    await debouncer.write("world ")
    await debouncer.write("from ")
    await debouncer.write("Maulness!")

    # Wait for the debounced flush
    await asyncio.sleep(0.08)

    # First write triggers leading-edge flush for immediate UI feedback;
    # subsequent rapid writes are throttled into the second flush.
    assert len(flushed_chunks) == 2
    assert flushed_chunks[0][0] == "Hello "
    assert flushed_chunks[1][0] == "Hello world from Maulness!"
    assert flushed_chunks[1][1] is False

    # Close debouncer
    await debouncer.close()
    assert flushed_chunks[-1][1] is True


@pytest.mark.asyncio
async def test_debouncer_chunks_large_text():
    flushed_chunks = []

    async def flush_cb(text: str, is_final: bool):
        flushed_chunks.append((text, is_final))

    # Low chunk threshold for testing
    debouncer = MessageStreamDebouncer(
        flush_callback=flush_cb, interval_seconds=0.01, max_chunk_size=20
    )

    await debouncer.write("This is a long sentence that exceeds twenty chars.")
    await asyncio.sleep(0.03)
    await debouncer.close()

    assert len(flushed_chunks) >= 2


def test_balance_code_blocks():
    from maulness.core.debouncer import balance_code_blocks

    # Balanced remains unchanged
    text = "Here is some code:\n```python\nprint('hello')\n```\nDone."
    closed, reopen = balance_code_blocks(text)
    assert closed == text
    assert reopen == ""

    # Single unclosed block gets closed
    unclosed = "Here is unclosed:\n```python\nprint('hello')"
    closed, reopen = balance_code_blocks(unclosed)
    assert closed.endswith("\n```")
    assert closed.count("```") == 2
    assert reopen == "```python\n"

    # Multiple blocks with odd count
    multi_unclosed = "```bash\necho 1\n```\nSome text\n```python\nx = 1"
    closed, reopen = balance_code_blocks(multi_unclosed)
    assert closed.endswith("\n```")
    assert closed.count("```") == 4
    assert reopen == "```python\n"


@pytest.mark.asyncio
async def test_debouncer_overflow_triggers_callback():
    overflow_chunks = []
    normal_flushes = []

    async def flush_cb(text: str, is_final: bool, is_overflow: bool = False):
        if is_overflow:
            overflow_chunks.append(text)
        else:
            normal_flushes.append(text)

    debouncer = MessageStreamDebouncer(
        flush_callback=flush_cb,
        interval_seconds=0.01,
        max_chunk_size=50,
    )

    # Write enough text to cause an overflow
    await debouncer.write("Chunk 1: " + "a" * 60 + "\n")
    await asyncio.sleep(0.03)

    assert len(overflow_chunks) >= 1
    await debouncer.close()


@pytest.mark.asyncio
async def test_debouncer_in_flight_flush_synchronization():
    """Verify that when a flush is in-flight, a subsequent write and close()
    properly await the active flush and deliver the final complete text."""
    flushed_history = []

    async def slow_flush(text: str, is_final: bool = False, is_overflow: bool = False):
        await asyncio.sleep(0.05)  # Simulate Discord HTTP edit latency
        flushed_history.append((text, is_final))

    debouncer = MessageStreamDebouncer(flush_callback=slow_flush, interval_seconds=0.02)

    # Initial write triggers leading flush
    await debouncer.write("Part 1")
    # Wait for interval so delayed flush triggers
    await asyncio.sleep(0.03)

    # Write more text while delayed flush is running
    await debouncer.write(" Part 2")
    # Close immediately while flush may be active
    await debouncer.close()

    assert len(flushed_history) >= 2
    # The final flush MUST be marked is_final=True and contain the complete text
    assert flushed_history[-1][1] is True
    assert flushed_history[-1][0] == "Part 1 Part 2"


@pytest.mark.asyncio
async def test_debouncer_properties_and_reset():
    """Verify full_text property and reset() functionality with active tasks."""
    flushes = []

    async def flush_cb(text: str, is_final: bool, is_overflow: bool = False):
        flushes.append(text)

    debouncer = MessageStreamDebouncer(flush_callback=flush_cb, interval_seconds=0.1)

    # Initial write flushes leading edge
    await debouncer.write("Alpha ")
    assert debouncer.full_text == "Alpha "

    # Rapid second write schedules a delayed task and buffers
    await debouncer.write("Beta")
    assert debouncer.full_text == "Alpha Beta"

    # Reset clears buffer and cancels active task
    debouncer.reset()
    assert debouncer.full_text == ""

    # Streaming can cleanly resume after reset
    await debouncer.write("Gamma")
    assert debouncer.full_text == "Gamma"
    await debouncer.close()


@pytest.mark.asyncio
async def test_debouncer_write_after_close():
    """Verify writes after close() are safely ignored."""
    flushes = []

    async def flush_cb(text: str, is_final: bool, is_overflow: bool = False):
        flushes.append(text)

    debouncer = MessageStreamDebouncer(flush_callback=flush_cb, interval_seconds=0.01)
    await debouncer.write("First")
    await debouncer.close()

    # Writing to closed debouncer must be a no-op
    await debouncer.write("Ignored")
    assert "Ignored" not in debouncer.full_text


@pytest.mark.asyncio
async def test_debouncer_empty_flush_and_legacy_signature():
    """Verify empty flush early return and fallback for legacy 2-arg callbacks."""
    legacy_flushes = []

    async def legacy_callback(text: str, is_final: bool):
        # 2-arg callback that raises TypeError if called with is_overflow
        legacy_flushes.append((text, is_final))

    debouncer = MessageStreamDebouncer(
        flush_callback=legacy_callback, interval_seconds=0.01
    )

    # Empty flush when not final should be a no-op
    await debouncer._flush(is_final=False)
    assert len(legacy_flushes) == 0

    # Write normal chunk
    await debouncer.write("Legacy Test")
    await asyncio.sleep(0.02)
    await debouncer.close()

    assert len(legacy_flushes) >= 1
    assert legacy_flushes[-1][0] == "Legacy Test"
    assert legacy_flushes[-1][1] is True


@pytest.mark.asyncio
async def test_debouncer_legacy_signature_overflow():
    """Verify legacy 2-arg callback fallback during an overflow event."""
    legacy_flushes = []

    async def legacy_callback(text: str, is_final: bool):
        legacy_flushes.append((text, is_final))

    debouncer = MessageStreamDebouncer(
        flush_callback=legacy_callback, interval_seconds=0.01, max_chunk_size=30
    )
    await debouncer.write("A" * 40)
    await asyncio.sleep(0.02)
    await debouncer.close()

    assert len(legacy_flushes) >= 2


def test_balance_code_blocks_no_newline():
    """Verify code block balance when no newline exists after language tag."""
    from maulness.core.debouncer import balance_code_blocks

    text = "```python"
    closed, reopen = balance_code_blocks(text)
    assert closed == "```python\n```"
    assert reopen == "```python\n"


@pytest.mark.asyncio
async def test_debouncer_write_cancels_pending_delayed_flush():
    """Verify that when interval has elapsed, a direct write cancels any pending delayed task."""
    flushes = []

    async def flush_cb(text: str, is_final: bool, is_overflow: bool = False):
        flushes.append(text)

    debouncer = MessageStreamDebouncer(flush_callback=flush_cb, interval_seconds=1.0)

    await debouncer.write("Start")
    # Rapid write schedules a delayed flush task with long delay
    await debouncer.write(" -> Delayed")
    assert debouncer._flush_task is not None and not debouncer._flush_task.done()

    # Yield control to event loop so _delayed_flush actually starts and enters asyncio.sleep
    await asyncio.sleep(0.01)

    # Artificially age the last flush time so next write triggers leading flush
    debouncer._last_flush_time -= 2.0
    task = debouncer._flush_task
    await debouncer.write(" -> Immediate")
    assert task.cancelling() > 0 or task.done()
    await asyncio.sleep(0.01)
    assert task.done()
    await debouncer.close()

    assert "Start -> Delayed -> Immediate" in flushes[-1]


# ==============================================================================
# Race Condition & Concurrency Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_debouncer_concurrent_writers_race():
    """Stress test debouncer with 10 concurrent async producers writing interleaved tokens."""
    flushed_history = []
    active_flushes = 0
    max_concurrent_flushes = 0

    async def instrumented_flush(text: str, is_final: bool, is_overflow: bool = False):
        nonlocal active_flushes, max_concurrent_flushes
        active_flushes += 1
        max_concurrent_flushes = max(max_concurrent_flushes, active_flushes)
        try:
            await asyncio.sleep(0.01)  # Simulate network hop
            flushed_history.append((text, is_final))
        finally:
            active_flushes -= 1

    debouncer = MessageStreamDebouncer(
        flush_callback=instrumented_flush, interval_seconds=0.02
    )

    async def worker(worker_id: int):
        for i in range(10):
            await debouncer.write(f"[W{worker_id}:{i}]")
            await asyncio.sleep(0.005)

    # Launch 10 workers concurrently
    await asyncio.gather(*(worker(w) for w in range(10)))
    await debouncer.close()

    # Invariant 1: Exactly 1 flush running at any time (strict serialization)
    assert max_concurrent_flushes == 1
    # Invariant 2: Final flush must be marked final
    assert flushed_history[-1][1] is True
    # Invariant 3: All 100 worker tokens must be present in the final message
    final_text = flushed_history[-1][0]
    for w in range(10):
        for i in range(10):
            assert f"[W{w}:{i}]" in final_text


@pytest.mark.asyncio
async def test_debouncer_mutual_exclusion_invariant():
    """Verify that rapid concurrent writes, delayed flushes, and close never violate the single-lock invariant."""
    active_executions = 0
    max_concurrency = 0

    async def slow_mock_discord_edit(
        text: str, is_final: bool, is_overflow: bool = False
    ):
        nonlocal active_executions, max_concurrency
        active_executions += 1
        max_concurrency = max(max_concurrency, active_executions)
        try:
            await asyncio.sleep(0.02)
        finally:
            active_executions -= 1

    debouncer = MessageStreamDebouncer(
        flush_callback=slow_mock_discord_edit, interval_seconds=0.01
    )

    for i in range(30):
        await debouncer.write(f"chunk_{i} ")
        if i % 5 == 0:
            await asyncio.sleep(0.015)

    await debouncer.close()
    assert max_concurrency == 1
    assert active_executions == 0


# ==============================================================================
# High-Throughput & Stress Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_debouncer_high_throughput_stream_stress():
    """Simulate a full LLM response stream (200 rapid tokens) with simulated Discord API latency."""
    delivered_versions = []

    async def mock_discord_api(text: str, is_final: bool, is_overflow: bool = False):
        await asyncio.sleep(0.01)  # 10ms Discord round-trip latency
        delivered_versions.append(text)

    debouncer = MessageStreamDebouncer(
        flush_callback=mock_discord_api, interval_seconds=0.03
    )

    expected_full_text = ""
    for i in range(200):
        token = f"tok{i}_"
        expected_full_text += token
        await debouncer.write(token)
        if i % 10 == 0:
            await asyncio.sleep(0.005)

    await debouncer.close()

    # Monotonic length growth check
    for j in range(1, len(delivered_versions)):
        assert len(delivered_versions[j]) >= len(delivered_versions[j - 1])

    # Final delivered version must match 100% of streamed tokens
    assert delivered_versions[-1] == expected_full_text


@pytest.mark.asyncio
async def test_debouncer_massive_overflow_chain_stress():
    """Stream large content exceeding chunk threshold with code blocks to stress test chaining."""
    overflow_chunks = []
    normal_flushes = []

    async def chunking_callback(text: str, is_final: bool, is_overflow: bool = False):
        if is_overflow:
            overflow_chunks.append(text)
        else:
            normal_flushes.append(text)

    debouncer = MessageStreamDebouncer(
        flush_callback=chunking_callback,
        interval_seconds=0.005,
        max_chunk_size=40,
    )

    # Stream text with markdown code fences that spans multiple chunks
    part1 = "Some intro text.\n```python\ndef fn():\n"
    part2 = "    return 42\n" * 8
    part3 = "```\nFinal conclusion."

    for chunk in [part1, part2, part3]:
        for ch in chunk.splitlines(keepends=True):
            await debouncer.write(ch)
            await asyncio.sleep(0.006)

    await debouncer.close()

    assert len(overflow_chunks) >= 3
    # Check that all code blocks in overflow chunks are balanced
    for chunk in overflow_chunks:
        fence_count = chunk.count("```")
        assert fence_count % 2 == 0, f"Unbalanced code block found in chunk: {chunk}"


def test_chunk_markdown_message_boundaries_and_fences():
    from maulness.core.debouncer import chunk_markdown_message

    # Empty and short
    assert chunk_markdown_message("") == []
    assert chunk_markdown_message("Hello world", max_size=50) == ["Hello world"]

    # Paragraph split
    text = "Paragraph one is here.\n\nParagraph two is here."
    chunks = chunk_markdown_message(text, max_size=30)
    assert len(chunks) == 2
    assert "Paragraph one is here." in chunks[0]
    assert "Paragraph two is here." in chunks[1]

    # Code block splitting balances fences in both chunks
    code_text = "Intro text.\n```python\nline1 = 1\nline2 = 2\nline3 = 3\n```\nOutro."
    code_chunks = chunk_markdown_message(code_text, max_size=40)
    assert len(code_chunks) >= 2
    for c in code_chunks:
        assert c.count("```") % 2 == 0, f"Unbalanced fences in chunk: {c}"

    # Single contiguous string without spaces splits safely without infinite loop
    long_str = "x" * 150
    str_chunks = chunk_markdown_message(long_str, max_size=50)
    assert len(str_chunks) >= 3
    assert "".join(str_chunks) == long_str
