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
