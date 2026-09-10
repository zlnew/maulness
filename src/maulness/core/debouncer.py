import asyncio
import time
from typing import Any, Callable, Coroutine, Optional


def balance_code_blocks(text: str) -> tuple[str, str]:
    """If text has an unclosed code block fence, close it for the chunk and reopen for the next chunk."""
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        last_idx = text.rfind("```")
        newline_idx = text.find("\n", last_idx)
        lang = ""
        if newline_idx != -1:
            lang = text[last_idx + 3 : newline_idx].strip()
        closed_chunk = text + "\n```"
        reopen_prefix = f"```{lang}\n" if lang else "```\n"
        return closed_chunk, reopen_prefix
    return text, ""


class MessageStreamDebouncer:
    """Buffers streaming token deltas and flushes them to a Discord message edit
    at a throttled rate (default 900ms) to respect Discord's rate limits (~5 edits / 5s).
    Automatically chains messages when approaching the 2,000-character ceiling.
    """

    def __init__(
        self,
        flush_callback: Callable[..., Coroutine[Any, Any, None]],
        interval_seconds: float = 0.9,
        max_chunk_size: int = 1800,
    ):
        self.flush_callback = flush_callback
        self.interval = interval_seconds
        self.max_chunk_size = max_chunk_size

        self._buffer: list[str] = []
        self._current_text = ""
        self._last_flush_time = 0.0
        self._flush_task: Optional[asyncio.Task] = None
        self._is_active = True

    @property
    def full_text(self) -> str:
        return self._current_text + "".join(self._buffer)

    async def write(self, delta: str):
        """Append text delta to the active buffer and schedule a throttled flush."""
        if not self._is_active:
            return

        self._buffer.append(delta)
        now = time.monotonic()

        if now - self._last_flush_time >= self.interval:
            await self._flush(is_final=False)
        elif self._flush_task is None or self._flush_task.done():
            remaining = self.interval - (now - self._last_flush_time)
            self._flush_task = asyncio.create_task(self._delayed_flush(remaining))

    async def _delayed_flush(self, delay: float):
        await asyncio.sleep(delay)
        if self._is_active:
            await self._flush(is_final=False)

    async def _flush(self, is_final: bool = False):
        if not self._buffer and not is_final:
            return

        new_text = "".join(self._buffer)
        self._buffer.clear()
        self._current_text += new_text
        self._last_flush_time = time.monotonic()

        # Check if text exceeds chunk threshold
        if len(self._current_text) >= self.max_chunk_size and not is_final:
            chunk_to_flush, prefix = balance_code_blocks(self._current_text)
            self._current_text = prefix
            try:
                await self.flush_callback(chunk_to_flush, is_final=False, is_overflow=True)
            except TypeError:
                await self.flush_callback(chunk_to_flush, is_final=False)
        else:
            try:
                await self.flush_callback(self._current_text, is_final=is_final, is_overflow=False)
            except TypeError:
                await self.flush_callback(self._current_text, is_final=is_final)

    async def close(self):
        """Finalize the stream and flush all remaining buffered content."""
        self._is_active = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self._flush(is_final=True)
