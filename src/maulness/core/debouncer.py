import asyncio
import time
from typing import Any, Callable, Coroutine, Optional


def balance_code_blocks(text: str) -> tuple[str, str]:
    """If text has an unclosed code block fence, close it for the chunk and reopen for the next chunk."""
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        last_idx = text.rfind("```")
        newline_idx = text.find("\n", last_idx)
        if newline_idx != -1:
            lang = text[last_idx + 3 : newline_idx].strip()
        else:
            lang = text[last_idx + 3 :].strip()
        closed_chunk = text + "\n```"
        reopen_prefix = f"```{lang}\n" if lang else "```\n"
        return closed_chunk, reopen_prefix
    return text, ""


def chunk_markdown_message(text: str, max_size: int = 1900) -> list[str]:
    """Split text into chunks of at most max_size characters, respecting
    code blocks, paragraphs, and word boundaries without losing characters.
    """
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    current = text

    while current:
        if len(current) <= max_size:
            chunks.append(current)
            break

        # Reserve small margin for potential fence closure (\n```)
        margin = 15 if max_size > 50 else 0
        target_limit = max(10, max_size - margin) if max_size > 10 else max_size

        # Find best split boundary within target_limit:
        # 1. Paragraph boundary: \n\n
        split_idx = current.rfind("\n\n", 0, target_limit)
        if split_idx != -1 and split_idx >= target_limit // 4:
            split_idx += 2
        else:
            # 2. Line boundary: \n
            split_idx = current.rfind("\n", 0, target_limit)
            if split_idx != -1 and split_idx >= target_limit // 4:
                split_idx += 1
            else:
                # 3. Word boundary: ' '
                split_idx = current.rfind(" ", 0, target_limit)
                if split_idx != -1 and split_idx >= target_limit // 4:
                    split_idx += 1
                else:
                    split_idx = target_limit

        head = current[:split_idx]
        tail = current[split_idx:]

        # Check code fence balance in head
        fence_count = head.count("```")
        if fence_count % 2 == 1:
            last_fence = head.rfind("```")
            newline_after = head.find("\n", last_fence)
            if newline_after != -1:
                lang = head[last_fence + 3 : newline_after].strip()
            else:
                lang = head[last_fence + 3 :].strip()

            closed_head = head.rstrip() + "\n```"
            reopen_tail = f"```{lang}\n" + tail.lstrip("\r\n")
            chunks.append(closed_head)
            current = reopen_tail
        else:
            chunks.append(head.rstrip("\r\n"))
            current = tail.lstrip("\r\n")

    return [c for c in chunks if c.strip()]


class MessageStreamDebouncer:
    """Buffers streaming token deltas and flushes them to a Discord message edit
    at a throttled rate (default 1.5s) to respect Discord's rate limits (~5 edits / 5s).
    Automatically chains messages when approaching the 2,000-character ceiling.
    """

    def __init__(
        self,
        flush_callback: Callable[..., Coroutine[Any, Any, None]],
        interval_seconds: float = 1.5,
        max_chunk_size: int = 1750,
    ):
        self.flush_callback = flush_callback
        self.interval = interval_seconds
        self.max_chunk_size = max_chunk_size

        self._buffer: list[str] = []
        self._history_parts: list[str] = []
        self._current_text = ""
        self._last_flush_time = 0.0
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._is_active = True

    @property
    def full_text(self) -> str:
        return "".join(self._history_parts) + self._current_text + "".join(self._buffer)

    def reset(self):
        """Reset internal text buffer to start streaming into a fresh message container."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._buffer.clear()
        self._history_parts.clear()
        self._current_text = ""
        self._is_active = True

    async def write(self, delta: str):
        """Append text delta to the active buffer and schedule a throttled flush."""
        if not self._is_active:
            return

        self._buffer.append(delta)
        now = time.monotonic()

        if now - self._last_flush_time >= self.interval and not self._lock.locked():
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
            await self._flush(is_final=False)
        elif self._flush_task is None or self._flush_task.done():
            remaining = max(0.0, self.interval - (now - self._last_flush_time))
            self._flush_task = asyncio.create_task(self._delayed_flush(remaining))

    async def _delayed_flush(self, delay: float):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._is_active:
            await asyncio.shield(self._flush(is_final=False))

    async def _flush(self, is_final: bool = False):
        async with self._lock:
            if not self._buffer and not is_final:
                return

            new_text = "".join(self._buffer)
            self._buffer.clear()
            self._current_text += new_text
            self._last_flush_time = time.monotonic()

            if is_final:
                if not self._current_text.strip():
                    return
                if len(self._current_text) > self.max_chunk_size:
                    chunks = chunk_markdown_message(
                        self._current_text, max_size=self.max_chunk_size
                    )
                    if not chunks:
                        return
                    for i, chunk in enumerate(chunks):
                        is_last = i == len(chunks) - 1
                        is_over = not is_last
                        try:
                            await self.flush_callback(
                                chunk, is_final=is_last, is_overflow=is_over
                            )
                        except TypeError:
                            await self.flush_callback(chunk, is_final=is_last)
                else:
                    try:
                        await self.flush_callback(
                            self._current_text, is_final=True, is_overflow=False
                        )
                    except TypeError:
                        await self.flush_callback(self._current_text, is_final=True)
            else:
                # Intermediate streaming flush
                if len(self._current_text) >= self.max_chunk_size:
                    chunks = chunk_markdown_message(
                        self._current_text, max_size=self.max_chunk_size
                    )
                    if len(chunks) > 1:
                        for chunk in chunks[:-1]:
                            self._history_parts.append(chunk)
                            try:
                                await self.flush_callback(
                                    chunk, is_final=False, is_overflow=True
                                )
                            except TypeError:
                                await self.flush_callback(chunk, is_final=False)
                        self._current_text = chunks[-1]
                        try:
                            await self.flush_callback(
                                self._current_text, is_final=False, is_overflow=False
                            )
                        except TypeError:
                            await self.flush_callback(
                                self._current_text, is_final=False
                            )
                    else:
                        chunk_to_flush, prefix = balance_code_blocks(self._current_text)
                        self._history_parts.append(chunk_to_flush)
                        self._current_text = prefix
                        try:
                            await self.flush_callback(
                                chunk_to_flush, is_final=False, is_overflow=True
                            )
                        except TypeError:
                            await self.flush_callback(chunk_to_flush, is_final=False)
                else:
                    try:
                        await self.flush_callback(
                            self._current_text, is_final=False, is_overflow=False
                        )
                    except TypeError:
                        await self.flush_callback(self._current_text, is_final=False)

    async def close(self):
        """Finalize the stream and flush all remaining buffered content."""
        self._is_active = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush under lock, guaranteeing sequential delivery after any in-flight flush completes
        await self._flush(is_final=True)
