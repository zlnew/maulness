import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from maulness.core.memory import AutonomousMemoryExtractor


def test_append_facts_to_memory_deduplication(tmp_path: Path):
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text("# Initial Notes\n- User uses Linux\n")

    extractor = AutonomousMemoryExtractor(memory_file=mem_file)

    # 1. Add new facts
    added = extractor._append_facts_to_memory([
        "- Uses pnpm instead of npm",
        "- Never push to remote without approval",
    ])
    assert len(added) == 2

    content = mem_file.read_text()
    assert "## Auto-Learned Knowledge & Decisions" in content
    assert "- Uses pnpm instead of npm" in content
    assert "- Never push to remote without approval" in content

    # 2. Add duplicate facts - should be ignored
    added_dup = extractor._append_facts_to_memory([
        "- Uses pnpm instead of npm",
        "- user uses linux",  # case-insensitive match
    ])
    assert len(added_dup) == 0


@pytest.mark.asyncio
async def test_extract_and_update_too_few_messages(tmp_path: Path):
    mem_file = tmp_path / "MEMORY.md"
    extractor = AutonomousMemoryExtractor(memory_file=mem_file)

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = await extractor.extract_and_update(messages)
    assert result == []


@pytest.mark.asyncio
async def test_extract_and_update_mock_provider(tmp_path: Path):
    mem_file = tmp_path / "MEMORY.md"
    extractor = AutonomousMemoryExtractor(memory_file=mem_file)

    messages = [
        {"role": "user", "content": "I prefer using service-name addressing in Docker"},
        {"role": "assistant", "content": "Understood, personal-postgres:5432."},
        {"role": "user", "content": "And never expose ports to host."},
        {"role": "assistant", "content": "Noted."},
    ]

    mock_provider = AsyncMock()
    mock_provider.run.return_value = "- Prefers service-name addressing\n- Never expose ports to host"

    with patch("maulness.core.providers.factory.get_provider_for_profile", return_value=mock_provider):
        added = await extractor.extract_and_update(messages, profile_name="default")
        assert len(added) == 2
        assert "- Prefers service-name addressing" in added
        assert "- Never expose ports to host" in added


@pytest.mark.asyncio
async def test_extract_and_update_edge_cases(tmp_path: Path):
    mem_file = tmp_path / "MEMORY.md"
    extractor = AutonomousMemoryExtractor(memory_file=mem_file)

    # 1. Empty message contents
    empty_msgs = [{"role": "user", "content": ""}, {"role": "assistant", "content": "   "}] * 3
    res = await extractor.extract_and_update(empty_msgs)
    assert res == []

    # Valid msgs for subsequent tests
    valid_msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]

    # 2. Provider returns NONE
    mock_none = AsyncMock()
    mock_none.run.return_value = "NONE"
    with patch("maulness.core.providers.factory.get_provider_for_profile", return_value=mock_none):
        res = await extractor.extract_and_update(valid_msgs)
        assert res == []

    # 3. Provider returns prose without bullets
    mock_prose = AsyncMock()
    mock_prose.run.return_value = "Here is what happened in the conversation."
    with patch("maulness.core.providers.factory.get_provider_for_profile", return_value=mock_prose):
        res = await extractor.extract_and_update(valid_msgs)
        assert res == []

    # 4. Provider raises Exception
    mock_err = AsyncMock()
    mock_err.run.side_effect = RuntimeError("Extraction provider timed out")
    with patch("maulness.core.providers.factory.get_provider_for_profile", return_value=mock_err):
        res = await extractor.extract_and_update(valid_msgs)
        assert res == []


def test_append_facts_existing_header(tmp_path: Path):
    mem_file = tmp_path / "MEMORY.md"
    mem_file.write_text("# Notes\n## Auto-Learned Knowledge & Decisions\n- Fact 1\n")

    extractor = AutonomousMemoryExtractor(memory_file=mem_file)
    added = extractor._append_facts_to_memory(["- Fact 2"])
    assert added == ["- Fact 2"]

    content = mem_file.read_text()
    # Ensure header wasn't duplicated
    assert content.count("## Auto-Learned Knowledge & Decisions") == 1
    assert "- Fact 2" in content

