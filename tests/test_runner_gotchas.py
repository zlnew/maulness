from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from maulness.core.models import RepoGotcha
from maulness.core.runner import TaskRunner
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_runner_gotcha_injection_and_retrospective(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    # Pre-seed an active gotcha
    gotcha = RepoGotcha(
        id="g-cors",
        repo_path=str(tmp_path),
        component="cors",
        symptom="OPTIONS 403 on preflight",
        resolution="Allow headers Authorization and Content-Type",
        commit_hash="deadbeef",
        is_active=True,
    )
    await storage.save_repo_gotcha(gotcha)

    runner = TaskRunner(storage=storage)

    with (
        patch("maulness.core.runner.get_provider_for_profile") as mock_get_provider,
        patch("maulness.core.runner.TestFreezeGate") as mock_freeze,
    ):
        mock_provider = AsyncMock()
        mock_get_provider.return_value = mock_provider
        mock_freeze.return_value.verify_no_tampering = lambda: None

        task_record = await runner.run_direct(
            prompt="Build feature",
            workspace_path=tmp_path,
            verbose=False,
        )

        assert task_record is not None

        # Verify gotcha was injected into the prompt executed on provider
        call_kwargs = mock_provider.run.call_args[1]
        executed_prompt = call_kwargs["prompt"]
        assert "[Known Workspace Gotchas]:" in executed_prompt
        assert "**cors**: OPTIONS 403 on preflight" in executed_prompt
        assert "Allow headers Authorization and Content-Type" in executed_prompt

        # Verify task retrospective was saved
        retros = await storage.get_task_retrospectives(str(tmp_path))
        assert len(retros) == 1
        assert retros[0].passed is True
        assert "Build feature" in retros[0].summary
