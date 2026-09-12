from pathlib import Path

import pytest

from maulness.core.models import RepoGotcha, TaskRetrospective
from maulness.storage.db import StorageManager


@pytest.mark.asyncio
async def test_repo_gotchas_crud(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    repo_dir = str(tmp_path / "my_repo")

    g1 = RepoGotcha(
        id="gotcha-1",
        repo_path=repo_dir,
        component="auth_middleware",
        symptom="Cookie missing in CORS requests",
        resolution="Set SameSite=None and Secure=True in auth cookie config",
        commit_hash="abc1234",
        is_active=True,
    )
    await storage.save_repo_gotcha(g1)

    active = await storage.get_active_gotchas(repo_dir)
    assert len(active) == 1
    assert active[0].id == "gotcha-1"
    assert active[0].component == "auth_middleware"
    assert (
        active[0].resolution
        == "Set SameSite=None and Secure=True in auth cookie config"
    )


@pytest.mark.asyncio
async def test_invalidate_gotchas_for_files(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    repo_dir = str(tmp_path / "my_repo")

    g1 = RepoGotcha(
        id="gotcha-auth",
        repo_path=repo_dir,
        component="auth_middleware",
        symptom="JWT expired",
        resolution="Refresh token",
        commit_hash="1111111",
        is_active=True,
    )
    g2 = RepoGotcha(
        id="gotcha-db",
        repo_path=repo_dir,
        component="postgres_pool",
        symptom="Connection pool exhaustion",
        resolution="Set max_connections=50",
        commit_hash="2222222",
        is_active=True,
    )
    await storage.save_repo_gotcha(g1)
    await storage.save_repo_gotcha(g2)

    # Invalidate by touching auth_middleware file
    invalidated_count = await storage.invalidate_gotchas_for_files(
        repo_dir, ["src/auth_middleware/jwt.py", "README.md"]
    )
    assert invalidated_count == 1

    active = await storage.get_active_gotchas(repo_dir)
    assert len(active) == 1
    assert active[0].id == "gotcha-db"


@pytest.mark.asyncio
async def test_task_retrospectives_crud(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    repo_dir = str(tmp_path / "my_repo")

    retro = TaskRetrospective(
        task_id="task-101",
        repo_path=repo_dir,
        summary="Refactored database pooling layer",
        passed=True,
        total_steps=8,
        cost_usd=0.035,
    )
    await storage.save_task_retrospective(retro)

    retros = await storage.get_task_retrospectives(repo_dir)
    assert len(retros) == 1
    assert retros[0].task_id == "task-101"
    assert retros[0].passed is True
    assert retros[0].total_steps == 8
    assert retros[0].cost_usd == 0.035
