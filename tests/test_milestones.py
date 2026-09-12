import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from maulness.core.models import Milestone, MilestonePlan, ChangeSet, ReviewVerdict
from maulness.core.pipeline import PipelineOrchestrator, parse_milestone_plan
from maulness.core.pipelines import PipelineDefinition, PipelineStage
from maulness.storage.db import StorageManager


def test_milestone_models():
    m1 = Milestone(
        id="M1",
        title="Setup project scaffold",
        files_to_modify=["pyproject.toml"],
        verification_command="uv run pytest",
        acceptance_criteria="All baseline tests pass",
    )
    plan = MilestonePlan(task_id="task_123", summary="Test plan", milestones=[m1])
    assert len(plan.milestones) == 1
    assert plan.milestones[0].id == "M1"

    cs = ChangeSet(
        task_id="task_123",
        milestone_id="M1",
        touched_files=["pyproject.toml"],
        git_diff_stat="1 file changed, 2 insertions(+)",
    )
    assert cs.verification_passed is True

    verdict = ReviewVerdict(decision="PASS", flaws=[], target_rework_files=[])
    assert verdict.decision == "PASS"


def test_parse_milestone_plan_from_json():
    json_plan = """
Here is the architectural milestone plan:
```json
{
  "task_id": "t1",
  "summary": "Partitioned build plan",
  "milestones": [
    {
      "id": "M1",
      "title": "Database Schema",
      "files_to_modify": ["schema.sql"],
      "verification_command": "echo 'schema ok'",
      "acceptance_criteria": "DDL applied cleanly"
    },
    {
      "id": "M2",
      "title": "API Handlers",
      "files_to_modify": ["api.py"],
      "verification_command": "echo 'api ok'",
      "acceptance_criteria": "Endpoints return 200"
    }
  ]
}
```
    """
    parsed = parse_milestone_plan(json_plan, "t1")
    assert parsed is not None
    assert len(parsed.milestones) == 2
    assert parsed.milestones[0].id == "M1"
    assert parsed.milestones[1].id == "M2"


def test_parse_milestone_plan_from_markdown():
    md_plan = """
# Implementation Plan

### Milestone 1: Setup storage layer
Verification: `pytest tests/test_db.py`
Criteria: Storage initialized and tables created.

### Milestone 2: Implement endpoint routing
Verification: `pytest tests/test_api.py`
Criteria: All routing handlers return correct response payload.
    """
    parsed = parse_milestone_plan(md_plan, "t2")
    assert parsed is not None
    assert len(parsed.milestones) == 2
    assert parsed.milestones[0].title == "Setup storage layer"
    assert parsed.milestones[0].verification_command == "pytest tests/test_db.py"


@pytest.mark.asyncio
async def test_pipeline_milestone_sub_sessions_runner(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "test_ms.db")
    await storage.initialize()

    plan_json = """
```json
{
  "task_id": "ms_task",
  "milestones": [
    {"id": "M1", "title": "Milestone One", "verification_command": "echo 'm1'"},
    {"id": "M2", "title": "Milestone Two", "verification_command": "echo 'm2'"}
  ]
}
```
    """

    pipe_def = PipelineDefinition(
        name="ms_pipeline",
        stages=[
            PipelineStage(name="plan", profile="planner", prompt="Plan", output_key="plan"),
            PipelineStage(name="build", profile="builder", prompt="Build with plan: {plan}", run_milestones=True),
        ],
    )

    mock_planner = MagicMock()
    mock_planner.run = AsyncMock(return_value=plan_json)

    mock_builder = MagicMock()
    mock_builder.run = AsyncMock(return_value="Milestone executed cleanly")

    def mock_get_provider(prof):
        if prof.name == "planner":
            return mock_planner
        return mock_builder

    orchestrator = PipelineOrchestrator(storage=storage)

    with patch("maulness.core.pipeline.get_provider_for_profile", side_effect=mock_get_provider):
        task = await orchestrator.run_pipeline(
            title="Milestone Sub-Session Test",
            prompt="Build feature",
            workspace_path=tmp_path,
            pipeline_def=pipe_def,
            auto_proceed=True,
        )

        assert task.status.value == "done"
        # Builder was called once for each milestone (2 times)
        assert mock_builder.run.call_count == 2
