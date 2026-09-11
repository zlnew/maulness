import pytest
from pathlib import Path
from maulness.core.pipelines import (
    PipelineDefinition,
    PipelineManager,
    PipelineStage,
    SafeFormatDict,
    StageTransitions,
    check_is_fail_verdict,
    check_is_pass_verdict,
    check_is_rework_verdict,
)


def test_safe_format_dict():
    context = {"repo_name": "expense-tracker", "title": "Add caching"}
    template = "Repo: {repo_name}, Title: {title}, Plan: {plan}"
    rendered = template.format_map(SafeFormatDict(context))
    assert rendered == "Repo: expense-tracker, Title: Add caching, Plan: {plan}"


def test_pipeline_manager_discovery(tmp_path: Path):
    manager = PipelineManager(pipelines_dir=tmp_path)
    pipelines = manager.list_pipelines()
    pipeline_names = {p.name for p in pipelines}

    assert "standard" in pipeline_names
    assert "quick" in pipeline_names
    assert "plan_only" in pipeline_names
    assert "audit" in pipeline_names

    standard = manager.get_pipeline("standard")
    assert standard.name == "standard"
    assert len(standard.stages) == 3
    assert standard.stages[0].name == "planning"
    assert standard.stages[0].profile == "planner"
    assert standard.stages[0].gate is not None
    assert standard.stages[1].name == "building"
    assert standard.stages[1].profile == "builder"
    assert standard.stages[2].name == "review"
    assert standard.stages[2].requires_diff is True


def test_user_pipeline_override(tmp_path: Path):
    custom_yaml = """
name: custom_test
description: "A custom test pipeline"
stages:
  - name: test_stage
    profile: builder
    prompt: "Test prompt: {title}"
"""
    (tmp_path / "custom_test.yaml").write_text(custom_yaml)

    manager = PipelineManager(pipelines_dir=tmp_path)
    custom = manager.get_pipeline("custom_test")
    assert custom.name == "custom_test"
    assert len(custom.stages) == 1
    assert custom.stages[0].profile == "builder"


@pytest.mark.asyncio
async def test_orchestrator_execution(tmp_path: Path, monkeypatch):
    from unittest.mock import AsyncMock
    from maulness.core.models import TaskStatus
    from maulness.core.pipeline import PipelineOrchestrator
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "test.db")
    pipeline_manager = PipelineManager(pipelines_dir=tmp_path)

    # Define a test pipeline with 2 stages
    custom_pipeline = PipelineDefinition(
        name="test_flow",
        description="test",
        stages=[
            PipelineStage(
                name="step1",
                profile="planner",
                status="planning",
                prompt="Plan for {title}",
                output_key="step1_out",
            ),
            PipelineStage(
                name="step2",
                profile="builder",
                status="building",
                prompt="Build with plan: {step1_out}",
                output_key="step2_out",
            ),
        ],
    )

    # Mock provider
    executed_prompts = []

    class MockProvider:
        async def run(self, prompt, **kwargs):
            executed_prompts.append(prompt)
            return f"output_for_{len(executed_prompts)}"

    from maulness.core import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_provider_for_profile", lambda p: MockProvider())

    orchestrator = PipelineOrchestrator(storage=storage, pipeline_manager=pipeline_manager)
    task = await orchestrator.run_pipeline(
        repo_name="maulness",
        title="Test Pipeline Task",
        prompt="Sample requirements",
        pipeline_def=custom_pipeline,
        auto_proceed=True,
    )

    assert task.status == TaskStatus.DONE
    assert len(executed_prompts) == 2
    assert "Plan for Test Pipeline Task" in executed_prompts[0]
    assert "Build with plan: output_for_1" in executed_prompts[1]


def test_verdict_helpers():
    # Rework verdicts
    assert check_is_rework_verdict("[DECISION: REWORK]\nChanges needed.")
    assert check_is_rework_verdict("[VERDICT: REWORK]")
    assert check_is_rework_verdict("**Decision:** REWORK")
    assert check_is_rework_verdict("Decision: **REWORK**")
    assert check_is_rework_verdict("**Audit Scorecard:** REWORK")
    assert check_is_rework_verdict("Scorecard: rework")
    assert not check_is_rework_verdict("[DECISION: PASS]")
    assert not check_is_rework_verdict("Decision: PASS")
    assert not check_is_rework_verdict("Provide an audit scorecard: PASS or REWORK with reasons.")
    assert not check_is_rework_verdict("")

    # Pass verdicts
    assert check_is_pass_verdict("[DECISION: PASS]\nLooks great.")
    assert check_is_pass_verdict("[VERDICT: PASS]")
    assert check_is_pass_verdict("**Decision:** PASS")
    assert check_is_pass_verdict("Decision: **PASS**")
    assert not check_is_pass_verdict("[DECISION: REWORK]")
    assert not check_is_pass_verdict("")

    # Fail verdicts
    assert check_is_fail_verdict("[DECISION: FAIL]")
    assert check_is_fail_verdict("Verdict: FAIL")
    assert not check_is_fail_verdict("[DECISION: PASS]")


@pytest.mark.asyncio
async def test_orchestrator_cyclic_rework_loop(tmp_path: Path, monkeypatch):
    from maulness.core.models import TaskStatus
    from maulness.core.pipeline import PipelineOrchestrator
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "cyclic_test.db")
    pipeline_manager = PipelineManager(pipelines_dir=tmp_path)

    cyclic_pipeline = PipelineDefinition(
        name="cyclic_flow",
        description="test cyclic transitions",
        stages=[
            PipelineStage(
                name="planning",
                profile="planner",
                status="planning",
                prompt="Plan: {title}",
                output_key="plan",
            ),
            PipelineStage(
                name="building",
                profile="builder",
                status="building",
                prompt="Build: {plan}\n{rework_feedback_block}",
                output_key="build_out",
            ),
            PipelineStage(
                name="review",
                profile="reviewer",
                status="review",
                prompt="Review changes",
                output_key="review_out",
                transitions=StageTransitions(
                    rework_target="building",
                    max_reworks=2,
                ),
            ),
        ],
    )

    call_count = 0
    executed_prompts = []

    class CyclicMockProvider:
        async def run(self, prompt, session_id=None, **kwargs):
            nonlocal call_count
            call_count += 1
            executed_prompts.append((session_id, prompt))
            if call_count == 1:
                # planning
                return "Architecture Plan v1"
            elif call_count == 2:
                # building attempt 1
                return "Implementation v1"
            elif call_count == 3:
                # review attempt 1 -> requests rework
                return "[DECISION: REWORK]\nMissing test cases for negative balance."
            elif call_count == 4:
                # building attempt 2 (after rework loopback)
                return "Implementation v2 with negative balance tests"
            elif call_count == 5:
                # review attempt 2 -> passes
                return "[DECISION: PASS]\nAll requirements and edge tests verified."
            return "unexpected call"

    from maulness.core import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_provider_for_profile", lambda p: CyclicMockProvider())

    orchestrator = PipelineOrchestrator(storage=storage, pipeline_manager=pipeline_manager)
    task = await orchestrator.run_pipeline(
        repo_name="cyclic_repo",
        title="Cyclic Feature Task",
        prompt="Add balance validation",
        workspace_path=tmp_path / "ws1",
        pipeline_def=cyclic_pipeline,
        auto_proceed=True,
    )

    assert task.status == TaskStatus.DONE
    assert len(executed_prompts) == 5

    # Check stage 1 (planning)
    assert "Plan: Cyclic Feature Task" in executed_prompts[0][1]

    # Check stage 2 (building - attempt 1): rework block should be empty
    assert "Architecture Plan v1" in executed_prompts[1][1]
    assert "Rework Cycle" not in executed_prompts[1][1]

    # Check stage 3 (review - attempt 1)
    assert "Review changes" in executed_prompts[2][1]

    # Check stage 4 (building - attempt 2 after loopback): contains critique block
    assert "Missing test cases for negative balance." in executed_prompts[3][1]
    assert "Rework Cycle 1/2" in executed_prompts[3][1]
    assert "r1" in executed_prompts[3][0]

    # Check stage 5 (review - attempt 2): pass verdict
    assert "Review changes" in executed_prompts[4][1]


@pytest.mark.asyncio
async def test_orchestrator_cyclic_max_reworks_cap(tmp_path: Path, monkeypatch):
    from maulness.core.models import TaskStatus
    from maulness.core.pipeline import PipelineOrchestrator
    from maulness.storage.db import StorageManager

    storage = StorageManager(db_path=tmp_path / "cyclic_cap_test.db")
    pipeline_manager = PipelineManager(pipelines_dir=tmp_path)

    cyclic_pipeline = PipelineDefinition(
        name="cyclic_cap_flow",
        description="test rework ceiling halt",
        stages=[
            PipelineStage(
                name="building",
                profile="builder",
                status="building",
                prompt="Build: {title}\n{rework_feedback_block}",
            ),
            PipelineStage(
                name="review",
                profile="reviewer",
                status="review",
                prompt="Review changes",
                transitions=StageTransitions(
                    rework_target="building",
                    max_reworks=2,
                ),
            ),
        ],
    )

    call_count = 0
    executed_prompts = []

    class StubbornRejectProvider:
        async def run(self, prompt, session_id=None, **kwargs):
            nonlocal call_count
            call_count += 1
            executed_prompts.append((session_id, prompt))
            if session_id and "review" in session_id:
                return f"[DECISION: REWORK]\nDefects persist on attempt {call_count}."
            return f"Built code iteration {call_count}"

    from maulness.core import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "get_provider_for_profile", lambda p: StubbornRejectProvider())

    orchestrator = PipelineOrchestrator(storage=storage, pipeline_manager=pipeline_manager)
    task = await orchestrator.run_pipeline(
        repo_name="stubborn_repo",
        title="Stubborn Task",
        prompt="Do something impossible",
        workspace_path=tmp_path / "ws2",
        pipeline_def=cyclic_pipeline,
        auto_proceed=True,
    )

    # Must halt with FAILED when max_reworks ceiling is hit
    assert task.status == TaskStatus.FAILED

    # 6. review rework 2 -> hits cap (2 >= 2) -> halts
    assert len(executed_prompts) == 6


@pytest.mark.asyncio
async def test_pipeline_rework_with_rollback(tmp_path: Path, monkeypatch):
    from maulness.core.pipeline import PipelineOrchestrator
    from maulness.core.worktree import WorktreeManager
    from maulness.core.models import TaskStatus
    from maulness.storage.db import StorageManager

    repo_path = tmp_path / "rollback_repo"
    wm = WorktreeManager()
    # Setup git repo
    repo_path.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(["git", "init", "-b", "main"], cwd=str(repo_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@maulness.local"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "config", "user.name", "Maulness Tester"], cwd=str(repo_path), check=True)
    (repo_path / "README.md").write_text("# Initial\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=str(repo_path), check=True)

    storage = StorageManager(db_path=tmp_path / "test.db")
    pipeline_manager = PipelineManager(pipelines_dir=tmp_path / "pipes")

    rollback_pipeline = PipelineDefinition(
        name="rollback_pipe",
        description="Pipeline testing rollback on rework",
        stages=[
            PipelineStage(
                name="building",
                profile="builder",
                status="building",
                checkpoint_before_stage=True,
                prompt="Build feature: {title}",
            ),
            PipelineStage(
                name="review",
                profile="reviewer",
                status="review",
                prompt="Review changes",
                transitions=StageTransitions(
                    rework_target="building",
                    max_reworks=1,
                    rollback_on_rework=True,
                ),
            ),
        ],
    )

    call_count = 0

    class RollbackMockProvider:
        async def run(self, prompt, session_id=None, workspace_path=None, **kwargs):
            nonlocal call_count
            call_count += 1
            ws = Path(workspace_path)
            if "review" in (session_id or ""):
                return "[DECISION: REWORK]\nFlawed logic, restart needed."
            else:
                # Building stage: write a file
                (ws / f"attempt_{call_count}.txt").write_text("Dirty state")
                return f"Built attempt {call_count}"

    from maulness.core import pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "get_provider_for_profile", lambda p: RollbackMockProvider())

    orchestrator = PipelineOrchestrator(storage=storage, pipeline_manager=pipeline_manager, worktree_manager=wm)
    task = await orchestrator.run_pipeline(
        repo_name="rollback_repo",
        title="Rollback Test",
        prompt="Build and review",
        workspace_path=repo_path,
        pipeline_def=rollback_pipeline,
        auto_proceed=True,
    )

    # Building was called twice: attempt 1 and attempt 2 (after rework)
    # Because rollback_on_rework was True, attempt_1.txt must have been rolled back when rework occurred!
    assert not (repo_path / "attempt_1.txt").exists()
    assert (repo_path / "attempt_3.txt").exists()  # attempt 3 is call_count 3 (building rework)


def test_sync_task_ledger(tmp_path: Path):
    from maulness.core.pipeline import sync_task_ledger

    definition = PipelineDefinition(
        name="standard",
        description="Standard Kanban flow",
        stages=[
            PipelineStage(name="planning", profile="planner", prompt="Plan"),
            PipelineStage(name="building", profile="builder", prompt="Build"),
            PipelineStage(name="review", profile="reviewer", prompt="Review"),
        ],
    )

    ledger = sync_task_ledger(
        workspace=tmp_path,
        task_id="task_12345",
        title="Add caching layer",
        definition=definition,
        current_stage_idx=1,
        rework_counts={"building": 1},
        context={"reviewer_feedback": "Check cache invalidation"},
    )

    assert (tmp_path / ".maulness" / "task.md").exists()
    assert "# Task Ledger: Add caching layer" in ledger
    assert "- [x] **Planning**" in ledger
    assert "- [/] **Building** (`builder`) - Active (Rework 1)" in ledger
    assert "- [ ] **Review**" in ledger
    assert "Check cache invalidation" in ledger

