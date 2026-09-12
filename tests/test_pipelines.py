import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from maulness.core.models import TaskStatus
from maulness.core.pipeline import PipelineOrchestrator
from maulness.storage.db import StorageManager
from maulness.core.pipelines import (
    PipelineDefinition,
    PipelineGate,
    PipelineManager,
    PipelineStage,
    SafeFormatDict,
    StageTransitions,
    VerificationGate,
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


@pytest.mark.asyncio
async def test_pipeline_user_pauses_at_gate(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="pause_test",
        stages=[
            PipelineStage(
                name="planning",
                profile="planner",
                prompt="Plan feature",
                gate=PipelineGate(prompt="Approve plan?"),
            ),
        ],
    )

    orchestrator = PipelineOrchestrator(storage=storage)

    # 1. on_gate returns False
    async def mock_on_gate(prompt):
        return False

    task = await orchestrator.run_pipeline(
        repo_name="pause_repo",
        title="Pause Test",
        prompt="Plan",
        workspace_path=tmp_path,
        pipeline_def=definition,
        auto_proceed=False,
        on_gate=mock_on_gate,
    )
    # Should return early
    assert task is not None


@pytest.mark.asyncio
async def test_pipeline_with_git_diff_and_gate_only(tmp_path: Path):
    import subprocess
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    # Init git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "file.txt").write_text("initial")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    # Create uncommitted diff
    (tmp_path / "file.txt").write_text("modified content")

    definition = PipelineDefinition(
        name="diff_test",
        stages=[
            PipelineStage(
                name="diff_inspector",
                profile="reviewer",
                prompt="Check diff: {git_diff}",
                requires_diff=True,
            ),
            PipelineStage(
                name="gate_only_stage",
                profile="reviewer",
                is_gate_only=True,
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Stage passed")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="diff_repo",
            title="Diff Test",
            prompt="Inspect diff",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE
        # Verify git diff was passed into prompt
        called_prompt = mock_provider.run.call_args[1]["prompt"]
        assert "modified content" in called_prompt or "file.txt" in called_prompt


@pytest.mark.asyncio
async def test_pipeline_worktree_isolation(tmp_path: Path):
    import subprocess
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    # Init git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "main.txt").write_text("master branch")
    subprocess.run(["git", "add", "main.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, check=True)

    definition = PipelineDefinition(
        name="wt_pipe",
        stages=[
            PipelineStage(name="build", profile="builder", prompt="Build in worktree"),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Done in worktree")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="wt_repo",
            title="Worktree Test",
            prompt="Build something",
            workspace_path=tmp_path,
            pipeline_def=definition,
            use_worktree=True,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE
        events = await storage.get_agent_events(task.id)
        branch_events = [e for e in events if e["event_type"] == "worktree_branch"]
        assert len(branch_events) >= 1
        assert "branch" in branch_events[0]["payload"]


@pytest.mark.asyncio
async def test_pipeline_verification_gate_success(tmp_path: Path):
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="vg_success",
        stages=[
            PipelineStage(
                name="build",
                profile="builder",
                prompt="Build",
                verification_gate=VerificationGate(command="echo 'ok'"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Code built")

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        return_value=GateResult(passed=True, command="echo 'ok'", exit_code=0, summary="ok")
    )

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Gate Success Test",
            prompt="Build",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_pipeline_verification_gate_rework_and_recovery(tmp_path: Path):
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="vg_rework",
        stages=[
            PipelineStage(
                name="build",
                profile="builder",
                prompt="Build",
                verification_gate=VerificationGate(command="pytest", auto_rework_on_fail=True),
                transitions=StageTransitions(max_reworks=2, rework_target="build"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Code built")

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        side_effect=[
            GateResult(passed=False, command="pytest", exit_code=1, summary="1 failed"),
            GateResult(passed=True, command="pytest", exit_code=0, summary="1 passed"),
        ]
    )

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Gate Rework Test",
            prompt="Build",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE
        assert orchestrator.gate_runner.run_gate.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_verification_gate_max_reworks_exceeded(tmp_path: Path):
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="vg_max_fail",
        stages=[
            PipelineStage(
                name="build",
                profile="builder",
                prompt="Build",
                verification_gate=VerificationGate(command="pytest", auto_rework_on_fail=True),
                transitions=StageTransitions(max_reworks=1, rework_target="build"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Code built")

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        return_value=GateResult(passed=False, command="pytest", exit_code=1, summary="1 failed")
    )

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Gate Max Fail Test",
            prompt="Build",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_pipeline_verification_gate_invalid_rework_target(tmp_path: Path):
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="vg_invalid_rework",
        stages=[
            PipelineStage(
                name="build",
                profile="builder",
                prompt="Build",
                verification_gate=VerificationGate(command="pytest", auto_rework_on_fail=True),
                transitions=StageTransitions(max_reworks=2, rework_target="nonexistent"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Code built")

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        return_value=GateResult(passed=False, command="pytest", exit_code=1, summary="1 failed")
    )

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Invalid Target Test",
            prompt="Build",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_pipeline_transitions_fail_target(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="trans_fail",
        stages=[
            PipelineStage(
                name="review",
                profile="reviewer",
                prompt="Review",
                transitions=StageTransitions(fail_target="cleanup"),
            ),
            PipelineStage(
                name="skipped_step",
                profile="builder",
                prompt="Skip me",
            ),
            PipelineStage(
                name="cleanup",
                profile="builder",
                prompt="Cleanup",
            ),
        ],
    )

    executed_stages = []

    class MockProvider:
        async def run(self, prompt, **kwargs):
            if "Review" in prompt:
                executed_stages.append("review")
                return "[DECISION: FAIL] Code is broken"
            elif "Cleanup" in prompt:
                executed_stages.append("cleanup")
                return "Cleaned up"
            executed_stages.append("other")
            return "ok"

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=MockProvider()):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Fail Target Test",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE
        assert executed_stages == ["review", "cleanup"]


@pytest.mark.asyncio
async def test_pipeline_transitions_invalid_fail_target(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="invalid_fail_target",
        stages=[
            PipelineStage(
                name="review",
                profile="reviewer",
                prompt="Review",
                transitions=StageTransitions(fail_target="missing_target"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="[VERDICT: FAIL] Critical issues")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Invalid Fail Target",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_pipeline_transitions_pass_target(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="trans_pass",
        stages=[
            PipelineStage(
                name="step1",
                profile="planner",
                prompt="Step 1",
                transitions=StageTransitions(pass_target="step3"),
            ),
            PipelineStage(
                name="step2",
                profile="builder",
                prompt="Step 2 (should be skipped)",
            ),
            PipelineStage(
                name="step3",
                profile="builder",
                prompt="Step 3",
            ),
        ],
    )

    executed_stages = []

    class MockProvider:
        async def run(self, prompt, **kwargs):
            if "Step 1" in prompt:
                executed_stages.append("step1")
                return "Passed step 1"
            elif "Step 3" in prompt:
                executed_stages.append("step3")
                return "Passed step 3"
            executed_stages.append("step2")
            return "step2"

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=MockProvider()):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Pass Target Test",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE
        assert executed_stages == ["step1", "step3"]


@pytest.mark.asyncio
async def test_pipeline_stage_exception(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="exc_test",
        stages=[
            PipelineStage(name="failing_stage", profile="builder", prompt="Fail"),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(side_effect=RuntimeError("Provider exploded"))

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        with pytest.raises(RuntimeError, match="Provider exploded"):
            await orchestrator.run_pipeline(
                repo_name="test_repo",
                title="Exception Test",
                prompt="Test",
                workspace_path=tmp_path,
                pipeline_def=definition,
                auto_proceed=True,
            )

        tasks = await storage.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_pipeline_callbacks_and_stdin_gate(tmp_path: Path, monkeypatch):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="callback_test",
        stages=[
            PipelineStage(
                name="gated_stage",
                profile="builder",
                prompt="Build",
                gate=PipelineGate(prompt="Continue?"),
            ),
        ],
    )

    started_stages = []

    async def on_stage_start(stage, idx, total):
        started_stages.append((stage.name, idx, total))

    # Mock stdin input to say 'y'
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Stage done")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Callback Test",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=False,
            on_stage_start=on_stage_start,
        )
        assert task.status == TaskStatus.DONE
        assert len(started_stages) == 1
        assert started_stages[0] == ("gated_stage", 1, 1)


def test_sync_task_ledger_exception(tmp_path: Path):
    from maulness.core.pipeline import sync_task_ledger

    # Make .maulness a file so mkdir fails
    maulness_dir = tmp_path / ".maulness"
    maulness_dir.write_text("not a dir")

    definition = PipelineDefinition(
        name="test_pipe",
        stages=[PipelineStage(name="s1", profile="builder")],
    )

    res = sync_task_ledger(
        workspace=tmp_path,
        task_id="t1",
        title="Ledger Error Test",
        definition=definition,
        current_stage_idx=0,
        rework_counts={},
        context={},
    )
    assert res == ""


@pytest.mark.asyncio
async def test_pipeline_transitions_invalid_rework_target(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="invalid_rework",
        stages=[
            PipelineStage(
                name="review",
                profile="reviewer",
                prompt="Review",
                transitions=StageTransitions(rework_target="nonexistent"),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="[VERDICT: REWORK] Needs fix")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Invalid Rework Test",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_pipeline_on_stage_finish_callback(tmp_path: Path):
    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    definition = PipelineDefinition(
        name="finish_cb",
        stages=[PipelineStage(name="s1", profile="builder", prompt="Do work")],
    )

    finished = []

    async def on_finish(stage, output):
        finished.append((stage.name, output))

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Output result")

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        orchestrator = PipelineOrchestrator(storage=storage)
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Finish Callback Test",
            prompt="Test",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
            on_stage_finish=on_finish,
        )
        assert task.status == TaskStatus.DONE
        assert finished == [("s1", "Output result")]


@pytest.mark.asyncio
async def test_pipeline_verification_gate_with_rollback(tmp_path: Path):
    import subprocess
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_test.db")
    await storage.initialize()

    # Init git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base")
    subprocess.run(["git", "add", "base.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    definition = PipelineDefinition(
        name="rollback_pipe",
        stages=[
            PipelineStage(
                name="build",
                profile="builder",
                prompt="Build code",
                checkpoint_before_stage=True,
                verification_gate=VerificationGate(command="pytest", auto_rework_on_fail=True),
                transitions=StageTransitions(max_reworks=2, rework_target="build", rollback_on_rework=True),
            ),
        ],
    )

    mock_provider = MagicMock()
    mock_provider.run = AsyncMock(return_value="Code built")

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        side_effect=[
            GateResult(passed=False, command="pytest", exit_code=1, summary="1 failed"),
            GateResult(passed=True, command="pytest", exit_code=0, summary="1 passed"),
        ]
    )

    with patch("maulness.core.pipeline.get_provider_for_profile", return_value=mock_provider):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Rollback Test",
            prompt="Build",
            workspace_path=tmp_path,
            pipeline_def=definition,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE




@pytest.mark.asyncio
async def test_pipeline_coverage_branches(tmp_path: Path):
    import subprocess
    from maulness.core.kernel.gates import GateResult

    storage = StorageManager(db_path=tmp_path / "pipe_cov.db")
    await storage.initialize()

    # Init git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "base.txt").write_text("base")
    subprocess.run(["git", "add", "base.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    # 1. Stage with invalid status enum value (lines 233-234), is_gate_only=True (line 380), stage.use_worktree=True with use_worktree=None (line 492)
    definition = PipelineDefinition(
        name="cov_pipe",
        stages=[
            PipelineStage(
                name="gate_stage",
                profile="builder",
                status="invalid_nonexistent_status",
                prompt="Verify",
                is_gate_only=True,
                use_worktree=True,
                verification_gate=VerificationGate(command="echo pass"),
            ),
        ],
    )

    orchestrator = PipelineOrchestrator(storage=storage)
    orchestrator.gate_runner.run_gate = AsyncMock(
        return_value=GateResult(passed=True, command="echo pass", exit_code=0, summary="All pass"),
    )

    # 1. Exercise subprocess.run exception on branch query (lines 511-512)
    orig_run = subprocess.run
    def custom_run_fail_branch(cmd, *args, **kwargs):
        if "--show-current" in cmd:
            raise RuntimeError("git branch query failed")
        return orig_run(cmd, *args, **kwargs)

    with patch("subprocess.run", side_effect=custom_run_fail_branch):
        await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Branch Fail Test",
            prompt="Run",
            workspace_path=tmp_path,
            pipeline_def=definition,
            use_worktree=None,
            auto_proceed=True,
        )

    # 2. Exercise storage record_agent_event error when branch exists (lines 525-526)
    with patch.object(storage, "record_agent_event", side_effect=RuntimeError("event error")):
        task = await orchestrator.run_pipeline(
            repo_name="test_repo",
            title="Cov Test",
            prompt="Run",
            workspace_path=tmp_path,
            pipeline_def=definition,
            use_worktree=None,
            auto_proceed=True,
        )
        assert task.status == TaskStatus.DONE


def test_pipelines_manager_fallbacks_and_errors(tmp_path: Path):
    # Line 90: check_is_fail_verdict on empty
    assert not check_is_fail_verdict("")
    assert not check_is_fail_verdict(None)

    # Lines 134-135: _ensure_user_pipelines exception handling
    with patch.object(Path, "mkdir", side_effect=PermissionError("no access")):
        mgr = PipelineManager(pipelines_dir=tmp_path / "uncreatable")

    # Line 164: get_pipeline falls back to standard
    mgr_std = PipelineManager(pipelines_dir=tmp_path / "std_mgr")
    std_pipe = mgr_std.get_pipeline("nonexistent_fallback_to_standard")
    assert std_pipe.name == "standard"

    # Lines 163-167: get_pipeline fallback when standard is missing
    mgr_empty = PipelineManager(pipelines_dir=tmp_path / "empty_mgr")
    with patch.object(mgr_empty, "list_pipelines", return_value=[]):
        fb_pipe = mgr_empty.get_pipeline("completely_unknown")
        assert fb_pipe.name == "completely_unknown"
        assert fb_pipe.stages[0].profile == "builder"

    # Lines 190-192: _load_file exception handling
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text(": bad yaml")
    assert mgr_empty._load_file(bad_yaml) is None
