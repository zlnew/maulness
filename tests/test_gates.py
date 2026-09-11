import asyncio
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch

from maulness.core.kernel.gates import (
    DeterministicGateRunner,
    GateFailureItem,
    GateResult,
    SemanticErrorNormalizer,
)
from maulness.core.kernel.models import KernelEventType
from maulness.core.models import TaskStatus
from maulness.core.pipeline import PipelineOrchestrator
from maulness.core.pipelines import (
    PipelineDefinition,
    PipelineManager,
    PipelineStage,
    StageTransitions,
    VerificationGate,
)
from maulness.storage.db import StorageManager


def test_semantic_error_normalizer_pytest_failure():
    sample_pytest_output = """
============================= test session starts ==============================
collected 2 items

tests/test_sample.py .F                                                  [100%]

=================================== FAILURES ===================================
__________________________________ test_fail ___________________________________

    def test_fail():
>       assert 1 == 2
E       AssertionError: assert 1 == 2

tests/test_sample.py:6: AssertionError
=========================== short test summary info ============================
FAILED tests/test_sample.py::test_fail - AssertionError: assert 1 == 2
========================= 1 failed, 1 passed in 0.12s ==========================
"""
    result = SemanticErrorNormalizer.normalize(
        command="pytest tests/",
        exit_code=1,
        stdout=sample_pytest_output,
        stderr="",
    )

    assert result.passed is False
    assert result.exit_code == 1
    assert "1 failed, 1 passed" in result.summary
    assert len(result.failures) == 1
    assert result.failures[0].identifier == "tests/test_sample.py::test_fail"
    assert "AssertionError: assert 1 == 2" in result.failures[0].message

    feedback = result.to_feedback_prompt()
    assert "Deterministic Verification Gate FAILED" in feedback
    assert "tests/test_sample.py::test_fail" in feedback
    assert "AssertionError: assert 1 == 2" in feedback


def test_semantic_error_normalizer_pytest_success():
    sample_output = """
============================= test session starts ==============================
collected 10 items

tests/test_sample.py ..........                                          [100%]

============================== 10 passed in 0.45s ==============================
"""
    result = SemanticErrorNormalizer.normalize(
        command="pytest tests/",
        exit_code=0,
        stdout=sample_output,
        stderr="",
    )

    assert result.passed is True
    assert result.exit_code == 0
    assert "10 passed" in result.summary
    assert len(result.failures) == 0


def test_semantic_error_normalizer_cargo_failure():
    cargo_output = """
running 2 tests
test tests::test_add ... ok
test tests::test_sub ... FAILED

failures:

---- tests::test_sub stdout ----
thread 'tests::test_sub' panicked at 'assertion failed: `(left == right)`
  left: `2`,
 right: `3`', src/lib.rs:14:9

failures:
    tests::test_sub

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
"""
    result = SemanticErrorNormalizer.normalize(
        command="cargo test",
        exit_code=101,
        stdout=cargo_output,
        stderr="",
    )

    assert result.passed is False
    assert "1 failed" in result.summary
    assert len(result.failures) == 1
    assert result.failures[0].identifier == "tests::test_sub"
    assert "assertion failed" in result.failures[0].message


def test_semantic_error_normalizer_go_failure():
    go_output = """
=== RUN   TestAdd
--- PASS: TestAdd (0.00s)
=== RUN   TestMultiply
--- FAIL: TestMultiply (0.01s)
    math_test.go:23: Expected 10, got 12
FAIL
FAIL	example.com/math	0.015s
FAIL
"""
    result = SemanticErrorNormalizer.normalize(
        command="go test ./...",
        exit_code=1,
        stdout=go_output,
        stderr="",
    )

    assert result.passed is False
    assert len(result.failures) == 1
    assert result.failures[0].identifier == "TestMultiply"
    assert "Failed in 0.01s" in result.failures[0].message


def test_semantic_error_normalizer_linter_failure():
    mypy_output = """
src/app.py:14:1: error: Missing return statement  [return]
src/core/utils.py:42:15: error: Incompatible types in assignment (expression has type "int", variable has type "str")  [assignment]
Found 2 errors in 2 files (checked 20 source files)
"""
    result = SemanticErrorNormalizer.normalize(
        command="mypy src/",
        exit_code=1,
        stdout=mypy_output,
        stderr="",
    )

    assert result.passed is False
    assert len(result.failures) == 2
    assert result.failures[0].identifier == "src/app.py:14"
    assert "Missing return statement" in result.failures[0].message
    assert result.failures[1].identifier == "src/core/utils.py:42"


def test_semantic_error_normalizer_generic_failure():
    output = "fatal: pathspec 'nonexistent.txt' did not match any files"
    result = SemanticErrorNormalizer.normalize(
        command="git checkout nonexistent.txt",
        exit_code=1,
        stdout="",
        stderr=output,
    )

    assert result.passed is False
    assert len(result.failures) == 1
    assert "fatal: pathspec" in result.failures[0].message


@pytest.mark.asyncio
async def test_deterministic_gate_runner_execution(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    runner = DeterministicGateRunner(storage=storage)

    # 1. Successful command
    pass_res = await runner.run_gate(
        task_id="task_gate_1",
        stage="building",
        step_index=1,
        command="python3 -c \"print('Verification passed')\"",
        workspace_path=tmp_path,
        sandbox_mode="none",
    )

    assert pass_res.passed is True
    assert pass_res.exit_code == 0

    events = await storage.get_agent_events("task_gate_1", stage="building")
    assert len(events) == 1
    assert events[0]["event_type"] == KernelEventType.GATE_EVAL.value
    assert events[0]["payload"]["passed"] is True

    # 2. Failing command
    fail_res = await runner.run_gate(
        task_id="task_gate_1",
        stage="building",
        step_index=2,
        command="python3 -c \"import sys; sys.stderr.write('SyntaxError: invalid syntax\\n'); sys.exit(1)\"",
        workspace_path=tmp_path,
        sandbox_mode="none",
    )

    assert fail_res.passed is False
    assert fail_res.exit_code == 1

    events_all = await storage.get_agent_events("task_gate_1", stage="building")
    assert len(events_all) == 2
    assert events_all[1]["payload"]["passed"] is False


@pytest.mark.asyncio
async def test_deterministic_gate_runner_timeout(tmp_path: Path):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    runner = DeterministicGateRunner(storage=storage)

    timeout_res = await runner.run_gate(
        task_id="task_timeout",
        stage="building",
        step_index=1,
        command="python3 -c \"import time; time.sleep(5)\"",
        workspace_path=tmp_path,
        timeout_seconds=1,
        sandbox_mode="none",
    )

    assert timeout_res.passed is False
    assert timeout_res.exit_code == -1
    assert "timed out" in timeout_res.summary.lower()


@pytest.mark.asyncio
async def test_orchestrator_deterministic_gate_auto_rework(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test.db"
    storage = StorageManager(db_path=db_path)
    await storage.initialize()

    pipeline_manager = PipelineManager(pipelines_dir=tmp_path)

    pipeline_def = PipelineDefinition(
        name="gate_pipeline",
        stages=[
            PipelineStage(
                name="builder_stage",
                profile="builder",
                prompt="Build task: {prompt}",
                verification_gate=VerificationGate(
                    command="python3 test_script.py",
                    auto_rework_on_fail=True,
                    sandbox_mode="none",
                ),
                transitions=StageTransitions(
                    rework_target="builder_stage",
                    max_reworks=2,
                ),
            ),
            PipelineStage(
                name="reviewer_stage",
                profile="reviewer",
                prompt="Review task: {prompt}",
            ),
        ],
    )

    # First run fails test_script.py, second run fixes it
    run_count = 0

    class MockProvider:
        async def run(self, prompt, **kwargs):
            nonlocal run_count
            run_count += 1
            script_file = tmp_path / "test_script.py"
            if run_count == 1:
                # Write failing script
                script_file.write_text("import sys; sys.stderr.write('AssertionError: test failed\\n'); sys.exit(1)")
            else:
                # Write passing script
                script_file.write_text("print('All tests passed')")
            return f"builder output cycle {run_count}"

    monkeypatch.setattr(
        "maulness.core.pipeline.get_provider_for_profile",
        lambda profile: MockProvider(),
    )

    orchestrator = PipelineOrchestrator(
        storage=storage,
        pipeline_manager=pipeline_manager,
    )

    task = await orchestrator.run_pipeline(
        repo_name="test-repo",
        title="Test Verification Gate Auto-Rework",
        prompt="Build with verification gate",
        workspace_path=tmp_path,
        pipeline_def=pipeline_def,
        auto_proceed=True,
        verbose=False,
    )

    assert task.status == TaskStatus.DONE
    # Builder should have run twice (cycle 1 failed gate -> rework -> cycle 2 passed gate)
    # Then reviewer ran once = 3 total runs
    assert run_count == 3
