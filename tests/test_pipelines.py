import pytest
from pathlib import Path
from maulness.core.pipelines import (
    PipelineDefinition,
    PipelineManager,
    PipelineStage,
    SafeFormatDict,
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
