import os
import shutil
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field

from maulness.config import config

USER_PIPELINES_DIR = config.config_dir / "pipelines"
TEMPLATE_PIPELINES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates" / "pipelines"


import re

class PipelineGate(BaseModel):
    type: str = "confirm"  # "confirm" (requires explicit confirmation) or "none"
    prompt: str = "Proceed to next stage?"


class StageTransitions(BaseModel):
    pass_target: Optional[str] = None
    rework_target: Optional[str] = None
    fail_target: Optional[str] = None
    max_reworks: int = 2
    rollback_on_rework: bool = False


class PipelineStage(BaseModel):
    name: str
    profile: str
    status: str = "building"
    output_key: Optional[str] = None
    requires_diff: bool = False
    requires_approval: bool = False
    use_worktree: bool = False
    checkpoint_before_stage: bool = False
    prompt: str
    gate: Optional[PipelineGate] = None
    transitions: Optional[StageTransitions] = None


def check_is_rework_verdict(text: str) -> bool:
    """Check if output text requests rework."""
    if not text:
        return False
    clean = re.sub(r"[\*_`#]", "", text)
    if re.search(r"\[(?:DECISION|VERDICT)\s*:\s*REWORK\]", clean, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:DECISION|VERDICT|AUDIT SCORECARD|SCORECARD)\s*:\s*REWORK\b",
        clean,
        re.IGNORECASE,
    ):
        return True
    return False


def check_is_pass_verdict(text: str) -> bool:
    """Check if output text indicates a passing verdict."""
    if not text:
        return False
    clean = re.sub(r"[\*_`#]", "", text)
    if re.search(r"\[(?:DECISION|VERDICT)\s*:\s*PASS\]", clean, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:DECISION|VERDICT|AUDIT SCORECARD|SCORECARD)\s*:\s*PASS\b",
        clean,
        re.IGNORECASE,
    ):
        return True
    return False


def check_is_fail_verdict(text: str) -> bool:
    """Check if output text indicates an explicit failure verdict."""
    if not text:
        return False
    clean = re.sub(r"[\*_`#]", "", text)
    if re.search(r"\[(?:DECISION|VERDICT)\s*:\s*FAIL\]", clean, re.IGNORECASE):
        return True
    if re.search(
        r"\b(?:DECISION|VERDICT|AUDIT SCORECARD|SCORECARD)\s*:\s*FAIL\b",
        clean,
        re.IGNORECASE,
    ):
        return True
    return False


class PipelineDefinition(BaseModel):
    name: str
    description: str = ""
    stages: list[PipelineStage] = Field(default_factory=list)


class SafeFormatDict(dict):
    """Fallback dictionary that retains unformatted placeholders without crashing."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


class PipelineManager:
    """Manages discovery, loading, and templating for declarative pipelines."""

    def __init__(self, pipelines_dir: Optional[Path] = None):
        self.pipelines_dir = pipelines_dir or USER_PIPELINES_DIR
        self._ensure_user_pipelines()

    def _ensure_user_pipelines(self):
        """Seed user pipelines directory with built-in templates if not yet initialized."""
        try:
            if not self.pipelines_dir.exists():
                self.pipelines_dir.mkdir(parents=True, exist_ok=True)

            if TEMPLATE_PIPELINES_DIR.exists():
                for tpl in TEMPLATE_PIPELINES_DIR.glob("*.yaml"):
                    dest = self.pipelines_dir / tpl.name
                    if not dest.exists():
                        shutil.copy(tpl, dest)
        except Exception:
            pass

    def list_pipelines(self) -> list[PipelineDefinition]:
        """List all discovered pipelines (user definitions override templates)."""
        pipelines: dict[str, PipelineDefinition] = {}

        # 1. Load built-in templates first
        if TEMPLATE_PIPELINES_DIR.exists():
            for file in sorted(TEMPLATE_PIPELINES_DIR.glob("*.yaml")):
                p = self._load_file(file)
                if p:
                    pipelines[p.name] = p

        # 2. Overlay user pipelines in ~/.config/maulness/pipelines
        if self.pipelines_dir.exists():
            for file in sorted(self.pipelines_dir.glob("*.yaml")):
                p = self._load_file(file)
                if p:
                    pipelines[p.name] = p

        return list(pipelines.values())

    def get_pipeline(self, name: str) -> PipelineDefinition:
        """Get a pipeline by name. Falls back to 'standard' or default pipeline if not found."""
        pipelines = {p.name: p for p in self.list_pipelines()}
        if name in pipelines:
            return pipelines[name]

        if "standard" in pipelines:
            return pipelines["standard"]

        # Minimal fallback pipeline if templates are missing
        return PipelineDefinition(
            name=name,
            description="Fallback single builder pipeline",
            stages=[
                PipelineStage(
                    name="building",
                    profile="builder",
                    status="building",
                    prompt="{prompt}",
                    requires_approval=True,
                )
            ],
        )

    def render_stage_prompt(self, stage: PipelineStage, context: dict[str, Any]) -> str:
        """Interpolate variables into stage prompt template safely."""
        return stage.prompt.format_map(SafeFormatDict(context))

    def _load_file(self, path: Path) -> Optional[PipelineDefinition]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return PipelineDefinition(**data)
        except Exception:
            pass
        return None
