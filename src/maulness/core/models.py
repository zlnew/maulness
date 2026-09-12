from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskMode(str, Enum):
    DIRECT = "direct"
    MULTI = "multi"


class TaskStatus(str, Enum):
    PLANNING = "planning"
    BUILDING = "building"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"
    SUSPENDED_AFK = "suspended_afk"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class AgentThoughtEvent:
    delta: str
    session_id: str


@dataclass
class AgentMessageEvent:
    delta: str
    session_id: str


@dataclass
class AgentToolCallEvent:
    call_id: str
    tool_name: str
    args: dict[str, Any]
    session_id: str


@dataclass
class ApprovalRequestEvent:
    request_id: int
    call_id: str
    tool_name: str
    args: dict[str, Any]
    session_id: str


@dataclass
class TaskRecord:
    id: str
    title: str
    repo_name: Optional[str] = None
    workspace_path: Optional[str] = None
    mode: TaskMode = TaskMode.DIRECT
    status: TaskStatus = TaskStatus.PLANNING
    origin_platform: str = "cli"
    origin_channel_id: Optional[str] = None
    origin_thread_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def discord_thread_id(self) -> Optional[int]:
        if self.origin_platform == "discord" and self.origin_thread_id:
            try:
                return int(self.origin_thread_id)
            except ValueError:
                return None
        return None


@dataclass
class SessionRecord:
    id: str
    task_id: Optional[str]
    profile: str
    engine: str
    acp_session_id: Optional[str] = None
    pid: Optional[int] = None
    status: str = "active"
    created_at: Optional[datetime] = None


@dataclass
class ApprovalRecord:
    id: str
    task_id: str
    rpc_request_id: int
    tool_name: str
    tool_args: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    platform: str = "discord"
    platform_message_id: Optional[str] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    @property
    def discord_message_id(self) -> Optional[int]:
        if self.platform == "discord" and self.platform_message_id:
            try:
                return int(self.platform_message_id)
            except ValueError:
                return None
        return None


# ==============================================================================
# Phase 2: Decoupled Multi-Agent Artifact Contracts & Milestone Sub-Sessions
# ==============================================================================


class Milestone(BaseModel):
    """An atomic, verifiable unit of work within a larger task."""

    id: str
    title: str
    files_to_modify: list[str] = Field(default_factory=list)
    verification_command: str = ""
    acceptance_criteria: str = ""


class MilestonePlan(BaseModel):
    """Structured architectural plan partitioned into verifiable milestones."""

    task_id: str
    summary: str = ""
    milestones: list[Milestone] = Field(default_factory=list)


class ChangeSet(BaseModel):
    """Structured change payload emitted by the builder."""

    task_id: str
    milestone_id: Optional[str] = None
    touched_files: list[str] = Field(default_factory=list)
    git_diff_stat: str = ""
    verification_passed: bool = True
    verification_output: str = ""


class ReviewVerdict(BaseModel):
    """Structured evaluation returned by the reviewer."""

    decision: str  # "PASS" or "REWORK"
    flaws: list[str] = Field(default_factory=list)
    target_rework_files: list[str] = Field(default_factory=list)
    guidance: str = ""


@dataclass
class RepoGotcha:
    """Commit-anchored associative knowledge about environment/repo quirks."""

    id: str
    repo_path: str
    component: str
    symptom: str
    resolution: str
    commit_hash: str
    is_active: bool = True
    created_at: Optional[datetime] = None


@dataclass
class TaskRetrospective:
    """Post-task retrospective record capturing outcome and execution metrics."""

    task_id: str
    repo_path: str
    summary: str
    passed: bool
    total_steps: int
    cost_usd: float
    created_at: Optional[datetime] = None
