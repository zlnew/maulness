from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class TaskMode(str, Enum):
    DIRECT = "direct"
    MULTI = "multi"


class TaskStatus(str, Enum):
    PLANNING = "planning"
    BUILDING = "building"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"


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
    repo_name: str
    workspace_path: str
    mode: TaskMode
    status: TaskStatus
    discord_thread_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


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
    discord_message_id: Optional[int] = None
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

