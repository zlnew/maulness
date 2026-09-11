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

