from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class KernelEventType(str, Enum):
    PROMPT = "prompt"
    THOUGHT = "thought"
    MODEL_TURN = "model_turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GATE_EVAL = "gate_eval"
    CHECKPOINT = "checkpoint"
    STAGE_TRANSITION = "stage_transition"
    FINAL_RESPONSE = "final_response"


@dataclass
class StepResult:
    value: Any
    from_cache: bool
    event_id: int
    idempotency_key: str
    step_index: int
    event_type: str


@dataclass
class AgentEventRecord:
    id: int
    task_id: str
    stage: str
    step_index: int
    event_type: str
    payload: Any
    idempotency_key: Optional[str]
    created_at: str


@dataclass
class ModelTurnOutput:
    content: str = ""
    thought: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    simulated_tool_call: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "thought": self.thought,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "simulated_tool_call": self.simulated_tool_call,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelTurnOutput":
        return cls(
            content=data.get("content", ""),
            thought=data.get("thought"),
            tool_calls=data.get("tool_calls", []),
            finish_reason=data.get("finish_reason"),
            simulated_tool_call=bool(data.get("simulated_tool_call", False)),
        )
