from dataclasses import dataclass
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
