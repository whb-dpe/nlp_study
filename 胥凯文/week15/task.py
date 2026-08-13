from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import uuid
import time


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    description: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self, agent_id: str):
        self.status = TaskStatus.RUNNING
        self.assigned_agent = agent_id
        self.started_at = time.time()

    def complete(self, result: str):
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str):
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "assigned_agent": self.assigned_agent,
            "duration": self.duration,
            "metadata": self.metadata,
        }