from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class RootCause(str, Enum):
    DATABASE_OVERLOAD  = "database_overload"
    MEMORY_LEAK        = "memory_leak"
    NETWORK_PARTITION  = "network_partition"
    DEPENDENCY_TIMEOUT = "dependency_timeout"
    MISCONFIGURATION   = "misconfiguration"
    TRAFFIC_SPIKE      = "traffic_spike"
    CERT_EXPIRY        = "cert_expiry"
    UNKNOWN            = "unknown"


class Severity(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Team(str, Enum):
    BACKEND        = "backend"
    FRONTEND       = "frontend"
    INFRASTRUCTURE = "infrastructure"
    DATABASE       = "database"
    SECURITY       = "security"
    PLATFORM       = "platform"
    NETWORKING     = "networking"


class IncidentObservation(BaseModel):
    incident_id: str = Field(..., description="Unique incident identifier")
    timestamp: str = Field(..., description="ISO-8601 datetime")
    service_name: str = Field(..., description="Primary service reporting anomalies")
    error_rate: float = Field(..., ge=0.0, le=1.0)
    p99_latency_ms: int = Field(..., ge=0)
    log_snippet: str = Field(..., description="Recent relevant log lines")
    affected_endpoints: List[str] = Field(default_factory=list)
    step_count: int = Field(0, ge=0)


class DiagnoseAction(BaseModel):
    kind: Literal["diagnose"] = "diagnose"
    service: str
    root_cause: RootCause


class SetSeverityAction(BaseModel):
    kind: Literal["set_severity"] = "set_severity"
    level: Severity


class EscalateAction(BaseModel):
    kind: Literal["escalate"] = "escalate"
    team: Team


class ResolveAction(BaseModel):
    kind: Literal["resolve"] = "resolve"
    message: str = Field(..., min_length=20)


IncidentAction = Union[
    DiagnoseAction,
    SetSeverityAction,
    EscalateAction,
    ResolveAction,
]


class RewardBreakdown(BaseModel):
    correct_service: float = Field(0.0, ge=0.0, le=0.35)
    correct_severity: float = Field(0.0, ge=0.0, le=0.25)
    correct_escalation: float = Field(0.0, ge=0.0, le=0.25)
    resolution_quality: float = Field(0.0, ge=0.0, le=0.15)
    step_penalty: float = Field(0.0, le=0.0)
    repeat_penalty: float = Field(0.0, le=0.0)


class IncidentReward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0)
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


class StepResult(BaseModel):
    observation: IncidentObservation
    reward: Optional[float] = None
    reward_detail: Optional[IncidentReward] = None
    done: bool = False
    info: dict = Field(default_factory=dict)


class EpisodeState(BaseModel):
    task_id: str
    scenario_id: str
    step_count: int
    done: bool
    ground_truth: dict = Field(description="Hidden ground truth")
    actions_taken: List[dict] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    last_action_kind: Optional[str] = None
