from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models import (
    EpisodeState,
    IncidentAction,
    IncidentObservation,
    IncidentReward,
    RewardBreakdown,
    StepResult,
)

app = FastAPI(
    title="IncidentTriageEnv",
    version="1.0.0",
    description="OpenEnv environment for production API incident triage",
)

SCENARIOS_DIR = BASE_DIR / "tasks"
GRADERS_DIR   = BASE_DIR / "tasks" / "graders"

_episode: Optional[EpisodeState] = None
_scenario_cache: dict = {}


def _get_episode() -> EpisodeState:
    if _episode is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _episode


def _load_scenario(task_id: str) -> dict:
    if task_id in _scenario_cache:
        return _scenario_cache[task_id]

    name = task_id.replace("task_", "")
    scenario_file = SCENARIOS_DIR / f"{name}_scenario.json"

    if not scenario_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Scenario file not found: {scenario_file}. Valid task_ids: task_easy, task_medium, task_hard",
        )

    with open(scenario_file) as f:
        data = json.load(f)

    _scenario_cache[task_id] = data
    return data


def _load_grader(task_id: str):
    name = task_id.replace("task_", "")
    grader_file = GRADERS_DIR / f"{name}_grader.py"

    if not grader_file.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"{name}_grader", grader_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scenario_to_observation(scenario: dict, step_count: int = 0) -> IncidentObservation:
    return IncidentObservation(
        incident_id=scenario.get("incident_id", f"INC-{uuid.uuid4().hex[:8].upper()}"),
        timestamp=scenario.get("timestamp", datetime.now(timezone.utc).isoformat()),
        service_name=scenario["initial_service"],
        error_rate=scenario["initial_error_rate"],
        p99_latency_ms=scenario["initial_p99_latency_ms"],
        log_snippet=scenario["initial_log_snippet"],
        affected_endpoints=scenario.get("affected_endpoints", []),
        step_count=step_count,
    )


def _compute_reward(episode: EpisodeState) -> float:
    grader = _load_grader(episode.task_id)
    if grader is None:
        return 0.0

    base_score = grader.grade_episode(episode.actions_taken)

    threshold = 10
    if episode.step_count > threshold:
        base_score -= 0.10 * (episode.step_count - threshold)

    kinds = [a.get("kind") for a in episode.actions_taken]
    for kind in set(kinds):
        count = kinds.count(kind)
        if count > 1:
            base_score -= 0.05 * (count - 1)

    return max(0.0, min(1.0, base_score))


def _is_done(episode: EpisodeState, max_steps: int) -> bool:
    return episode.last_action_kind == "resolve" or episode.step_count >= max_steps


class ResetRequest(BaseModel):
    task_id: str = "task_easy"


@app.post("/reset", response_model=IncidentObservation)
def reset(request: ResetRequest) -> IncidentObservation:
    global _episode

    scenario = _load_scenario(request.task_id)

    _episode = EpisodeState(
        task_id=request.task_id,
        scenario_id=scenario.get("incident_id", request.task_id),
        step_count=0,
        done=False,
        ground_truth=scenario.get("ground_truth", {}),
        actions_taken=[],
        cumulative_reward=0.0,
        reward_breakdown=RewardBreakdown(),
        last_action_kind=None,
    )

    return _scenario_to_observation(scenario, step_count=0)


@app.post("/step", response_model=StepResult)
def step(action: IncidentAction) -> StepResult:
    episode = _get_episode()

    if episode.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    scenario = _load_scenario(episode.task_id)

    episode.step_count += 1
    episode.last_action_kind = action.kind
    episode.actions_taken.append(action.model_dump())

    total_reward = _compute_reward(episode)
    episode.cumulative_reward = total_reward
    episode.done = _is_done(episode, scenario.get("max_steps", 20))

    obs = _scenario_to_observation(scenario, step_count=episode.step_count)
    reward_obj = IncidentReward(total=total_reward, breakdown=episode.reward_breakdown)

    return StepResult(
        observation=obs,
        reward=total_reward,
        reward_detail=reward_obj,
        done=episode.done,
        info={"step_count": episode.step_count, "actions_taken": len(episode.actions_taken)},
    )


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    return _get_episode()


@app.get("/health")
def health():
    return {"status": "ok"}
