from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# ── What the agent CAN do ──────────────────────────────────────────────
class TrafficAction(BaseModel):
    signal_phase: str = Field(
        ...,
        description="Which direction gets green: 'NS' or 'EW' for task1; "
                    "'N','S','E','W','PED' for task2/3"
    )
    duration: int = Field(
        default=30,
        ge=5,
        le=60,
        description="How many seconds to hold this green phase (5–60)"
    )

# ── Sub-model: one approach arm of the junction ───────────────────────
class ApproachState(BaseModel):
    queue_length: int          # vehicles waiting
    total_pcu: float           # passenger-car units (trucks count more)
    wait_time_avg_s: float     # average wait in seconds
    has_emergency: bool        # ambulance / fire engine present
    overflow_risk: bool        # queue backing up past detector



# ── Current episode metadata ───────────────────────────────────────────
class TrafficState(BaseModel):
    episode_id: str
    task: str
    step_count: int
    total_reward: float
    done: bool

class TrafficObservation(BaseModel):
    # ── Add these two fields ──────────────────────────────────────────
    reward: float = 0.0
    done: bool = False
    # ── Your existing fields (keep all of them) ───────────────────────
    step: int
    time_of_day: str
    valid_phases: List[str]
    approaches: Dict[str, ApproachState]
    vehicles_cleared_total: int
    throughput_ratio: float
    emergency_violations: int
    pedestrian_phases_given: int
    incident_active: bool
    description: str