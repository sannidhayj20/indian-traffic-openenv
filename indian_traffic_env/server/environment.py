import uuid
import random
from typing import Tuple

from openenv.core.env_server import Environment
from indian_traffic_env.models import (
    TrafficAction, TrafficObservation, TrafficState, ApproachState
)

class TrafficEnvironment(Environment):
    """
    Indian urban traffic signal controller.
    The agent controls signal phases at a single junction (UVH-26 stats).
    """

    def __init__(self):
        super().__init__()
        self._state = TrafficState(
            episode_id=str(uuid.uuid4()),
            task="single_junction_basic",
            step_count=0,
            total_reward=0.0,
            done=False
        )
        self._sim = self._build_sim("single_junction_basic", seed=42)  # ← initialize here

    # ── Called at episode start ────────────────────────────────────────
    def reset(self, task: str = "single_junction_basic", seed: int = 42) -> TrafficObservation:
        self._state = TrafficState(
            episode_id=str(uuid.uuid4()),
            task=task,
            step_count=0,
            total_reward=0.0,
            done=False
        )
        self._sim = self._build_sim(task, seed)   # ← now returns a real object
        return self._make_observation()

    # ── Called every agent decision ────────────────────────────────────
    def step(self, action: TrafficAction) -> TrafficObservation:
        if self._sim is None:
            self._sim = self._build_sim(self._state.task or "single_junction_basic", seed=42)
        
        reward, done = self._sim.advance(action.signal_phase, action.duration)
        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.done = done
        return self._make_observation(reward=reward, done=done)

    # ── Polled any time by framework / client ──────────────────────────
    @property
    def state(self) -> TrafficState:
        return self._state

    # ── Helpers ────────────────────────────────────────────────────────
    def _build_sim(self, task, seed):
        from traffic_sim import TrafficSimulation
        return TrafficSimulation(task=task, seed=seed)

    def _make_observation(self, reward: float = 0.0, done: bool = False) -> TrafficObservation:
        raw = self._sim.get_state()
        approaches = {
            arm: ApproachState(**raw["approaches"][arm])
            for arm in ["N", "S", "E", "W"]
        }
        return TrafficObservation(
            reward=reward,          # ← new
            done=done,              # ← new
            step=raw["step"],
            time_of_day=raw["time"],
            valid_phases=raw["valid_phases"],
            approaches=approaches,
            vehicles_cleared_total=raw["cleared"],
            throughput_ratio=raw["throughput"],
            emergency_violations=raw["emergency_violations"],
            pedestrian_phases_given=raw["ped_phases"],
            incident_active=raw["incident"],
            description=self._describe(raw),
        )

    def _describe(self, raw) -> str:
        # Generate a plain-English summary the LLM can read
        lines = [f"Time: {raw['time']}. Step {self._state.step_count}."]
        for arm, data in raw["approaches"].items():
            lines.append(
                f"  {arm}: {data['queue_length']} vehicles queued"
                f"{', EMERGENCY' if data['has_emergency'] else ''}"
            )
        lines.append(f"Throughput so far: {raw['throughput']:.0%}")
        return " ".join(lines)