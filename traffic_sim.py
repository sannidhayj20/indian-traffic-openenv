# traffic_sim.py  —  Complete self-contained traffic simulation
import random
from typing import Dict, Tuple

# ── Realistic Indian junction constants (UVH-26 based) ─────────────────
VEHICLE_MIX = {
    "two_wheeler": 0.52,
    "car":         0.30,
    "bus":         0.08,
    "truck":       0.10,
}
# PCU = Passenger Car Unit (how much road space each type takes)
PCU = {"two_wheeler": 0.5, "car": 1.0, "bus": 2.5, "truck": 3.0}

# Arrival rates (vehicles per cycle) by time of day
ARRIVAL_RATES = {
    "morning_peak":  {"N": 18, "S": 14, "E": 20, "W": 16},
    "afternoon":     {"N":  8, "S":  8, "E":  8, "W":  8},
    "evening_peak":  {"N": 16, "S": 20, "E": 14, "W": 18},
    "night":         {"N":  3, "S":  3, "E":  3, "W":  3},
}

PHASE_CLEARS = {
    # How many vehicles can leave per second of green
    "NS":  {"N": 0.8, "S": 0.8, "E": 0.0, "W": 0.0},
    "EW":  {"N": 0.0, "S": 0.0, "E": 0.8, "W": 0.8},
    "N":   {"N": 1.0, "S": 0.0, "E": 0.0, "W": 0.0},
    "S":   {"N": 0.0, "S": 1.0, "E": 0.0, "W": 0.0},
    "E":   {"N": 0.0, "S": 0.0, "E": 1.0, "W": 0.0},
    "W":   {"N": 0.0, "S": 0.0, "E": 0.0, "W": 1.0},
    "PED": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
}


class ArmState:
    def __init__(self):
        self.queue: int = 0
        self.total_pcu: float = 0.0
        self.wait_time_s: float = 0.0
        self.has_emergency: bool = False
        self.overflow_risk: bool = False
        self.vehicle_counts: Dict[str, int] = {k: 0 for k in VEHICLE_MIX}

    def add_arrivals(self, count: int):
        for _ in range(count):
            vtype = random.choices(
                list(VEHICLE_MIX.keys()),
                weights=list(VEHICLE_MIX.values())
            )[0]
            self.vehicle_counts[vtype] += 1
            self.queue += 1
            self.total_pcu += PCU[vtype]
        self.overflow_risk = self.queue > 25

    def clear_vehicles(self, rate: float, duration: int) -> int:
        cleared = min(self.queue, int(rate * duration))
        # Recalculate PCU proportionally
        if self.queue > 0:
            pcu_per_vehicle = self.total_pcu / self.queue
            self.total_pcu = max(0, self.total_pcu - cleared * pcu_per_vehicle)
        self.queue -= cleared
        self.vehicle_counts = {
            k: max(0, int(v * (self.queue / max(1, self.queue + cleared))))
            for k, v in self.vehicle_counts.items()
        }
        self.overflow_risk = self.queue > 25
        return cleared

    def to_dict(self) -> dict:
        return {
            "queue_length":    self.queue,
            "total_pcu":       round(self.total_pcu, 2),
            "wait_time_avg_s": round(self.wait_time_s, 1),
            "has_emergency":   self.has_emergency,
            "overflow_risk":   self.overflow_risk,
            "vehicle_counts":  dict(self.vehicle_counts),
        }


class TrafficSimulation:
    """
    Simulates a single 4-arm Indian urban junction.

    Tasks:
      single_junction_basic   — 2-phase (NS/EW only), 20 steps
      priority_routing        — 4-arm + emergency vehicles, 30 steps
      adaptive_congestion     — peak hour + random incident, 40 steps
    """

    TASK_CONFIG = {
        "single_junction_basic": {
            "valid_phases": ["NS", "EW"],
            "max_steps":    20,
            "time_period":  "morning_peak",
            "emergency_prob": 0.0,
            "incident_prob":  0.0,
        },
        "priority_routing": {
            "valid_phases": ["N", "S", "E", "W", "PED"],
            "max_steps":    30,
            "time_period":  "morning_peak",
            "emergency_prob": 0.12,
            "incident_prob":  0.0,
        },
        "adaptive_congestion": {
            "valid_phases": ["N", "S", "E", "W", "PED"],
            "max_steps":    40,
            "time_period":  "evening_peak",
            "emergency_prob": 0.08,
            "incident_prob":  0.15,
        },
    }

    def __init__(self, task: str = "single_junction_basic", seed: int = 42):
        random.seed(seed)
        self.task = task
        cfg = self.TASK_CONFIG[task]
        self.valid_phases: list     = cfg["valid_phases"]
        self.max_steps: int         = cfg["max_steps"]
        self.time_period: str       = cfg["time_period"]
        self.emergency_prob: float  = cfg["emergency_prob"]
        self.incident_prob: float   = cfg["incident_prob"]

        self.step_count: int        = 0
        self.vehicles_cleared: int  = 0
        self.vehicles_arrived: int  = 0
        self.emergency_violations: int = 0
        self.ped_phases_given: int  = 0
        self.ped_phases_needed: int = 0
        self.incident_arm: str      = None
        self.done: bool             = False

        self.arms: Dict[str, ArmState] = {
            arm: ArmState() for arm in ["N", "S", "E", "W"]
        }

        # Seed initial queues
        rates = ARRIVAL_RATES[self.time_period]
        for arm in self.arms:
            self.arms[arm].add_arrivals(rates[arm] // 2)

    # ── Core simulation step ───────────────────────────────────────────
    def advance(self, phase: str, duration: int) -> Tuple[float, bool]:
        """Run one signal cycle. Returns (reward, done)."""
        if phase not in self.valid_phases:
            return -0.2, False   # penalty for invalid phase

        self.step_count += 1
        rates = ARRIVAL_RATES[self.time_period]

        # 1. Spawn emergency vehicle randomly
        for arm in self.arms:
            self.arms[arm].has_emergency = False
        if random.random() < self.emergency_prob:
            emerg_arm = random.choice(["N", "S", "E", "W"])
            self.arms[emerg_arm].has_emergency = True

        # 2. Spawn incident (blocks one arm)
        if self.incident_prob > 0 and self.step_count == 5:
            self.incident_arm = random.choice(["N", "S", "E", "W"])

        # 3. Arrive new vehicles
        for arm in self.arms:
            count = rates[arm]
            if arm == self.incident_arm:
                count = int(count * 1.5)   # backlog builds behind incident
            arrivals = random.randint(
                max(0, count - 4), count + 4
            )
            self.arms[arm].add_arrivals(arrivals)
            self.vehicles_arrived += arrivals

        # 4. Clear vehicles on green phase
        cleared_this_step = 0
        clear_rates = PHASE_CLEARS.get(phase, {})
        for arm, rate in clear_rates.items():
            if rate > 0:
                n = self.arms[arm].clear_vehicles(rate, duration)
                cleared_this_step += n
                self.vehicles_cleared += n

        # 5. Update wait times
        for arm in self.arms:
            if clear_rates.get(arm, 0) == 0:
                self.arms[arm].wait_time_s += duration

        if phase == "PED":
            self.ped_phases_given += 1

        # 6. Count pedestrian need
        if self.step_count % 5 == 0:
            self.ped_phases_needed += 1

        # 7. Check emergency violations
        for arm in self.arms:
            if self.arms[arm].has_emergency:
                arm_is_green = clear_rates.get(arm, 0) > 0
                if not arm_is_green:
                    self.emergency_violations += 1

        # 8. Compute reward
        reward = self._compute_reward(phase, cleared_this_step, duration)

        # 9. Episode end
        done = self.step_count >= self.max_steps
        self.done = done
        return reward, done

    def _compute_reward(self, phase: str, cleared: int, duration: int) -> float:
        reward = 0.0

        # Throughput reward (main signal)
        if self.vehicles_arrived > 0:
            throughput = self.vehicles_cleared / self.vehicles_arrived
            reward += throughput * 0.5

        # Bonus for clearing vehicles this step
        reward += min(cleared / 20.0, 0.3)

        # Emergency penalty
        reward -= self.emergency_violations * 0.15

        # Overflow penalty
        overflow_arms = sum(
            1 for a in self.arms.values() if a.overflow_risk
        )
        reward -= overflow_arms * 0.05

        # Pedestrian bonus
        if self.ped_phases_needed > 0:
            ped_ratio = self.ped_phases_given / self.ped_phases_needed
            reward += ped_ratio * 0.1

        return round(max(0.0, min(1.0, reward)), 4)

    # ── State snapshot (what the server reads) ────────────────────────
    def get_state(self) -> dict:
        time_labels = {
            "morning_peak": "08:30",
            "afternoon":    "13:00",
            "evening_peak": "18:00",
            "night":        "23:00",
        }
        throughput = (
            self.vehicles_cleared / self.vehicles_arrived
            if self.vehicles_arrived > 0 else 0.0
        )
        return {
            "time":                 time_labels[self.time_period],
            "valid_phases":         self.valid_phases,
            "approaches":           {arm: self.arms[arm].to_dict()
                                     for arm in self.arms},
            "cleared":              self.vehicles_cleared,
            "arrived":              self.vehicles_arrived,
            "throughput":           round(throughput, 4),
            "emergency_violations": self.emergency_violations,
            "ped_phases":           self.ped_phases_given,
            "incident":             self.incident_arm is not None,
            "incident_arm":         self.incident_arm,
            "step":                 self.step_count,
            "done":                 self.done,
        }