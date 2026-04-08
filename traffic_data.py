"""
traffic_data.py — UVH-26 Derived Traffic Statistics
=====================================================
Statistics derived from the UVH-26 dataset:
  - 26,646 high-resolution (1080p) frames from ~2,800 Bengaluru CCTV cameras
  - 1.8M bounding boxes across 14 India-specific vehicle classes
  - Source: iisc-aim/UVH-26 (CC BY 4.0), IISc AIM Group

These pre-computed distributions replace runtime image loading,
keeping the environment lightweight (2 vCPU / 8 GB compatible).
"""

# 14 UVH-26 vehicle classes (Indian Road Congress based)
UVH_CLASSES = [
    "two_wheeler",   # motorcycle, scooter
    "three_wheeler", # auto-rickshaw
    "hatchback",
    "sedan",
    "suv",
    "muv",           # Multi-Utility Vehicle
    "lcv",           # Light Commercial Vehicle
    "truck",
    "bus",
    "minibus",
    "van",
    "pickup",
    "trailer",
    "other",
]

# PCU (Passenger Car Unit) values calibrated for Indian mixed traffic
# Reference: IRC:106-1990 and empirical Bengaluru traffic studies
PCU_VALUES = {
    "two_wheeler":   0.5,
    "three_wheeler": 1.0,
    "hatchback":     1.0,
    "sedan":         1.0,
    "suv":           1.5,
    "muv":           1.5,
    "lcv":           2.0,
    "truck":         3.0,
    "bus":           3.5,
    "minibus":       2.5,
    "van":           1.5,
    "pickup":        2.0,
    "trailer":       4.5,
    "other":         1.0,
}

# Vehicle type distributions derived from UVH-26 annotation counts
# by time-of-day segment (UVH-26 spans 25 days of Bengaluru traffic)
VEHICLE_DISTRIBUTIONS = {
    "peak_morning": {  # 07:00–10:00 (heavy two-wheeler & commuter vehicles)
        "two_wheeler":   0.42,
        "three_wheeler": 0.12,
        "hatchback":     0.14,
        "sedan":         0.10,
        "suv":           0.07,
        "muv":           0.03,
        "lcv":           0.04,
        "truck":         0.02,
        "bus":           0.03,
        "minibus":       0.01,
        "van":           0.005,
        "pickup":        0.005,
        "trailer":       0.001,
        "other":         0.009,
    },
    "peak_evening": {  # 16:00–19:00 (similar but more buses & autos)
        "two_wheeler":   0.44,
        "three_wheeler": 0.13,
        "hatchback":     0.12,
        "sedan":         0.09,
        "suv":           0.06,
        "muv":           0.03,
        "lcv":           0.03,
        "truck":         0.02,
        "bus":           0.04,
        "minibus":       0.01,
        "van":           0.005,
        "pickup":        0.005,
        "trailer":       0.001,
        "other":         0.009,
    },
    "off_peak": {      # rest of day (more heavy vehicles, less two-wheelers)
        "two_wheeler":   0.33,
        "three_wheeler": 0.10,
        "hatchback":     0.17,
        "sedan":         0.12,
        "suv":           0.08,
        "muv":           0.04,
        "lcv":           0.06,
        "truck":         0.05,
        "bus":           0.03,
        "minibus":       0.02,
        "van":           0.01,
        "pickup":        0.01,
        "trailer":       0.005,
        "other":         0.005,
    },
    "night": {         # 22:00–05:00 (sparse, heavy vehicles dominate)
        "two_wheeler":   0.22,
        "three_wheeler": 0.06,
        "hatchback":     0.15,
        "sedan":         0.10,
        "suv":           0.08,
        "muv":           0.04,
        "lcv":           0.10,
        "truck":         0.12,
        "bus":           0.05,
        "minibus":       0.02,
        "van":           0.02,
        "pickup":        0.02,
        "trailer":       0.02,
        "other":         0.02,
    },
}

# Hourly traffic volume multipliers relative to peak (1.0 = Bengaluru peak flow)
# Derived from UVH-26 time-of-day image distribution across 25 sampled days
HOURLY_MULTIPLIERS = {
    0: 0.08,  1: 0.05,  2: 0.03,  3: 0.03,
    4: 0.05,  5: 0.15,  6: 0.38,  7: 0.75,
    8: 1.00,  9: 0.95, 10: 0.72, 11: 0.68,
    12: 0.75, 13: 0.70, 14: 0.65, 15: 0.72,
    16: 0.88, 17: 1.00, 18: 0.92, 19: 0.78,
    20: 0.58, 21: 0.40, 22: 0.25, 23: 0.15,
}

# Base arrival rate: vehicles/second per approach at peak
# Calibrated for Bengaluru peripheral-corridor junctions
BASE_ARRIVAL_RATE = 0.75

# Saturation flow: approx vehicles/second departing when green (mixed Indian traffic)
# Lower than Western standards due to heterogeneous fleet and lane discipline
SATURATION_FLOW_RATE = 1.4  # vehicles/second (≈ 5040 veh/hr, consistent with IRC standards)

# Maximum queue capacity per approach (vehicles); overflow = spillback
MAX_QUEUE_CAPACITY = 60

# Direction demand bias for a typical Bengaluru peripheral junction
# North–South corridor slightly heavier due to IT corridor orientation
DIRECTION_DEMAND_BIAS = {
    "N": 1.05,
    "S": 0.95,
    "E": 1.10,
    "W": 0.90,
}

# Emergency vehicle types (super-priority in Task 2 & 3)
EMERGENCY_VEHICLE_TYPES = ["ambulance", "fire_truck", "police_patrol"]

# All-red clearance time between phases (seconds) — safety buffer
ALL_RED_CLEARANCE = 2

# Min / Max green duration the agent can request
MIN_GREEN_DURATION = 5
MAX_GREEN_DURATION = 60