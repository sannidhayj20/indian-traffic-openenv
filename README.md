---
title: Indian Traffic Env
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - india
  - real-world
---

# Indian Traffic Signal — OpenEnv Environment

**Bengaluru junction signal control** grounded in real Indian traffic statistics from the UVH-26 dataset (IISc AIM Group).

---

## Environment Description & Motivation

India's urban traffic is among the most heterogeneous in the world — two-wheelers, auto-rickshaws, cars, SUVs, buses, and trucks all compete for the same road space. Bengaluru consistently ranks among the world's most congested cities (TomTom index). This environment simulates a signalised 4-arm urban junction where an LLM agent controls signal phases to maximise vehicle throughput while handling emergencies, pedestrian phases, and traffic incidents.

Vehicle arrival rates and type distributions are derived from **UVH-26** (IISc AIM Group — 26,646 annotated frames from ~2,800 Bengaluru CCTV cameras, 1.8M bounding boxes, 14 India-specific vehicle classes, [arXiv:2511.02563](https://arxiv.org/abs/2511.02563), CC-BY 4.0).

**Why this works well for LLM agents:**
- Observations are rich natural language (queue lengths, PCU counts, emergency flags, time of day)
- Action space is small but strategic (which phase, how long)
- Partial rewards at every step — not just sparse end-of-episode signal
- Clean difficulty scaling across 3 tasks

---

## Tasks

### Task 1 — `single_junction_basic` (Easy)

| Property | Value |
|---|---|
| Valid phases | `NS`, `EW` |
| Max steps | 20 |
| Traffic period | Morning peak (08:30) |
| Emergencies | None |
| Score | `vehicles_cleared / vehicles_arrived` |

Choose between North-South and East-West green phases to maximise throughput. A random agent scores ~0.40; a queue-aware agent scores ~0.70+.

---

### Task 2 — `priority_routing` (Medium)

| Property | Value |
|---|---|
| Valid phases | `N`, `S`, `E`, `W`, `PED` |
| Max steps | 30 |
| Traffic period | Morning peak, standard rates |
| Emergencies | ~12% probability per step |
| Score | `0.5 × throughput + 0.3 × emergency_clearance + 0.2 × overflow_avoidance` |

4-arm per-direction control with emergency vehicles (ambulance/fire/police). Agent must clear emergency approaches within 2 signal cycles or incur escalating penalties.

---

### Task 3 — `adaptive_congestion` (Hard)

| Property | Value |
|---|---|
| Valid phases | `N`, `S`, `E`, `W`, `PED` |
| Max steps | 40 |
| Traffic period | Evening peak, 1.5× base arrival rate |
| Incident | At step 5: surge on one random approach |
| Emergencies | ~8% probability per step |
| Pedestrian | Mandatory PED phase every 5 steps |
| Score | `0.35 × throughput + 0.25 × emergency_score + 0.25 × incident_response + 0.15 × pedestrian_compliance` |

Maintain throughput through a traffic surge, respond to the incident approach, issue periodic pedestrian phases, and handle emergency vehicles simultaneously.

---

## Action Space

```json
{
  "signal_phase": "NS",
  "duration": 30
}
```

| Field | Type | Description |
|---|---|---|
| `signal_phase` | string | One of the task's `valid_phases` |
| `duration` | int | Green time in seconds — range `[5, 60]` |

---

## Observation Space

```json
{
  "step": 7,
  "time_of_day": "08:30",
  "valid_phases": ["NS", "EW"],
  "approaches": {
    "N": {
      "queue_length": 14,
      "total_pcu": 18.5,
      "wait_time_avg_s": 62.0,
      "has_emergency": false,
      "overflow_risk": false,
      "vehicle_counts": {"two_wheeler": 6, "car": 4, "bus": 1, "truck": 3}
    }
  },
  "vehicles_cleared_total": 47,
  "throughput_ratio": 0.566,
  "emergency_violations": 0,
  "pedestrian_phases_given": 1,
  "incident_active": false,
  "reward": 0.61,
  "done": false,
  "description": "Time: 08:30. Step 7. N: 14 vehicles queued. S: 9 vehicles queued. ..."
}
```

---

## Reward Function

| Signal | Value |
|---|---|
| Vehicles cleared this step | `+min(cleared/20, 0.30)` |
| Overall throughput ratio | `+throughput × 0.50` |
| Emergency vehicle not cleared | `−0.15 per violation` |
| Approach in overflow | `−0.05 per approach` |
| Pedestrian compliance | `+ped_ratio × 0.10` |
| All rewards clipped to | `[0.0, 1.0]` |

---

## Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

| Task | Score |
|---|---|
| `single_junction_basic` | ~0.58 |
| `priority_routing` | ~0.44 |
| `adaptive_congestion` | ~0.32 |

---

## Setup & Usage

### Run locally

```bash
git clone https://github.com/Sannidhay/indian-traffic-openenv
cd indian-traffic-openenv
pip install -e .
uvicorn indian_traffic_env.server.app:app --host 0.0.0.0 --port 7860
```

Test it:
```bash
curl http://localhost:7860/health
```

### Run the baseline inference script

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export SPACE_URL=https://sannidhay-indian-traffic-env.hf.space

python inference.py
```

### Docker

```bash
docker build -t indian-traffic-env .
docker run -p 7860:7860 indian-traffic-env
curl http://localhost:7860/health
```

### Validate

```bash
openenv validate
```

---

## Project Structure

```
indian-traffic-openenv/
├── indian_traffic_env/
│   ├── __init__.py          # Exports TrafficEnv, TrafficAction, TrafficObservation
│   ├── models.py            # Pydantic action/observation/state models
│   ├── client.py            # EnvClient subclass
│   └── server/
│       ├── app.py           # create_fastapi_app() entry point
│       └── environment.py   # TrafficEnvironment simulation logic
├── traffic_sim.py           # Discrete-event simulation core
├── traffic_data.py          # UVH-26 derived constants
├── inference.py             # Baseline LLM agent (OpenAI client)
├── server/
│   └── app.py               # Root-level server entry point
├── models.py                # Root-level re-export
├── client.py                # Root-level re-export
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container for HF Spaces
├── pyproject.toml
└── README.md
```

---

## Dataset Attribution

```
@techreport{sharma2025uvh26,
  title       = {Towards Image Annotations and Accurate Vision Models for Indian Traffic},
  author      = {Akash Sharma and Chinmay Mhatre and Sankalp Gawali and others},
  institution = {Indian Institute of Science},
  year        = {2025},
  doi         = {10.48550/arXiv.2511.02563}
}
```

Dataset: [iisc-aim/UVH-26](https://huggingface.co/datasets/iisc-aim/UVH-26) — CC-BY 4.0
