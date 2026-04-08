"""
inference.py — Indian Traffic Signal OpenEnv
Uses HF Inference Router (OpenAI-compatible) to drive the agent.
"""
import asyncio
import os
import json

from openai import OpenAI
from indian_traffic_env import TrafficEnv, TrafficAction

# ── Credentials & config ───────────────────────────────────────────────
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL     = os.getenv("SPACE_URL",  "https://YOUR_USERNAME-indian-traffic-env.hf.space")
MAX_STEPS     = 20
BENCHMARK     = "indian-traffic-env"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    ("single_junction_basic",  20),
    ("priority_routing",       30),
    ("adaptive_congestion",    40),
]

def get_llm_action(obs_description: str, valid_phases: list) -> dict:
    """Ask the LLM what signal phase to set."""
    prompt = f"""You are a traffic signal controller AI.

Current junction state:
{obs_description}

Valid phases you can choose: {valid_phases}

Respond with ONLY a JSON object like:
{{"signal_phase": "NS", "duration": 35}}

Choose the phase that will clear the most urgent queue.
Emergency vehicles MUST be cleared first."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


async def run_task(task_name: str, max_steps: int):
    rewards = []
    error_str = "null"

    async with TrafficEnv(base_url=SPACE_URL) as env:
        result = await env.reset(task=task_name, seed=42)
        obs = result.observation

        # MANDATORY stdout format
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

        for step in range(1, max_steps + 1):
            try:
                action_dict = get_llm_action(
                    obs.description, obs.valid_phases
                )
                action = TrafficAction(**action_dict)
                action_str = f"{action.signal_phase}/{action.duration}s"
            except Exception as e:
                # Fallback action if LLM returns garbage
                action = TrafficAction(signal_phase=obs.valid_phases[0], duration=30)
                action_str = f"{action.signal_phase}/30s"
                error_str = str(e)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward
            done = result.done
            rewards.append(reward)

            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}"
            )
            error_str = "null"

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= 0.6   # your threshold
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step} "
            f"score={score:.2f} rewards={rewards_str}"
        )
        return score


async def main():
    for task_name, max_steps in TASKS:
        await run_task(task_name, max_steps)

if __name__ == "__main__":
    asyncio.run(main())