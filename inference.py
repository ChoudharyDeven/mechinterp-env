"""
inference.py — mechinterp-env baseline inference script.

MANDATORY: This file must be named exactly 'inference.py' and placed at project root.

Runs an LLM agent against all 3 tasks using the OpenAI client.
Produces the mandatory [START]/[STEP]/[END] stdout log format.

Environment variables required:
  HF_TOKEN or API_KEY  — your Hugging Face / API key
  API_BASE_URL         — LLM endpoint (default: HF router)
  MODEL_NAME           — model identifier (default: Qwen2.5-72B-Instruct)
  MECHINTERP_ENV_URL   — deployed HF Space URL

Usage:
  MECHINTERP_ENV_URL=https://your-space.hf.space python inference.py
"""

import asyncio
import json
import os
import re
import sys
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

# ── Environment configuration ─────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("MECHINTERP_ENV_URL", "http://localhost:7860")
BENCHMARK    = "mechinterp-env"

TASK_CONFIGS = [
    {"task_id": "head-identification",  "max_steps": 4},
    {"task_id": "circuit-localization", "max_steps": 12},
    {"task_id": "full-hypothesis",      "max_steps": 20},
]

# ── Mandatory stdout log format ───────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err  = error if error else "null"
    done_str = str(done).lower()
    # Sanitize action string — no spaces, newlines, special chars
    action_clean = re.sub(r'\s+', '_', str(action))[:80]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── System prompt ─────────────────────────────────────────────────────────────
# This is the most important part of inference.py.
# It teaches MI concepts and gives a clear decision framework.

SYSTEM_PROMPT = """You are a mechanistic interpretability (MI) research agent.
Your goal: identify which attention heads in a small transformer model are responsible
for a specific behavior. These heads form the "circuit" for that behavior.

=== BACKGROUND ===
A transformer model has layers. Each layer has attention heads.
An attention head moves information between token positions.
A "circuit" is the minimal set of heads that implements a behavior.
When you ABLATE (remove) a circuit head, the behavior drops dramatically.
When you ablate a noise head, the behavior barely changes.

=== YOUR TOOLS ===

1. ablate_head
   Zeroes out a head's output entirely.
   If behavioral_delta is very negative (e.g. -0.8), that head is CRITICAL.
   If delta is near zero (e.g. -0.03), the head is NOT in the circuit.
   This is your primary tool.

2. patch_activation
   Replaces an activation with one from a different prompt.
   More precise than ablation for identifying which position matters.

3. query_attn_pattern
   Shows which tokens this head attends to.
   Reveals the head's function (e.g. "always attends to previous token").

4. query_logit_lens
   Shows what the model predicts at intermediate layers.
   Useful for understanding where information gets written.

5. submit_hypothesis
   Submit your final answer.
   Format: {"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(L, H)": score}}}
   Example: {"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(0, 2)": 1.0}}}
   Importance score is 0.0-1.0. Heads you are confident about get 1.0.
   ALWAYS submit before running out of steps.

=== STRATEGY ===
Step 1: Ablate heads layer by layer (L0H0, L0H1, L0H2, ... L1H0, L1H1, ...)
Step 2: Note which heads have |behavioral_delta| > 0.25 — these are candidates
Step 3: Once you have candidates, submit your hypothesis
Step 4: If unsure, ablate a few more candidates to compare deltas

=== DECISION RULE ===
behavioral_delta < -0.3  → IMPORTANT head, include in circuit with score 0.8-1.0
behavioral_delta < -0.1  → Possibly relevant, include with score 0.3-0.5
behavioral_delta > -0.1  → Not in circuit, do NOT include

=== RESPONSE FORMAT ===
Respond with EXACTLY ONE JSON object per turn. Nothing else. No explanation.
Valid responses:
  {"action_type": "ablate_head", "layer": 0, "head": 2}
  {"action_type": "ablate_head", "layer": 1, "head": 3}
  {"action_type": "query_attn_pattern", "layer": 0, "head": 1, "prompt_id": 0}
  {"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(0, 2)": 1.0}}}
  {"action_type": "submit_hypothesis", "circuit_mask": {"components": {"(0, 1)": 0.6, "(1, 2)": 1.0}}}
"""

# ── Observation → user prompt ─────────────────────────────────────────────────

def build_user_prompt(obs: dict, task_id: str) -> str:
    lines = []

    lines.append(f"=== TASK: {task_id} ===")
    lines.append(f"Step {obs.get('step_number', 0)} | Steps remaining: {obs.get('steps_remaining', 0)}")
    lines.append("")

    lines.append("TARGET BEHAVIOR:")
    lines.append(obs.get("target_behavior", ""))
    lines.append("")

    # Latest action result
    result = obs.get("action_result", {})
    if result:
        lines.append("LATEST RESULT:")
        # Special handling for initial prerun results (task 1)
        if "prerun_ablation_results" in result:
            lines.append("Pre-run ablation results (all heads already ablated for you):")
            for key, data in result["prerun_ablation_results"].items():
                delta = data.get("behavioral_delta", 0)
                interp = data.get("interpretation", "")
                lines.append(f"  {key}: delta={delta:.4f} ({interp})")
            lines.append("")
            lines.append("READ the results above carefully. Which head has the most negative delta?")
            lines.append("Submit your hypothesis immediately.")
        else:
            # Regular action result
            for k, v in result.items():
                if k not in ("full_pattern",):  # skip large matrices
                    lines.append(f"  {k}: {v}")
        lines.append("")

    # Experiment history — only show high-signal entries
    history = obs.get("experiment_history", [])
    if history:
        important = [
            e for e in history
            if abs(e.get("behavioral_delta", 0)) > 0.08
        ]
        if important:
            lines.append("KEY FINDINGS SO FAR (|delta| > 0.08):")
            for e in important[-8:]:
                l = e.get("layer", "?")
                h = e.get("head", "?")
                d = e.get("behavioral_delta", 0)
                s = e.get("result_summary", "")
                lines.append(f"  Step {e['step']}: L{l}H{h} → delta={d:.4f} | {s}")
            lines.append("")

        # Also show what heads have been tried (to avoid redundancy)
        tried = set()
        for e in history:
            if e.get("layer") is not None and e.get("head") is not None:
                tried.add(f"L{e['layer']}H{e['head']}")
        if tried:
            lines.append(f"Already tested: {', '.join(sorted(tried))}")
            lines.append("")

    # Urgency warning
    remaining = obs.get("steps_remaining", 999)
    if remaining <= 3:
        lines.append("⚠️  WARNING: Only {remaining} steps left! Submit your hypothesis NOW.")
        lines.append("")

        # Auto-build best guess from history
        candidates = sorted(
            [e for e in history if e.get("behavioral_delta", 0) < -0.1],
            key=lambda e: e.get("behavioral_delta", 0),
        )[:5]
        if candidates:
            lines.append("Best candidates based on your experiments:")
            for c in candidates:
                lines.append(f"  L{c['layer']}H{c['head']}: delta={c['behavioral_delta']:.4f}")
            lines.append("")

    lines.append("What is your next action? Respond with JSON only.")
    return "\n".join(lines)

# ── Action parsing ─────────────────────────────────────────────────────────────

def parse_action(response_text: str, obs: dict, history: List[dict]) -> Optional[dict]:
    """
    Parse the LLM's response into an action dict.
    Returns None if parsing fails (caller handles fallback).
    """
    text = response_text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return data
    except Exception:
        pass

    # Try extracting a JSON object from surrounding text
    matches = re.findall(r'\{[^{}]*\}', text, re.DOTALL)
    for m in matches:
        try:
            data = json.loads(m)
            if "action_type" in data:
                return data
        except Exception:
            continue

    # Try extracting nested JSON (for submit_hypothesis with circuit_mask)
    try:
        deep_match = re.search(r'\{.*\}', text, re.DOTALL)
        if deep_match:
            data = json.loads(deep_match.group())
            if "action_type" in data:
                return data
    except Exception:
        pass

    return None


def fallback_action(obs: dict, tried_keys: set) -> dict:
    """
    Produce a sensible fallback action when LLM output can't be parsed.
    Systematically ablates untried heads, or submits if all are tried.
    """
    available = obs.get("available_actions", [])
    n_layers  = 1  # default
    n_heads   = 4

    # Infer model size from task_id
    task_id = obs.get("task_id", "")
    if "localization" in task_id:
        n_layers, n_heads = 2, 4
    elif "hypothesis" in task_id:
        n_layers, n_heads = 4, 8

    # Try untried heads in layer-major order
    for l in range(n_layers):
        for h in range(n_heads):
            key = (l, h)
            if key not in tried_keys:
                return {"action_type": "ablate_head", "layer": l, "head": h}

    # All heads tried — submit best guess from history
    history = obs.get("experiment_history", [])
    candidates = sorted(
        [e for e in history if e.get("behavioral_delta", 0) < -0.1],
        key=lambda e: e.get("behavioral_delta", 0),
    )
    components = {}
    for c in candidates[:5]:
        l, h = c.get("layer"), c.get("head")
        if l is not None and h is not None:
            importance = min(1.0, abs(c["behavioral_delta"]))
            components[f"({l}, {h})"] = round(importance, 2)

    if not components:
        # Absolute fallback: guess head (0, 0)
        components = {"(0, 0)": 0.5}

    return {
        "action_type":  "submit_hypothesis",
        "circuit_mask": {"components": components},
    }


def action_to_str(data: dict) -> str:
    """Human-readable action string for log_step."""
    at = data.get("action_type", "unknown")
    l  = data.get("layer")
    h  = data.get("head")
    if at == "submit_hypothesis":
        mask = data.get("circuit_mask", {})
        comp = mask.get("components", {}) if isinstance(mask, dict) else {}
        return f"submit_hypothesis({list(comp.keys())})"
    if l is not None and h is not None:
        return f"{at}(L={l},H={h})"
    return at

# ── HTTP helpers (direct HTTP, no client package needed for inference) ─────────

async def http_reset(client: httpx.AsyncClient, task_id: str) -> dict:
    resp = await client.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()

async def http_step(client: httpx.AsyncClient, action_data: dict) -> dict:
    resp = await client.post("/step", json=action_data)
    resp.raise_for_status()
    return resp.json()

# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_task(
    llm:      OpenAI,
    task_cfg: dict,
) -> dict:
    task_id   = task_cfg["task_id"]
    max_steps = task_cfg["max_steps"]

    log_start(task_id, MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    async with httpx.AsyncClient(base_url=ENV_URL, timeout=60.0) as http:
        try:
            obs = await http_reset(http, task_id)

            # Conversation history for the LLM
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            tried_keys: set = set()

            for step in range(1, max_steps + 1):
                if obs.get("done", False):
                    break

                # Build user prompt from observation
                user_content = build_user_prompt(obs, task_id)
                messages.append({"role": "user", "content": user_content})

                # LLM call
                llm_response = ""
                llm_error    = None
                try:
                    completion = llm.chat.completions.create(
                        model       = MODEL_NAME,
                        messages    = messages,
                        temperature = 0.2,
                        max_tokens  = 300,
                    )
                    llm_response = (completion.choices[0].message.content or "").strip()
                    messages.append({"role": "assistant", "content": llm_response})
                except Exception as e:
                    llm_error    = str(e)[:60]
                    llm_response = ""

                # Parse action
                action_data = parse_action(llm_response, obs, messages) if llm_response else None
                if action_data is None:
                    action_data = fallback_action(obs, tried_keys)

                # Track tried heads
                if action_data.get("action_type") == "ablate_head":
                    l = action_data.get("layer")
                    h = action_data.get("head")
                    if l is not None and h is not None:
                        tried_keys.add((l, h))

                # Step the environment
                action_str = action_to_str(action_data)
                try:
                    obs = await http_step(http, action_data)
                except Exception as e:
                    llm_error = f"env_error:{str(e)[:40]}"
                    obs = {"done": False, "reward": 0.0, "action_result": {}, "step_number": step}

                reward     = float(obs.get("reward", 0.0))
                done       = bool(obs.get("done", False))
                rewards.append(reward)
                steps_taken = step

                log_step(step, action_str, reward, done, llm_error)

                # If episode ended via submit_hypothesis, extract score
                if done:
                    result = obs.get("action_result", {})
                    score  = float(result.get("circuit_f1", 0.0))
                    success = score >= 0.5
                    break

            # If budget exhausted without submitting, force submit
            if not obs.get("done", False):
                auto_action = fallback_action(obs, tried_keys)
                auto_str    = action_to_str(auto_action)
                try:
                    obs = await http_step(http, auto_action)
                except Exception:
                    obs = {"done": True, "reward": 0.0, "action_result": {}}

                reward  = float(obs.get("reward", 0.0))
                result  = obs.get("action_result", {})
                score   = float(result.get("circuit_f1", 0.0))
                success = score >= 0.5
                rewards.append(reward)
                steps_taken += 1
                log_step(steps_taken, auto_str, reward, True, None)

        except Exception as e:
            # Ensure [END] is always emitted even on unexpected failure
            pass

        finally:
            log_end(
                success = success,
                steps   = steps_taken,
                score   = score,
                rewards = rewards,
            )

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"# mechinterp-env baseline inference", flush=True)
    print(f"# model={MODEL_NAME}", flush=True)
    print(f"# env={ENV_URL}", flush=True)
    print("", flush=True)

    all_results = []
    for task_cfg in TASK_CONFIGS:
        result = await run_task(llm, task_cfg)
        all_results.append(result)
        print("", flush=True)   # blank line between tasks

    # Summary
    print("# === SUMMARY ===", flush=True)
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        print(
            f"# {status} {r['task_id']}: score={r['score']:.3f}, steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
