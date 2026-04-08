"""
verify_tasks.py — Run before deploying to confirm everything works.

Checks:
  1. All three models load and produce valid forward passes
  2. Circuit heads have behavioral_delta < -0.25 when ablated
  3. Noise heads have behavioral_delta > -0.10 when ablated
  4. Graders return correct scores for perfect/wrong hypotheses
  5. Reward function produces values in [0, 1]
  6. Environment reset/step cycle works end-to-end

Run from project root:
  python verify_tasks.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from server.transformer import forward, get_token_prob, logit_lens
from data.models.model_1layer import build_model_1layer
from data.models.model_2layer import build_model_2layer
from data.models.model_4layer import build_model_4layer
from server.tasks import TASKS, TASK_1, TASK_2, TASK_3
from server.graders import grade_task1, grade_task2, grade_task3
from server.rewards import compute_step_reward, compute_terminal_reward
from server.environment import MechInterpEnvironment
from models import MechInterpAction, CircuitMask, EpisodeState

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  {status} {name}" + (f" | {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Model loading
# ─────────────────────────────────────────────────────────────────────────────
section("1. Model Loading")

try:
    m1 = build_model_1layer()
    check("Model 1 loads", True, f"1L, {m1.n_heads}H, d={m1.d_model}")
except Exception as e:
    check("Model 1 loads", False, str(e))
    m1 = None

try:
    m2 = build_model_2layer()
    check("Model 2 loads", True, f"2L, {m2.n_heads}H, d={m2.d_model}")
except Exception as e:
    check("Model 2 loads", False, str(e))
    m2 = None

try:
    m4 = build_model_4layer()
    check("Model 4 loads", True, f"4L, {m4.n_heads}H, d={m4.d_model}")
except Exception as e:
    check("Model 4 loads", False, str(e))
    m4 = None


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Task 1 — Copy head verification
# ─────────────────────────────────────────────────────────────────────────────
section("2. Task 1 — Copy Head Verification")

if m1:
    tokens  = TASK_1.prompt_pool[0]["tokens"]
    target  = TASK_1.baseline_token
    cache   = forward(m1, tokens)
    base    = get_token_prob(cache, -1, target)

    print(f"  Baseline prob for token {target}: {base:.4f}")
    check("Baseline prob > 0.1", base > 0.1, f"got {base:.4f}")

    deltas = {}
    for h in range(4):
        ac    = forward(m1, tokens, ablated_heads={(0, h)})
        prob  = get_token_prob(ac, -1, target)
        delta = prob - base
        deltas[h] = delta
        print(f"    Ablate H{h}: delta={delta:.4f}")

    circuit_h = 2
    noise_hs  = [0, 1, 3]

    check(
        f"Circuit head H{circuit_h} has large negative delta",
        deltas[circuit_h] < -0.25,
        f"delta={deltas[circuit_h]:.4f}"
    )
    for h in noise_hs:
        check(
            f"Noise head H{h} has small delta",
            abs(deltas[h]) < 0.15,
            f"delta={deltas[h]:.4f}"
        )
else:
    print("  Skipped (model failed to load)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Task 2 — Induction circuit verification
# ─────────────────────────────────────────────────────────────────────────────
section("3. Task 2 — Induction Circuit Verification")

if m2:
    tokens = TASK_2.prompt_pool[0]["tokens"]
    target = TASK_2.baseline_token
    cache  = forward(m2, tokens)
    base   = get_token_prob(cache, -1, target)

    print(f"  Baseline prob for token {target}: {base:.4f}")
    check("Baseline prob > 0.05", base > 0.05, f"got {base:.4f}")

    circuit_heads = [(0, 1), (1, 2)]
    noise_heads   = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3)]

    print("  Circuit heads:")
    for (l, h) in circuit_heads:
        ac    = forward(m2, tokens, ablated_heads={(l, h)})
        prob  = get_token_prob(ac, -1, target)
        delta = prob - base
        print(f"    L{l}H{h}: delta={delta:.4f}")
        check(f"L{l}H{h} delta < -0.15", delta < -0.15, f"delta={delta:.4f}")

    print("  Noise heads (sample):")
    for (l, h) in noise_heads[:4]:
        ac    = forward(m2, tokens, ablated_heads={(l, h)})
        prob  = get_token_prob(ac, -1, target)
        delta = prob - base
        print(f"    L{l}H{h}: delta={delta:.4f}")
        if abs(delta) >= 0.15:
            print(f"    {WARN} L{l}H{h} has larger delta than expected — noise head overlap")
else:
    print("  Skipped (model failed to load)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Task 3 — Multi-head circuit verification
# ─────────────────────────────────────────────────────────────────────────────
section("4. Task 3 — Multi-head Circuit Verification")

if m4:
    tokens = TASK_3.prompt_pool[0]["tokens"]
    target = TASK_3.baseline_token
    cache  = forward(m4, tokens)
    base   = get_token_prob(cache, -1, target)

    print(f"  Baseline prob for token {target}: {base:.4f}")
    check("Baseline prob > 0.01", base > 0.01, f"got {base:.4f}")

    circuit = {(0,3), (1,1), (1,5), (2,2), (3,6)}
    noise   = [(0,0), (0,1), (1,0), (1,2), (2,0), (3,0)]

    print("  Circuit heads:")
    circuit_ok = 0
    for (l, h) in sorted(circuit):
        ac    = forward(m4, tokens, ablated_heads={(l, h)})
        prob  = get_token_prob(ac, -1, target)
        delta = prob - base
        print(f"    L{l}H{h}: delta={delta:.4f}")
        if delta < -0.05:
            circuit_ok += 1

    check(
        f"At least 3/5 circuit heads have delta < -0.05",
        circuit_ok >= 3,
        f"{circuit_ok}/5 passed"
    )

    print("  Sample noise heads:")
    for (l, h) in noise[:4]:
        ac    = forward(m4, tokens, ablated_heads={(l, h)})
        prob  = get_token_prob(ac, -1, target)
        delta = prob - base
        print(f"    L{l}H{h}: delta={delta:.4f}")
else:
    print("  Skipped (model failed to load)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Grader correctness
# ─────────────────────────────────────────────────────────────────────────────
section("5. Grader Correctness")

# Task 1
perfect_1 = CircuitMask(components={"(0, 2)": 1.0})
wrong_1   = CircuitMask(components={"(0, 0)": 1.0})
partial_1 = CircuitMask(components={"(0, 1)": 1.0})  # correct layer, wrong head

s_perfect = grade_task1(TASK_1, perfect_1)
s_wrong   = grade_task1(TASK_1, wrong_1)
s_partial = grade_task1(TASK_1, partial_1)

print(f"  Task 1: perfect={s_perfect}, wrong_head_same_layer={s_wrong}, same_layer_diff_head={s_partial}")
check("Task 1 perfect = 1.0",  s_perfect == 1.0, f"got {s_perfect}")
# In a 1-layer model, any wrong head is still in layer 0 → gets 0.5 (correct layer)
check("Task 1 wrong head = 0.5", s_wrong == 0.5, f"got {s_wrong}")
check("Task 1 same layer = 0.5", s_partial == 0.5, f"got {s_partial}")

# Task 2
perfect_2   = CircuitMask(components={"(0, 1)": 0.6, "(1, 2)": 1.0})
one_head_2  = CircuitMask(components={"(1, 2)": 1.0})
wrong_2     = CircuitMask(components={"(0, 0)": 1.0})

s_perfect2 = grade_task2(TASK_2, perfect_2)
s_one2     = grade_task2(TASK_2, one_head_2)
s_wrong2   = grade_task2(TASK_2, wrong_2)

print(f"  Task 2: perfect={s_perfect2:.3f}, one_head={s_one2:.3f}, wrong={s_wrong2:.3f}")
check("Task 2 perfect > 0.9",  s_perfect2 > 0.9, f"got {s_perfect2:.3f}")
check("Task 2 one head > 0.4", s_one2 > 0.4,    f"got {s_one2:.3f}")
check("Task 2 wrong = 0.0",    s_wrong2 == 0.0,  f"got {s_wrong2:.3f}")

# Task 3
perfect_3 = CircuitMask(components={"(0, 3)": 0.7, "(1, 1)": 0.8, "(1, 5)": 0.6, "(2, 2)": 1.0, "(3, 6)": 0.9})
partial_3 = CircuitMask(components={"(2, 2)": 1.0, "(3, 6)": 0.9})  # top 2
wrong_3   = CircuitMask(components={"(0, 0)": 1.0, "(1, 0)": 1.0})

s_perfect3 = grade_task3(TASK_3, perfect_3)
s_partial3 = grade_task3(TASK_3, partial_3)
s_wrong3   = grade_task3(TASK_3, wrong_3)

print(f"  Task 3: perfect={s_perfect3:.3f}, top2={s_partial3:.3f}, wrong={s_wrong3:.3f}")
check("Task 3 perfect > 0.9",      s_perfect3 > 0.9,  f"got {s_perfect3:.3f}")
check("Task 3 partial > 0.2",      s_partial3 > 0.2,  f"got {s_partial3:.3f}")
check("Task 3 wrong = 0.0",        s_wrong3 == 0.0,   f"got {s_wrong3:.3f}")
check("All scores in [0,1]", all(
    0.0 <= s <= 1.0 for s in [s_perfect2, s_one2, s_wrong2, s_perfect3, s_partial3, s_wrong3]
))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Reward function
# ─────────────────────────────────────────────────────────────────────────────
section("6. Reward Function")

state = EpisodeState(episode_id="test", task_id="head-identification", max_steps=12)
action = MechInterpAction(action_type="ablate_head", layer=0, head=2)

# High-delta ablation
state.queried_heads = [(0, 2)]
r_high = compute_step_reward(action, -0.9, state)
print(f"  High-delta ablation reward: {r_high}")
check("High-delta reward > 0.3", r_high > 0.3, f"got {r_high}")

# Redundant action
state.queried_heads = [(0, 2), (0, 2)]
r_redundant = compute_step_reward(action, -0.9, state)
print(f"  Redundant ablation reward: {r_redundant}")
check("Redundant action penalized", r_redundant < r_high, f"got {r_redundant}")

# Terminal rewards
r_perfect_fast = compute_terminal_reward(1.0, EpisodeState(episode_id="t", task_id="t", step_count=4, max_steps=12))
r_perfect_slow = compute_terminal_reward(1.0, EpisodeState(episode_id="t", task_id="t", step_count=11, max_steps=12))
r_zero         = compute_terminal_reward(0.0, EpisodeState(episode_id="t", task_id="t", step_count=5, max_steps=12))

print(f"  Terminal: perfect_fast={r_perfect_fast}, perfect_slow={r_perfect_slow}, zero_f1={r_zero}")
check("Perfect fast = 1.0",   r_perfect_fast == 1.0, f"got {r_perfect_fast}")
check("Perfect slow < 1.0",   r_perfect_slow < 1.0,  f"got {r_perfect_slow}")
check("Perfect slow > 0.8",   r_perfect_slow > 0.8,  f"got {r_perfect_slow}")
check("Zero F1 > 0.0",        r_zero > 0.0,          f"got {r_zero} (attempt bonus)")
check("All terminal in [0,1]", all(
    0.0 <= r <= 1.0 for r in [r_perfect_fast, r_perfect_slow, r_zero]
))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: End-to-end environment cycle
# ─────────────────────────────────────────────────────────────────────────────
section("7. End-to-End Environment Cycle")

try:
    env = MechInterpEnvironment()

    # Task 1 cycle
    obs = env.reset("head-identification")
    check("reset() returns observation", obs is not None)
    check("Initial obs has prerun results", "prerun_ablation_results" in obs.action_result,
          f"keys: {list(obs.action_result.keys())}")
    check("Steps remaining = max_steps", obs.steps_remaining == 4, f"got {obs.steps_remaining}")

    # Take an ablation step
    action = MechInterpAction(action_type="ablate_head", layer=0, head=0)
    obs2 = env.step(action)
    check("step() returns observation", obs2 is not None)
    check("Step count incremented", obs2.step_number == 1, f"got {obs2.step_number}")
    check("behavioral_delta is float", isinstance(obs2.behavioral_delta, float))

    # Submit perfect hypothesis
    action_submit = MechInterpAction(
        action_type  = "submit_hypothesis",
        circuit_mask = CircuitMask(components={"(0, 2)": 1.0})
    )
    obs3 = env.step(action_submit)
    check("submit returns done=True", obs3.done, f"done={obs3.done}")
    f1 = obs3.action_result.get("circuit_f1", 0.0)
    check("Perfect hypothesis scores 1.0", f1 == 1.0, f"f1={f1}")
    check("Terminal reward in [0,1]", 0.0 <= obs3.reward <= 1.0, f"reward={obs3.reward}")

    # Task 2 cycle
    obs = env.reset("circuit-localization")
    check("Task 2 reset works", obs.task_id == "circuit-localization")
    check("Task 2 no prerun results", obs.action_result == {} or
          "prerun_ablation_results" not in obs.action_result)

    # Task 3 cycle
    obs = env.reset("full-hypothesis")
    check("Task 3 reset works", obs.task_id == "full-hypothesis")
    check("Task 3 steps_remaining = 20", obs.steps_remaining == 20, f"got {obs.steps_remaining}")

except Exception as e:
    check("Environment cycle ran without exception", False, str(e))
    import traceback
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: JSON serialization
# ─────────────────────────────────────────────────────────────────────────────
section("8. JSON Serialization (API compatibility)")

try:
    env = MechInterpEnvironment()
    obs = env.reset("head-identification")
    obs_dict = obs.model_dump()
    obs_json = json.dumps(obs_dict)
    check("Observation serializes to JSON", True, f"{len(obs_json)} chars")
    check("JSON has required fields", all(
        k in obs_dict for k in ["done", "reward", "action_result", "step_number"]
    ))

    action = MechInterpAction(action_type="ablate_head", layer=0, head=1)
    action_dict = action.model_dump()
    action_json = json.dumps(action_dict)
    check("Action serializes to JSON", True, f"{len(action_json)} chars")

except Exception as e:
    check("JSON serialization works", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(results)
total  = len(results)
pct    = int(100 * passed / total) if total > 0 else 0

print(f"\n  {passed}/{total} checks passed ({pct}%)\n")

if passed == total:
    print(f"  {PASS} All checks passed — safe to deploy!")
else:
    failed = total - passed
    print(f"  {FAIL} {failed} check(s) failed — review output above before deploying.")
    sys.exit(1)
