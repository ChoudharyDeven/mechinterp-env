"""
Dense reward function for mechinterp-env.

Step-level rewards fire after every action (except submit_hypothesis).
Terminal rewards fire only on submit_hypothesis.

Design goals:
  - Random agent: ~0.05 average reward
  - Systematic but uninformed agent: ~0.30
  - Agent using good MI strategy: ~0.70+
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import MechInterpAction, ActionType, EpisodeState


# ─────────────────────────────────────────────────────────────────────────────
# Step-level reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    action:           MechInterpAction,
    behavioral_delta: float,
    state:            EpisodeState,
) -> float:
    """
    Compute reward for a single step (non-terminal action).

    Components:
      1. Information gain  — |behavioral_delta| × 0.3 for ablations/patches
      2. High-delta bonus  — +0.10 for finding a clearly important head
      3. Redundancy penalty — -0.10 for re-querying the same (layer, head)
      4. Efficiency penalty — -0.05 per step after step 8
    """
    reward = 0.0

    # 1. Information gain reward
    if action.action_type in (ActionType.ABLATE_HEAD, ActionType.PATCH_ACTIVATION):
        info_gain = abs(behavioral_delta) * 0.3
        reward += info_gain

    # 2. High-delta bonus: found a clearly important head
    if behavioral_delta < -0.35:
        reward += 0.10

    # 3. Redundancy penalty: ablating/patching the same head more than once
    if action.action_type in (ActionType.ABLATE_HEAD, ActionType.PATCH_ACTIVATION):
        key = (action.layer, action.head)
        # queried_heads already includes the current action (added before reward)
        times_seen = sum(1 for q in state.queried_heads if q == key)
        if times_seen > 1:
            reward -= 0.10

    # 4. Efficiency penalty after step 8
    if state.step_count > 8:
        reward -= 0.05

    return round(reward, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal reward
# ─────────────────────────────────────────────────────────────────────────────

def compute_terminal_reward(
    circuit_f1: float,
    state:      EpisodeState,
) -> float:
    """
    Compute reward on submit_hypothesis().

    Components:
      1. Circuit F1 × 2.0     — primary signal (0 to 2.0)
      2. Speed bonus +0.30    — if F1 >= 0.8 and steps <= half budget
      3. Attempt bonus +0.05  — small reward just for submitting (vs timing out)

    Normalized to [0, 1]:
      Max possible = (2.0 + 0.30 + 0.05) / 2.35 = 1.0
    """
    reward = 0.0

    # 1. Primary: circuit F1
    reward += circuit_f1 * 2.0

    # 2. Speed bonus: found the circuit efficiently
    half_budget = state.max_steps / 2
    if circuit_f1 >= 0.8 and state.step_count <= half_budget:
        reward += 0.30

    # 3. Attempt bonus: submitted something (vs running out of steps silently)
    reward += 0.05

    # Normalize to [0, 1]
    MAX_REWARD = 2.35
    normalized = reward / MAX_REWARD
    return round(min(max(normalized, 0.0), 1.0), 4)
