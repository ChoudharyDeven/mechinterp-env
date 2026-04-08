"""
Task definitions for mechinterp-env.
Each task specifies a model, prompt pool, ground truth circuit,
target behavior description, and (for task 1) prerun results.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Task:
    task_id:              str
    model_name:           str              # key into MODELS dict
    max_steps:            int
    difficulty:           str              # 'easy' | 'medium' | 'hard'
    target_behavior:      str
    prompt_pool:          List[Dict]
    ground_truth_circuit: Dict[str, float] # {str(layer, head): importance}
    baseline_token:       int              # token the model should predict
    corrupted_token:      int              # token for patching source
    prerun_results:       Dict             # only populated for task 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Head Identification (Easy)
# ─────────────────────────────────────────────────────────────────────────────
# Model: 1-layer, 4-head (d=32)
# Circuit: single copy head at (0, 2)
# Prerun ablation results are injected into the first observation.
# Agent just needs to read the results and submit.

TASK_1 = Task(
    task_id    = "head-identification",
    model_name = "model_1layer",
    max_steps  = 4,
    difficulty = "easy",
    target_behavior = (
        "This 1-layer transformer has 4 attention heads (H0, H1, H2, H3). "
        "It is trained to copy a key token from position 2 to the final position. "
        "Normally it assigns >0.7 probability to the correct token. "
        "Pre-run ablation results are provided in your first observation — "
        "they show behavioral_delta (change in target probability) when each head is ablated. "
        "Your job: identify which single head is responsible for this behavior and submit. "
        "Hint: the circuit head has behavioral_delta near -0.9 when ablated."
    ),
    prompt_pool = [
        {"id": 0, "tokens": [1, 2, 8, 4, 5, 6, 0], "text": "The color is RED . The color is [?]"},
        {"id": 1, "tokens": [1, 2, 9, 4, 5, 6, 0], "text": "The color is BLUE . The color is [?]"},
        {"id": 2, "tokens": [1, 2, 10, 4, 5, 6, 0], "text": "The color is GREEN . The color is [?]"},
    ],
    ground_truth_circuit = {"(0, 2)": 1.0},
    baseline_token  = 8,   # RED
    corrupted_token = 0,   # PAD (corrupt source for patching)
    prerun_results  = {
        # These are hardcoded to match what build_model_1layer() actually produces.
        # They will be verified and regenerated if model weights change.
        "ablate_(0, 0)": {
            "ablated_head":    "(0, 0)",
            "baseline_prob":   0.82,
            "ablated_prob":    0.80,
            "behavioral_delta": -0.02,
        },
        "ablate_(0, 1)": {
            "ablated_head":    "(0, 1)",
            "baseline_prob":   0.82,
            "ablated_prob":    0.81,
            "behavioral_delta": -0.01,
        },
        "ablate_(0, 2)": {
            "ablated_head":    "(0, 2)",
            "baseline_prob":   0.82,
            "ablated_prob":    0.05,
            "behavioral_delta": -0.77,   # Will be updated by verify_tasks.py
        },
        "ablate_(0, 3)": {
            "ablated_head":    "(0, 3)",
            "baseline_prob":   0.82,
            "ablated_prob":    0.80,
            "behavioral_delta": -0.02,
        },
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Circuit Localization (Medium)
# ─────────────────────────────────────────────────────────────────────────────
# Model: 2-layer, 4-head each (d=48)
# Circuit: induction circuit — L0H1 (prev-token) + L1H2 (induction)
# No prerun results. Agent must design and run its own experiments.
# Budget: 12 steps. Circuit has 2 heads; systematic ablation of 8 heads takes 8 steps.

TASK_2 = Task(
    task_id    = "circuit-localization",
    model_name = "model_2layer",
    max_steps  = 12,
    difficulty = "medium",
    target_behavior = (
        "This 2-layer transformer (4 heads per layer) completes repeating sequences. "
        "Given the pattern [A B C A B C A], it predicts B at the final position. "
        "This is an induction behavior implemented by two heads working in composition. "
        "NO pre-run results are provided — you must design and run your own ablation experiments. "
        "Strategy: ablate each head in each layer and measure behavioral_delta. "
        "Heads with delta < -0.2 are likely part of the circuit. "
        "There are exactly 2 circuit heads, one in each layer. "
        "Submit your circuit mask once you have identified them. "
        "Budget: 12 steps total."
    ),
    prompt_pool = [
        {"id": 0, "tokens": [3, 4, 5, 6, 7, 8, 1, 0], "text": "N N N N N N SIGNAL PAD [?]"},
        {"id": 1, "tokens": [3, 5, 4, 7, 6, 8, 1, 0], "text": "N N N N N N SIGNAL PAD [?]"},
        {"id": 2, "tokens": [4, 3, 6, 5, 8, 7, 1, 0], "text": "N N N N N N SIGNAL PAD [?]"},
    ],
    ground_truth_circuit = {"(0, 1)": 0.6, "(1, 2)": 1.0},
    baseline_token  = 2,   # tok2 (transcoded output of tok1)
    corrupted_token = 0,   # PAD (used as source for patching experiments)
    prerun_results  = {},  # Empty — agent runs its own experiments
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Full Hypothesis (Hard)
# ─────────────────────────────────────────────────────────────────────────────
# Model: 4-layer, 8-head (d=64)
# Circuit: 5 heads across 3 layers
# Search space: 4 × 8 = 32 heads. Budget: 20 steps.
# Agent cannot ablate all heads — must be strategic.

TASK_3 = Task(
    task_id    = "full-hypothesis",
    model_name = "model_4layer",
    max_steps  = 20,
    difficulty = "hard",
    target_behavior = (
        "This 4-layer transformer (8 heads per layer) completes subject-verb-object sentences. "
        "Given 'The chef cooked the [?]', it predicts food-related tokens with high probability. "
        "The circuit implementing this behavior spans multiple layers and involves 5 attention heads. "
        "Search space: 4 layers × 8 heads = 32 components total. "
        "NO pre-run results are provided. Budget: 20 steps — you cannot ablate every head. "
        "Strategic advice: "
        "  1. Start by ablating one head per layer to identify which layers matter. "
        "  2. Then drill into the important layers and ablate all heads there. "
        "  3. Look for heads with behavioral_delta < -0.25. "
        "  4. The circuit has heads in layers 0, 1, 2, and 3 (not all layers are equal). "
        "  5. Submit your best hypothesis before running out of steps. "
        "Scoring uses weighted F1 — partial credit for finding some circuit heads."
    ),
    prompt_pool = [
        {"id": 0, "tokens": [2, 6, 11, 2, 0],  "text": "The chef cooked the [?]"},
        {"id": 1, "tokens": [2, 7, 12, 2, 0],  "text": "The pilot flew the [?]"},
        {"id": 2, "tokens": [2, 8, 13, 2, 0],  "text": "The artist painted the [?]"},
    ],
    ground_truth_circuit = {
        "(0, 3)": 0.7,
        "(1, 1)": 0.8,
        "(1, 5)": 0.6,
        "(2, 2)": 1.0,
        "(3, 6)": 0.9,
    },
    baseline_token  = 16,  # food token
    corrupted_token = 0,   # PAD
    prerun_results  = {},  # Empty
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, Task] = {
    t.task_id: t for t in [TASK_1, TASK_2, TASK_3]
}
