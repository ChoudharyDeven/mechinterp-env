"""
Deterministic graders for all three mechinterp-env tasks.
All graders return a float strictly between 0 and 1 (exclusive).
No external API calls — fully deterministic.
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import CircuitMask
from server.tasks import Task


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) — never exactly 0.0 or 1.0."""
    return round(min(max(score, 0.001), 0.999), 4)


def grade_task(task: Task, hypothesis: Optional[CircuitMask]) -> float:
    """Dispatch to the correct grader based on task_id."""
    if hypothesis is None:
        return _clamp(0.0)
    if task.task_id == "head-identification":
        return grade_task1(task, hypothesis)
    elif task.task_id == "circuit-localization":
        return grade_task2(task, hypothesis)
    elif task.task_id == "full-hypothesis":
        return grade_task3(task, hypothesis)
    return _clamp(0.0)


def grade_task1(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 1: Identify the single responsible head.

    Scoring:
      ~1.0 — exact match (correct layer AND correct head)
      0.5  — correct layer, wrong head
      ~0.0 — wrong layer or no hypothesis
    """
    if not hypothesis.components:
        return _clamp(0.0)

    gt_key = list(task.ground_truth_circuit.keys())[0]
    try:
        gt_parts = gt_key.strip("()").split(",")
        gt_layer = int(gt_parts[0].strip())
        gt_head  = int(gt_parts[1].strip())
    except Exception:
        return _clamp(0.0)

    pred = hypothesis.top_head()
    if pred is None:
        return _clamp(0.0)

    pred_layer, pred_head = pred

    if pred_layer == gt_layer and pred_head == gt_head:
        return _clamp(1.0)
    if pred_layer == gt_layer:
        return _clamp(0.5)
    return _clamp(0.0)


def grade_task2(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 2: Identify the two-head induction circuit.
    Scoring: F1 on predicted vs true heads (threshold 0.3).
    """
    true_heads = set(task.ground_truth_circuit.keys())
    pred_heads = hypothesis.heads_above(threshold=0.3)

    if not true_heads and not pred_heads:
        return _clamp(1.0)
    if not pred_heads or not true_heads:
        return _clamp(0.0)

    tp = len(pred_heads & true_heads)
    precision = tp / len(pred_heads) if pred_heads else 0.0
    recall    = tp / len(true_heads) if true_heads else 0.0

    if precision + recall == 0:
        return _clamp(0.0)

    f1 = 2 * precision * recall / (precision + recall)
    return _clamp(f1)


def grade_task3(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 3: Identify a 5-head multi-layer circuit.
    Scoring: Weighted F1.
    """
    true_heads = set(task.ground_truth_circuit.keys())
    pred_heads = hypothesis.heads_above(threshold=0.3)

    if not true_heads and not pred_heads:
        return _clamp(1.0)
    if not pred_heads or not true_heads:
        return _clamp(0.0)

    correct = pred_heads & true_heads

    total_gt_importance = sum(task.ground_truth_circuit.values())
    correct_importance  = sum(task.ground_truth_circuit.get(h, 0.0) for h in correct)
    weighted_precision  = correct_importance / total_gt_importance if total_gt_importance > 0 else 0.0

    recall = len(correct) / len(true_heads) if true_heads else 0.0

    if weighted_precision + recall == 0:
        return _clamp(0.0)

    score = 2 * weighted_precision * recall / (weighted_precision + recall)
    return _clamp(score)

