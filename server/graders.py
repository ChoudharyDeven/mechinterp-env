"""
Deterministic graders for all three mechinterp-env tasks.
All graders return a float in [0.0, 1.0].
No external API calls — fully deterministic.
"""

from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import CircuitMask
from server.tasks import Task


def grade_task(task: Task, hypothesis: Optional[CircuitMask]) -> float:
    """Dispatch to the correct grader based on task_id."""
    if hypothesis is None:
        return 0.0
    if task.task_id == "head-identification":
        return grade_task1(task, hypothesis)
    elif task.task_id == "circuit-localization":
        return grade_task2(task, hypothesis)
    elif task.task_id == "full-hypothesis":
        return grade_task3(task, hypothesis)
    return 0.0


def grade_task1(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 1: Identify the single responsible head.
    Ground truth: one head with importance 1.0.

    Scoring:
      1.0 — exact match (correct layer AND correct head)
      0.5 — correct layer, wrong head
      0.0 — wrong layer or no hypothesis
    """
    if not hypothesis.components:
        return 0.0

    # Ground truth is a single head
    gt_key = list(task.ground_truth_circuit.keys())[0]  # "(0, 2)"
    try:
        gt_parts = gt_key.strip("()").split(",")
        gt_layer = int(gt_parts[0].strip())
        gt_head  = int(gt_parts[1].strip())
    except Exception:
        return 0.0

    # Agent's top prediction
    pred = hypothesis.top_head()
    if pred is None:
        return 0.0

    pred_layer, pred_head = pred

    if pred_layer == gt_layer and pred_head == gt_head:
        return 1.0
    if pred_layer == gt_layer:
        return 0.5
    return 0.0


def grade_task2(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 2: Identify the two-head induction circuit.
    Ground truth: 2 heads with different importance scores.

    Scoring: F1 on the set of predicted vs true heads (threshold 0.3).
    Precision and recall equally weighted.
    """
    true_heads = set(task.ground_truth_circuit.keys())
    pred_heads = hypothesis.heads_above(threshold=0.3)

    if not true_heads and not pred_heads:
        return 1.0
    if not pred_heads or not true_heads:
        return 0.0

    tp = len(pred_heads & true_heads)
    precision = tp / len(pred_heads) if pred_heads else 0.0
    recall    = tp / len(true_heads) if true_heads else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(min(f1, 1.0), 4)


def grade_task3(task: Task, hypothesis: CircuitMask) -> float:
    """
    Task 3: Identify a 5-head multi-layer circuit.
    Ground truth: 5 heads with importance weights.

    Scoring: Weighted F1.
      - Weighted precision: sum of GT importance for correct predictions / total GT importance
      - Standard recall: correct predictions / num true heads
    This means identifying the most important heads (L2H2 at 1.0) matters more.
    """
    true_heads = set(task.ground_truth_circuit.keys())
    pred_heads = hypothesis.heads_above(threshold=0.3)

    if not true_heads and not pred_heads:
        return 1.0
    if not pred_heads or not true_heads:
        return 0.0

    correct = pred_heads & true_heads

    # Weighted precision: importance of correct heads / total importance of all true heads
    total_gt_importance  = sum(task.ground_truth_circuit.values())
    correct_importance   = sum(task.ground_truth_circuit.get(h, 0.0) for h in correct)
    weighted_precision   = correct_importance / total_gt_importance if total_gt_importance > 0 else 0.0

    # Standard recall: fraction of true heads found
    recall = len(correct) / len(true_heads) if true_heads else 0.0

    if weighted_precision + recall == 0:
        return 0.0

    score = 2 * weighted_precision * recall / (weighted_precision + recall)
    return round(min(score, 1.0), 4)
