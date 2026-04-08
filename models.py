"""
Pydantic type definitions for mechinterp-env.
Shared between server (environment) and client (inference script).
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ActionType(str, Enum):
    ABLATE_HEAD        = "ablate_head"
    PATCH_ACTIVATION   = "patch_activation"
    QUERY_LOGIT_LENS   = "query_logit_lens"
    QUERY_ATTN_PATTERN = "query_attn_pattern"
    SUBMIT_HYPOTHESIS  = "submit_hypothesis"


class CircuitMask(BaseModel):
    """
    A proposed or ground-truth circuit.
    Maps "(layer, head)" string keys → importance score in [0, 1].
    Example: {"(0, 2)": 1.0, "(1, 1)": 0.6}
    """
    components: Dict[str, float] = Field(default_factory=dict)

    def top_head(self) -> Optional[Tuple[int, int]]:
        """Return the (layer, head) with highest importance score."""
        if not self.components:
            return None
        key = max(self.components, key=lambda k: self.components[k])
        try:
            parts = key.strip("()").split(",")
            return (int(parts[0].strip()), int(parts[1].strip()))
        except Exception:
            return None

    def heads_above(self, threshold: float = 0.3) -> set:
        """Return set of keys where importance >= threshold."""
        return {k for k, v in self.components.items() if v >= threshold}

    def top_layer(self) -> Optional[int]:
        head = self.top_head()
        return head[0] if head else None


class MechInterpAction(BaseModel):
    """
    One action the agent can take.

    action_type determines which fields are required:
      ablate_head        → layer, head
      patch_activation   → layer, head, position, source_prompt_id
      query_logit_lens   → layer, position
      query_attn_pattern → layer, head, prompt_id
      submit_hypothesis  → circuit_mask
    """
    action_type: ActionType

    # For ablate_head, patch_activation, query_attn_pattern
    layer: Optional[int] = None
    head:  Optional[int] = None

    # For patch_activation
    position:         Optional[int] = None
    source_prompt_id: Optional[int] = None

    # For query_logit_lens (also uses layer)
    # position is shared with patch_activation

    # For query_attn_pattern (also uses layer, head)
    prompt_id: Optional[int] = None

    # For submit_hypothesis
    circuit_mask: Optional[CircuitMask] = None


class ExperimentRecord(BaseModel):
    """One entry in experiment_history."""
    step:             int
    action_type:      str
    layer:            Optional[int]
    head:             Optional[int]
    behavioral_delta: float
    result_summary:   str


class MechInterpObservation(BaseModel):
    """
    Everything the agent sees after each step (or after reset).
    """
    # What happened
    action_taken:     str
    action_result:    Dict[str, Any]
    behavioral_delta: float

    # Task context
    target_behavior:  str
    task_id:          str
    step_number:      int
    steps_remaining:  int

    # Navigation aids
    available_actions:  List[str]
    experiment_history: List[Dict[str, Any]]
    prompt_pool:        List[Dict[str, Any]]

    # Episode terminal fields (required by OpenEnv spec)
    done:   bool  = False
    reward: float = 0.0


class EpisodeState(BaseModel):
    """Internal mutable state, one instance per episode."""
    episode_id:         str
    task_id:            str
    step_count:         int   = 0
    max_steps:          int   = 12
    ablated_heads:      List  = Field(default_factory=list)
    queried_heads:      List  = Field(default_factory=list)   # includes all ablate/patch queries
    experiment_history: List  = Field(default_factory=list)
    total_reward:       float = 0.0
    done:               bool  = False
