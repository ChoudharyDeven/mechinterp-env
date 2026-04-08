"""
MechInterpEnvironment — the core OpenEnv environment class.

Implements reset(), step(), and state property per the OpenEnv spec.
All game logic lives here. No external dependencies except numpy.
"""

from __future__ import annotations
import uuid
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    MechInterpAction, MechInterpObservation, ActionType,
    EpisodeState, CircuitMask,
)
from server.tasks import TASKS, Task
from server.graders import grade_task
from server.rewards import compute_step_reward, compute_terminal_reward
from server import transformer as T

from data.models.model_1layer import build_model_1layer
from data.models.model_2layer import build_model_2layer
from data.models.model_4layer import build_model_4layer

# Load all models at startup — they're tiny, this is instant
MODELS = {
    "model_1layer": build_model_1layer(),
    "model_2layer": build_model_2layer(),
    "model_4layer": build_model_4layer(),
}


class MechInterpEnvironment:
    """
    OpenEnv-compliant environment for mechanistic interpretability debugging.

    Each episode:
      1. reset(task_id) — loads the task, returns initial observation
      2. step(action) × N — agent runs experiments, accumulates evidence
      3. step(submit_hypothesis) — grader scores the circuit, episode ends

    Internal state: EpisodeState (cleared on each reset)
    """

    def __init__(self):
        self._state: EpisodeState | None = None
        self._task:  Task | None         = None
        self._model: T.NumpyTransformer | None = None
        self._baseline: float = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "head-identification") -> MechInterpObservation:
        if task_id not in TASKS:
            task_id = "head-identification"

        self._task  = TASKS[task_id]
        self._model = MODELS[self._task.model_name]
        self._state = EpisodeState(
            episode_id = str(uuid.uuid4()),
            task_id    = task_id,
            max_steps  = self._task.max_steps,
        )

        # Compute baseline behavior metric
        self._baseline = self._measure_behavior(
            self._task.prompt_pool[0]["tokens"]
        )

        # For task 1: inject prerun ablation results into initial observation
        initial_result: dict = {}
        if self._task.prerun_results:
            # Recompute prerun results live to ensure consistency with weights
            live_prerun = self._compute_live_prerun()
            initial_result = {"prerun_ablation_results": live_prerun}

        return self._build_observation(
            reward           = 0.0,
            behavioral_delta = 0.0,
            action_result    = initial_result,
            action_desc      = (
                "INITIAL OBSERVATION — Pre-run ablation results provided below."
                if self._task.prerun_results
                else "INITIAL OBSERVATION — No pre-run results. Run your own experiments."
            ),
        )

    # ─────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────

    def step(self, action: MechInterpAction) -> MechInterpObservation:
        if self._state is None:
            # Safety: reset if somehow called before reset()
            self.reset()

        if self._state.done:
            return self._build_observation(0.0, 0.0, {}, "Episode already done.")

        self._state.step_count += 1
        action_result    = {}
        behavioral_delta = 0.0

        # ── Dispatch ──────────────────────────────────────────────────────
        if action.action_type == ActionType.ABLATE_HEAD:
            behavioral_delta, action_result = self._handle_ablate(action)

        elif action.action_type == ActionType.PATCH_ACTIVATION:
            behavioral_delta, action_result = self._handle_patch(action)

        elif action.action_type == ActionType.QUERY_LOGIT_LENS:
            behavioral_delta, action_result = self._handle_logit_lens(action)

        elif action.action_type == ActionType.QUERY_ATTN_PATTERN:
            behavioral_delta, action_result = self._handle_attn_pattern(action)

        elif action.action_type == ActionType.SUBMIT_HYPOTHESIS:
            return self._handle_submission(action)

        # ── Step reward ───────────────────────────────────────────────────
        step_reward = compute_step_reward(action, behavioral_delta, self._state)
        self._state.total_reward += step_reward

        # ── Log experiment ────────────────────────────────────────────────
        self._state.experiment_history.append({
            "step":             self._state.step_count,
            "action_type":      action.action_type.value,
            "layer":            action.layer,
            "head":             action.head,
            "behavioral_delta": round(behavioral_delta, 4),
            "result_summary":   self._summarize(action_result),
        })

        # ── Check budget exhaustion ───────────────────────────────────────
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        return self._build_observation(
            reward           = step_reward,
            behavioral_delta = behavioral_delta,
            action_result    = action_result,
            action_desc      = f"{action.action_type.value}(L={action.layer}, H={action.head})",
        )

    # ─────────────────────────────────────────────────────────────────────
    # state property (OpenEnv spec)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> dict:
        if self._state is None:
            return {"episode_id": "none", "step_count": 0}
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "task_id":    self._state.task_id,
            "done":       self._state.done,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────────────

    def _handle_ablate(self, action: MechInterpAction):
        l, h = action.layer or 0, action.head or 0
        l = min(l, len(self._model.layers) - 1)
        h = min(h, self._model.n_heads - 1)

        # Track for redundancy penalty
        self._state.queried_heads.append((l, h))
        self._state.ablated_heads.append((l, h))

        prompt_tokens = self._task.prompt_pool[0]["tokens"]
        ablated_cache = T.forward(self._model, prompt_tokens, ablated_heads={(l, h)})
        ablated_prob  = T.get_token_prob(ablated_cache, -1, self._task.baseline_token)
        delta         = ablated_prob - self._baseline

        return delta, {
            "ablated_head":     f"({l}, {h})",
            "baseline_prob":    round(self._baseline, 4),
            "ablated_prob":     round(ablated_prob, 4),
            "behavioral_delta": round(delta, 4),
            "interpretation": (
                "IMPORTANT — likely circuit component"  if delta < -0.3
                else "Possibly relevant"                 if delta < -0.1
                else "Likely NOT part of circuit"
            ),
        }

    def _handle_patch(self, action: MechInterpAction):
        l   = min(action.layer or 0, len(self._model.layers) - 1)
        h   = min(action.head  or 0, self._model.n_heads - 1)
        pos = action.position or 0
        src = action.source_prompt_id or 1
        src = min(src, len(self._task.prompt_pool) - 1)

        # Track for redundancy penalty
        self._state.queried_heads.append((l, h))

        base_tokens  = self._task.prompt_pool[0]["tokens"]
        src_tokens   = self._task.prompt_pool[src]["tokens"]

        # Get source activation (from corrupted prompt)
        src_cache = T.forward(self._model, src_tokens)
        src_attn  = np.array(src_cache["layers"][l]["attn_out"][h])
        src_val   = src_attn[min(pos, len(src_tokens) - 1)]

        # Run patched forward pass
        patch_key    = (l, h, min(pos, len(base_tokens) - 1))
        patched_cache = T.forward(self._model, base_tokens, patch_map={patch_key: src_val})
        patched_prob  = T.get_token_prob(patched_cache, -1, self._task.baseline_token)
        delta         = patched_prob - self._baseline

        return delta, {
            "patched_component":  f"({l}, {h}, pos={pos})",
            "source_prompt_id":   src,
            "baseline_prob":      round(self._baseline, 4),
            "patched_prob":       round(patched_prob, 4),
            "behavioral_delta":   round(delta, 4),
            "interpretation": (
                "IMPORTANT — activation at this position matters"  if abs(delta) > 0.2
                else "This position has limited effect"
            ),
        }

    def _handle_logit_lens(self, action: MechInterpAction):
        l   = min(action.layer or 0, len(self._model.layers))
        pos = action.position if action.position is not None else -1

        tokens = self._task.prompt_pool[0]["tokens"]
        pos    = pos if pos >= 0 else len(tokens) - 1
        pos    = min(pos, len(tokens) - 1)

        probs  = T.logit_lens(self._model, tokens, layer=l, position=pos)
        top5   = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:5]

        return 0.0, {
            "layer":    l,
            "position": pos,
            "top5_tokens": [
                {"token_id": tid, "probability": round(p, 4)}
                for tid, p in top5
            ],
            "note": f"Token {self._task.baseline_token} prob at this layer: {round(float(probs[self._task.baseline_token]), 4)}",
        }

    def _handle_attn_pattern(self, action: MechInterpAction):
        l   = min(action.layer or 0, len(self._model.layers) - 1)
        h   = min(action.head  or 0, self._model.n_heads - 1)
        pid = action.prompt_id or 0
        pid = min(pid, len(self._task.prompt_pool) - 1)

        tokens = self._task.prompt_pool[pid]["tokens"]
        cache  = T.forward(self._model, tokens)
        pattern = cache["layers"][l]["attn_patterns"][h]

        # Summarize: for each destination position, where does it attend most?
        summary = []
        if pattern:
            pat = np.array(pattern)
            for dst in range(len(tokens)):
                src = int(pat[dst].argmax())
                summary.append({
                    "dst_pos": dst,
                    "max_src_pos": src,
                    "max_attn": round(float(pat[dst, src]), 4),
                })

        return 0.0, {
            "layer":     l,
            "head":      h,
            "prompt_id": pid,
            "prompt_text": self._task.prompt_pool[pid]["text"],
            "attention_summary": summary,
            "full_pattern": pattern,   # (seq, seq) matrix
        }

    def _handle_submission(self, action: MechInterpAction) -> MechInterpObservation:
        mask = action.circuit_mask or CircuitMask(components={})
        score = grade_task(self._task, mask)

        terminal_reward = compute_terminal_reward(score, self._state)
        self._state.done          = True
        self._state.total_reward += terminal_reward

        return MechInterpObservation(
            action_taken      = "submit_hypothesis",
            action_result     = {
                "circuit_f1":         round(score, 4),
                "your_hypothesis":    mask.components,
                "ground_truth":       self._task.ground_truth_circuit,
                "correct_heads":      list(mask.heads_above(0.3) & set(self._task.ground_truth_circuit.keys())),
                "missed_heads":       list(set(self._task.ground_truth_circuit.keys()) - mask.heads_above(0.3)),
                "false_positives":    list(mask.heads_above(0.3) - set(self._task.ground_truth_circuit.keys())),
            },
            behavioral_delta  = 0.0,
            target_behavior   = self._task.target_behavior,
            task_id           = self._task.task_id,
            step_number       = self._state.step_count,
            steps_remaining   = 0,
            available_actions = [],
            experiment_history = list(self._state.experiment_history[-8:]),
            prompt_pool       = self._task.prompt_pool,
            done              = True,
            reward            = terminal_reward,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _measure_behavior(self, tokens: list) -> float:
        cache = T.forward(self._model, tokens)
        return T.get_token_prob(cache, -1, self._task.baseline_token)

    def _compute_live_prerun(self) -> dict:
        """Recompute prerun ablation results using actual model weights."""
        tokens   = self._task.prompt_pool[0]["tokens"]
        baseline = self._baseline
        results  = {}
        n_layers = len(self._model.layers)
        n_heads  = self._model.n_heads
        for l in range(n_layers):
            for h in range(n_heads):
                cache = T.forward(self._model, tokens, ablated_heads={(l, h)})
                prob  = T.get_token_prob(cache, -1, self._task.baseline_token)
                delta = prob - baseline
                key   = f"ablate_({l}, {h})"
                results[key] = {
                    "ablated_head":     f"({l}, {h})",
                    "baseline_prob":    round(baseline, 4),
                    "ablated_prob":     round(prob, 4),
                    "behavioral_delta": round(delta, 4),
                    "interpretation": (
                        "IMPORTANT"         if delta < -0.3
                        else "Possibly relevant" if delta < -0.1
                        else "Not in circuit"
                    ),
                }
        return results

    def _build_observation(
        self,
        reward:           float,
        behavioral_delta: float,
        action_result:    dict,
        action_desc:      str,
    ) -> MechInterpObservation:
        return MechInterpObservation(
            action_taken       = action_desc,
            action_result      = action_result,
            behavioral_delta   = behavioral_delta,
            target_behavior    = self._task.target_behavior,
            task_id            = self._task.task_id,
            step_number        = self._state.step_count,
            steps_remaining    = self._state.max_steps - self._state.step_count,
            available_actions  = self._available_actions(),
            experiment_history = list(self._state.experiment_history[-8:]),
            prompt_pool        = self._task.prompt_pool,
            done               = self._state.done,
            reward             = reward,
        )

    def _available_actions(self) -> list:
        n_l = len(self._model.layers)
        n_h = self._model.n_heads
        actions = []
        for l in range(n_l):
            for h in range(n_h):
                actions.append(f"ablate_head(layer={l}, head={h})")
                actions.append(f"query_attn_pattern(layer={l}, head={h}, prompt_id=0)")
        for l in range(n_l + 1):
            actions.append(f"query_logit_lens(layer={l}, position=-1)")
        actions.append("patch_activation(layer=L, head=H, position=P, source_prompt_id=1)")
        actions.append("submit_hypothesis(circuit_mask={components: {'(L, H)': importance, ...}})")
        return actions

    def _summarize(self, result: dict) -> str:
        if "behavioral_delta" in result:
            delta = result["behavioral_delta"]
            interp = result.get("interpretation", "")
            return f"delta={delta:.4f} — {interp}"
        if "top5_tokens" in result:
            top = result["top5_tokens"][:2]
            return f"logit_lens top tokens: {top}"
        if "attention_summary" in result:
            n = len(result.get("attention_summary", []))
            return f"attn_pattern: {n} positions"
        return str(result)[:80]
